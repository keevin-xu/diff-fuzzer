//! Properties of a case that a trigger rule might key on.
//!
//! # What a feature is, and what it deliberately is not
//!
//! A **feature** is a boolean property computable from a case **alone** — never from the
//! results of running it. That restriction is the whole point. A rule computed from
//! outputs is just another description of the *symptom*, which is what `signature.rs`
//! already does. A rule computed from the input makes a falsifiable claim about **cases
//! that do not exist yet**, and can therefore be tested by generating them.
//!
//! # Why hand-written rather than learned
//!
//! The vocabulary is the only a-priori artifact in `PHASE-7B` and the only place new domain
//! insight enters. Predicates are *derived* from data; features are *written*, because
//! naming what might matter is the part that requires knowing floating-point arithmetic.
//!
//! # Bit order is part of the on-disk format
//!
//! A [`Predicate`](crate::predicate) is a bitmask over [`FEATURES`]. **Appending is safe;
//! reordering silently invalidates every recorded predicate** — the mask would still match,
//! just against different meanings. Nothing would error. A test in `predicate.rs` guards
//! this by rebuilding a known predicate from a case that produces it.

use crate::input::{TensorOp, TensorValue};

/// The vocabulary. **Index is bit position, and that mapping is durable — append only.**
///
/// Seventeen features: eight describing *values*, nine describing *shape*. The split
/// matters because they answer different questions — what the numbers are, versus which
/// kernel the shape will select.
pub const FEATURES: [&str; 20] = [
    // --- value features: standard floating-point failure modes ---
    "overflow_product",
    "mixed_sign_overflow",
    "partial_sum_overflow",
    "cancellation",
    "subnormal_present",
    "input_special",
    "zero_present",
    "magnitude_ratio_extreme",
    // --- shape features: proxies for which kernel runs ---
    "rank_ge_3",
    "m_eq_1",
    "n_eq_1",
    "output_is_vector",
    "k_large",
    "degenerate_dim",
    "all_same_sign",
    "m_not_multiple_of_tile",
    "n_not_multiple_of_tile",
    // --- broadcast features (PHASE-7C): which shape-inference path an elementwise case
    //     takes. Added *before* any broadcast findings exist, so they cannot be fitted to
    //     what they will be scored on — the condition every previous search here has failed.
    "broadcast_present",
    "broadcast_whole_operand",
    "broadcast_both_operands",
];

/// A compile-time guard on the vocabulary size.
///
/// `FeatureVec` is a `u32`, so bit 32 would silently shift out and every predicate mentioning
/// it would match nothing while looking perfectly reasonable. Widening to `u64` is a
/// deliberate decision — the search space doubles per bit — and must be made on purpose
/// rather than discovered from a rule that mysteriously never fires.
const _: () = assert!(
    FEATURES.len() <= 32,
    "FEATURES exceeds the bits in FeatureVec; widen it to u64 deliberately"
);

/// One case's features, one bit each.
///
/// A bitmask rather than a set, so matching a predicate is a single `AND` — chosen because
/// it is trivially explainable, not because seventeen booleans need optimising.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureVec(pub u32);

impl FeatureVec {
    /// Whether the named feature holds. Returns `false` for an unknown name rather than
    /// panicking, since names arrive from recorded predicates that may predate a rename.
    pub fn has(&self, name: &str) -> bool {
        match FEATURES.iter().position(|f| *f == name) {
            Some(bit) => self.0 & (1 << bit) != 0,
            None => false,
        }
    }

    /// The features that hold, by name — for reports, where a bitmask means nothing.
    pub fn names(&self) -> Vec<&'static str> {
        FEATURES
            .iter()
            .enumerate()
            .filter(|(bit, _)| self.0 & (1 << bit) != 0)
            .map(|(_, name)| *name)
            .collect()
    }

    fn set(&mut self, name: &str) {
        if let Some(bit) = FEATURES.iter().position(|f| *f == name) {
            self.0 |= 1 << bit;
        }
    }
}

// --- thresholds -------------------------------------------------------------------
//
// Every constant here is recorded in `DECISIONS.md` with its rationale. **None is tuned
// per finding** — that is the fitting-to-data error the tolerance policy exists to prevent,
// and it would be no less wrong applied to a feature than to a bound.

/// Above this, a product of two values can exceed `f32`'s range.
const F32_MAX: f64 = f32::MAX as f64;

/// A magnitude spread this wide is where cancellation and accumulation-order effects live.
///
/// Twelve decades: chosen because `f32` carries roughly seven decimal digits, so a spread
/// beyond that guarantees the smaller terms cannot influence the sum at all.
const EXTREME_RATIO: f64 = 1e12;

/// A contraction dimension above this is "large" — enough terms for accumulation order to
/// matter, and enough to cross a blocking threshold in most kernels.
const LARGE_K: usize = 32;

/// A sum this much smaller than its largest term has suffered catastrophic cancellation.
///
/// The ratio is `1e-6`, roughly `f32`'s precision: below it, the sum's leading digits have
/// been cancelled away and what remains is rounding error.
const CANCELLATION_RATIO: f64 = 1e-6;

/// Micro-kernel tile dimensions for libtorch's `f32` GEMM on this machine.
///
/// **Measured, not assumed** — `SPECS.md` §3.1. The number of output elements that disagree
/// with a uniformly-fusing implementation is exactly `(m mod 4) × (n mod 8)`, which
/// predicted every shape tested.
///
/// **These are the only backend-specific constants in the vocabulary**, and they earn their
/// place: they are the confirmed root cause of the project's one real finding. The design
/// document listed tile alignment under "candidates not yet included, to be added only when
/// a finding demands one" — a finding demanded one.
const TILE_ROWS: usize = 4;
const TILE_COLS: usize = 8;

/// Compute every feature of a case.
///
/// **Reads the case only.** Nothing here runs a backend or inspects an output.
pub fn extract(case: &TensorOp) -> FeatureVec {
    let mut features = FeatureVec::default();

    value_features(case, &mut features);
    shape_features(case, &mut features);

    features
}

/// Properties of the numbers themselves.
fn value_features(case: &TensorOp, features: &mut FeatureVec) {
    let operands = operands(case);
    let all: Vec<f32> = operands
        .iter()
        .flat_map(|o| o.data().iter().copied())
        .collect();

    if all.iter().any(|v| !v.is_finite()) {
        features.set("input_special");
    }
    // `contains` rather than `any`: -0.0 == 0.0, so both signs of zero are caught, which
    // is what this feature means.
    if all.contains(&0.0) {
        features.set("zero_present");
    }
    if all.iter().any(|v| *v != 0.0 && v.abs() < f32::MIN_POSITIVE) {
        features.set("subnormal_present");
    }

    let finite: Vec<f64> = all
        .iter()
        .filter(|v| v.is_finite() && **v != 0.0)
        .map(|v| v.abs() as f64)
        .collect();
    if let (Some(smallest), Some(largest)) = (
        finite.iter().cloned().reduce(f64::min),
        finite.iter().cloned().reduce(f64::max),
    ) && largest / smallest > EXTREME_RATIO
    {
        features.set("magnitude_ratio_extreme");
    }

    if operands
        .iter()
        .any(|o| o.data().iter().all(|v| *v >= 0.0) || o.data().iter().all(|v| *v <= 0.0))
    {
        features.set("all_same_sign");
    }

    // The dot-product features only mean anything where dot products happen.
    if let TensorOp::Matmul { lhs, rhs } = case {
        dot_product_features(lhs, rhs, features);
    }
    if let TensorOp::Reduce { arg, .. } = case {
        accumulation_features(arg.data(), features);
    }
}

/// Features that require walking each dot product individually.
///
/// Separated because they are the expensive ones and only `matmul` has them — and because
/// **`mixed_sign_overflow` is per dot product, not per case**. A case containing one
/// positively-overflowing product and one negatively-overflowing product in *different*
/// output elements is not the same thing at all, and conflating them was an early error
/// this project made in prose before catching it in measurement.
fn dot_product_features(lhs: &TensorValue, rhs: &TensorValue, features: &mut FeatureVec) {
    let (ls, rs) = (lhs.shape(), rhs.shape());
    if ls.len() < 2 || rs.len() < 2 {
        return;
    }
    let (m, k) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let n = rs[rs.len() - 1];
    let batch: usize = ls[..ls.len() - 2].iter().product();
    let (lhs_stride, rhs_stride) = (m * k, k * n);

    for b in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut positive_overflow = false;
                let mut negative_overflow = false;
                let mut running = 0.0f64;
                let mut largest_term = 0.0f64;

                for t in 0..k {
                    let a = lhs.data()[b * lhs_stride + i * k + t] as f64;
                    let c = rhs.data()[b * rhs_stride + t * n + j] as f64;
                    let product = a * c;

                    if product > F32_MAX {
                        positive_overflow = true;
                    } else if product < -F32_MAX {
                        negative_overflow = true;
                    }
                    largest_term = largest_term.max(product.abs());

                    running += product;
                    if running.abs() > F32_MAX {
                        features.set("partial_sum_overflow");
                    }
                }

                if positive_overflow || negative_overflow {
                    features.set("overflow_product");
                }
                // Both signs **within one dot product** — the condition that makes
                // `inf + (-inf)` reachable, and the reason a per-case check would be wrong.
                if positive_overflow && negative_overflow {
                    features.set("mixed_sign_overflow");
                }
                if largest_term > 0.0 && running.abs() / largest_term < CANCELLATION_RATIO {
                    features.set("cancellation");
                }
            }
        }
    }
}

/// The same accumulation properties, for a reduction rather than a dot product.
fn accumulation_features(values: &[f32], features: &mut FeatureVec) {
    let mut running = 0.0f64;
    let mut largest = 0.0f64;

    for value in values {
        running += *value as f64;
        largest = largest.max((*value as f64).abs());
        if running.abs() > F32_MAX {
            features.set("partial_sum_overflow");
        }
    }
    if largest > 0.0 && running.abs() / largest < CANCELLATION_RATIO {
        features.set("cancellation");
    }
}

/// Properties of the shape — proxies for which kernel the backend will select.
/// Which broadcasting an elementwise case does, if any.
///
/// Separate from `degenerate_dim`, which fires whenever *any* operand has an extent of 1 —
/// including when both do and nothing stretches. These say something narrower and more
/// useful: that a backend had to **reuse** an operand's elements, which is a different code
/// path from an ordinary elementwise loop.
fn broadcast_features(case: &TensorOp, features: &mut FeatureVec) {
    let TensorOp::Binary { lhs, rhs, .. } = case else {
        return;
    };
    let Some(result) = crate::ops::broadcast::result_shape(lhs.shape(), rhs.shape()) else {
        return;
    };

    let lhs_stretches = lhs.shape() != result.as_slice();
    let rhs_stretches = rhs.shape() != result.as_slice();

    if lhs_stretches || rhs_stretches {
        features.set("broadcast_present");
    }
    // One operand is a single element stretched across the whole result — the extreme case,
    // and the one most likely to take a scalar fast path rather than a general one.
    if (lhs_stretches && lhs.data().len() == 1) || (rhs_stretches && rhs.data().len() == 1) {
        features.set("broadcast_whole_operand");
    }
    // Both sides stretch, on different axes — neither operand has the result's shape, so no
    // backend can simply loop over one of them.
    if lhs_stretches && rhs_stretches {
        features.set("broadcast_both_operands");
    }
}

fn shape_features(case: &TensorOp, features: &mut FeatureVec) {
    let operands = operands(case);
    if operands.iter().any(|o| o.rank() >= 3) {
        features.set("rank_ge_3");
    }
    if operands.iter().any(|o| o.shape().contains(&1)) {
        features.set("degenerate_dim");
    }

    broadcast_features(case, features);

    let TensorOp::Matmul { lhs, rhs } = case else {
        return;
    };
    let (ls, rs) = (lhs.shape(), rhs.shape());
    if ls.len() < 2 || rs.len() < 2 {
        return;
    }
    let (m, k) = (ls[ls.len() - 2], ls[ls.len() - 1]);
    let n = rs[rs.len() - 1];

    if m == 1 {
        features.set("m_eq_1");
    }
    if n == 1 {
        features.set("n_eq_1");
    }
    if m == 1 && n == 1 {
        features.set("output_is_vector");
    }
    if k > LARGE_K {
        features.set("k_large");
    }
    // The tile-remainder condition: `SPECS.md` §3.1 measured that disagreeing elements
    // number `(m mod 4) × (n mod 8)`, so **both** must be non-zero for the corner to exist.
    // Recorded as two features rather than one so the search can discover the conjunction
    // rather than having it assumed.
    if !m.is_multiple_of(TILE_ROWS) {
        features.set("m_not_multiple_of_tile");
    }
    if !n.is_multiple_of(TILE_COLS) {
        features.set("n_not_multiple_of_tile");
    }
}

fn operands(case: &TensorOp) -> Vec<&TensorValue> {
    match case {
        TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => vec![arg],
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => vec![lhs, rhs],
    }
}

#[cfg(test)]
mod tests {
    /// The three broadcast features, on cases built to isolate each.
    #[test]
    fn broadcast_features_distinguish_the_shape_of_the_stretch() {
        use crate::input::BinaryOp;

        let build = |ls: &[usize], rs: &[usize]| {
            let l = TensorValue::new(ls.to_vec(), vec![1.0; ls.iter().product()]);
            let r = TensorValue::new(rs.to_vec(), vec![1.0; rs.iter().product()]);
            extract(&TensorOp::binary(BinaryOp::Add, l, r))
        };

        // No stretch: equal shapes.
        let equal = build(&[3, 4], &[3, 4]);
        assert!(!equal.has("broadcast_present"));

        // One axis stretched on one side.
        let one_axis = build(&[3, 1], &[3, 4]);
        assert!(one_axis.has("broadcast_present"));
        assert!(!one_axis.has("broadcast_whole_operand"));
        assert!(!one_axis.has("broadcast_both_operands"));

        // A single element stretched across the whole result.
        let scalar = build(&[1, 1], &[3, 4]);
        assert!(scalar.has("broadcast_present"));
        assert!(scalar.has("broadcast_whole_operand"));

        // Both sides stretch, on different axes — neither operand has the result's shape.
        let both = build(&[3, 1], &[1, 4]);
        assert!(both.has("broadcast_present"));
        assert!(both.has("broadcast_both_operands"));
        assert!(!both.has("broadcast_whole_operand"));
    }

    /// **`degenerate_dim` is not a broadcast test**, and conflating them would make the new
    /// features redundant. A `[3,1] + [3,1]` case has an extent of 1 and stretches nothing.
    #[test]
    fn an_extent_of_one_on_both_sides_is_not_broadcasting() {
        use crate::input::BinaryOp;
        let v = |s: &[usize]| TensorValue::new(s.to_vec(), vec![1.0; s.iter().product()]);
        let features = extract(&TensorOp::binary(BinaryOp::Add, v(&[3, 1]), v(&[3, 1])));

        assert!(features.has("degenerate_dim"));
        assert!(!features.has("broadcast_present"));
    }

    use super::*;
    use crate::input::UnaryOp;

    fn value(shape: &[usize], data: &[f32]) -> TensorValue {
        TensorValue::new(shape.to_vec(), data.to_vec())
    }

    fn matmul(lhs: (&[usize], &[f32]), rhs: (&[usize], &[f32])) -> TensorOp {
        TensorOp::matmul(value(lhs.0, lhs.1), value(rhs.0, rhs.1))
    }

    fn unary(shape: &[usize], data: &[f32]) -> TensorOp {
        TensorOp::unary(UnaryOp::Neg, value(shape, data))
    }

    /// **The property the whole phase depends on.** A feature computed from results would
    /// describe the symptom, which `signature.rs` already does. This cannot be asserted
    /// directly — `extract` takes only a case, so the type system enforces it — but the
    /// test states the intent so a future change has to break it deliberately.
    #[test]
    fn extraction_reads_only_the_case() {
        let case = unary(&[2], &[1.0, 2.0]);
        assert_eq!(extract(&case), extract(&case.clone()));
    }

    #[test]
    fn feature_extraction_is_deterministic() {
        let case = matmul((&[1, 2], &[1e30, -1e30]), (&[2, 1], &[1e30, 1e30]));
        assert_eq!(extract(&case), extract(&case));
    }

    /// Bit positions are the on-disk format; a duplicate name would make `has` ambiguous.
    #[test]
    fn feature_names_are_unique_and_fit_the_mask() {
        let mut names = FEATURES.to_vec();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "a feature name appears twice");
        assert!(count <= 32, "FeatureVec is a u32");
    }

    // --- value features ---------------------------------------------------------------

    /// The burn#5284 case: both products overflow, with opposite signs, in one dot product.
    #[test]
    fn the_filed_overflow_case_has_the_features_that_describe_it() {
        let f = extract(&matmul((&[1, 2], &[1e30, -1e30]), (&[2, 1], &[1e30, 1e30])));

        assert!(f.has("overflow_product"));
        assert!(f.has("mixed_sign_overflow"));
        assert!(f.has("output_is_vector"), "1x1 output");
        assert!(f.has("m_eq_1") && f.has("n_eq_1"));
    }

    /// **Mixed sign is per dot product, not per case.** Two overflows of opposite sign in
    /// *different* output elements are not the condition that makes `inf + (-inf)`
    /// reachable, and treating them as equivalent would make the feature meaningless.
    #[test]
    fn opposite_overflows_in_different_output_elements_are_not_mixed_sign() {
        // Two separate 1-term dot products: one overflows positive, the other negative.
        let f = extract(&matmul((&[2, 1], &[1e30, -1e30]), (&[1, 1], &[1e30])));

        assert!(f.has("overflow_product"), "both do overflow");
        assert!(
            !f.has("mixed_sign_overflow"),
            "but never within one dot product"
        );
    }

    #[test]
    fn subnormals_and_specials_are_distinguished() {
        assert!(extract(&unary(&[1], &[1e-45])).has("subnormal_present"));
        assert!(!extract(&unary(&[1], &[1e-45])).has("input_special"));

        assert!(extract(&unary(&[1], &[f32::NAN])).has("input_special"));
        assert!(!extract(&unary(&[1], &[f32::NAN])).has("subnormal_present"));
    }

    /// Zero is not subnormal — conflating them would fire on a large share of every
    /// campaign for no reason.
    #[test]
    fn zero_is_not_subnormal() {
        let f = extract(&unary(&[2], &[0.0, 1.0]));
        assert!(f.has("zero_present"));
        assert!(!f.has("subnormal_present"));
    }

    #[test]
    fn an_extreme_magnitude_spread_is_detected() {
        assert!(extract(&unary(&[2], &[1e15, 1e-3])).has("magnitude_ratio_extreme"));
        assert!(!extract(&unary(&[2], &[100.0, 1.0])).has("magnitude_ratio_extreme"));
    }

    /// Cancellation is about the *sum*, not the terms: large values summing to near zero.
    #[test]
    fn catastrophic_cancellation_is_detected() {
        let f = extract(&TensorOp::reduce(
            crate::input::ReduceOp::Sum,
            value(&[2], &[1e10, -1e10]),
            0,
        ));
        assert!(f.has("cancellation"));

        let ordinary = extract(&TensorOp::reduce(
            crate::input::ReduceOp::Sum,
            value(&[2], &[1.0, 2.0]),
            0,
        ));
        assert!(!ordinary.has("cancellation"));
    }

    // --- shape features ---------------------------------------------------------------

    /// The tile-remainder condition from `SPECS.md` §3.1. Both dimensions must leave a
    /// remainder for the trailing corner to exist — `(m mod 4) × (n mod 8)`.
    #[test]
    fn tile_alignment_matches_the_measured_remainder_rule() {
        let ones = |n: usize| vec![1.0f32; n];

        // 14 mod 4 = 2, 27 mod 8 = 3 — the case that diverges.
        let corner = extract(&matmul((&[14, 4], &ones(56)), (&[4, 27], &ones(108))));
        assert!(corner.has("m_not_multiple_of_tile"));
        assert!(corner.has("n_not_multiple_of_tile"));

        // 16 mod 4 = 0, 32 mod 8 = 0 — the case that agrees.
        let aligned = extract(&matmul((&[16, 4], &ones(64)), (&[4, 32], &ones(128))));
        assert!(!aligned.has("m_not_multiple_of_tile"));
        assert!(!aligned.has("n_not_multiple_of_tile"));

        // 17 mod 4 = 1, 32 mod 8 = 0 — one remainder is not enough; measured to agree.
        let half = extract(&matmul((&[17, 4], &ones(68)), (&[4, 32], &ones(128))));
        assert!(half.has("m_not_multiple_of_tile"));
        assert!(!half.has("n_not_multiple_of_tile"));
    }

    #[test]
    fn rank_and_degeneracy_are_read_from_the_shape() {
        let batched = extract(&unary(&[2, 2, 2], &[1.0; 8]));
        assert!(batched.has("rank_ge_3"));
        assert!(!batched.has("degenerate_dim"));

        let flat = extract(&unary(&[1, 4], &[1.0; 4]));
        assert!(!flat.has("rank_ge_3"));
        assert!(flat.has("degenerate_dim"));
    }

    #[test]
    fn a_large_contraction_dimension_is_flagged() {
        let ones = |n: usize| vec![1.0f32; n];
        assert!(extract(&matmul((&[1, 64], &ones(64)), (&[64, 1], &ones(64)))).has("k_large"));
        assert!(!extract(&matmul((&[1, 4], &ones(4)), (&[4, 1], &ones(4)))).has("k_large"));
    }

    /// Names are what a report shows; a bitmask means nothing to a reader.
    #[test]
    fn features_can_be_listed_by_name() {
        let names = extract(&unary(&[1], &[f32::NAN])).names();
        assert!(names.contains(&"input_special"));
    }

    /// An unknown name is `false`, not a panic — recorded predicates may predate a rename.
    #[test]
    fn an_unknown_feature_name_is_absent_rather_than_fatal() {
        assert!(!extract(&unary(&[1], &[1.0])).has("no_such_feature"));
    }
}
