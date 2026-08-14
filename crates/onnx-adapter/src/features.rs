//! What an input *is*, computed from the case alone.
//!
//! # The one rule this module has
//!
//! **Every feature is a function of the case. None may look at a result.** A feature that
//! consults what the runtimes said is a signature wearing a disguise: it can describe a
//! divergence that already happened and cannot predict one that has not. The distinction is the
//! whole point of PHASE-N11, and it is enforced by the signature of [`features`] — it takes an
//! `&OnnxCase` and nothing else.
//!
//! # The vocabulary was written before the findings, on purpose
//!
//! `PHASE-N11-predicate-grouping.md` lists its candidate atoms, and that list was written during
//! project planning — **before N7 found anything**. It is implemented here as written.
//!
//! That matters because of the risk the same phase file names: *"fitting the vocabulary to the
//! findings you already have — a run in which everything is neatly explained is a warning sign,
//! not a success."* Three of this domain's four problems are known to involve signed zero or an
//! empty tensor, and it would be easy to write atoms that spell them out and then report a
//! triumphant fit. Using the pre-registered list is what makes a neat result mean something and
//! a poor one informative.
//!
//! **One deviation, recorded rather than silently taken.** The planned list includes `opset_ge_N`.
//! Every case this generator currently produces is at opset 22, so that feature would be **true
//! for every case in the corpus** — constant. A constant feature is worse than a missing one: the
//! search can put it in the `required` set of any rule at no cost, where it reads as a condition
//! and constrains nothing. It is omitted until opset becomes a generation axis (`PENDING` 2.6,
//! decided but not implemented), at which point it becomes meaningful and should be added.
//!
//! # The hazard: bit order is load-bearing
//!
//! A predicate is a bitmask over [`FEATURES`] by **index**. Reordering this array silently
//! changes the meaning of every recorded predicate — the masks still match, just against
//! different properties, and nothing errors. `predicate.rs` carries the registry test that
//! stands between that and a confidently wrong report.

use crate::case::{ElemType, OnnxCase, TensorData};
use crate::ops;

/// The feature vocabulary. **Order is part of the format** — append only, never reorder.
///
/// Fourteen atoms, from the pre-registered list in `PHASE-N11-predicate-grouping.md`. See the
/// module note for why `opset_ge_N` is absent.
pub const FEATURES: [&str; 14] = [
    // ── value features: what is in the numbers ──
    "has_nan_input",
    "has_inf_input",
    "has_negative_zero",
    "has_subnormal",
    "has_boundary_magnitude",
    // ── shape features: what the tensors look like ──
    "empty_tensor",
    "rank_0",
    "has_size_1_dim",
    "broadcasting_required",
    "output_larger_than_input",
    // ── type features ──
    "integer_dtype",
    "float_dtype",
    // ── operator and attribute features ──
    "quantized_op",
    "attribute_at_bound",
];

/// A case's features as a bitmask over [`FEATURES`].
///
/// `u64` rather than `u32`, matching the tensor domain: fourteen atoms fit either, and the wider
/// word removes a ceiling that would otherwise have to be noticed later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PartialOrd, Ord)]
pub struct FeatureVec(pub u64);

impl FeatureVec {
    /// Whether the named feature holds.
    ///
    /// Returns `false` for an unknown name rather than panicking: names arrive from recorded
    /// predicates that may predate a rename, and a triage run should degrade rather than die.
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

    /// How many features hold.
    pub fn count(&self) -> u32 {
        self.0.count_ones()
    }
}

/// Compute a case's features.
///
/// **Takes the case and nothing else.** That signature is the enforcement mechanism for this
/// module's one rule; there is no result available to accidentally consult.
pub fn features(case: &OnnxCase) -> FeatureVec {
    let mut bits = 0u64;
    let mut set = |name: &str| {
        if let Some(bit) = FEATURES.iter().position(|f| *f == name) {
            bits |= 1 << bit;
        }
    };

    // ── value features ────────────────────────────────────────────────────────────────
    //
    // Every float element of every input, examined once. `-0.0` is tested on its **bit
    // pattern**, for the reason this project has recorded in three separate places: `-0.0 ==
    // 0.0` is true, so a value comparison silently answers "no" for the one case the feature
    // exists to name.
    let mut nan = false;
    let mut inf = false;
    let mut negative_zero = false;
    let mut subnormal = false;
    let mut boundary = false;

    for input in &case.inputs {
        match &input.data {
            TensorData::F32(values) => {
                for v in values {
                    nan |= v.is_nan();
                    inf |= v.is_infinite();
                    negative_zero |= v.to_bits() == (-0.0f32).to_bits();
                    subnormal |= v.is_subnormal();
                    boundary |= *v == f32::MAX || *v == f32::MIN;
                }
            }
            TensorData::F64(values) => {
                for v in values {
                    nan |= v.is_nan();
                    inf |= v.is_infinite();
                    negative_zero |= v.to_bits() == (-0.0f64).to_bits();
                    subnormal |= v.is_subnormal();
                    boundary |= *v == f64::MAX || *v == f64::MIN;
                }
            }
            // The integer types have no `NaN`, no infinity and no signed zero. They do have
            // boundaries, and those are where wrapping and saturation part company — the
            // `int32::MIN / -1` overflow (`SPECS.md` §2.11) lives exactly there.
            TensorData::I32(values) => {
                boundary |= values.iter().any(|v| *v == i32::MIN || *v == i32::MAX);
            }
            TensorData::I64(values) => {
                boundary |= values.iter().any(|v| *v == i64::MIN || *v == i64::MAX);
            }
            TensorData::I8(values) => {
                boundary |= values.iter().any(|v| *v == i8::MIN || *v == i8::MAX);
            }
            TensorData::U8(values) => {
                boundary |= values.iter().any(|v| *v == u8::MIN || *v == u8::MAX);
            }
            TensorData::Bool(_) => {}
        }
    }
    if nan {
        set("has_nan_input");
    }
    if inf {
        set("has_inf_input");
    }
    if negative_zero {
        set("has_negative_zero");
    }
    if subnormal {
        set("has_subnormal");
    }
    if boundary {
        set("has_boundary_magnitude");
    }

    // ── shape features ────────────────────────────────────────────────────────────────
    if case.inputs.iter().any(|i| i.element_count() == 0) {
        set("empty_tensor");
    }
    if ops::data_rank(case) == 0 {
        set("rank_0");
    }
    if case.inputs.iter().any(|i| i.dims.contains(&1)) {
        set("has_size_1_dim");
    }
    // **Broadcasting is a property of a pair**, so it needs two shapes that differ. A single
    // input can never require it, whatever its shape — the same "a property of the combination"
    // point the capability census keys on.
    let shapes: Vec<&Vec<i64>> = case.inputs.iter().map(|i| &i.dims).collect();
    if shapes.len() > 1 && shapes.iter().any(|s| *s != shapes[0]) {
        set("broadcasting_required");
    }
    // `output_spec` is the adapter's inferred output, validated against ONNX's own
    // `shape_inference` over 400 models at N10.1 — 323 shapes produced, 323 agreements.
    let (output_elem, output_dims) = ops::output_spec(case);
    let output_count: i64 = output_dims.iter().product::<i64>().max(0);
    let input_count = case
        .inputs
        .first()
        .map(|i| i.element_count() as i64)
        .unwrap_or(0);
    if output_count > input_count {
        set("output_larger_than_input");
    }

    // ── type features ─────────────────────────────────────────────────────────────────
    //
    // Keyed on `data_elem_type`, not `inputs[0]`: for `Where` the first input is the boolean
    // condition rather than the data, and for the shape-input operators it is an `int64` shape
    // vector. Getting this wrong would label most structural cases `integer_dtype`.
    let elem = ops::data_elem_type(case);
    if elem.is_floating() {
        set("float_dtype");
    } else if !matches!(elem, ElemType::Bool) {
        set("integer_dtype");
    }

    // ── operator and attribute features ───────────────────────────────────────────────
    if ops::spec(case.op).tier == ops::Tier::Q {
        set("quantized_op");
    }
    if attribute_at_bound(case) {
        set("attribute_at_bound");
    }

    let _ = output_elem;
    FeatureVec(bits)
}

/// Whether any integer attribute sits at an extreme of its legal range.
///
/// "At bound" means the value is the first or last position it may legally take, which for an
/// axis is `0` or `rank - 1`. Boundaries are where off-by-one handling differs between
/// implementations, which is the reason the atom is on the list.
///
/// Negative axes are normalised first: ONNX allows `-1` to mean the last axis, so a rule keyed on
/// the raw value would treat two spellings of the same axis as different cases.
fn attribute_at_bound(case: &OnnxCase) -> bool {
    let rank = ops::data_rank(case) as i64;
    for (name, value) in case.attrs.iter() {
        if let crate::attrs::AttrValue::Int(raw) = value {
            // Only axis-like attributes have a rank-derived range. `to` on a `Cast` is a type
            // code and has no ordering worth calling a bound.
            if name == "axis" && rank > 0 {
                let axis = if *raw < 0 { raw + rank } else { *raw };
                if axis == 0 || axis == rank - 1 {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attrs::Attrs;
    use crate::case::{OpKind, TensorValue};

    const OPSET: i64 = 22;

    fn unary(op: OpKind, dims: Vec<i64>, data: TensorData) -> OnnxCase {
        OnnxCase::new(op, OPSET, vec![TensorValue::new("a", dims, data)])
    }

    /// **The registry test, half one.** The vocabulary must have no duplicates and no blanks.
    ///
    /// A duplicated name makes `position` return the first index, so the second copy is a bit
    /// nothing can ever set — a feature that is silently always false.
    #[test]
    fn the_vocabulary_is_well_formed() {
        let mut seen: Vec<&str> = Vec::new();
        for name in FEATURES {
            assert!(!name.is_empty(), "a feature name is blank");
            assert!(!seen.contains(&name), "duplicate feature name {name}");
            seen.push(name);
        }
        assert!(
            FEATURES.len() <= 64,
            "the vocabulary no longer fits the FeatureVec word"
        );
    }

    /// `-0.0` must be found by its bit pattern, not by comparison.
    ///
    /// `-0.0 == 0.0` is true, so `v == -0.0` answers *yes for every zero* and the feature would
    /// be set for ordinary positive zeros. Two of this domain's four problems are signed-zero
    /// problems, so an atom that cannot tell the two zeros apart would be the one that matters
    /// most and works least.
    #[test]
    fn negative_zero_is_detected_by_bit_pattern() {
        let negative = unary(OpKind::Sign, vec![1], TensorData::F32(vec![-0.0]));
        let positive = unary(OpKind::Sign, vec![1], TensorData::F32(vec![0.0]));
        assert!(features(&negative).has("has_negative_zero"));
        assert!(
            !features(&positive).has("has_negative_zero"),
            "positive zero must not set the negative-zero feature"
        );
    }

    /// Broadcasting needs two shapes that differ; one input can never require it.
    #[test]
    fn broadcasting_is_a_property_of_a_pair() {
        let single = unary(OpKind::Abs, vec![2, 3], TensorData::F32(vec![1.0; 6]));
        assert!(!features(&single).has("broadcasting_required"));

        let same = OnnxCase::new(
            OpKind::Add,
            OPSET,
            vec![
                TensorValue::new("a", vec![2, 3], TensorData::F32(vec![1.0; 6])),
                TensorValue::new("b", vec![2, 3], TensorData::F32(vec![1.0; 6])),
            ],
        );
        assert!(!features(&same).has("broadcasting_required"));

        let differing = OnnxCase::new(
            OpKind::Add,
            OPSET,
            vec![
                TensorValue::new("a", vec![2, 3], TensorData::F32(vec![1.0; 6])),
                TensorValue::new("b", vec![3], TensorData::F32(vec![1.0; 3])),
            ],
        );
        assert!(features(&differing).has("broadcasting_required"));
    }

    /// The type atoms key on the **data** input, not on `inputs[0]`.
    ///
    /// `Where` takes its boolean condition first. Keying on `inputs[0]` would label every
    /// float `Where` case as neither float nor integer, and the atom would be wrong on exactly
    /// the operator one of the four problems lives in.
    #[test]
    fn type_features_follow_the_data_input_not_the_first_one() {
        let case = OnnxCase::new(
            OpKind::Where,
            OPSET,
            vec![
                TensorValue::new("cond", vec![2], TensorData::Bool(vec![true, false])),
                TensorValue::new("x", vec![2], TensorData::F32(vec![1.0, 2.0])),
                TensorValue::new("y", vec![2], TensorData::F32(vec![3.0, 4.0])),
            ],
        );
        let f = features(&case);
        assert!(f.has("float_dtype"), "Where on floats is a float case");
        assert!(!f.has("integer_dtype"));
    }

    /// A boolean case is neither `float_dtype` nor `integer_dtype`.
    ///
    /// Deliberate: `Bool` is not an arithmetic type, and folding it into `integer_dtype` would
    /// make that atom mean "not float", which is a different and much weaker claim.
    #[test]
    fn a_boolean_case_claims_neither_numeric_type() {
        let case = unary(OpKind::Not, vec![2], TensorData::Bool(vec![true, false]));
        let f = features(&case);
        assert!(!f.has("float_dtype"));
        assert!(!f.has("integer_dtype"));
    }

    /// An empty tensor sets `empty_tensor`, and a rank-0 tensor does not.
    ///
    /// They are easy to conflate: both have unusual shapes and both have shown up in findings.
    /// A scalar holds **one** element; an empty tensor holds none.
    #[test]
    fn empty_and_rank_zero_are_different_things() {
        let empty = unary(OpKind::Abs, vec![0], TensorData::F32(vec![]));
        let scalar = unary(OpKind::Abs, vec![], TensorData::F32(vec![1.0]));
        assert!(features(&empty).has("empty_tensor"));
        assert!(!features(&empty).has("rank_0"), "rank 1 of extent 0");
        assert!(features(&scalar).has("rank_0"));
        assert!(!features(&scalar).has("empty_tensor"), "a scalar holds one");
    }

    /// A negative axis names the same axis as its positive spelling, so it must set the same bit.
    #[test]
    fn a_negative_axis_is_normalised_before_the_bound_is_tested() {
        let last_positive = OnnxCase::new(
            OpKind::Concat,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![2, 3],
                TensorData::F32(vec![1.0; 6]),
            )],
        )
        .with_attrs(Attrs::new().int("axis", 1));
        let last_negative = OnnxCase::new(
            OpKind::Concat,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![2, 3],
                TensorData::F32(vec![1.0; 6]),
            )],
        )
        .with_attrs(Attrs::new().int("axis", -1));
        assert!(features(&last_positive).has("attribute_at_bound"));
        assert!(
            features(&last_negative).has("attribute_at_bound"),
            "-1 and rank-1 are the same axis and must agree"
        );
    }

    /// **Features are a function of the case alone**, so the same case always gives the same
    /// vector. Trivially true today; the test exists so that it stays true when someone is
    /// tempted to pass a result in "just for this one atom".
    #[test]
    fn features_are_deterministic() {
        let case = unary(
            OpKind::Sign,
            vec![3],
            TensorData::F32(vec![-0.0, 1.0, -1.0]),
        );
        assert_eq!(features(&case), features(&case));
    }
}
