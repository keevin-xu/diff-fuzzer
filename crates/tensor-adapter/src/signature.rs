//! What makes two findings the same underlying problem.
//!
//! A single defect is reachable from an enormous number of inputs. A fuzzer that finds
//! one will find it again within seconds and keep finding it — so without collapsing
//! them, a campaign's output becomes a thousand copies of one thing, and any genuinely
//! *second* problem is invisible in the noise.
//!
//! # The trade-off, which is the whole difficulty
//!
//! A signature is a deliberate loss of information, and it can fail in two directions:
//!
//! - **Too coarse** — two genuinely different bugs share a signature, and the second is
//!   silently discarded. This is the dangerous failure: it looks exactly like success.
//! - **Too fine** — one bug produces many signatures, and the noise problem returns.
//!   Annoying, but visible, and therefore self-correcting.
//!
//! Given the asymmetry, this errs toward **finer**. The components below are the ones
//! that plausibly distinguish *different causes*; everything that merely distinguishes
//! *different inputs to the same cause* is deliberately excluded.

use crate::input::TensorOp;
use crate::normalize::CanonicalTensor;
use diff_fuzzer_core::{Agreement, ApproxEq, Tolerance};

/// A stable identifier for the kind of problem a case exhibits.
///
/// Built from four things:
///
/// 1. **The operation** — `matmul` disagreeing is not the same problem as `exp`
///    disagreeing.
/// 2. **The rank** — rank-specific code paths are genuinely separate implementations in
///    most libraries, so a rank-1 failure and a rank-4 failure may well have different
///    causes.
/// 3. **How the results disagree** — a structural mismatch, an undefined-versus-number
///    disagreement, and a numeric difference are three different phenomena that happen to
///    share a verdict.
/// 4. **The order of magnitude of the error**, as a power of ten. Coarse on purpose: two
///    cases differing by `3.1e-6` and `7.4e-6` are almost certainly the same problem
///    reached by different inputs, while one differing by `1e-1` is not.
///
/// **Deliberately excluded: the shapes and the values.** Those distinguish *inputs*, not
/// *causes*, and including them would give nearly every case its own signature — which is
/// the same as no de-duplication at all.
pub fn signature(
    case: &TensorOp,
    left: &CanonicalTensor,
    right: &CanonicalTensor,
    tolerance: Tolerance,
) -> String {
    let kind = disagreement_kind(left, right, tolerance);
    format!("{}/rank{}/{}", case.name(), case.rank(), kind)
}

/// Which two implementations a signature was computed from.
///
/// Recorded **beside** the signature rather than inside it — see [`signature_across`] for
/// why the names must stay out of the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisagreeingPair {
    pub left: String,
    pub right: String,
}

/// A signature for a case run on **any number** of implementations.
///
/// # The problem this solves
///
/// [`signature`] takes exactly two results, which was unambiguous while there were two
/// backends. With three, callers were passing the *first two* — so when the GPU was the
/// one that disagreed, the label was computed from two CPU backends that agreed, and a
/// diverging case came out labelled `.../agree`. **Something wrong was recorded, which is
/// worse than nothing, because it looks like a normal result.**
///
/// This picks the pair that actually disagreed — the **worst** one, by kind first and
/// magnitude second — and returns its name alongside the signature.
///
/// # Why the implementation names stay out of the signature string
///
/// It is tempting to write `div/rank2/ndarray-vs-wgpu/numeric/1e-7`. Resist it:
///
/// - **The signature is a de-duplication key, and names make it depend on the experiment.**
///   The same `matmul` overflow would be `matmul/rank2/undefined` in a two-backend campaign
///   and `matmul/rank2/ndarray-vs-tch/undefined` in a three-backend one. `known.rs` matches
///   exact strings, so a long-settled problem would return labelled *new* the moment a
///   backend is added.
/// - **It inflates the class count.** One GPU rounding difference disagrees on
///   `ndarray↔wgpu` *and* `tch↔wgpu`; with names in the key that is two classes for one
///   phenomenon, triaged twice.
/// - **Nothing is lost**, because the pair is returned separately and can be grouped or
///   displayed on demand.
///
/// > **A signature should describe the *problem*, not the *observation setup*.** Which
/// > backends were running is a fact about the experiment, not about the bug.
///
/// Returns `None` for the pair when everything agreed.
pub fn signature_across(
    case: &TensorOp,
    outputs: &[(String, CanonicalTensor)],
    tolerance: Tolerance,
) -> (String, Option<DisagreeingPair>) {
    let mut worst: Option<(Severity, usize, usize)> = None;

    for left in 0..outputs.len() {
        for right in (left + 1)..outputs.len() {
            let severity = severity_of(&outputs[left].1, &outputs[right].1, tolerance);
            if severity == Severity::AGREE {
                continue;
            }
            // Strictly greater, so the first pair wins a tie and the choice is
            // deterministic — a signature that varied with iteration order would be
            // useless as a key.
            if worst.as_ref().is_none_or(|(w, _, _)| severity > *w) {
                worst = Some((severity, left, right));
            }
        }
    }

    match worst {
        Some((_, left, right)) => (
            signature(case, &outputs[left].1, &outputs[right].1, tolerance),
            Some(DisagreeingPair {
                left: outputs[left].0.clone(),
                right: outputs[right].0.clone(),
            }),
        ),
        // Everything agreed. Fall back to the first pair so the function is total; callers
        // reach this only when asking for a signature on a case that did not diverge.
        None => match outputs {
            [(_, a), (_, b), ..] => (signature(case, a, b, tolerance), None),
            _ => (format!("{}/unpaired", case.name()), None),
        },
    }
}

/// How bad a disagreement is, for choosing which pair names a finding.
///
/// Ordered so that a difference *in kind* always outranks one of degree: two backends
/// returning different shapes is a more fundamental disagreement than two returning
/// slightly different numbers, however large the numeric gap.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct Severity(u8, f64);

impl Severity {
    const AGREE: Severity = Severity(0, 0.0);
}

fn severity_of(left: &CanonicalTensor, right: &CanonicalTensor, tolerance: Tolerance) -> Severity {
    match left.approx_compare(right, tolerance) {
        Agreement::Agree(_) => Severity::AGREE,
        Agreement::Structural { .. } => Severity(3, 0.0),
        Agreement::Disagree(comparison) => {
            if involves_undefined(left, right) {
                Severity(2, 0.0)
            } else {
                // Magnitude breaks ties within the numeric class, so the pair that differs
                // most is the one that names the finding.
                let error = comparison.max_relative_error;
                Severity(1, if error.is_finite() { error } else { f64::MAX })
            }
        }
    }
}

/// How two results disagree, in the coarsest terms that still separate causes.
fn disagreement_kind(
    left: &CanonicalTensor,
    right: &CanonicalTensor,
    tolerance: Tolerance,
) -> String {
    match left.approx_compare(right, tolerance) {
        Agreement::Agree(_) => "agree".to_string(),

        // Shape or element type. These are not numeric problems at all, and a shape
        // mismatch has nothing in common with a value being slightly off.
        Agreement::Structural { .. } => "structural".to_string(),

        Agreement::Disagree(comparison) => {
            // A disagreement about whether an answer exists is a different phenomenon
            // from one about its value, even though both land in this branch.
            if involves_undefined(left, right) {
                return "undefined".to_string();
            }

            numeric_kind(comparison.max_relative_error)
        }
    }
}

/// A numeric disagreement's severity, on a scale whose boundaries mean something.
///
/// # Why not the order of magnitude
///
/// This used to return `numeric/1e{floor(log10(error))}` — one class per decade. That was
/// recorded as suspect from the start (`PENDING` 2.8) and a third backend proved it: GPU
/// rounding differences landed at `1e-7` **and** `1e-8`, splitting **one phenomenon across
/// two classes**. A campaign produced 44 classes where perhaps 15 were real, and triage
/// pays for every spurious one with a wasted investigation.
///
/// The problem is that a decade boundary is arbitrary — nothing distinguishes `9.9e-8` from
/// `1.01e-7` except which side of a power of ten they fall on.
///
/// # The two boundaries, and why each has a reason
///
/// **`ROUNDING_SCALE` (16 ε).** Machine epsilon is the natural unit of legitimate
/// floating-point disagreement: two implementations that are both correct but round
/// differently land within a handful of ULP. Sixteen is generous enough to cover several
/// roundings in sequence and far below anything a real defect produces. **Errors here are
/// the arithmetic being itself.**
///
/// **`TOTAL` (relative error ≥ 1.0).** At this point the results are no longer a perturbed
/// version of each other — one is at least as far from the other as the other is from zero.
/// A subnormal flushed to zero gives *exactly* 1.0; a sign flip gives 2.0. **Errors here
/// are a different answer, not an inaccurate one.**
///
/// Neither boundary was chosen by looking at a histogram of observed errors. That
/// distinction matters: picking cut-points to make the class count look tidy is the same
/// fitting-to-data error the tolerance policy exists to prevent.
///
/// # What is deliberately given up
///
/// Three buckets cannot distinguish `1e-2` from `1e-1`; both are `significant`. This is a
/// **coarsening**, and coarsening is the direction that hides things — so it is only
/// defensible because the finer number is not lost: every report's summary carries the
/// exact `max relative error`. The *signature* is a grouping key, not the evidence.
fn numeric_kind(error: f64) -> String {
    /// Sixteen machine epsilons — a handful of roundings.
    const ROUNDING_SCALE: f64 = 16.0 * f32::EPSILON as f64;
    /// At a relative error of 1, the results are not near each other in any sense.
    const TOTAL: f64 = 1.0;

    // `is_finite` first, so `NaN` is caught explicitly rather than by a negated comparison
    // — with floats the two are not the same thing, and the reader should not have to work
    // that out.
    if !error.is_finite() || error <= 0.0 {
        return "numeric/unmeasurable".to_string();
    }
    if error <= ROUNDING_SCALE {
        return "numeric/rounding".to_string();
    }
    if error >= TOTAL {
        return "numeric/total".to_string();
    }
    "numeric/significant".to_string()
}

/// Whether either result contains a value that is not finite.
///
/// Checked on the results rather than inferred from the error, because an
/// undefined-versus-number disagreement produces an error magnitude that is either zero
/// or not finite — neither of which describes what actually happened.
fn involves_undefined(left: &CanonicalTensor, right: &CanonicalTensor) -> bool {
    left.values
        .iter()
        .chain(&right.values)
        .any(|v| !v.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{BinaryOp, TensorValue, UnaryOp};

    fn canon(shape: &[usize], values: &[f32]) -> CanonicalTensor {
        CanonicalTensor {
            shape: shape.to_vec(),
            dtype: "F32".to_string(),
            values: values.to_vec(),
        }
    }

    fn case(shape: &[usize]) -> TensorOp {
        let count = shape.iter().product();
        TensorOp::unary(
            UnaryOp::Neg,
            TensorValue::new(shape.to_vec(), vec![1.0; count]),
        )
    }

    /// **The property de-duplication exists for.** The same problem reached by different
    /// inputs must collapse to one signature — otherwise a campaign reports the same
    /// thing repeatedly and the count stops meaning anything.
    #[test]
    fn the_same_problem_from_different_inputs_shares_a_signature() {
        let a = signature(
            &case(&[4]),
            &canon(&[4], &[1.0, 2.0, 3.0, 4.0]),
            &canon(&[4], &[1.000003, 2.0, 3.0, 4.0]),
            Tolerance::EXACT,
        );
        let b = signature(
            &case(&[4]),
            &canon(&[4], &[9.0, 8.0, 7.0, 6.0]),
            &canon(&[4], &[9.000067, 8.0, 7.0, 6.0]),
            Tolerance::EXACT,
        );

        assert_eq!(a, b, "the same kind of error produced different signatures");
    }

    /// Different operations are different problems, even with identical numbers.
    #[test]
    fn different_operations_do_not_share_a_signature() {
        let left = canon(&[2], &[1.0, 2.0]);
        let right = canon(&[2], &[1.5, 2.0]);

        let unary = signature(&case(&[2]), &left, &right, Tolerance::EXACT);
        let binary = signature(
            &TensorOp::binary(
                BinaryOp::Add,
                TensorValue::new(vec![2], vec![1.0, 1.0]),
                TensorValue::new(vec![2], vec![1.0, 1.0]),
            ),
            &left,
            &right,
            Tolerance::EXACT,
        );

        assert_ne!(unary, binary);
    }

    /// Rank-specific code paths are separate implementations in most libraries, so a
    /// failure at one rank may have a different cause from the same operation at another.
    #[test]
    fn different_ranks_do_not_share_a_signature() {
        let left = canon(&[4], &[1.0; 4]);
        let right = canon(&[4], &[1.5, 1.0, 1.0, 1.0]);

        let flat = signature(&case(&[4]), &left, &right, Tolerance::EXACT);
        let square = signature(&case(&[2, 2]), &left, &right, Tolerance::EXACT);

        assert_ne!(flat, square);
    }

    /// Errors orders of magnitude apart are almost certainly different problems.
    #[test]
    fn errors_of_different_magnitude_do_not_share_a_signature() {
        let base = canon(&[1], &[1.0]);

        let tiny = signature(
            &case(&[1]),
            &base,
            &canon(&[1], &[1.000001]),
            Tolerance::EXACT,
        );
        let large = signature(&case(&[1]), &base, &canon(&[1], &[1.5]), Tolerance::EXACT);

        assert_ne!(tiny, large);
    }

    /// A disagreement about whether an answer *exists* is a different phenomenon from one
    /// about its value — the distinction the `matmul` overflow finding turns on.
    #[test]
    fn an_undefined_result_gets_its_own_signature() {
        let numeric = signature(
            &case(&[1]),
            &canon(&[1], &[1.0]),
            &canon(&[1], &[1.5]),
            Tolerance::EXACT,
        );
        let undefined = signature(
            &case(&[1]),
            &canon(&[1], &[f32::NEG_INFINITY]),
            &canon(&[1], &[f32::NAN]),
            Tolerance::EXACT,
        );

        assert_ne!(numeric, undefined);
        assert!(undefined.contains("undefined"), "{undefined}");
    }

    /// A shape mismatch has nothing in common with a value being slightly off.
    #[test]
    fn a_structural_difference_gets_its_own_signature() {
        let structural = signature(
            &case(&[2]),
            &canon(&[2], &[1.0, 2.0]),
            &canon(&[3], &[1.0, 2.0, 3.0]),
            Tolerance::EXACT,
        );

        assert!(structural.contains("structural"), "{structural}");
    }

    fn named(name: &str, shape: &[usize], values: &[f32]) -> (String, CanonicalTensor) {
        (name.to_string(), canon(shape, values))
    }

    /// **The bug this function exists to fix.** Callers used to pass the first two outputs,
    /// so when the third implementation was the one that disagreed, the label was computed
    /// from two that agreed — and a diverging case came out labelled `.../agree`.
    #[test]
    fn a_third_implementation_disagreeing_is_not_labelled_agree() {
        let outputs = [
            named("cpu-a", &[1], &[1.0]),
            named("cpu-b", &[1], &[1.0]),
            named("gpu", &[1], &[1.5]),
        ];

        let (signature, pair) = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);

        assert!(
            !signature.contains("agree"),
            "a diverging case must not be labelled agree: {signature}"
        );
        let pair = pair.expect("a disagreeing pair exists");
        assert_eq!(pair.right, "gpu", "the pair must include the dissenter");
    }

    /// **The property that keeps `known.rs` from orphaning when a backend is added.** The
    /// same problem must produce the same key whether two or three implementations ran.
    #[test]
    fn adding_an_agreeing_implementation_does_not_change_the_signature() {
        let two = [named("cpu-a", &[1], &[1.0]), named("gpu", &[1], &[1.5])];
        let three = [
            named("cpu-a", &[1], &[1.0]),
            named("cpu-b", &[1], &[1.0]),
            named("gpu", &[1], &[1.5]),
        ];

        let (from_two, _) = signature_across(&case(&[1]), &two, Tolerance::EXACT);
        let (from_three, _) = signature_across(&case(&[1]), &three, Tolerance::EXACT);

        assert_eq!(
            from_two, from_three,
            "the signature must describe the problem, not which backends were running"
        );
    }

    /// Implementation names must stay out of the key — see `signature_across`'s docs for
    /// the two failures that causes.
    #[test]
    fn the_signature_never_contains_an_implementation_name() {
        // **The real names**, not a placeholder. The point is that the names this project
        // actually uses stay out of the key; testing it with `"burn-ndarray"` — removed at
        // PHASE-7A — proved it about a backend nobody runs.
        let outputs = [
            named(crate::backends::FLEX_NAME, &[1], &[1.0]),
            named(crate::backends::WGPU_NAME, &[1], &[1.5]),
        ];
        let (signature, _) = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);

        for name in [
            crate::backends::FLEX_NAME,
            crate::backends::LIBTORCH_NAME,
            crate::backends::WGPU_NAME,
        ] {
            assert!(!signature.contains(name), "{signature} contains {name}");
        }
    }

    /// A difference in *kind* outranks one of degree, however large the numeric gap: two
    /// backends returning different shapes disagree more fundamentally than two returning
    /// different numbers.
    #[test]
    fn a_structural_disagreement_outranks_a_large_numeric_one() {
        let outputs = [
            named("a", &[1], &[1.0]),
            named("b", &[1], &[1e30]),
            named("c", &[2], &[1.0, 2.0]),
        ];

        let (signature, pair) = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);

        assert!(signature.contains("structural"), "{signature}");
        assert_eq!(pair.expect("a pair").right, "c");
    }

    /// Deterministic on ties, or the key would vary with iteration order.
    #[test]
    fn the_chosen_pair_is_deterministic() {
        let outputs = [
            named("a", &[1], &[1.0]),
            named("b", &[1], &[2.0]),
            named("c", &[1], &[2.0]),
        ];

        let first = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);
        let again = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);
        assert_eq!(first.1, again.1);
        assert_eq!(
            first.1.expect("a pair").right,
            "b",
            "the earlier pair wins a tie"
        );
    }

    /// Agreement yields no pair — callers should not be told a disagreement happened.
    #[test]
    fn full_agreement_reports_no_disagreeing_pair() {
        let outputs = [named("a", &[1], &[1.0]), named("b", &[1], &[1.0])];
        let (_, pair) = signature_across(&case(&[1]), &outputs, Tolerance::EXACT);
        assert!(pair.is_none());
    }

    /// **The fix for `PENDING` 2.8, as a test.** Two GPU rounding differences an order of
    /// magnitude apart are the *same* phenomenon — the arithmetic being itself — and used to
    /// land in `numeric/1e-7` and `numeric/1e-8`, splitting one problem across two classes.
    #[test]
    fn rounding_differences_an_order_of_magnitude_apart_share_a_signature() {
        let base = canon(&[1], &[1.0]);
        let one_ulp = signature(
            &case(&[1]),
            &base,
            &canon(&[1], &[1.0 + f32::EPSILON]),
            Tolerance::EXACT,
        );
        // Eight ULP, not a fraction of one: `1.0 + EPSILON/8.0` rounds back to exactly
        // `1.0` in f32, so the two results would be identical and the test would compare
        // nothing. An order of magnitude apart in error, both still ordinary rounding.
        let eight_ulp = signature(
            &case(&[1]),
            &base,
            &canon(&[1], &[1.0 + 8.0 * f32::EPSILON]),
            Tolerance::EXACT,
        );

        assert_eq!(one_ulp, eight_ulp, "both are ordinary rounding");
        assert!(one_ulp.contains("rounding"), "{one_ulp}");
    }

    /// **The distinction the coarsening must not destroy.** A subnormal flushed to zero is a
    /// *relative error of exactly 1.0* — a different answer, not an inaccurate one — and
    /// must never be grouped with rounding noise.
    #[test]
    fn a_flushed_subnormal_is_not_grouped_with_rounding() {
        let flushed = signature(
            &case(&[1]),
            &canon(&[1], &[1e-45]),
            &canon(&[1], &[0.0]),
            Tolerance::EXACT,
        );
        let rounding = signature(
            &case(&[1]),
            &canon(&[1], &[1.0]),
            &canon(&[1], &[1.0 + f32::EPSILON]),
            Tolerance::EXACT,
        );

        assert_ne!(flushed, rounding);
        assert!(flushed.contains("total"), "{flushed}");
    }

    /// The boundaries are anchored to machine epsilon and to a relative error of 1, not to
    /// powers of ten. Pinned so a future adjustment has to be deliberate — and so that
    /// nobody quietly retunes them to make a class count look tidy.
    #[test]
    fn the_severity_boundaries_are_where_they_are_claimed_to_be() {
        let epsilon = f32::EPSILON as f64;

        assert_eq!(numeric_kind(epsilon), "numeric/rounding");
        assert_eq!(numeric_kind(15.0 * epsilon), "numeric/rounding");
        assert_eq!(numeric_kind(100.0 * epsilon), "numeric/significant");
        assert_eq!(numeric_kind(0.5), "numeric/significant");
        assert_eq!(numeric_kind(1.0), "numeric/total", "a flushed subnormal");
        assert_eq!(numeric_kind(2.0), "numeric/total", "a sign flip");
        assert_eq!(numeric_kind(0.0), "numeric/unmeasurable");
        assert_eq!(numeric_kind(f64::INFINITY), "numeric/unmeasurable");
    }

    /// Signatures must be stable across runs, or de-duplication would fail to duplicate.
    #[test]
    fn signatures_are_deterministic() {
        let build = || {
            signature(
                &case(&[3]),
                &canon(&[3], &[1.0, 2.0, 3.0]),
                &canon(&[3], &[1.1, 2.0, 3.0]),
                Tolerance::EXACT,
            )
        };
        assert_eq!(build(), build());
    }
}
