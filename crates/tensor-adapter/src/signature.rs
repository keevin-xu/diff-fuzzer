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

            // Otherwise, the order of magnitude of the relative error. `1e-6` and `1e-1`
            // are almost certainly different problems; `3.1e-6` and `7.4e-6` are almost
            // certainly not.
            let error = comparison.max_relative_error;
            if error > 0.0 && error.is_finite() {
                format!("numeric/1e{}", error.log10().floor() as i32)
            } else {
                "numeric/unmeasurable".to_string()
            }
        }
    }
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
