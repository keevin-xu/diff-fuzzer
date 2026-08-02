//! How much difference each operation is allowed.
//!
//! Every number here is **derived, then checked against measurement** — never fitted to
//! it. That ordering is the whole point. A threshold set just above the largest
//! observed noise has no argument behind it: it is guaranteed to produce no false
//! positives on the data it was fitted to, no margin for cases not yet generated, and
//! it would silently absorb a real bug that happened to be smaller than noise already
//! seen. A threshold derived from how floating-point arithmetic works, which then turns
//! out to cover the observed noise with room to spare, is a claim that can be defended.
//!
//! Three classes, for three genuinely different reasons.
//!
//! # Exactly equal: `add`, `sub`, `mul`, `div`, `sqrt`, `neg`, `abs`
//!
//! Not "very close" — identical, and provably so. IEEE-754 **requires** addition,
//! subtraction, multiplication, division and square root to be *correctly rounded*: the
//! result must be the representable number nearest the true answer. There is exactly
//! one such number, so any two conforming implementations must produce the same bits.
//! `neg` and `abs` only touch the sign bit. Measurement agrees: zero error across
//! 14,000 cases.
//!
//! Holding these to exact equality is therefore not strictness for its own sake. A
//! difference here would be a genuine violation, and giving them slack would only hide
//! it.
//!
//! # One rounding step: `exp`
//!
//! `exp` is conspicuously *not* in the list IEEE-754 requires to be correctly rounded —
//! doing so is expensive, and libraries choose their own approximations. So two
//! correct implementations may land on adjacent representable numbers. Measurement
//! shows exactly that: a hard ceiling at `1.192e-7`, which is precisely one unit in the
//! last place. Two units are allowed, since each side may round its own approximation.
//!
//! # Accumulated error: `sum`, `matmul`
//!
//! Adding many numbers is where implementations legitimately part company, because
//! floating-point addition is not associative — a different summation order gives a
//! different answer, and neither is wrong. The standard bound for summing `n` terms is
//!
//! ```text
//!     |computed - exact|  <=  n * eps * sum|x_i|
//! ```
//!
//! and two implementations may sit on opposite sides of the true value, so the gap
//! between *them* can reach twice that. This is computed **per case** from the actual
//! shapes and values rather than from a global worst case, which keeps the tolerance
//! tight on small inputs instead of applying the loosest case everywhere.
//!
//! The absolute term matters more than it looks here. Summing mixed-sign values can
//! land near zero while the terms are large — cancellation — and a tiny absolute error
//! then becomes an enormous *relative* one. Measurement shows `sum` reaching `1.2e-3`
//! relative while its absolute error stays at `7.6e-6`: the error did not grow, the
//! denominator shrank.

use crate::input::TensorOp;
use diff_fuzzer_core::{Tolerance, TolerancePolicy};

/// One rounding step for `f32`, as a `f64` so the arithmetic below does not itself
/// round.
const EPSILON: f64 = f32::EPSILON as f64;

/// Why an operation is allowed the tolerance it gets.
///
/// Named for the *reason* rather than the operations, since the reason is what a new
/// operation has to be classified by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// IEEE-754 requires a correctly rounded result, so implementations must agree
    /// bit-for-bit.
    CorrectlyRounded,
    /// Approximated by each library independently; results may differ by a rounding
    /// step.
    Approximated,
    /// Sums many terms, so results depend on summation order.
    Accumulating,
}

impl TensorOp {
    /// Which tolerance class this case falls into.
    pub fn class(&self) -> OpClass {
        use crate::input::{ReduceOp, UnaryOp};

        match self {
            TensorOp::Unary { kind, .. } => match kind {
                UnaryOp::Exp => OpClass::Approximated,
                // `sqrt` is correctly rounded by IEEE-754; `neg` and `abs` only touch
                // the sign bit.
                UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sqrt => OpClass::CorrectlyRounded,
            },
            // Every elementwise binary operation is one correctly rounded arithmetic
            // operation per element, with nothing accumulated.
            TensorOp::Binary { .. } => OpClass::CorrectlyRounded,
            TensorOp::Reduce { kind, .. } => match kind {
                ReduceOp::Sum => OpClass::Accumulating,
            },
            TensorOp::Matmul { .. } => OpClass::Accumulating,
        }
    }
}

/// Chooses a tolerance from the operation and the size of its arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorTolerancePolicy;

impl TolerancePolicy<TensorOp> for TensorTolerancePolicy {
    fn tolerance_for(&self, input: &TensorOp) -> Tolerance {
        match input.class() {
            OpClass::CorrectlyRounded => Tolerance::EXACT,

            OpClass::Approximated => approximated_tolerance(input),

            OpClass::Accumulating => accumulating_tolerance(input),
        }
    }
}

/// Tolerance for an approximated function, scaled by how hard the function is to
/// evaluate at the argument it was given.
///
/// The governing idea is the **condition number**: how much a small relative
/// perturbation of the input is magnified in the output. For `exp(x)` it is `|x|`,
/// because `exp(x + d) = exp(x) * e^d` — a tiny error in the argument becomes a relative
/// error of roughly `d` in the result. Implementations reduce the argument before
/// approximating (`x = k*ln2 + r`), and the error in that reduction grows with `|x|`, so
/// two libraries drift further apart the larger the argument.
///
/// Hence `(1 + |x|) * eps` for one implementation — a rounding step plus the
/// condition-number term — doubled because two implementations may sit on opposite sides
/// of the true value.
///
/// **This replaces a fixed `2 * eps`, which was wrong in an instructive way.** That
/// constant was derived from data measured with `|x| <= 10`, and it held perfectly
/// there. Run at `|x| <= 1000` it produced 235 false positives, because it did not
/// scale with the thing that actually drives the error. *Fixed thresholds inherit the
/// scope of the evidence they were derived from* — the same trap the accumulating class
/// avoided by being computed per case from the outset.
///
/// The bound is deliberately looser than measurement at small arguments (roughly 20x the
/// worst observed at `|x| <= 10`). That gap is honest: the model bounds what is
/// *permissible* for a function the standard does not require to be correctly rounded,
/// while measurement shows what these two particular libraries *happen* to do today. A
/// threshold tightened to the latter would be fitted to an implementation detail.
fn approximated_tolerance(input: &TensorOp) -> Tolerance {
    let largest = match input {
        TensorOp::Unary { arg, .. } => largest_magnitude(arg.data()),
        // Every approximated operation is unary; anything else is misclassified.
        other => unreachable!("{} is not an approximated unary operation", other.name()),
    };

    Tolerance::new(2.0 * (1.0 + largest) * EPSILON, 0.0)
}

/// Tolerance for an operation that sums `terms` values of magnitude up to `largest`.
///
/// The factor of two accounts for the two implementations sitting on opposite sides of
/// the true value: each may be off by the bound, so the gap between them can be twice
/// it.
fn bound(terms: usize, largest: f64) -> Tolerance {
    let terms = terms as f64;
    // Relative component: covers results that are large, where the error scales with
    // the answer.
    let rtol = 2.0 * terms * EPSILON;
    // Absolute component: covers results near zero, where cancellation has destroyed
    // the scale that a relative tolerance would need. `terms * largest` bounds the sum
    // of magnitudes.
    let atol = 2.0 * terms * EPSILON * (terms * largest);
    Tolerance::new(rtol, atol)
}

fn accumulating_tolerance(input: &TensorOp) -> Tolerance {
    match input {
        TensorOp::Reduce { arg, axis, .. } => {
            // Each output element sums exactly the values along the collapsed axis.
            let terms = arg.shape()[*axis];
            bound(terms, largest_magnitude(arg.data()))
        }
        TensorOp::Matmul { lhs, rhs } => {
            // Each output element sums `k` products, where `k` is the shared inner
            // dimension. A product of two values is bounded by the product of their
            // largest magnitudes.
            let shape = lhs.shape();
            let terms = shape[shape.len() - 1];
            let largest = largest_magnitude(lhs.data()) * largest_magnitude(rhs.data());
            bound(terms, largest)
        }
        // Every accumulating operation is handled above; anything else is misclassified.
        other => unreachable!("{} is not an accumulating operation", other.name()),
    }
}

/// Largest absolute value present, ignoring anything not finite.
///
/// A non-finite input would make the bound meaningless rather than merely large, so it
/// is excluded; such cases are judged by the NaN and infinity rules instead.
fn largest_magnitude(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| v.abs() as f64)
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{BinaryOp, ReduceOp, TensorValue, UnaryOp};

    fn value(shape: &[usize], fill: f32) -> TensorValue {
        let count = shape.iter().product();
        TensorValue::new(shape.to_vec(), vec![fill; count])
    }

    fn tolerance_for(op: &TensorOp) -> Tolerance {
        TensorTolerancePolicy.tolerance_for(op)
    }

    #[test]
    fn correctly_rounded_operations_get_no_slack() {
        for op in [
            TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::binary(BinaryOp::Div, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::unary(UnaryOp::Sqrt, value(&[4], 4.0)),
            TensorOp::unary(UnaryOp::Neg, value(&[4], 1.0)),
        ] {
            assert_eq!(
                tolerance_for(&op),
                Tolerance::EXACT,
                "{} should be exact",
                op.name()
            );
        }
    }

    #[test]
    fn exp_is_allowed_at_least_two_rounding_steps() {
        let tolerance = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1.0)));

        assert_eq!(tolerance.atol, 0.0);
        // Comfortably above the measured ceiling of one unit in the last place.
        assert!(tolerance.rtol > f32::EPSILON as f64);
        assert!(tolerance.rtol < 1e-6);
    }

    /// The fix for the 235 false positives found at wide bounds: `exp`'s allowance must
    /// grow with the argument, because that is what drives its error. A fixed constant
    /// is only valid over the range of arguments it was measured on.
    #[test]
    fn exp_tolerance_scales_with_argument_magnitude() {
        let small = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1.0)));
        let large = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1000.0)));

        assert!(large.rtol > small.rtol * 100.0, "{large:?} vs {small:?}");
        // Still an entirely relative allowance — `exp` of a large argument is large, so
        // an absolute floor would be meaningless here.
        assert_eq!(large.atol, 0.0);
    }

    /// The derived bound must cover what was actually measured at wide bounds, with
    /// margin. If this fails, either the model is wrong or something is happening that
    /// is not rounding.
    #[test]
    fn the_exp_bound_covers_the_measured_worst_case_at_large_arguments() {
        let tolerance = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1000.0)));

        // Measured worst relative error for `exp` across 4,000 wide-bounds cases.
        let measured_worst = 1.633e-4;
        assert!(
            tolerance.rtol > measured_worst,
            "derived rtol {:e} does not cover measured {:e}",
            tolerance.rtol,
            measured_worst
        );
        // And is not absurdly loose: within an order of magnitude of what occurs.
        assert!(tolerance.rtol < measured_worst * 10.0);
    }

    /// The argument's magnitude drives the allowance, not the tensor's size — twice as
    /// many values of the same magnitude are no harder to evaluate.
    #[test]
    fn exp_tolerance_ignores_how_many_values_there_are() {
        let few = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[2], 5.0)));
        let many = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[64], 5.0)));

        assert_eq!(few.rtol, many.rtol);
    }

    /// The derived bound must cover what was actually measured, with margin. If this
    /// ever fails, either the derivation is wrong or something is happening that is not
    /// rounding — both worth stopping for.
    #[test]
    fn the_derived_bound_covers_the_measured_worst_case_for_sum() {
        // Worst case within the generator's limits: eight terms of magnitude ten.
        let op = TensorOp::reduce(ReduceOp::Sum, value(&[8], 10.0), 0);
        let tolerance = tolerance_for(&op);

        // Measured worst absolute error for `sum` across 20,000 cases.
        let measured_worst = 7.63e-6;
        assert!(
            tolerance.atol > measured_worst,
            "derived atol {:e} does not cover measured {:e}",
            tolerance.atol,
            measured_worst
        );
        // ... and is not absurdly loose either. Ten times the observed worst is margin;
        // ten thousand times would be a licence to miss real bugs.
        assert!(tolerance.atol < measured_worst * 1_000.0);
    }

    #[test]
    fn the_derived_bound_covers_the_measured_worst_case_for_matmul() {
        let op = TensorOp::matmul(value(&[8, 8], 10.0), value(&[8, 8], 10.0));
        let tolerance = tolerance_for(&op);

        let measured_worst = 3.05e-5;
        assert!(
            tolerance.atol > measured_worst,
            "derived atol {:e} does not cover measured {:e}",
            tolerance.atol,
            measured_worst
        );
        assert!(tolerance.atol < measured_worst * 1_000.0);
    }

    /// The point of computing per case rather than from a global worst case: a small
    /// input must not inherit the tolerance a large one needs.
    #[test]
    fn smaller_inputs_get_a_tighter_tolerance() {
        let small = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[2], 1.0), 0));
        let large = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[8], 10.0), 0));

        assert!(
            small.atol < large.atol,
            "small {:e} vs large {:e}",
            small.atol,
            large.atol
        );
        assert!(small.rtol < large.rtol);
    }

    /// Values, not just shapes, must move the bound — the error depends on the
    /// magnitude of what is being added.
    #[test]
    fn larger_values_get_a_looser_absolute_tolerance() {
        let modest = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[4], 1.0), 0));
        let big = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[4], 1000.0), 0));

        assert!(big.atol > modest.atol);
        // The relative component depends only on how many terms are summed, so it is
        // unchanged by their size.
        assert_eq!(big.rtol, modest.rtol);
    }

    /// Reducing a different axis sums a different number of terms, so the tolerance
    /// must follow the axis rather than the tensor's total size.
    #[test]
    fn the_tolerance_follows_the_reduced_axis() {
        let arg = value(&[2, 8], 1.0);
        let short = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, arg.clone(), 0));
        let long = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, arg, 1));

        assert!(short.atol < long.atol, "axis 0 sums 2, axis 1 sums 8");
    }

    #[test]
    fn every_operation_class_is_reachable() {
        assert_eq!(
            TensorOp::unary(UnaryOp::Exp, value(&[2], 1.0)).class(),
            OpClass::Approximated
        );
        assert_eq!(
            TensorOp::binary(BinaryOp::Mul, value(&[2], 1.0), value(&[2], 1.0)).class(),
            OpClass::CorrectlyRounded
        );
        assert_eq!(
            TensorOp::matmul(value(&[2, 2], 1.0), value(&[2, 2], 1.0)).class(),
            OpClass::Accumulating
        );
    }
}
