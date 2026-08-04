//! Cases sitting exactly on a boundary, where implementations most plausibly differ.
//!
//! A **boundary** is a point where an operation's behaviour changes character: `abs` at
//! zero, where the sign flips; `sqrt` at zero, where the derivative becomes infinite;
//! any subtraction of equal values, where the result cancels exactly to zero. These are
//! the places a library is most likely to make a different choice from another one, so
//! they are worth testing deliberately rather than waiting for a random generator to
//! land on them — which it essentially never will, since the chance of drawing exactly
//! `0.0` from a continuous range is nil.
//!
//! The technique this replaces is ∇Fuzz's **neighbour sampling**: when a difference
//! appears at a non-differentiable point, sample nearby points, and if the difference
//! occurs *only* exactly at the boundary it is an artifact rather than a bug. That
//! machinery is built for a *metamorphic gradient* oracle, where the numerical gradient
//! `(f(x+h) - f(x)) / h` is genuinely meaningless at a kink. **A differential oracle
//! comparing forward values has no such problem**: both backends receive bit-identical
//! inputs, so there is no perturbation for a kink to amplify.
//!
//! So rather than build a filter for a problem that may not exist here, these tests
//! *measure whether one exists*. If boundary cases agree, that is the finding, and the
//! filter is not needed until the metamorphic oracle arrives.

use diff_fuzzer_core::{Agreement, ApproxEq, Implementation, Normalizer, TolerancePolicy};
use tensor_adapter::{
    BinaryOp, ReduceOp, TensorNormalizer, TensorOp, TensorTolerancePolicy, TensorValue, UnaryOp,
    libtorch, ndarray,
};

/// Run a case on both backends and return their canonical results.
fn on_both(case: &TensorOp) -> (Vec<f32>, Vec<f32>) {
    let cpu = TensorNormalizer.normalize(ndarray().run(case).expect("valid case"));
    let torch = TensorNormalizer.normalize(libtorch().run(case).expect("valid case"));
    (cpu.values, torch.values)
}

/// Do the backends agree under the project's own tolerance policy?
fn agree(case: &TensorOp) -> bool {
    let cpu = TensorNormalizer.normalize(ndarray().run(case).expect("valid case"));
    let torch = TensorNormalizer.normalize(libtorch().run(case).expect("valid case"));
    let tolerance = TensorTolerancePolicy.tolerance_for(case, ("burn-ndarray", "burn-tch"));

    matches!(cpu.approx_compare(&torch, tolerance), Agreement::Agree(_))
}

fn value(data: &[f32]) -> TensorValue {
    TensorValue::new(vec![data.len()], data.to_vec())
}

/// The zeros and near-zeros most likely to expose a difference.
const BOUNDARY_VALUES: [f32; 7] = [
    0.0,
    -0.0,
    f32::MIN_POSITIVE, // smallest normal
    -f32::MIN_POSITIVE,
    1e-45, // smallest subnormal
    1.0,
    -1.0,
];

/// Unary operations evaluated exactly at their boundaries.
///
/// `abs` at zero (the sign flips), `sqrt` at zero (the derivative is infinite), `neg` at
/// zero (produces the other zero), `exp` at zero (exactly 1).
#[test]
fn unary_operations_agree_at_their_boundaries() {
    for kind in [UnaryOp::Abs, UnaryOp::Neg, UnaryOp::Exp] {
        let case = TensorOp::unary(kind, value(&BOUNDARY_VALUES));
        assert!(agree(&case), "{} disagreed at a boundary", case.name());
    }

    // `sqrt` only over its defined domain.
    let non_negative: Vec<f32> = BOUNDARY_VALUES
        .iter()
        .copied()
        .filter(|v| *v >= 0.0)
        .collect();
    let case = TensorOp::unary(UnaryOp::Sqrt, value(&non_negative));
    assert!(agree(&case), "sqrt disagreed at a boundary");
}

/// Exact cancellation: `x - x` and `x + (-x)` both collapse to zero regardless of how
/// large `x` was. This is the boundary that matters most for a differential oracle,
/// because it is where relative error stops being meaningful.
#[test]
fn exact_cancellation_agrees() {
    let magnitudes = value(&[1.0, 1e10, 1e-10, 3.7, f32::MAX]);
    let negated = value(&[-1.0, -1e10, -1e-10, -3.7, -f32::MAX]);

    let subtracted = TensorOp::binary(BinaryOp::Sub, magnitudes.clone(), magnitudes.clone());
    let added = TensorOp::binary(BinaryOp::Add, magnitudes, negated);

    for case in [subtracted, added] {
        let (cpu, torch) = on_both(&case);
        assert!(
            cpu.iter().all(|v| *v == 0.0),
            "expected exact zeros: {cpu:?}"
        );
        assert_eq!(
            cpu,
            torch,
            "{} disagreed on exact cancellation",
            case.name()
        );
    }
}

/// A sum whose terms cancel to exactly zero — the case where a relative tolerance would
/// be useless and the absolute one has to carry the comparison.
#[test]
fn a_sum_cancelling_to_zero_agrees() {
    let case = TensorOp::reduce(ReduceOp::Sum, value(&[1.0, -1.0, 2.5, -2.5, 0.0]), 0);

    let (cpu, torch) = on_both(&case);
    assert_eq!(cpu, torch, "sum disagreed when its terms cancelled");
    assert!(agree(&case));
}

/// Multiplication producing a signed zero, which is where the two zeros are most easily
/// distinguished.
#[test]
fn multiplication_producing_signed_zero_agrees() {
    let case = TensorOp::binary(
        BinaryOp::Mul,
        value(&[0.0, -0.0, 0.0, -0.0]),
        value(&[1.0, 1.0, -1.0, -1.0]),
    );
    assert!(agree(&case), "mul disagreed producing signed zeros");
}

/// **The one boundary difference that would be invisible to the comparison.**
///
/// `0.0 == -0.0` is true in floating-point, so two backends disagreeing about the *sign*
/// of a zero would be reported as agreement. That is a defensible policy — the values
/// are numerically equal, and nothing downstream in this project divides by them — but
/// it is a difference the tool cannot see, and an invisible blind spot is worse than a
/// known one.
///
/// This test inspects the sign bits directly, outside the comparison, to establish
/// whether such a difference actually occurs. It asserts what was measured rather than
/// what would be convenient.
#[test]
fn the_backends_agree_on_the_sign_of_zero() {
    let producing_zeros = [
        TensorOp::unary(UnaryOp::Neg, value(&[0.0, -0.0])),
        TensorOp::unary(UnaryOp::Abs, value(&[0.0, -0.0])),
        TensorOp::unary(UnaryOp::Sqrt, value(&[0.0])),
        TensorOp::binary(BinaryOp::Mul, value(&[0.0, -0.0]), value(&[-1.0, -1.0])),
        TensorOp::binary(BinaryOp::Sub, value(&[0.0, 5.0]), value(&[0.0, 5.0])),
    ];

    for case in producing_zeros {
        let (cpu, torch) = on_both(&case);

        for (index, (a, b)) in cpu.iter().zip(&torch).enumerate() {
            if *a == 0.0 && *b == 0.0 {
                assert_eq!(
                    a.is_sign_negative(),
                    b.is_sign_negative(),
                    "{} element {index}: zeros of different sign ({a} vs {b}) — \
                     numerically equal, so the comparison would report agreement",
                    case.name()
                );
            }
        }
    }
}

/// Subnormals: numbers below the smallest normal value, which trade precision for range.
/// Some hardware and some libraries flush them to zero as an optimisation, which is a
/// legitimate implementation choice and would be a real, visible difference here.
#[test]
fn subnormal_inputs_agree() {
    let subnormals = value(&[1e-45, 5e-44, 1e-40, f32::MIN_POSITIVE / 2.0]);

    for kind in [UnaryOp::Abs, UnaryOp::Neg] {
        let case = TensorOp::unary(kind, subnormals.clone());
        assert!(agree(&case), "{} disagreed on subnormals", case.name());
    }

    let doubled = TensorOp::binary(BinaryOp::Add, subnormals.clone(), subnormals);
    assert!(agree(&doubled), "add disagreed on subnormals");
}

/// Neither backend may flush subnormals to zero without the other doing the same. Worth
/// checking separately from agreement: if *both* flushed, they would agree while
/// silently behaving differently from the standard.
#[test]
fn subnormals_are_not_silently_flushed_to_zero() {
    let case = TensorOp::unary(UnaryOp::Abs, value(&[1e-45, 5e-44]));
    let (cpu, torch) = on_both(&case);

    assert!(
        cpu.iter().all(|v| *v != 0.0),
        "cpu backend flushed subnormals to zero: {cpu:?}"
    );
    assert!(
        torch.iter().all(|v| *v != 0.0),
        "libtorch backend flushed subnormals to zero: {torch:?}"
    );
}
