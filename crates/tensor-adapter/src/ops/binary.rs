//! Two arguments of identical shape, combined elementwise.
//!
//! The shape constraint is satisfied by generating **one** shape and using it for both
//! operands. That is the whole idea of correct-by-construction in miniature: rather
//! than generating two shapes and checking whether they happen to match — which they
//! almost never would — the constraint is built into how the case is made.
//!
//! **Since PHASE-7C the shapes need not be equal.** Operands are drawn as a compatible
//! *pair* — see [`crate::ops::broadcast`] — so a `[3, 1]` and a `[3, 4]` combine by
//! stretching the length-1 axis. The correct-by-construction principle is unchanged: the
//! pair is derived from a single result shape, so it cannot be incompatible.
//!
//! Equal shapes remain the most common case by design. They are the ordinary path in real
//! use and are where every finding so far has come from, so a generator that always
//! broadcast would have quietly stopped testing what it used to.

use crate::input::{BinaryOp, TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, broadcast, element_count, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every binary operation the generator may pick.
pub const ALL: [BinaryOp; 4] = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];

/// The values the *right-hand* operand is defined on. Only division restricts it, and
/// only on that side — a zero numerator is perfectly well behaved.
///
/// When restrictions are lifted, a divisor may be zero and the result becomes infinite
/// (or `NaN`, for `0/0`) — the case the special-value policy exists to judge.
fn right_domain(kind: BinaryOp, bounds: &Bounds) -> Domain {
    match kind {
        BinaryOp::Div if bounds.restrict_domains => Domain::NonZero,
        BinaryOp::Div | BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => Domain::Any,
    }
}

/// Build a valid elementwise binary case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];

    // A compatible pair, derived from one result shape. As with the single-shape version
    // it replaced, the constraint cannot be violated because the operands are *built from*
    // the answer rather than checked against it.
    let (lhs_shape, rhs_shape) = broadcast::pair(rng, bounds);

    // Each operand carries only its own elements — that is the point of broadcasting, and
    // it is why a stretched operand is cheaper to generate than the result it produces.
    let lhs_count = element_count(&lhs_shape);
    let rhs_count = element_count(&rhs_shape);

    let lhs = TensorValue::new(lhs_shape, values(rng, lhs_count, Domain::Any, bounds));
    let rhs = TensorValue::new(
        rhs_shape,
        values(rng, rhs_count, right_domain(kind, bounds), bounds),
    );

    TensorOp::binary(kind, lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    /// Renamed from `operands_always_share_a_shape` at PHASE-7C: they no longer must, but
    /// they must still **combine**, which is the constraint that actually matters.
    #[test]
    fn operands_always_combine() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Binary { lhs, rhs, .. } = &case else {
                panic!("binary generator produced {case:?}");
            };
            assert!(
                broadcast::compatible(lhs.shape(), rhs.shape()),
                "generated incompatible operands: {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        });
    }

    /// Broadcasting must actually occur, or the change is cosmetic.
    #[test]
    fn broadcasting_cases_are_produced_alongside_equal_shaped_ones() {
        let bounds = Bounds::default();
        let mut broadcasting = 0;
        let mut equal = 0;
        for seed in 0..2_000u64 {
            let mut rng = SeededRng::from_seed(seed);
            if let TensorOp::Binary { lhs, rhs, .. } = generate(&mut rng, &bounds) {
                if lhs.shape() == rhs.shape() {
                    equal += 1;
                } else {
                    broadcasting += 1;
                }
            }
        }
        assert!(
            broadcasting > 100,
            "too few broadcast cases: {broadcasting}"
        );
        assert!(equal > 100, "equal shapes became rare: {equal}");
    }

    #[test]
    #[ignore = "superseded at PHASE-7C by operands_always_combine"]
    fn operands_always_share_a_shape() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Binary { lhs, rhs, .. } = &case else {
                panic!("expected a binary case, got {case:?}");
            };
            assert_eq!(lhs.shape(), rhs.shape());
            assert_eq!(lhs.len(), rhs.len());
        });
    }

    /// A divisor of zero would produce infinity, which is a statement about
    /// floating-point range rather than about either backend.
    #[test]
    fn division_never_receives_a_zero_divisor() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            if let TensorOp::Binary {
                kind: BinaryOp::Div,
                rhs,
                ..
            } = &case
            {
                assert!(rhs.data().iter().all(|&v| v != 0.0), "{rhs:?}");
            }
        });
    }

    /// The restriction applies to the divisor only. If it had leaked onto the left
    /// operand, a whole class of inputs — dividing zero by something — would silently
    /// never be tested.
    #[test]
    fn division_still_allows_a_zero_numerator_region() {
        let bounds = Bounds::default();
        let mut small_numerators = 0;
        for seed in 0..500 {
            let case = generate(&mut SeededRng::from_seed(seed), &bounds);
            if let TensorOp::Binary {
                kind: BinaryOp::Div,
                lhs,
                ..
            } = &case
                && lhs.data().iter().any(|&v| v.abs() < 0.5)
            {
                small_numerators += 1;
            }
        }
        assert!(
            small_numerators > 0,
            "the divisor restriction appears to have leaked onto the numerator"
        );
    }

    #[test]
    fn every_binary_operation_gets_generated() {
        let bounds = Bounds::default();
        let mut seen = std::collections::HashSet::new();
        for seed in 0..500 {
            if let TensorOp::Binary { kind, .. } =
                generate(&mut SeededRng::from_seed(seed), &bounds)
            {
                seen.insert(kind);
            }
        }
        for kind in ALL {
            assert!(seen.contains(&kind), "{kind:?} was never generated");
        }
    }
}
