//! Two arguments of identical shape, combined elementwise.
//!
//! The shape constraint is satisfied by generating **one** shape and using it for both
//! operands. That is the whole idea of correct-by-construction in miniature: rather
//! than generating two shapes and checking whether they happen to match — which they
//! almost never would — the constraint is built into how the case is made.
//!
//! Note this only covers equal shapes. Real libraries also allow *broadcasting*,
//! where a `[3, 1]` and a `[3, 4]` operand combine by stretching the length-1 axis.
//! That is a richer constraint and a good source of bugs, and it is deliberately left
//! for a later batch rather than added here as an afterthought.

use crate::input::{BinaryOp, TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, element_count, shape, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every binary operation the generator may pick.
pub const ALL: [BinaryOp; 4] = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div];

/// The values the *right-hand* operand is defined on. Only division restricts it, and
/// only on that side — a zero numerator is perfectly well behaved.
fn right_domain(kind: BinaryOp) -> Domain {
    match kind {
        BinaryOp::Div => Domain::NonZero,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => Domain::Any,
    }
}

/// Build a valid elementwise binary case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];

    // One shape, used twice. The constraint cannot be violated because there is only
    // ever one shape to violate it with.
    let shape = shape(rng, bounds);
    let count = element_count(&shape);

    let lhs = TensorValue::new(shape.clone(), values(rng, count, Domain::Any, bounds));
    let rhs = TensorValue::new(shape, values(rng, count, right_domain(kind), bounds));

    TensorOp::binary(kind, lhs, rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    #[test]
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
