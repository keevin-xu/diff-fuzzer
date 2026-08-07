//! One argument in, a result of the same shape out.
//!
//! The only constraint is on *values*, not shapes: `sqrt` is undefined below zero, so
//! its argument is drawn from the non-negative side. Every other unary operation
//! accepts anything finite.

use crate::input::{TensorOp, TensorValue, UnaryOp};
use crate::ops::{Bounds, Domain, element_count, shape, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every unary operation the generator may pick.
pub const ALL: [UnaryOp; 5] = [
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Exp,
    UnaryOp::Sqrt,
    UnaryOp::Log,
];

/// The values an operation is defined on.
///
/// When domain restrictions are lifted, `sqrt` is offered negatives and produces `NaN` —
/// which is the interesting case, now that the comparison has an explicit policy for
/// undefined results.
fn domain(kind: UnaryOp, bounds: &Bounds) -> Domain {
    match kind {
        UnaryOp::Sqrt if bounds.restrict_domains => Domain::NonNegative,
        UnaryOp::Sqrt => Domain::Any,
        // `log` has the same domain as `sqrt` — undefined below zero, and `-inf` at zero,
        // which the unrestricted setting deliberately reaches.
        UnaryOp::Log if bounds.restrict_domains => Domain::NonNegative,
        UnaryOp::Log => Domain::Any,
        // `exp` is defined everywhere; it overflows to infinity for large arguments,
        // but the magnitude bound keeps generated inputs well short of that. Removing
        // the bound is one of the interesting things to try later.
        UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Exp => Domain::Any,
    }
}

/// Build a valid unary case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];
    let shape = shape(rng, bounds);
    let data = values(rng, element_count(&shape), domain(kind, bounds), bounds);

    TensorOp::unary(kind, TensorValue::new(shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    #[test]
    fn generated_cases_are_unary_and_within_bounds() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Unary { arg, .. } = &case else {
                panic!("expected a unary case, got {case:?}");
            };
            assert!((1..=bounds.max_rank).contains(&arg.rank()));
            assert_eq!(arg.len(), element_count(arg.shape()));
        });
    }

    /// The constraint that actually matters here: `sqrt` must never be handed a
    /// negative number while domains are restricted, or it would produce `NaN` and the
    /// comparison would report a difference that says nothing about either backend.
    #[test]
    fn sqrt_never_receives_a_negative_argument() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            if let TensorOp::Unary {
                kind: UnaryOp::Sqrt,
                arg,
            } = &case
            {
                assert!(arg.data().iter().all(|&v| v >= 0.0), "{arg:?}");
            }
        });
    }

    /// Every operation must actually get generated. A `match` arm that is never
    /// reached is an operation that is never tested, and nothing would otherwise say
    /// so.
    #[test]
    fn every_unary_operation_gets_generated() {
        let bounds = Bounds::default();
        let mut seen = std::collections::HashSet::new();
        for seed in 0..500 {
            let case = generate(&mut SeededRng::from_seed(seed), &bounds);
            if let TensorOp::Unary { kind, .. } = case {
                seen.insert(kind);
            }
        }
        for kind in ALL {
            assert!(seen.contains(&kind), "{kind:?} was never generated");
        }
    }
}
