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
pub const ALL: [UnaryOp; 6] = [
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Exp,
    UnaryOp::Sqrt,
    UnaryOp::Log,
    UnaryOp::Erf,
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
        // `erf` is defined on the whole real line and bounded in [-1, 1]; nothing to
        // restrict.
        UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Exp | UnaryOp::Erf => Domain::Any,
    }
}

/// Where `libm`'s `erff` switches between rational approximations.
///
/// **A switch point is a property of the input that selects a code path** — the same shape as
/// the tile remainder behind the one bug filed upstream. Two implementations choosing
/// different boundaries disagree most sharply just either side of one, so the generator aims
/// there rather than waiting to land nearby by chance.
///
/// Read from `burn-flex`'s source, not guessed: `SPECS.md` §2b.5.
const ERF_SWITCH_POINT: f32 = 0.84375;

/// How often an `erf` case is placed right at that boundary.
const ERF_AT_SWITCH_SHARE: f64 = 0.35;

/// Values straddling `libm`'s switch point, within a few ULP either side.
fn erf_switch_values(rng: &mut SeededRng, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            // A handful of ULP either side, so both branches are exercised and the pairing is
            // as close as the format allows.
            let steps = rng.random_range(0..8i32) - 4;
            let mut value = ERF_SWITCH_POINT;
            for _ in 0..steps.abs() {
                value = if steps > 0 {
                    f32::from_bits(value.to_bits() + 1)
                } else {
                    f32::from_bits(value.to_bits() - 1)
                };
            }
            sign * value
        })
        .collect()
}

/// Build a valid unary case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];
    let shape = shape(rng, bounds);
    let count = element_count(&shape);
    // **`erf` is aimed at its switch point**; everything else is drawn from its domain.
    let data = if kind == UnaryOp::Erf && rng.random_bool(ERF_AT_SWITCH_SHARE) {
        erf_switch_values(rng, count)
    } else {
        values(rng, count, domain(kind, bounds), bounds)
    };

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
