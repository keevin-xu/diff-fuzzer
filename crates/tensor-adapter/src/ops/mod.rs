//! Building valid arguments for each operation.
//!
//! The principle is **correct by construction**: generate arguments that already
//! satisfy an operation's rules, rather than generating freely and discarding what
//! fails. The difference matters more than it sounds. An input rejected as malformed
//! exercises nothing but the validation code, so a generator with a low validity rate
//! spends its time proving that shape checks work — while the kernels it was built to
//! test go unexercised.
//!
//! Modules are organised by **constraint shape**, not one per operation, because that
//! is how the constraints actually cluster: `add`, `sub`, `mul` and `div` share
//! "operands must have equal shapes", and would otherwise be four copies of one rule.
//!
//! - [`unary`] — one argument, result keeps the shape
//! - [`binary`] — two arguments of identical shape
//! - [`reduce`] — one argument plus an axis that must be within its rank
//! - [`matmul`] — inner dimensions must agree, batch dimensions must match

pub mod binary;
pub mod matmul;
pub mod reduce;
pub mod unary;

use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Limits on what the generator may produce.
///
/// Kept small on purpose. Tiny tensors execute quickly, and a fuzzer's yield depends
/// on how many cases it gets through — but more importantly, a divergence found on a
/// 2x3 tensor is already nearly a minimal reproduction, while the same bug found on a
/// 500x500 one would need shrinking before anyone could act on it.
///
/// The competing risk is generating only trivial cases and so missing everything, and
/// these bounds are the dial between the two. They are widened once validity is
/// established rather than guessed at now.
#[derive(Debug, Clone, Copy)]
pub struct Bounds {
    /// Highest rank to generate. Cannot exceed the backend's `MAX_RANK`, since each
    /// rank is a separate dispatch arm there.
    pub max_rank: usize,
    /// Largest length of any single dimension. Dimensions of 1 are allowed and
    /// deliberately common — degenerate shapes are a classic source of bugs.
    pub max_dim: usize,
    /// Values are drawn from roughly `-magnitude..magnitude`.
    pub magnitude: f32,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            max_rank: crate::backends::MAX_RANK,
            max_dim: 8,
            magnitude: 10.0,
        }
    }
}

/// Which values an operation is willing to accept.
///
/// Some operations are undefined on part of the number line. Rather than let them
/// produce `NaN` and `inf`, arguments are drawn from the region where the operation is
/// defined — for now. This is a sequencing choice, not a permanent one: those extremes
/// are exactly where implementations tend to part company, so the restriction gets
/// lifted deliberately once there is a policy for comparing a `NaN` against a `NaN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    /// Any finite value.
    Any,
    /// Zero or above — `sqrt` is undefined below it.
    NonNegative,
    /// Bounded away from zero — divisors, which would otherwise produce infinities.
    NonZero,
}

/// A shape with a random rank and random dimensions, within `bounds`.
///
/// Every dimension is at least 1, so the shape always describes at least one element.
pub fn shape(rng: &mut SeededRng, bounds: &Bounds) -> Vec<usize> {
    let rank = rng.random_range(1..=bounds.max_rank);
    (0..rank)
        .map(|_| rng.random_range(1..=bounds.max_dim))
        .collect()
}

/// A shape of exactly `rank` dimensions.
///
/// Needed where rank is not free to vary — `matmul` requires at least two dimensions,
/// so it picks its rank first and then asks for a shape of that size.
pub fn shape_of_rank(rng: &mut SeededRng, rank: usize, bounds: &Bounds) -> Vec<usize> {
    (0..rank)
        .map(|_| rng.random_range(1..=bounds.max_dim))
        .collect()
}

/// `count` values drawn from `domain`.
///
/// Everything is `f32` for now. A second element type would double every case the
/// oracle has to reason about, and is worth adding only once one type is trustworthy.
pub fn values(rng: &mut SeededRng, count: usize, domain: Domain, bounds: &Bounds) -> Vec<f32> {
    let m = bounds.magnitude;
    (0..count)
        .map(|_| match domain {
            Domain::Any => rng.random_range(-m..m),
            Domain::NonNegative => rng.random_range(0.0..m),
            // A divisor near zero produces a huge quotient that says more about
            // floating-point range than about either backend, so the magnitude is kept
            // away from zero on both sides.
            Domain::NonZero => {
                let magnitude = rng.random_range(0.5..m);
                if rng.random_bool(0.5) {
                    magnitude
                } else {
                    -magnitude
                }
            }
        })
        .collect()
}

/// Total number of elements in a shape.
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `f` across many seeds. Constraints must hold for *every* generated case, so
    /// checking one is close to meaningless — these are cheap, so the count is high.
    pub(crate) fn for_many_seeds(mut f: impl FnMut(&mut SeededRng)) {
        for seed in 0..500 {
            f(&mut SeededRng::from_seed(seed));
        }
    }

    #[test]
    fn shapes_stay_within_bounds() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let shape = shape(rng, &bounds);
            assert!((1..=bounds.max_rank).contains(&shape.len()), "{shape:?}");
            assert!(
                shape.iter().all(|&d| (1..=bounds.max_dim).contains(&d)),
                "{shape:?}"
            );
        });
    }

    #[test]
    fn values_respect_their_domain() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            assert!(
                values(rng, 16, Domain::NonNegative, &bounds)
                    .iter()
                    .all(|&v| v >= 0.0)
            );
            assert!(
                values(rng, 16, Domain::NonZero, &bounds)
                    .iter()
                    .all(|&v| v.abs() >= 0.5)
            );
            assert!(
                values(rng, 16, Domain::Any, &bounds)
                    .iter()
                    .all(|&v| v.is_finite() && v.abs() <= bounds.magnitude)
            );
        });
    }

    /// Both signs must actually appear for `NonZero`, or the restriction has quietly
    /// become "positive divisors only" and half the cases would never be generated.
    #[test]
    fn non_zero_values_take_both_signs() {
        let bounds = Bounds::default();
        let mut rng = SeededRng::from_seed(0);
        let vs = values(&mut rng, 200, Domain::NonZero, &bounds);
        assert!(vs.iter().any(|&v| v > 0.0));
        assert!(vs.iter().any(|&v| v < 0.0));
    }

    #[test]
    fn generation_is_deterministic() {
        let bounds = Bounds::default();
        let run = |seed| {
            let mut rng = SeededRng::from_seed(seed);
            (
                shape(&mut rng, &bounds),
                values(&mut rng, 8, Domain::Any, &bounds),
            )
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
    }
}
