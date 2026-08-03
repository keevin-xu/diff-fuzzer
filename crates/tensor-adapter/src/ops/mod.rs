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

    /// How often a value is drawn from [`SPECIAL_VALUES`] instead of uniformly.
    ///
    /// Uniform sampling over a continuous range **never produces the interesting
    /// numbers**. The probability of drawing exactly `0.0`, or `1.0`, or a subnormal,
    /// is nil — so a million-case campaign can run without once testing what an
    /// operation does with zero. Bugs cluster at exactly those values, which is why
    /// they have to be injected deliberately rather than waited for.
    pub special_value_rate: f64,

    /// Whether arguments are confined to each operation's defined domain.
    ///
    /// When `true` (the default), `sqrt` receives only non-negatives and `div` only
    /// non-zero divisors, so no operation produces `NaN` or infinity. When `false`,
    /// those restrictions lift and undefined results occur — which is the point: those
    /// are the numerically interesting cases, and the comparison now has an explicit
    /// policy for them.
    pub restrict_domains: bool,
}

impl Default for Bounds {
    fn default() -> Self {
        Self {
            max_rank: crate::backends::MAX_RANK,
            max_dim: 8,
            magnitude: 10.0,
            // Roughly one value in eight. High enough that most cases contain at least
            // one interesting value, low enough that ordinary arithmetic still dominates
            // and the operations are exercised on realistic data too.
            special_value_rate: 0.125,
            restrict_domains: true,
        }
    }
}

/// How far from zero a divisor is kept while domain restrictions are in force.
///
/// Not merely non-zero: a divisor of `1e-45` is non-zero and still overflows the
/// quotient, which says more about floating-point range than about either backend.
pub const DIVISOR_FLOOR: f32 = 0.5;

/// Values worth testing on purpose, because random sampling will not find them.
///
/// Each is here for a reason: the zeros because sign is observable and division by them
/// is undefined; `±1` because they are the identities and a wrong one is easy to miss;
/// the smallest normal and the smallest subnormal because precision degrades below them
/// and some implementations flush them away; the extremes because they are where
/// overflow and underflow begin.
pub const SPECIAL_VALUES: [f32; 10] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    f32::MIN_POSITIVE,
    -f32::MIN_POSITIVE,
    1e-45, // smallest positive subnormal
    -1e-45,
    1e30,
    -1e30,
];

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
    /// Bounded away from zero by [`DIVISOR_FLOOR`] — divisors, which would otherwise
    /// produce overflowing quotients.
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
    (0..count)
        .map(|_| {
            if rng.random_bool(bounds.special_value_rate) {
                special_value(rng, domain, bounds)
            } else {
                uniform_value(rng, domain, bounds)
            }
        })
        .collect()
}

/// An ordinary value drawn uniformly from the operation's domain.
fn uniform_value(rng: &mut SeededRng, domain: Domain, bounds: &Bounds) -> f32 {
    let m = bounds.magnitude;
    match domain {
        Domain::Any => rng.random_range(-m..m),
        Domain::NonNegative => rng.random_range(0.0..m),
        // A divisor near zero produces a huge quotient that says more about
        // floating-point range than about either backend, so the magnitude is kept away
        // from zero on both sides.
        Domain::NonZero => {
            let magnitude = rng.random_range(DIVISOR_FLOOR..m);
            if rng.random_bool(0.5) {
                magnitude
            } else {
                -magnitude
            }
        }
    }
}

/// One of the deliberately interesting values, respecting the operation's domain.
///
/// The domain filter matters: offering `-1.0` to `sqrt` while domains are restricted
/// would break the very guarantee the restriction exists to provide. When restrictions
/// are lifted the domain is `Any`, and every special value becomes reachable.
fn special_value(rng: &mut SeededRng, domain: Domain, bounds: &Bounds) -> f32 {
    let allowed: Vec<f32> = SPECIAL_VALUES
        .iter()
        .copied()
        .filter(|v| match domain {
            Domain::Any => true,
            Domain::NonNegative => *v >= 0.0,
            // `NonZero` means *bounded away from* zero, not merely unequal to it. A
            // divisor of `1e-45` is non-zero and still produces an overflowing quotient,
            // which would say more about floating-point range than about either backend
            // — the exact noise this restriction exists to keep out while it is in
            // force. Matches the threshold the uniform path uses.
            Domain::NonZero => v.abs() >= DIVISOR_FLOOR,
        })
        .collect();

    // Every domain leaves some special values available, so this cannot be empty; the
    // fallback keeps the function total rather than relying on that reasoning holding
    // if the table changes.
    if allowed.is_empty() {
        return uniform_value(rng, domain, bounds);
    }

    allowed[rng.random_range(0..allowed.len())]
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
                    .all(|&v| v.abs() >= DIVISOR_FLOOR)
            );
            // Finite while domains are restricted — but *not* necessarily within
            // `magnitude`, because special values deliberately reach past it. That is
            // their purpose: the extremes are where overflow and underflow begin.
            assert!(
                values(rng, 16, Domain::Any, &bounds)
                    .iter()
                    .all(|&v| v.is_finite()
                        && (v.abs() <= bounds.magnitude || SPECIAL_VALUES.contains(&v)))
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

    /// The interesting values must actually turn up. Uniform sampling never produces
    /// them, so if injection were broken, zero and one would simply never be tested and
    /// nothing else in the suite would notice.
    #[test]
    fn special_values_actually_appear() {
        let bounds = Bounds::default();
        let mut rng = SeededRng::from_seed(0);
        let drawn = values(&mut rng, 5_000, Domain::Any, &bounds);

        for special in SPECIAL_VALUES {
            assert!(
                drawn.iter().any(|v| v.to_bits() == special.to_bits()),
                "{special} was never generated"
            );
        }
    }

    /// Turning the rate off must turn them off entirely — a knob that does nothing is
    /// worse than no knob, because it invites false confidence.
    #[test]
    fn a_zero_rate_produces_no_special_values() {
        let bounds = Bounds {
            special_value_rate: 0.0,
            ..Bounds::default()
        };
        let mut rng = SeededRng::from_seed(0);

        for value in values(&mut rng, 2_000, Domain::Any, &bounds) {
            assert!(value.abs() <= bounds.magnitude, "{value} exceeds the bound");
        }
    }

    /// Ordinary arithmetic must still dominate. If nearly every value were special, the
    /// operations would only ever be exercised on edge cases and never on realistic
    /// data.
    #[test]
    fn ordinary_values_still_dominate() {
        let bounds = Bounds::default();
        let mut rng = SeededRng::from_seed(0);
        let drawn = values(&mut rng, 5_000, Domain::Any, &bounds);

        let special = drawn
            .iter()
            .filter(|v| SPECIAL_VALUES.iter().any(|s| s.to_bits() == v.to_bits()))
            .count();
        assert!(
            special < drawn.len() / 2,
            "{special} of {} values were special",
            drawn.len()
        );
    }

    /// Domain restrictions must hold even for injected values — otherwise the
    /// restriction would be quietly defeated by the very mechanism meant to stress it.
    #[test]
    fn special_values_respect_domain_restrictions() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            assert!(
                values(rng, 16, Domain::NonNegative, &bounds)
                    .iter()
                    .all(|&v| v >= 0.0)
            );
        });
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
