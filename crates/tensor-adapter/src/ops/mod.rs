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

pub mod activation;
pub mod binary;
pub mod broadcast;
pub mod conv;
pub mod matmul;
pub mod reduce;
pub mod scan;
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

    /// The most elements one operand may have.
    ///
    /// **`max_rank` and `max_dim` multiply, so bounding each does not bound the case.** At
    /// rank 4 and `max_dim: 64` a shape reaches 64⁴ = 16.7 million elements, and a matmul's
    /// cost is a further factor of `n` on top. This is the only field that bounds the case
    /// itself.
    ///
    /// **It is a real trade, not free.** Measured: at `max_dim: 64`, lowering this to 4,096
    /// took the divergence rate from 9 in 2,000 to **0 in 2,000** — the large shapes that
    /// cost the time were the same ones that produced the disagreements. Raise it to find
    /// more per case; lower it to run more cases.
    pub max_elements: usize,

    /// **Which operation classes the generator may emit.**
    ///
    /// Added at PHASE-7F, and the reason is measured rather than anticipated: a six-hour
    /// campaign produced 1,834 findings, *every one* of them the same `max`/`min` ordering
    /// disagreement, while `softmax`, `log` and `mean` produced nothing. The easily-reached
    /// class saturated the corpus and crowded out the operations whose answers were unknown.
    ///
    /// **A campaign's configuration should exclude the axes whose answers are known.** The
    /// SQL adapter reached that conclusion independently, which is why the mechanism now
    /// lives in the engine as [`GenerationAxes`] rather than in either adapter.
    ///
    /// Every axis is on by default, so nothing changes unless a campaign deliberately
    /// narrows — and **enabling an axis adds cases, it never removes them**.
    pub unary_ops: bool,
    /// Elementwise binary operations, including broadcasting.
    pub binary_ops: bool,
    /// `sum` and `mean` — reductions that accumulate.
    pub accumulating_reductions: bool,
    /// `max` and `min` — reductions that select an input unchanged.
    ///
    /// The axis most likely to be switched *off*: its disagreement is understood, and
    /// generating more of it buys nothing.
    pub selecting_reductions: bool,
    /// `matmul`.
    pub matmul: bool,
    /// `softmax`.
    pub activations: bool,
    /// `cumsum` — running results along an axis.
    ///
    /// Its own axis rather than folded into the reductions, because it is the only operation
    /// whose backends may legitimately *associate differently*, and a campaign hunting a
    /// numeric disagreement wants it alone.
    pub scans: bool,
    /// `conv2d`.
    ///
    /// Its own axis because it is the only operation whose three backends run genuinely
    /// different *algorithms* rather than the same arithmetic in a different order, and a
    /// campaign hunting an algorithmic disagreement wants it alone.
    pub convolution: bool,

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
            // 8⁴ — the worst case the historical `max_dim: 8` regime already allowed, so
            // this default changes nothing about how the old campaigns behaved.
            max_elements: 4_096,
            // Every operation class on by default: narrowing is a deliberate act.
            unary_ops: true,
            binary_ops: true,
            accumulating_reductions: true,
            selecting_reductions: true,
            matmul: true,
            activations: true,
            scans: true,
            convolution: true,
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
///
/// # `NaN` and `±inf`, added at PHASE-7E — and why they were missing
///
/// Every entry above is **finite**, so no generated or decoded case could ever contain a
/// non-finite *input*. That was invisible until it hid a real result: `max([1, NaN, 3])`
/// returns `NaN` on both CPU backends and `3.0` on the GPU — a semantic disagreement no
/// tolerance can absorb, found by a hand-built probe and **unreachable by a campaign of any
/// length**. A four-hour run would have reported zero findings and looked clean.
///
/// The `input_special` feature had measured 0 of 20,000 since PHASE-7B and was noted as an
/// example of validation's "not reachable" outcome. It was a curiosity until an actual
/// finding turned out to live behind it.
///
/// **Gated on `restrict_domains`**, like every other route into the undefined region. The
/// seeded generator's default stays finite, so every distribution measured before this is
/// unchanged; the fuzzer runs unrestricted and reaches them.
///
/// **The cost is real and expected:** more campaign cases will be `Skipped` rather than
/// judged, because two backends both returning `NaN` have compared nothing. That is the
/// policy working — a skip is an honest "no opinion", not a pass — but it means the judged
/// fraction drops, and the skip column has to be read alongside the case count.
pub const SPECIAL_VALUES: [f32; 13] = [
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
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
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
    let raw = (0..rank)
        .map(|_| rng.random_range(1..=bounds.max_dim))
        .collect();
    clamp_to(raw, bounds.max_elements)
}

/// Shrink dimensions, largest first, until the total fits `budget`.
///
/// Clamping rather than rejecting: for the decoder a rejected input teaches the fuzzer
/// nothing, and for the generator a rejected draw would silently skew the distribution
/// toward small ranks. `matmul` passes a reduced budget, because its operand is the batch
/// dimensions *times* `m × k`.
pub fn clamp_to(mut shape: Vec<usize>, budget: usize) -> Vec<usize> {
    let budget = budget.max(1);
    while element_count(&shape) > budget {
        let Some(largest) = shape.iter_mut().max() else {
            break;
        };
        if *largest <= 1 {
            break;
        }
        *largest /= 2;
    }
    shape
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
        // **Non-finite inputs are gated on `restrict_domains`, like every other way of
        // reaching the undefined region.** Restricted mode means well-behaved arguments, and
        // a `NaN` input is exactly what it exists to exclude; the fuzzer runs unrestricted
        // and so reaches them. Without this gate, adding them to the table would have changed
        // every seeded distribution measured so far, for no gain — the campaign is where they
        // are wanted.
        .filter(|v| v.is_finite() || !bounds.restrict_domains)
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

impl diff_fuzzer_core::GenerationAxes for Bounds {
    fn axes(&self) -> Vec<(&'static str, bool)> {
        vec![
            ("unary", self.unary_ops),
            ("binary", self.binary_ops),
            ("accumulating_reductions", self.accumulating_reductions),
            ("selecting_reductions", self.selecting_reductions),
            ("matmul", self.matmul),
            ("activations", self.activations),
            ("scans", self.scans),
            ("convolution", self.convolution),
            ("unrestricted_domains", !self.restrict_domains),
        ]
    }

    fn scalars(&self) -> Vec<(&'static str, String)> {
        vec![
            ("max_rank", self.max_rank.to_string()),
            ("max_dim", self.max_dim.to_string()),
            ("magnitude", self.magnitude.to_string()),
            ("special_rate", self.special_value_rate.to_string()),
            ("max_elements", self.max_elements.to_string()),
        ]
    }
}

impl Bounds {
    /// Every operation class enabled — the default, and what every campaign before PHASE-7F
    /// ran.
    pub const ALL_OPERATIONS: Self = Self {
        unary_ops: true,
        binary_ops: true,
        accumulating_reductions: true,
        selecting_reductions: true,
        matmul: true,
        activations: true,
        scans: true,
        convolution: true,
        ..Self::DEFAULT
    };

    /// Everything except `max`/`min`.
    ///
    /// **The configuration the six-hour campaign should have used for its second half.**
    /// Those two saturate a corpus with a disagreement already understood; excluding them is
    /// what lets a run reach `softmax`, `log` and `mean`.
    pub const WITHOUT_SELECTING_REDUCTIONS: Self = Self {
        selecting_reductions: false,
        ..Self::ALL_OPERATIONS
    };

    /// Only the operations whose backends run genuinely different algorithms.
    ///
    /// `softmax` (three implementations), and the unary transcendentals `exp` and `log`
    /// (neither correctly rounded). The narrowest useful setting, for asking whether a
    /// *numeric* disagreement exists at all.
    pub const NUMERICALLY_INTERESTING: Self = Self {
        unary_ops: true,
        binary_ops: false,
        accumulating_reductions: true,
        selecting_reductions: false,
        matmul: false,
        activations: true,
        scans: true,
        // The strongest candidate for an algorithmic disagreement, so it belongs in the
        // narrowest interesting configuration rather than only in the widest.
        convolution: true,
        ..Self::DEFAULT
    };

    /// The plain default, as a `const` so the presets above can build on it.
    pub const DEFAULT: Self = Self {
        max_rank: crate::backends::MAX_RANK,
        max_dim: 8,
        magnitude: 10.0,
        special_value_rate: 0.125,
        max_elements: 4_096,
        unary_ops: true,
        binary_ops: true,
        accumulating_reductions: true,
        selecting_reductions: true,
        matmul: true,
        activations: true,
        scans: true,
        convolution: true,
        restrict_domains: true,
    };
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

        // The finite ones, under the default restricted bounds.
        for special in SPECIAL_VALUES.iter().filter(|v| v.is_finite()) {
            assert!(
                drawn.iter().any(|v| v.to_bits() == special.to_bits()),
                "{special} was never generated"
            );
        }
    }

    /// **`NaN` and the infinities appear only when domains are unrestricted**, which is the
    /// fuzzer's setting and not the seeded default.
    ///
    /// Both halves matter. Reaching them is what makes the `max`-versus-`NaN` disagreement
    /// findable by a campaign at all; *not* reaching them by default is what keeps every
    /// distribution measured before PHASE-7E comparable.
    #[test]
    fn non_finite_inputs_appear_only_when_domains_are_unrestricted() {
        let mut rng = SeededRng::from_seed(0);

        let restricted = values(&mut rng, 5_000, Domain::Any, &Bounds::default());
        assert!(
            restricted.iter().all(|v| v.is_finite()),
            "a non-finite value reached a restricted case"
        );

        let unrestricted = values(
            &mut rng,
            5_000,
            Domain::Any,
            &Bounds {
                restrict_domains: false,
                ..Bounds::default()
            },
        );
        assert!(unrestricted.iter().any(|v| v.is_nan()), "no NaN generated");
        assert!(
            unrestricted.contains(&f32::INFINITY),
            "no positive infinity generated"
        );
        assert!(
            unrestricted.contains(&f32::NEG_INFINITY),
            "no negative infinity generated"
        );
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
