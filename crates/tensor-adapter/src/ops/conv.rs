//! 2-D convolution — the first operation whose backends run different *algorithms*.
//!
//! # What this generator is aiming at, and why it is not uniform
//!
//! Every other generator here samples shapes roughly uniformly, because for `add` or `exp`
//! one shape is much like another. That is false for convolution. `burn-flex` selects among
//! **five algorithms** using the very parameters this module chooses
//! (`burn-flex/src/ops/conv.rs`): a 1×1 fast path, a depthwise path, a small-channel path, a
//! direct path behind a seven-condition guard, and the general tiled im2col + GEMM.
//!
//! A uniform sample over plausible shapes lands in the general path nearly every time — and
//! the general path is the best-tested one. So the generator picks a **profile** first and
//! shapes to fit it, which is the difference between testing five algorithms and testing one.
//!
//! The profiles are not invented. `groups > 1` together with `padding > 0` is the exact
//! trigger of burn#4727, a missing channel offset in the SIMD remainder path; 1×1 is where
//! burn#4591 lived.
//!
//! # Validity is guaranteed by construction, never by rejection
//!
//! A convolution has real constraints — channels divisible by groups, the weight's second
//! dimension equal to `in_channels / groups`, and a window that fits. An invalid case is not
//! a finding: burn panics on one, and under `cargo-fuzz` a panic is reported as a crash,
//! which would bury every real divergence under our own noise.
//!
//! So this module never generates-then-checks. [`layout`] picks padding and the dilated
//! kernel span *first*, then a spatial extent that cannot be too small.
//!
//! # One layout function, two front-ends
//!
//! There are two ways a case is born: the seeded generator, and the fuzzer's byte decoder.
//! Everywhere else in this crate those are **separate implementations of the same shape
//! rules**, and they have drifted — four operations were reachable from one and not the other
//! for an entire phase, and nothing failed.
//!
//! So the rules live once, in [`layout`], which turns [`Choices`] — eleven raw numbers — into
//! shapes and parameters. The generator draws those numbers from an RNG and the decoder reads
//! them from the fuzzer's bytes. Only the tensor *data* differs between the two, which is the
//! part that genuinely must.

use crate::input::{Conv2dParams, TensorOp, TensorValue, conv2d_output_size};
use crate::ops::{Bounds, Domain, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Which of `burn-flex`'s code paths a case is built to reach.
///
/// Named after the *guard* rather than the algorithm, because the guard is what a generator
/// can control and what a predicate would later be written over.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Profile {
    /// `groups > 1` **and** `padding > 0` — burn#4727's exact trigger.
    GroupedAndPadded,
    /// A 1×1 kernel, which skips im2col entirely. burn#4591's path.
    Pointwise,
    /// `groups == in_channels == out_channels`, the canonical depthwise convolution.
    Depthwise,
    /// One or two input channels, which `burn-flex` gives a dedicated kernel.
    FewChannels,
    /// Everything else: the general tiled im2col + GEMM path.
    General,
}

/// Every profile, in a fixed order so sampling is reproducible.
pub const ALL_PROFILES: [Profile; 5] = [
    Profile::GroupedAndPadded,
    Profile::Pointwise,
    Profile::Depthwise,
    Profile::FewChannels,
    Profile::General,
];

/// Sampling weights, deliberately non-uniform.
///
/// `General` is the best-tested path and gets the smallest share; the two profiles with a
/// *documented* upstream bug get the largest. A judgement about where to spend a budget, not
/// a claim about where bugs are — stated here rather than buried in a `match` so that
/// changing it is a visible decision.
const WEIGHTS: [u32; 5] = [30, 25, 20, 15, 10];

/// Ceiling on multiply-accumulates per case.
///
/// A convolution's cost is `batch × out_channels × out_h × out_w × (in_channels/groups) ×
/// kh × kw` — a product of seven bounded quantities, so bounding each one individually does
/// not bound the case. This is the only cap that bounds the work itself, and it exists for
/// the same reason [`Bounds::max_elements`] does.
pub const MAX_MULTIPLY_ACCUMULATES: usize = 200_000;

/// The raw numbers a convolution is built from, before any profile-specific folding.
///
/// Deliberately unvalidated: every field is folded into a legal range by [`layout`]. That is
/// what lets the fuzzer supply arbitrary bytes and the RNG supply arbitrary draws, without
/// either needing to know the constraints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choices {
    /// Selects the profile, taken modulo the total of [`WEIGHTS`].
    pub profile_ticket: u32,
    pub groups: u32,
    pub in_per_group: u32,
    pub out_per_group: u32,
    pub kernel: [u32; 2],
    pub padding: [u32; 2],
    pub dilation: [u32; 2],
    pub stride: [u32; 2],
    pub spatial: [u32; 2],
    pub batch: u32,
    pub bias: bool,
}

impl Choices {
    /// Draw every field from a seeded RNG.
    pub fn from_rng(rng: &mut SeededRng) -> Self {
        Choices {
            profile_ticket: rng.random_range(0..WEIGHTS.iter().sum::<u32>()),
            groups: rng.random_range(0..64),
            in_per_group: rng.random_range(0..64),
            out_per_group: rng.random_range(0..64),
            kernel: [rng.random_range(0..64), rng.random_range(0..64)],
            padding: [rng.random_range(0..64), rng.random_range(0..64)],
            dilation: [rng.random_range(0..64), rng.random_range(0..64)],
            stride: [rng.random_range(0..64), rng.random_range(0..64)],
            spatial: [rng.random_range(0..64), rng.random_range(0..64)],
            batch: rng.random_range(0..64),
            // Not a tuning knob but a divergence surface: a backend folding the bias into
            // its accumulator rounds differently from one adding it in a separate pass, so
            // both must occur.
            bias: rng.random_bool(0.5),
        }
    }
}

/// The shapes and parameters a set of [`Choices`] describes.
///
/// Every field is guaranteed to satisfy [`TensorOp::conv2d`]'s constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub image: Vec<usize>,
    pub weight: Vec<usize>,
    /// `Some([out_channels])` when the convolution carries a bias.
    pub bias: Option<Vec<usize>>,
    pub params: Conv2dParams,
}

/// Fold raw choices into a dimensionally valid convolution.
///
/// **The single definition of the shape rules**, shared by the seeded generator and the byte
/// decoder. The order of operations is what makes validity structural rather than checked:
/// groups and per-group channel counts first, so divisibility cannot be violated; then
/// padding and the dilated kernel span; then a spatial extent at least large enough for the
/// window to fit.
pub fn layout(choices: &Choices, bounds: &Bounds) -> Layout {
    let max_channel = bounds.max_dim.clamp(1, 8) as u32;
    let max_spatial = bounds.max_dim.clamp(1, 16) as u32;
    let profile = profile_for(choices.profile_ticket);

    // Maps any number into `low..=high`, so no field of `Choices` can be invalid.
    let fold = |value: u32, low: u32, high: u32| low + value % (high.saturating_sub(low) + 1);

    let (groups, in_per_group, out_per_group) = match profile {
        // groups == in_channels == out_channels means exactly one channel per group.
        Profile::Depthwise => (fold(choices.groups, 2, max_channel.max(2)), 1, 1),
        Profile::FewChannels => (
            1,
            fold(choices.in_per_group, 1, 2),
            fold(choices.out_per_group, 1, max_channel),
        ),
        Profile::GroupedAndPadded => (
            fold(choices.groups, 2, max_channel.max(2)),
            fold(choices.in_per_group, 1, max_channel.min(3)),
            fold(choices.out_per_group, 1, max_channel.min(3)),
        ),
        Profile::Pointwise | Profile::General => (
            fold(choices.groups, 1, max_channel.min(3)),
            fold(choices.in_per_group, 1, max_channel.min(4)),
            fold(choices.out_per_group, 1, max_channel.min(4)),
        ),
    };
    let in_channels = (groups * in_per_group) as usize;
    let out_channels = (groups * out_per_group) as usize;

    let kernel: [usize; 2] = std::array::from_fn(|axis| match profile {
        Profile::Pointwise => 1,
        _ => fold(choices.kernel[axis], 1, 3) as usize,
    });
    let padding: [usize; 2] = std::array::from_fn(|axis| match profile {
        // The defining condition of this profile, so it is folded strictly positive.
        Profile::GroupedAndPadded => fold(choices.padding[axis], 1, 2) as usize,
        Profile::Pointwise => 0,
        _ => fold(choices.padding[axis], 0, 1) as usize,
    });
    let dilation: [usize; 2] = std::array::from_fn(|axis| match profile {
        // Dilating a 1×1 kernel changes nothing, so it stays dense and the profile keeps
        // its meaning.
        Profile::Pointwise => 1,
        _ => fold(choices.dilation[axis], 1, 2) as usize,
    });
    let stride: [usize; 2] = std::array::from_fn(|axis| fold(choices.stride[axis], 1, 2) as usize);

    // **The step that makes validity structural.** A dilated kernel spans
    // `dilation * (kernel - 1) + 1`; padding contributes `2 * padding`. Choosing the spatial
    // extent to be at least their difference means the window always fits.
    let minimum: [usize; 2] = std::array::from_fn(|axis| {
        let span = dilation[axis] * (kernel[axis] - 1) + 1;
        span.saturating_sub(2 * padding[axis]).max(1)
    });
    let mut spatial: [usize; 2] = std::array::from_fn(|axis| {
        let high = (max_spatial as usize).max(minimum[axis]);
        minimum[axis] + choices.spatial[axis] as usize % (high - minimum[axis] + 1)
    });

    let mut batch = fold(choices.batch, 1, 2) as usize;
    let params = Conv2dParams {
        stride,
        padding,
        dilation,
        groups: groups as usize,
    };

    // **Trim rather than reject.** The decoder cannot retry — a byte string must always
    // decode to *some* case, or the fuzzer learns nothing from it — so an over-budget case is
    // shrunk deterministically instead. Batch first, since its size says least about which
    // algorithm runs; then the spatial extents, never below the minimum that keeps the window
    // fitting. Channels are left alone because reducing them changes the profile, which would
    // defeat the point of having chosen one.
    while cost_of(batch, in_channels, out_channels, &spatial, kernel, &params)
        > MAX_MULTIPLY_ACCUMULATES
    {
        if batch > 1 {
            batch -= 1;
        } else if spatial[0] > minimum[0] || spatial[1] > minimum[1] {
            for axis in 0..2 {
                if spatial[axis] > minimum[axis] {
                    spatial[axis] = (spatial[axis] / 2).max(minimum[axis]);
                }
            }
        } else {
            break;
        }
    }

    Layout {
        image: vec![batch, in_channels, spatial[0], spatial[1]],
        weight: vec![out_channels, in_per_group as usize, kernel[0], kernel[1]],
        bias: choices.bias.then(|| vec![out_channels]),
        params,
    }
}

/// Which profile a ticket selects, according to [`WEIGHTS`].
fn profile_for(ticket: u32) -> Profile {
    let mut remaining = ticket % WEIGHTS.iter().sum::<u32>();
    for (profile, weight) in ALL_PROFILES.iter().zip(WEIGHTS) {
        if remaining < weight {
            return *profile;
        }
        remaining -= weight;
    }
    Profile::General
}

/// Build a valid convolution from a seeded RNG.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let plan = layout(&Choices::from_rng(rng), bounds);
    let image = filled(rng, plan.image, bounds);
    let weight = filled(rng, plan.weight, bounds);
    let bias = plan.bias.map(|shape| filled(rng, shape, bounds));
    TensorOp::conv2d(image, weight, bias, plan.params)
}

/// A tensor of the given shape, filled from the shared value generator.
fn filled(rng: &mut SeededRng, shape: Vec<usize>, bounds: &Bounds) -> TensorValue {
    let count = shape.iter().product();
    TensorValue::new(shape, values(rng, count, Domain::Any, bounds))
}

/// Multiply-accumulates a case performs, as the cost ceiling measures it.
///
/// Public because 7G.7's shrinker needs the same number to know whether a candidate is
/// actually smaller, and two definitions of "cost" would drift.
pub fn cost(case: &TensorOp) -> usize {
    let TensorOp::Conv2d {
        input,
        weight,
        params,
        ..
    } = case
    else {
        return 0;
    };
    cost_of(
        input.shape()[0],
        input.shape()[1],
        weight.shape()[0],
        &[input.shape()[2], input.shape()[3]],
        [weight.shape()[2], weight.shape()[3]],
        params,
    )
}

/// The cost formula, over loose parts so [`layout`] can call it before a case exists.
fn cost_of(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    spatial: &[usize; 2],
    kernel: [usize; 2],
    params: &Conv2dParams,
) -> usize {
    let out: [usize; 2] = std::array::from_fn(|axis| {
        conv2d_output_size(
            spatial[axis],
            kernel[axis],
            params.stride[axis],
            params.padding[axis],
            params.dilation[axis],
        )
        .unwrap_or(0)
    });
    batch * out_channels * out[0] * out[1] * (in_channels / params.groups) * kernel[0] * kernel[1]
}

/// Which profile a case satisfies, for tests and for the feature vocabulary.
///
/// **Classification is by guard, not by how the case was built**, so a `General` case that
/// happens to be depthwise is reported as depthwise — which is what the backend will act on.
/// Checked in the order the guards overlap.
pub fn classify(case: &TensorOp) -> Profile {
    let TensorOp::Conv2d {
        input,
        weight,
        params,
        ..
    } = case
    else {
        return Profile::General;
    };
    let in_channels = input.shape()[1];
    let out_channels = weight.shape()[0];
    let pointwise = weight.shape()[2] == 1 && weight.shape()[3] == 1;
    let padded = params.padding.iter().any(|p| *p > 0);

    if params.groups > 1 && padded {
        Profile::GroupedAndPadded
    } else if pointwise {
        Profile::Pointwise
    } else if params.groups > 1 && params.groups == in_channels && params.groups == out_channels {
        Profile::Depthwise
    } else if in_channels <= 2 && params.groups == 1 {
        Profile::FewChannels
    } else {
        Profile::General
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn bounds() -> Bounds {
        Bounds::default()
    }

    /// **The property this module exists to guarantee.** [`TensorOp::conv2d`] asserts every
    /// constraint independently, so a generator that violated one would panic here — and a
    /// panic under `cargo-fuzz` is reported as a crash, which would bury real divergences.
    #[test]
    fn the_generator_only_emits_valid_cases() {
        for seed in 0..600 {
            let mut rng = SeededRng::from_seed(seed);
            let case = generate(&mut rng, &bounds());
            assert_eq!(case.name(), "conv2d", "seed {seed}");
        }
    }

    /// **The decoder shares the layout rules, so arbitrary numbers must also be valid.**
    /// This is the property free-form fuzzer bytes rely on: every field of [`Choices`] is
    /// folded into range rather than checked, so no byte string can produce a case that
    /// panics.
    #[test]
    fn arbitrary_choices_always_produce_a_valid_layout() {
        let b = bounds();
        for seed in 0..3_000u64 {
            // Deliberately wild values, far outside any range the RNG front-end would draw.
            let raw = seed.wrapping_mul(2_654_435_761).wrapping_add(seed << 17);
            let n = |shift: u32| (raw >> (shift % 48)) as u32;
            let choices = Choices {
                profile_ticket: n(0),
                groups: n(3),
                in_per_group: n(6),
                out_per_group: n(9),
                kernel: [n(12), n(15)],
                padding: [n(18), n(21)],
                dilation: [n(24), n(27)],
                stride: [n(30), n(33)],
                spatial: [n(36), n(39)],
                batch: n(5),
                bias: raw % 2 == 0,
            };
            let plan = layout(&choices, &b);

            let in_channels = plan.image[1];
            let out_channels = plan.weight[0];
            assert_eq!(in_channels % plan.params.groups, 0, "seed {seed}");
            assert_eq!(out_channels % plan.params.groups, 0, "seed {seed}");
            assert_eq!(
                plan.weight[1],
                in_channels / plan.params.groups,
                "seed {seed}"
            );
            if let Some(bias) = &plan.bias {
                assert_eq!(bias, &[out_channels], "seed {seed}");
            }
            for axis in 0..2 {
                assert!(
                    conv2d_output_size(
                        plan.image[2 + axis],
                        plan.weight[2 + axis],
                        plan.params.stride[axis],
                        plan.params.padding[axis],
                        plan.params.dilation[axis],
                    )
                    .is_some_and(|s| s > 0),
                    "seed {seed} axis {axis}: window does not fit"
                );
            }
        }
    }

    /// **Without this, a path can silently stop being covered.** That is not hypothetical:
    /// four operations were generator-reachable but fuzzer-unreachable for a whole phase
    /// because the decoder's slot table was not extended, and nothing failed.
    #[test]
    fn every_profile_is_reachable() {
        let mut seen: HashSet<Profile> = HashSet::new();
        for seed in 0..600 {
            let mut rng = SeededRng::from_seed(seed);
            seen.insert(classify(&generate(&mut rng, &bounds())));
        }
        for profile in ALL_PROFILES {
            assert!(seen.contains(&profile), "{profile:?} was never generated");
        }
    }

    /// burn#4727's trigger specifically — a conjunction, not either condition alone.
    #[test]
    fn the_grouped_and_padded_trigger_is_produced_often_enough_to_matter() {
        let hits = (0..600)
            .filter(|seed| {
                let mut rng = SeededRng::from_seed(*seed);
                matches!(
                    classify(&generate(&mut rng, &bounds())),
                    Profile::GroupedAndPadded
                )
            })
            .count();
        assert!(
            hits > 60,
            "only {hits}/600 cases hit groups>1 and padding>0; the profile weighting is not \
             reaching the region it exists for"
        );
    }

    #[test]
    fn no_case_exceeds_the_cost_ceiling() {
        for seed in 0..600 {
            let mut rng = SeededRng::from_seed(seed);
            let case = generate(&mut rng, &bounds());
            let spent = cost(&case);
            assert!(
                spent <= MAX_MULTIPLY_ACCUMULATES,
                "seed {seed} costs {spent} multiply-accumulates"
            );
        }
    }

    /// The same seed must give the same case — the project's most basic rule, and cheap to
    /// break with an unordered iteration or a stray `thread_rng`.
    #[test]
    fn generation_is_reproducible_from_its_seed() {
        for seed in [0, 7, 99] {
            let first = generate(&mut SeededRng::from_seed(seed), &bounds());
            let again = generate(&mut SeededRng::from_seed(seed), &bounds());
            assert_eq!(first, again);
        }
    }
}
