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
//! So this module never generates-then-checks. It picks padding and the dilated kernel width
//! *first*, then chooses a spatial extent that cannot be too small. [`TensorOp::conv2d`]
//! asserts the same constraints independently, and `the_generator_only_emits_valid_cases`
//! runs the pair against many seeds.

use crate::input::{Conv2dParams, TensorOp, TensorValue, conv2d_output_size};
use crate::ops::{Bounds, Domain, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Which of `burn-flex`'s code paths a case is built to reach.
///
/// Named after the *guard* rather than the algorithm, because the guard is what the generator
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

/// Every profile, in a fixed order so the sampling is reproducible from a seed.
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
/// *documented* upstream bug get the largest. These are a judgement about where to spend a
/// budget, not a claim about where bugs are — and they are stated here rather than buried in
/// a `match` so that changing them is a visible decision.
const WEIGHTS: [u32; 5] = [30, 25, 20, 15, 10];

/// Ceiling on multiply-accumulates per case.
///
/// A convolution's cost is `batch × out_channels × out_h × out_w × (in_channels/groups) ×
/// kh × kw`, which is a product of seven bounded quantities — so bounding each one
/// individually does not bound the case. This is the only cap that bounds the work itself,
/// and it exists for the same reason [`Bounds::max_elements`] does.
const MAX_MULTIPLY_ACCUMULATES: usize = 200_000;

/// How often a convolution carries a bias.
///
/// Not a tuning knob but a divergence surface: a backend folding the bias into its
/// accumulator rounds differently from one adding it in a separate pass, so both must occur.
const BIAS_SHARE: f64 = 0.5;

/// Build a valid convolution.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    // Retry rather than clamp: a case can be rejected only for exceeding the cost ceiling,
    // and shrinking it in place would bias every profile toward its smallest shape — which
    // is exactly the region where a tiled algorithm does *not* take its interesting path.
    // The loop terminates because `Pointwise` at minimum size is always affordable.
    for _ in 0..16 {
        let profile = pick_profile(rng);
        let case = build(rng, bounds, profile);
        if cost(&case) <= MAX_MULTIPLY_ACCUMULATES {
            return case;
        }
    }
    build(rng, bounds, Profile::Pointwise)
}

/// Draw a profile according to [`WEIGHTS`].
fn pick_profile(rng: &mut SeededRng) -> Profile {
    let total: u32 = WEIGHTS.iter().sum();
    let mut ticket = rng.random_range(0..total);
    for (profile, weight) in ALL_PROFILES.iter().zip(WEIGHTS) {
        if ticket < weight {
            return *profile;
        }
        ticket -= weight;
    }
    Profile::General
}

/// Build one case for a chosen profile.
///
/// Every path through this function produces a dimensionally valid convolution. The order
/// matters: `groups` and the per-group channel counts are chosen first so divisibility holds
/// by construction, then padding and the dilated kernel, then a spatial extent large enough
/// for the window to fit.
fn build(rng: &mut SeededRng, bounds: &Bounds, profile: Profile) -> TensorOp {
    let max_channel = bounds.max_dim.clamp(1, 8);
    let max_spatial = bounds.max_dim.clamp(1, 16);

    // `groups`, and the channels *per group*. Deriving the totals from these rather than
    // picking totals and testing divisibility is what makes the constraint unrepresentable.
    let (groups, in_per_group, out_per_group) = match profile {
        Profile::Depthwise => {
            // groups == in_channels == out_channels means exactly one channel per group.
            (rng.random_range(1..=max_channel), 1, 1)
        }
        Profile::FewChannels => (
            1,
            rng.random_range(1..=2),
            rng.random_range(1..=max_channel),
        ),
        Profile::GroupedAndPadded => (
            rng.random_range(2..=max_channel.max(2)),
            rng.random_range(1..=max_channel.min(3)),
            rng.random_range(1..=max_channel.min(3)),
        ),
        Profile::Pointwise | Profile::General => (
            rng.random_range(1..=max_channel.min(3)),
            rng.random_range(1..=max_channel.min(4)),
            rng.random_range(1..=max_channel.min(4)),
        ),
    };
    let in_channels = groups * in_per_group;
    let out_channels = groups * out_per_group;

    let kernel: [usize; 2] = match profile {
        Profile::Pointwise => [1, 1],
        _ => [rng.random_range(1..=3), rng.random_range(1..=3)],
    };

    let padding: [usize; 2] = match profile {
        // The defining condition of this profile, so it is drawn strictly positive.
        Profile::GroupedAndPadded => [rng.random_range(1..=2), rng.random_range(1..=2)],
        Profile::Pointwise => [0, 0],
        _ => [rng.random_range(0..=1), rng.random_range(0..=1)],
    };

    let dilation: [usize; 2] = match profile {
        // Dilating a 1×1 kernel changes nothing, so it stays dense and the profile keeps its
        // meaning.
        Profile::Pointwise => [1, 1],
        _ => [rng.random_range(1..=2), rng.random_range(1..=2)],
    };
    let stride: [usize; 2] = [rng.random_range(1..=2), rng.random_range(1..=2)];

    // **The step that makes validity structural.** A dilated kernel spans
    // `dilation * (kernel - 1) + 1`; padding contributes `2 * padding`. Choosing the spatial
    // extent to be at least the difference means the window always fits, so no case is ever
    // generated and then rejected.
    let spatial: [usize; 2] = std::array::from_fn(|axis| {
        let span = dilation[axis] * (kernel[axis] - 1) + 1;
        let minimum = span.saturating_sub(2 * padding[axis]).max(1);
        rng.random_range(minimum..=minimum.max(max_spatial))
    });

    let batch = rng.random_range(1..=2);
    let params = Conv2dParams {
        stride,
        padding,
        dilation,
        groups,
    };

    let image_shape = vec![batch, in_channels, spatial[0], spatial[1]];
    let weight_shape = vec![out_channels, in_per_group, kernel[0], kernel[1]];
    let image = tensor(rng, image_shape, bounds);
    let weight = tensor(rng, weight_shape, bounds);
    let bias = rng
        .random_bool(BIAS_SHARE)
        .then(|| tensor(rng, vec![out_channels], bounds));

    TensorOp::conv2d(image, weight, bias, params)
}

/// A tensor of the given shape filled from the shared value generator.
fn tensor(rng: &mut SeededRng, shape: Vec<usize>, bounds: &Bounds) -> TensorValue {
    let count = shape.iter().product();
    let data = values(rng, count, Domain::Any, bounds);
    TensorValue::new(shape, data)
}

/// Multiply-accumulates this case will perform, as the cost ceiling measures it.
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
    let (batch, in_channels) = (input.shape()[0], input.shape()[1]);
    let out_channels = weight.shape()[0];
    let (kh, kw) = (weight.shape()[2], weight.shape()[3]);

    let out: Vec<usize> = (0..2)
        .map(|axis| {
            conv2d_output_size(
                input.shape()[2 + axis],
                weight.shape()[2 + axis],
                params.stride[axis],
                params.padding[axis],
                params.dilation[axis],
            )
            .unwrap_or(0)
        })
        .collect();

    batch * out_channels * out[0] * out[1] * (in_channels / params.groups) * kh * kw
}

/// Which profile a case satisfies, for tests and for the feature vocabulary.
///
/// **Classification is by guard, not by how the case was built**, so a `General` case that
/// happens to be depthwise is reported as depthwise — which is what the backend will act on.
/// Checked in the order flex checks them, since the guards overlap.
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

            let TensorOp::Conv2d {
                input,
                weight,
                bias,
                params,
            } = &case
            else {
                panic!("conv generator produced {}", case.name());
            };

            let in_channels = input.shape()[1];
            let out_channels = weight.shape()[0];
            assert_eq!(in_channels % params.groups, 0, "seed {seed}");
            assert_eq!(out_channels % params.groups, 0, "seed {seed}");
            assert_eq!(
                weight.shape()[1],
                in_channels / params.groups,
                "seed {seed}"
            );
            if let Some(bias) = bias {
                assert_eq!(bias.shape(), [out_channels], "seed {seed}");
            }
            for axis in 0..2 {
                let size = conv2d_output_size(
                    input.shape()[2 + axis],
                    weight.shape()[2 + axis],
                    params.stride[axis],
                    params.padding[axis],
                    params.dilation[axis],
                );
                assert!(
                    size.is_some_and(|s| s > 0),
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
