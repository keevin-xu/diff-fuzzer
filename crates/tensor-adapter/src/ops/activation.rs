//! Normalisations along one axis.
//!
//! # What this generator is aiming at
//!
//! `softmax` is the first operation where the three backends run **three different
//! algorithms**, so the generator's job is to reach the places where those algorithms can
//! disagree rather than to sample uniformly.
//!
//! Two axes of attack, both deliberate:
//!
//! **The dimension.** `burn-flex` transposes when `dim != rank - 1`, normalises the last
//! axis, and transposes back; when `dim == rank - 1` it does not. That is two code paths
//! selected by a property of the input — structurally the same shape as the libtorch
//! tile-remainder bug, which is the one real bug this project has found. Uniform sampling
//! would spend half its budget on the branch with no plausible mechanism, so the transposing
//! branch is **over-sampled**.
//!
//! **The values.** `softmax` is scale-invariant in exact arithmetic and emphatically not in
//! floating point:
//!
//! - **Large positives** stress the max-subtraction, which is a stability measure rather than
//!   part of the definition — so implementations may apply it differently or at a different
//!   point.
//! - **Large negatives** drive `exp` toward underflow, where one backend may return a
//!   subnormal and another zero.
//! - **Identical values** have an exact answer (`1/n` everywhere) that no correct
//!   implementation may round, which makes them the strongest near-miss available: a
//!   disagreement there needs no tolerance argument at all.

use crate::input::{ActivationOp, TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, element_count, shape, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every activation the generator may pick.
pub const ALL: [ActivationOp; 1] = [ActivationOp::Softmax];

/// How often the normalised dimension is *not* the last one.
///
/// Above the uniform share on purpose — see the module docs. Not tuned against results:
/// chosen before any softmax case had been run, because it targets a mechanism read from
/// `burn-flex`'s source rather than a pattern observed in findings.
const OFF_LAST_DIM_SHARE: f64 = 0.6;

/// How often every value along the tensor is made identical.
///
/// These are the cases with an exact known answer. Kept a minority: they are valuable as
/// near-misses, but a generator dominated by them would stop exercising ordinary arithmetic.
const IDENTICAL_VALUES_SHARE: f64 = 0.15;

/// Build a valid activation case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];
    let shape = shape(rng, bounds);
    let rank = shape.len();

    // Rank 1 has only one dimension, so the split does not exist there.
    let dim = if rank > 1 && rng.random_bool(OFF_LAST_DIM_SHARE) {
        rng.random_range(0..rank - 1)
    } else {
        rank - 1
    };

    let count = element_count(&shape);
    let data = if rng.random_bool(IDENTICAL_VALUES_SHARE) {
        // One value repeated: softmax is exactly `1/n`, whatever the value.
        let single = values(rng, 1, Domain::Any, bounds);
        vec![single[0]; count]
    } else {
        values(rng, count, Domain::Any, bounds)
    };

    TensorOp::activation(kind, TensorValue::new(shape, data), dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    /// burn panics on an out-of-range dimension rather than returning an error, so the
    /// generator must make that unreachable rather than rely on being caught.
    #[test]
    fn the_generator_never_emits_an_out_of_range_dim() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Activation { arg, dim, .. } = &case else {
                panic!("activation generator produced {case:?}");
            };
            assert!(
                *dim < arg.rank(),
                "dim {dim} out of range for {:?}",
                arg.shape()
            );
        });
    }

    /// **Both sides of `burn-flex`'s path split must be reachable**, or the phase's whole
    /// premise goes untested — and the off-last branch is the one with a mechanism.
    #[test]
    fn both_dimension_paths_are_reachable_and_the_transposing_one_is_favoured() {
        let bounds = Bounds::default();
        let mut last = 0;
        let mut off_last = 0;

        for seed in 0..3_000u64 {
            let mut rng = SeededRng::from_seed(seed);
            let TensorOp::Activation { arg, dim, .. } = generate(&mut rng, &bounds) else {
                unreachable!()
            };
            if dim == arg.rank() - 1 {
                last += 1;
            } else {
                off_last += 1;
            }
        }

        assert!(last > 200, "the last-dimension path is rare: {last}");
        assert!(off_last > 200, "the transposing path is rare: {off_last}");
    }

    /// The exactly-known cases must actually occur; they are the strongest near-misses.
    #[test]
    fn rows_of_identical_values_are_produced() {
        let bounds = Bounds::default();
        let mut identical = 0;

        for seed in 0..3_000u64 {
            let mut rng = SeededRng::from_seed(seed);
            let TensorOp::Activation { arg, .. } = generate(&mut rng, &bounds) else {
                unreachable!()
            };
            if arg.data().len() > 1 && arg.data().iter().all(|v| *v == arg.data()[0]) {
                identical += 1;
            }
        }

        assert!(identical > 100, "no rows of identical values: {identical}");
    }
}
