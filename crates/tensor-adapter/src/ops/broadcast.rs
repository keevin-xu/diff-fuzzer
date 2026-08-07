//! Shapes that differ but still combine elementwise.
//!
//! # The rule
//!
//! **Both operands carry the same rank.** Two dimensions at the same index are compatible
//! when they are equal or one of them is `1`, and the result takes the larger:
//!
//! ```text
//!     [3, 1]  x  [3, 4]   ->  [3, 4]      a length-1 axis stretches
//!     [1, 1]  x  [3, 4]   ->  [3, 4]      every axis stretches
//!     [3, 4]  x  [3, 4]   ->  [3, 4]      the ordinary case
//!     [3, 2]  x  [3, 4]   ->  invalid     2 and 4 are neither equal nor 1
//! ```
//!
//! # Why rank-differing broadcast is *not* modelled
//!
//! NumPy right-aligns shapes and implies missing leading axes, so `[4]` combines with
//! `[3, 4]`. **burn does not offer that**, and the reason is its type system rather than an
//! omission: `Tensor<B, D>` carries its rank as a const generic, and `add(self, other: Self)`
//! requires *the same* `D` on both sides. `TensorCheck::binary_ops_ew_shape<D>` accordingly
//! loops `0..D` over both shapes with no alignment step
//! (`burn-tensor/src/tensor/api/check.rs`, retrieved 2026-08-06).
//!
//! Reaching it would mean calling `unsqueeze` in this adapter first — which would test our
//! own reshape call rather than burn's broadcasting, and would hand all three backends
//! identically-ranked tensors anyway. Nothing would be gained.
//!
//! # Why this is generated rather than checked
//!
//! Two independently drawn shapes are almost never compatible, so generate-and-reject would
//! spend nearly the whole budget on rejected cases. Instead a *result* shape is drawn first
//! and each operand is derived from it by stretching or dropping axes — every pair is
//! compatible by construction, and the interesting relationships are chosen deliberately
//! rather than waited for.
//!
//! # The budget applies to the result, not the operands
//!
//! **A broadcast result can be far larger than either input.** `[64, 1]` against `[1, 64]` is
//! 64 elements on each side and 4,096 in the output, and the output is what every backend
//! must allocate and compute. Drawing the result first and clamping *it* is what keeps a case
//! that looks small from behaving like a large one.

use crate::ops::{Bounds, clamp_to, element_count, shape_of_rank};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// How often both operands get the same shape.
///
/// **Kept high on purpose.** Equal shapes are the common path in real use and are where every
/// finding so far has come from; a generator that always broadcasts has quietly stopped
/// testing what it used to test. This is a distribution decision, not a tuning knob.
const IDENTICAL_SHARE: f64 = 0.4;

/// Chance that any one axis is stretched from a given side.
///
/// Applied per axis per side, so a rank-3 case usually stretches one or two axes rather than
/// all of them — the mixed cases are the interesting ones.
const STRETCH_SHARE: f64 = 0.25;

/// The shape an elementwise operation on these two produces, or `None` if they do not combine.
///
/// Computed here rather than read back from a backend, because **whether the backends agree
/// on the output shape is one of the things under test**. A test that asked a backend for the
/// answer could not detect a backend that got it wrong.
///
/// Differing ranks return `None` — see the module docs: burn's typed API cannot express them.
pub fn result_shape(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    if lhs.len() != rhs.len() {
        return None;
    }

    let mut out = Vec::with_capacity(lhs.len());
    for (&l, &r) in lhs.iter().zip(rhs) {
        if l != r && l != 1 && r != 1 {
            return None;
        }
        out.push(l.max(r));
    }

    Some(out)
}

/// Whether the two shapes combine elementwise.
pub fn compatible(lhs: &[usize], rhs: &[usize]) -> bool {
    result_shape(lhs, rhs).is_some()
}

/// Draw a compatible pair of operand shapes.
///
/// The result shape is drawn and clamped first; each operand is then derived from it, so the
/// pair cannot be incompatible and the *output* stays inside the element budget.
pub fn pair(rng: &mut SeededRng, bounds: &Bounds) -> (Vec<usize>, Vec<usize>) {
    let rank = rng.random_range(1..=bounds.max_rank);
    let target = shape_of_rank(rng, rank, bounds);

    if rng.random_bool(IDENTICAL_SHARE) {
        return (target.clone(), target);
    }

    let mut lhs = target.clone();
    let mut rhs = target;

    // Stretch axes: setting an extent to 1 on one side makes that side broadcast along it.
    // Never both sides at once — that would shrink the result rather than broadcast it, and
    // the result was already chosen and budgeted.
    let mut stretched_any = false;
    for i in 0..rank {
        if rng.random_bool(STRETCH_SHARE) {
            stretch(rng, &mut lhs, &mut rhs, i);
            stretched_any = true;
        }
    }

    // **Guarantee at least one stretch on this branch**, so `IDENTICAL_SHARE` means what its
    // name says. Without this, a case reaching here still emerges identical whenever no axis
    // happened to be picked — which at rank 1 is most of the time, and measured as 74%
    // identical against an intended 60%. A constant whose name misstates its effect is worse
    // than a differently-valued one.
    //
    // An axis of extent 1 in the target cannot be stretched (it is already 1), so a target
    // that is all ones legitimately yields an identical pair. That is rare and correct.
    if !stretched_any {
        let choices: Vec<usize> = (0..rank).filter(|&i| lhs[i] > 1).collect();
        if !choices.is_empty() {
            let i = choices[rng.random_range(0..choices.len())];
            stretch(rng, &mut lhs, &mut rhs, i);
        }
    }

    (lhs, rhs)
}

/// Set axis `i` to 1 on one side or the other, chosen evenly.
fn stretch(rng: &mut SeededRng, lhs: &mut [usize], rhs: &mut [usize], i: usize) {
    if rng.random_bool(0.5) {
        lhs[i] = 1;
    } else {
        rhs[i] = 1;
    }
}

/// How many elements the result of combining these two holds.
///
/// Panics if they are incompatible, which `pair` cannot produce.
pub fn result_count(lhs: &[usize], rhs: &[usize]) -> usize {
    element_count(&result_shape(lhs, rhs).expect("shapes were built to be compatible"))
}

/// Clamp a *result* shape to the element budget.
///
/// Exposed for callers that build a result shape by other means; `pair` already applies it.
pub fn clamp_result(shape: Vec<usize>, bounds: &Bounds) -> Vec<usize> {
    clamp_to(shape, bounds.max_elements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    // --- the rule itself -------------------------------------------------------------

    #[test]
    fn equal_shapes_combine_to_themselves() {
        assert_eq!(result_shape(&[3, 4], &[3, 4]), Some(vec![3, 4]));
    }

    #[test]
    fn a_length_one_axis_stretches() {
        assert_eq!(result_shape(&[3, 1], &[3, 4]), Some(vec![3, 4]));
        assert_eq!(result_shape(&[3, 4], &[3, 1]), Some(vec![3, 4]));
        assert_eq!(result_shape(&[1, 1], &[3, 4]), Some(vec![3, 4]));
    }

    /// **Differing ranks do not combine, and that is burn's constraint rather than ours.**
    /// `Tensor<B, D>` fixes the rank at compile time and `add` takes `Self`, so NumPy's
    /// right-alignment has no way to be expressed. Pinned as a test because it looks like an
    /// omission and is not — a future reader will otherwise "fix" it.
    #[test]
    fn differing_ranks_do_not_combine() {
        assert_eq!(result_shape(&[4], &[3, 4]), None);
        assert_eq!(result_shape(&[2, 1, 4], &[4]), None);
    }

    #[test]
    fn mismatched_axes_that_are_neither_equal_nor_one_do_not_combine() {
        assert_eq!(result_shape(&[3, 2], &[3, 4]), None);
        assert!(!compatible(&[5], &[3]));
    }

    /// The relation is symmetric: `a` combines with `b` exactly when `b` combines with `a`,
    /// and to the same result.
    #[test]
    fn the_rule_is_symmetric() {
        for (a, b) in [
            (vec![3, 1], vec![3, 4]),
            (vec![1, 4], vec![3, 4]),
            (vec![3, 2], vec![3, 4]),
            (vec![1], vec![7]),
        ] {
            assert_eq!(result_shape(&a, &b), result_shape(&b, &a));
        }
    }

    // --- generation ------------------------------------------------------------------

    #[test]
    fn every_generated_pair_is_compatible() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let (lhs, rhs) = pair(rng, &bounds);
            assert!(
                compatible(&lhs, &rhs),
                "generated an incompatible pair: {lhs:?} and {rhs:?}"
            );
        });
    }

    /// **Both relationships must actually occur.** A generator that technically supports
    /// broadcasting but produces it once in ten thousand cases has not added anything, and
    /// nothing else in the suite would notice.
    ///
    /// Ranks are asserted equal in the loop rather than counted: differing ranks are not a
    /// rarity to measure, they are a case this model must never emit.
    #[test]
    fn every_broadcast_relationship_is_reachable() {
        let bounds = Bounds::default();
        let mut identical = 0;
        let mut stretched = 0;

        for seed in 0..2_000u64 {
            let mut rng = SeededRng::from_seed(seed);
            let (lhs, rhs) = pair(&mut rng, &bounds);
            assert_eq!(lhs.len(), rhs.len(), "ranks must always match");
            if lhs == rhs {
                identical += 1;
            } else {
                stretched += 1;
            }
        }

        assert!(identical > 100, "identical shapes are rare: {identical}");
        assert!(stretched > 100, "stretched axes are rare: {stretched}");
    }

    /// **The budget binds the result, not the operands.** `[64,1]` against `[1,64]` has 64
    /// elements per side and 4,096 in the result; a check on the operands would let it past.
    #[test]
    fn the_result_stays_inside_the_element_budget() {
        let bounds = Bounds {
            max_dim: 64,
            max_elements: 4_096,
            ..Bounds::default()
        };
        for_many_seeds(|rng| {
            let (lhs, rhs) = pair(rng, &bounds);
            assert!(
                result_count(&lhs, &rhs) <= bounds.max_elements,
                "{lhs:?} x {rhs:?} produces {} elements, over budget",
                result_count(&lhs, &rhs)
            );
        });
    }

    /// A stretched pair really is smaller than its result — otherwise nothing is being
    /// broadcast and the feature is cosmetic.
    #[test]
    fn a_stretched_operand_holds_fewer_elements_than_the_result() {
        let lhs = vec![3, 1];
        let rhs = vec![3, 4];
        assert!(element_count(&lhs) < result_count(&lhs, &rhs));
    }

    #[test]
    fn generated_shapes_respect_the_rank_bound() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let (lhs, rhs) = pair(rng, &bounds);
            assert!(
                !lhs.is_empty() && !rhs.is_empty(),
                "rank 0 is not supported"
            );
            assert!(lhs.len() <= bounds.max_rank && rhs.len() <= bounds.max_rank);
        });
    }
}
