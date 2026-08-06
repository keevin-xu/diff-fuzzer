//! Matrix multiplication: the most constrained operation in the set.
//!
//! Three rules have to hold at once. Both operands need at least two dimensions; the
//! last dimension of the left must equal the second-to-last of the right; and any
//! leading batch dimensions must match. Written out: `[.., m, k]` times `[.., k, n]`
//! gives `[.., m, n]`, and the shared `k` is what makes the multiplication defined.
//!
//! Generating two shapes and hoping they line up would essentially never work — with
//! dimensions up to 8, two independently chosen `k` values agree about one time in
//! eight, before the batch dimensions are even considered. So `m`, `k` and `n` are
//! each drawn **once** and placed into both shapes. The constraint holds because there
//! is only one `k` in existence.

use crate::input::{TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, clamp_to, element_count, shape_of_rank, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Build a valid matmul case, batched when the rank exceeds two.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    // At least two dimensions, since a matrix multiplication of vectors is undefined.
    let rank = rng.random_range(2..=bounds.max_rank.max(2));

    // The leading dimensions, shared by both operands so they cannot disagree.
    let batch = shape_of_rank(rng, rank - 2, bounds);

    // Drawn once each, then placed into both shapes.
    let m = rng.random_range(1..=bounds.max_dim);
    let k = rng.random_range(1..=bounds.max_dim);
    let n = rng.random_range(1..=bounds.max_dim);

    // **The batch is clamped against what the matrix dimensions have already spent.** An
    // operand here is `batch × m × k`, so bounding the batch and `max_dim` separately does
    // not bound the case: at `max_dim: 64` a rank-3 matmul would otherwise reach
    // 4,096 × 4,096 elements, and its cost is a further factor of `n` on top.
    let per_matrix = (m * k).max(k * n);
    let batch = clamp_to(batch, bounds.max_elements / per_matrix.max(1));

    let mut lhs_shape = batch.clone();
    lhs_shape.extend([m, k]);
    let mut rhs_shape = batch;
    rhs_shape.extend([k, n]);

    let lhs_data = values(rng, element_count(&lhs_shape), Domain::Any, bounds);
    let rhs_data = values(rng, element_count(&rhs_shape), Domain::Any, bounds);

    TensorOp::matmul(
        TensorValue::new(lhs_shape, lhs_data),
        TensorValue::new(rhs_shape, rhs_data),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    #[test]
    fn inner_and_batch_dimensions_always_agree() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Matmul { lhs, rhs } = &case else {
                panic!("expected a matmul, got {case:?}");
            };
            let (ls, rs) = (lhs.shape(), rhs.shape());

            assert!(ls.len() >= 2 && rs.len() >= 2, "{ls:?} times {rs:?}");
            assert_eq!(ls.len(), rs.len(), "{ls:?} times {rs:?}");
            // The shared inner dimension.
            assert_eq!(ls[ls.len() - 1], rs[rs.len() - 2], "{ls:?} times {rs:?}");
            // The leading batch dimensions.
            assert_eq!(
                ls[..ls.len() - 2],
                rs[..rs.len() - 2],
                "{ls:?} times {rs:?}"
            );
        });
    }

    /// Batched multiplication is a separate kernel in most libraries from the plain
    /// two-dimensional case, so both need to be reached.
    #[test]
    fn both_plain_and_batched_cases_get_generated() {
        let bounds = Bounds::default();
        let (mut plain, mut batched) = (false, false);
        for seed in 0..500 {
            if let TensorOp::Matmul { lhs, .. } = generate(&mut SeededRng::from_seed(seed), &bounds)
            {
                match lhs.rank() {
                    2 => plain = true,
                    _ => batched = true,
                }
            }
        }
        assert!(plain, "no plain 2-D matmul was generated");
        assert!(batched, "no batched matmul was generated");
    }
}
