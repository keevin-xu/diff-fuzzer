//! One argument plus an axis to collapse.
//!
//! The constraint is that the axis must be a dimension the tensor actually has. It is
//! satisfied by generating the shape first and then choosing an axis from the range
//! that shape defines — so the axis cannot be out of range, because it is drawn from
//! the valid range by construction.

use crate::input::{ReduceOp, TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, element_count, shape, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every reduction the generator may pick.
pub const ALL: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];

/// Build a valid reduction case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];

    let shape = shape(rng, bounds);
    // Chosen *after* the shape, from the range the shape defines.
    let axis = rng.random_range(0..shape.len());
    let data = values(rng, element_count(&shape), Domain::Any, bounds);

    TensorOp::reduce(kind, TensorValue::new(shape, data), axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    #[test]
    fn the_axis_is_always_within_the_rank() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Reduce { arg, axis, .. } = &case else {
                panic!("expected a reduction, got {case:?}");
            };
            assert!(*axis < arg.rank(), "axis {axis} for rank {}", arg.rank());
        });
    }

    /// Reducing along the last axis is a different code path in most libraries from
    /// reducing along the first, because of how tensors are laid out in memory. If the
    /// generator only ever picked axis 0, that path would go untested.
    #[test]
    fn axes_other_than_the_first_get_chosen() {
        let bounds = Bounds::default();
        let mut saw_non_zero_axis = false;
        for seed in 0..500 {
            if let TensorOp::Reduce { axis, .. } =
                generate(&mut SeededRng::from_seed(seed), &bounds)
                && axis > 0
            {
                saw_non_zero_axis = true;
            }
        }
        assert!(saw_non_zero_axis, "only axis 0 was ever generated");
    }
}
