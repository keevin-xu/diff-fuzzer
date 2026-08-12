//! The numbers: filling a tensor once its shape and element type are decided.
//!
//! The second half of the shape-then-value split. Kept apart from `gen_shape.rs` so that the
//! **rate** of special values can vary independently of shape, which is what makes their
//! effect on yield measurable rather than confounded with everything else.
//!
//! Ordinary values only, for now. The adversarial pool arrives with the special-value axis at
//! N4, and it arrives with a **baseline** — a rate without a baseline is not a measurement.

use crate::case::{ElemType, TensorData};
use diff_fuzzer_core::rng::SeededRng;
use rand::RngExt;

/// Fill `count` elements of `elem` with ordinary values.
///
/// "Ordinary" means finite, of modest magnitude, and **distinct within the tensor** — a tensor
/// of identical values cannot reveal an operator that transposed, reversed, or reordered it.
pub fn ordinary(elem: ElemType, count: usize, rng: &mut SeededRng) -> TensorData {
    match elem {
        // A modest range: large enough to be real arithmetic, small enough that `Mul` does not
        // overflow to infinity on most cases and drown the special-value signal in ordinary
        // overflow once that axis is turned on.
        ElemType::F32 => TensorData::F32(
            (0..count)
                .map(|_| rng.random_range(-100.0..100.0))
                .collect(),
        ),
        ElemType::F64 => TensorData::F64(
            (0..count)
                .map(|_| f64::from(rng.random_range(-100.0f32..100.0)))
                .collect(),
        ),
        ElemType::I32 => TensorData::I32((0..count).map(|_| rng.random_range(-100..100)).collect()),
        ElemType::I64 => {
            TensorData::I64((0..count).map(|_| rng.random_range(-100i64..100)).collect())
        }
        ElemType::Bool => TensorData::Bool((0..count).map(|_| rng.random_bool(0.5)).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_values_are_finite_and_of_the_requested_type() {
        for elem in ElemType::ALL {
            let mut rng = SeededRng::from_seed(7);
            let data = ordinary(elem, 32, &mut rng);
            assert_eq!(data.elem_type(), elem);
            assert_eq!(data.len(), 32);
            if let Some(values) = data.as_f32() {
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{elem:?} produced a non-finite value"
                );
            }
        }
    }

    /// The same seed must give the same numbers, or no finding replays.
    #[test]
    fn value_generation_is_deterministic() {
        for elem in ElemType::ALL {
            let a = ordinary(elem, 16, &mut SeededRng::from_seed(42));
            let b = ordinary(elem, 16, &mut SeededRng::from_seed(42));
            assert_eq!(a, b, "{elem:?} was not reproducible");
        }
    }

    /// Values must vary within a tensor. A constant tensor cannot reveal an operator that
    /// reordered its elements, so a generator emitting one is testing less than it appears to.
    #[test]
    fn values_vary_within_a_tensor() {
        let mut rng = SeededRng::from_seed(3);
        let data = ordinary(ElemType::F32, 64, &mut rng);
        let values = data.as_f32().unwrap();
        let distinct = values
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            distinct.len() > 32,
            "only {} distinct values in 64",
            distinct.len()
        );
    }

    #[test]
    fn an_empty_tensor_is_producible() {
        let mut rng = SeededRng::from_seed(1);
        assert_eq!(ordinary(ElemType::F32, 0, &mut rng).len(), 0);
    }
}
