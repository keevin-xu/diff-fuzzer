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

/// The values worth injecting deliberately, because sampling never produces them.
///
/// Uniform sampling essentially never yields `0.0`, `±inf`, `NaN`, a subnormal or `f32::MAX`,
/// and **both of this project's prior real findings were special-value bugs** — a `matmul`
/// overflow giving `inf` on one backend and `NaN` on another, and a reduction seeded with the
/// smallest finite float instead of `−inf`. Neither would have been found by sampling.
///
/// Every entry is here because it has broken something somewhere: overflow to infinity, the
/// sign of zero, the subnormal boundary, and the largest finite magnitudes.
const SPECIAL_F32: [f32; 10] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MIN_POSITIVE, // the smallest normal — the subnormal boundary
    f32::MAX,
    f32::MIN,
];

/// The integer equivalents: boundaries where wrapping and saturation differ.
const SPECIAL_I64: [i64; 7] = [
    0,
    1,
    -1,
    i64::MAX,
    i64::MIN,
    i32::MAX as i64,
    i32::MIN as i64,
];

/// Values with special ones injected at `rate`.
///
/// `exclude_nan` exists for `Max` and `Min`, whose `NaN` behaviour **ONNX does not specify** —
/// the operator page never mentions it, and unlike `Add`/`Sub`/`Mul`/`Div`/`Sqrt` they are not
/// IEEE-754 basic operations, so IEEE does not supply the answer either. IEEE-754's own
/// `maxNum`/`minNum` semantics changed between its 2008 and 2019 revisions. A case whose answer
/// no document determines is a false finding waiting to be triaged, so it is not generated.
/// `SPECS.md` §2.2c, `PENDING` 1.13.
pub fn with_specials(
    elem: ElemType,
    count: usize,
    rate: f64,
    exclude_nan: bool,
    rng: &mut SeededRng,
) -> TensorData {
    match elem {
        ElemType::F32 => TensorData::F32(
            (0..count)
                .map(|_| {
                    if rng.random_bool(rate) {
                        pick_f32(exclude_nan, rng)
                    } else {
                        rng.random_range(-100.0..100.0)
                    }
                })
                .collect(),
        ),
        ElemType::F64 => TensorData::F64(
            (0..count)
                .map(|_| {
                    if rng.random_bool(rate) {
                        f64::from(pick_f32(exclude_nan, rng))
                    } else {
                        f64::from(rng.random_range(-100.0f32..100.0))
                    }
                })
                .collect(),
        ),
        ElemType::I32 => TensorData::I32(
            (0..count)
                .map(|_| {
                    if rng.random_bool(rate) {
                        // Saturating, so an i64 boundary lands on the i32 boundary rather than
                        // wrapping into an arbitrary value that tests nothing in particular.
                        SPECIAL_I64[rng.random_range(0..SPECIAL_I64.len())]
                            .clamp(i64::from(i32::MIN), i64::from(i32::MAX))
                            as i32
                    } else {
                        rng.random_range(-100..100)
                    }
                })
                .collect(),
        ),
        ElemType::I64 => TensorData::I64(
            (0..count)
                .map(|_| {
                    if rng.random_bool(rate) {
                        SPECIAL_I64[rng.random_range(0..SPECIAL_I64.len())]
                    } else {
                        rng.random_range(-100i64..100)
                    }
                })
                .collect(),
        ),
        // A boolean has no special values: both of its two values are ordinary.
        ElemType::Bool => ordinary(ElemType::Bool, count, rng),
    }
}

fn pick_f32(exclude_nan: bool, rng: &mut SeededRng) -> f32 {
    loop {
        let value = SPECIAL_F32[rng.random_range(0..SPECIAL_F32.len())];
        if exclude_nan && value.is_nan() {
            continue;
        }
        return value;
    }
}

/// Ordinary values with **no zeros**, for a divisor.
///
/// # Why this exists: an undetermined answer must not be generated
///
/// Integer division by zero was found, at N3, to make `tract` and `candle` panic while
/// `onnx.reference` returns `0`. That looks like a conformance finding and **may not be one**:
/// the reference's `Div` is a thin wrapper over numpy, so its `0` is numpy's answer, and
/// whether ONNX *specifies* integer division by zero has not been retrieved.
///
/// Until it is, the case's answer is not known to be determined — and
/// `03-CONCEPTS.md` §7 is explicit that the generator must refuse to produce cases whose
/// answer the specification does not pin down. A case permitting two correct answers is a
/// false finding paid for in triage.
///
/// This follows the precedent `02-METHODOLOGY.md` records: SQL needed to know whether
/// `PARTITION BY` and `GROUP BY` treat two `NULL`s alike, neither engine documented it, and
/// rather than assume, the relation **declined cases with a `NULL` key** — sound either way.
/// Declining here is sound either way too. If the specification turns out to pin the answer
/// down, this restriction is lifted and the finding is real; if it does not, we were right to
/// refuse. `PENDING` 1.11.
///
/// Floats are **not** restricted: division by zero is defined by IEEE-754 and produces
/// `±inf`/`NaN`, which is specified behaviour and exactly the surface this domain wants.
pub fn nonzero(elem: ElemType, count: usize, rng: &mut SeededRng) -> TensorData {
    match elem {
        ElemType::I32 => TensorData::I32(
            (0..count)
                .map(|_| {
                    let value = rng.random_range(-100..99);
                    if value >= 0 { value + 1 } else { value }
                })
                .collect(),
        ),
        ElemType::I64 => TensorData::I64(
            (0..count)
                .map(|_| {
                    let value = rng.random_range(-100i64..99);
                    if value >= 0 { value + 1 } else { value }
                })
                .collect(),
        ),
        // Floats and booleans are unrestricted: float division by zero is IEEE-754 defined,
        // and `Div` does not accept booleans at all.
        other => ordinary(other, count, rng),
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

    /// The divisor pool must contain no zeros, or the restriction is decoration.
    #[test]
    fn nonzero_never_produces_zero_for_integers() {
        for elem in [ElemType::I32, ElemType::I64] {
            let mut rng = SeededRng::from_seed(11);
            let data = nonzero(elem, 4_000, &mut rng);
            let zeros = data.to_bit_keys().iter().filter(|b| **b == 0).count();
            assert_eq!(zeros, 0, "{elem:?} divisor pool contained {zeros} zeros");
        }
    }

    /// ...and it must still produce both signs, or excluding zero has quietly excluded half
    /// the number line as well.
    #[test]
    fn nonzero_still_spans_both_signs() {
        let mut rng = SeededRng::from_seed(12);
        let TensorData::I64(values) = nonzero(ElemType::I64, 2_000, &mut rng) else {
            panic!("wrong variant");
        };
        assert!(values.iter().any(|v| *v > 0), "no positive divisors");
        assert!(values.iter().any(|v| *v < 0), "no negative divisors");
    }

    /// Floats keep their zeros: dividing by zero is IEEE-754 defined and is exactly the
    /// surface this domain exists to test. Restricting them would be giving away signal.
    #[test]
    fn nonzero_does_not_restrict_floats() {
        let mut rng = SeededRng::from_seed(13);
        let restricted = nonzero(ElemType::F32, 64, &mut rng);
        let mut rng = SeededRng::from_seed(13);
        let plain = ordinary(ElemType::F32, 64, &mut rng);
        assert_eq!(restricted, plain, "float divisors must be unrestricted");
    }

    #[test]
    fn an_empty_tensor_is_producible() {
        let mut rng = SeededRng::from_seed(1);
        assert_eq!(ordinary(ElemType::F32, 0, &mut rng).len(), 0);
    }
}
