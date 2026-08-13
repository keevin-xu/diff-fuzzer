//! The numbers: filling a tensor once its shape and element type are decided.
//!
//! The second half of the shape-then-value split. Kept apart from `gen_shape.rs` so that the
//! **rate** of special values can vary independently of shape, which is what makes their
//! effect on yield measurable rather than confounded with everything else.
//!
//! Ordinary values only, for now. The adversarial pool arrives with the special-value axis at
//! N4, and it arrives with a **baseline** — a rate without a baseline is not a measurement.

use crate::case::{ElemType, OpKind, TensorData};
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
        // Quantized types span their whole saturation range: unlike the wider integers, every
        // representable value is reachable in a handful of draws, so there is no reason to
        // sample a narrow band. `SPECS.md` §2q.1.
        ElemType::I8 => TensorData::I8((0..count).map(|_| rng.random_range(-128..=127)).collect()),
        ElemType::U8 => TensorData::U8((0..count).map(|_| rng.random_range(0..=255)).collect()),
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
/// Values an operator must never receive, because the specification does not determine the
/// answer and two runtimes will therefore legitimately differ.
///
/// # One definition, consulted everywhere
///
/// Defined here rather than inline at the call site for the same reason `ops::data_elem_type`
/// exists: two places computing the same rule independently is how they come to disagree. Every
/// entry corresponds to a row of [`crate::known::CATALOG`] handled by declining to generate it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Excluded {
    /// `NaN` must not appear in an input.
    pub nan: bool,
    /// `-0.0` must not appear in an input.
    pub negative_zero: bool,
}

/// What the specification leaves undetermined for this operator.
///
/// | operator | excluded | why |
/// |---|---|---|
/// | `Max`, `Min` | `NaN`, `-0.0` | the page never mentions either, in any of its five versions, and these are not IEEE-754 basic operations — `SPECS.md` §2.2c, §2.9 |
/// | `Sign` | `NaN` | the page defines `> 0`, `< 0` and `== 0` only; `NaN` satisfies none of them and is never mentioned — `SPECS.md` §2.10 |
///
/// `Sign(-0.0)` **is** determined: `-0.0 == 0` is true, so the specified answer is `0`. Only
/// `NaN` is excluded, and narrowing the exclusion to exactly what is undetermined is the point.
pub fn undetermined_for(op: OpKind) -> Excluded {
    match op {
        OpKind::Max | OpKind::Min => Excluded {
            nan: true,
            negative_zero: true,
        },
        OpKind::Sign => Excluded {
            nan: true,
            negative_zero: false,
        },
        _ => Excluded::default(),
    }
}

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
    exclude: Excluded,
    rng: &mut SeededRng,
) -> TensorData {
    match elem {
        ElemType::F32 => TensorData::F32(
            (0..count)
                .map(|_| {
                    if rng.random_bool(rate) {
                        pick_f32(exclude, rng)
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
                        f64::from(pick_f32(exclude, rng))
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
        // **The quantized types are all boundary.** `ordinary` already draws uniformly across
        // the whole saturation range, so every value — including both extremes — is reachable
        // in a few draws. There is no separate "special" pool to bias toward, and inventing one
        // would only re-weight a range that is already fully covered. `SPECS.md` §2q.1.
        ElemType::I8 | ElemType::U8 => ordinary(elem, count, rng),
    }
}

fn pick_f32(exclude: Excluded, rng: &mut SeededRng) -> f32 {
    loop {
        let value = SPECIAL_F32[rng.random_range(0..SPECIAL_F32.len())];
        if exclude.nan && value.is_nan() {
            continue;
        }
        // Checked on the **bit pattern**: `-0.0 == 0.0` is true, so a value comparison would
        // fail to exclude anything at all here.
        if exclude.negative_zero && value.to_bits() == (-0.0f32).to_bits() {
            continue;
        }
        return value;
    }
}

/// Values safe to feed a **float-to-integer `Cast`**.
///
/// # Why this exists: "undefined if OOR"
///
/// The ONNX `Cast` reference states, for a float-to-fixed-point conversion:
///
/// > "fixed point: **undefined if OOR**."
///
/// and its `saturate` attribute *"only applies for float 8 conversion"*, not to integers. So
/// casting a float outside the target integer's range has **no determined answer**: `tract`
/// saturates at `int32` bounds, ONNX Runtime at `int64` bounds, and both are legal.
///
/// Measured before the retrieval: this produced **17 divergences in 6,000 cases** that looked
/// exactly like a wrong answer in `tract`. They were ours.
///
/// So the special-value pool is filtered here to the values that remain *finite*: zeros, ±1, and
/// the subnormal boundary (which converts to 0). `±inf`, `NaN`, `f32::MAX` and `f32::MIN` are
/// excluded — every one of them is out of range for every integer target.
///
/// This keeps a real special-value surface for `Cast` rather than dropping to ordinary values
/// entirely, which would have been the easy fix and would have cost the coverage.
///
/// **Finite is necessary and not sufficient**: `-1.0` is finite and out of range for `uint8`.
/// The pool is filtered again per target in [`cast_safe`].
const CAST_SAFE_F32: [f32; 5] = [0.0, -0.0, 1.0, -1.0, f32::MIN_POSITIVE];

/// The magnitude ordinary `Cast` draws reach for, before the target's range narrows it.
const CAST_DRAW_MAGNITUDE: f64 = 100.0;

/// The draw range for a float `Cast` into `target`: the ordinary magnitude, clipped to whatever
/// the target can actually hold.
///
/// Returns a **half-open** range whose upper bound is one step above the largest value the
/// target represents, because that is what `random_range` wants; the clipping below keeps it
/// well inside every real bound.
fn cast_draw_range(target: ElemType) -> (f64, f64) {
    match target.cast_target_range() {
        Some((low, high)) => (low.max(-CAST_DRAW_MAGNITUDE), high.min(CAST_DRAW_MAGNITUDE)),
        // A float or boolean target is not a fixed-point conversion; nothing is out of range.
        None => (-CAST_DRAW_MAGNITUDE, CAST_DRAW_MAGNITUDE),
    }
}

/// Whether `value` survives a `Cast` into `target` with a determined answer.
fn in_cast_range(value: f32, target: ElemType) -> bool {
    match target.cast_target_range() {
        Some((low, high)) => {
            let v = f64::from(value);
            v.is_finite() && v >= low && v <= high
        }
        None => true,
    }
}

/// Values for a `Cast` from a float `source` into `target`.
///
/// # Why the target is a parameter
///
/// It used to draw `-100.0..100.0` regardless, which honours §2.5 for `int32` and violates it
/// for `uint8`, whose range is `[0, 255]`. A campaign duly reported ten `Cast` signatures where
/// `tract` clamped a negative to `0` and ONNX Runtime wrapped it — both legal, all ten ours.
/// See `SPECS.md` §2.5b. **The range belongs to the target, so it is asked of the target.**
///
/// Note what is *not* here: an integer source. Integer-to-integer narrowing **is** specified —
/// it wraps, two's complement (§2.5b) — so those cases have a right answer and are generated
/// ordinarily by the caller. Declining them would have hidden a real disagreement.
pub fn cast_safe(
    source: ElemType,
    target: ElemType,
    count: usize,
    rate: f64,
    rng: &mut SeededRng,
) -> TensorData {
    let (low, high) = cast_draw_range(target);
    // The specials that this target can hold. Never empty: `0.0` is in range for every integer
    // type, so the `rate` branch always has something to offer.
    let specials: Vec<f32> = CAST_SAFE_F32
        .iter()
        .copied()
        .filter(|v| in_cast_range(*v, target))
        .collect();

    let draw = |rng: &mut SeededRng| -> f64 {
        if rng.random_bool(rate) && !specials.is_empty() {
            f64::from(specials[rng.random_range(0..specials.len())])
        } else {
            rng.random_range(low..high)
        }
    };

    match source {
        ElemType::F32 => TensorData::F32((0..count).map(|_| draw(rng) as f32).collect()),
        ElemType::F64 => TensorData::F64((0..count).map(|_| draw(rng)).collect()),
        // Only float sources are undetermined out of range, so only they route through here.
        other => ordinary(other, count, rng),
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
    // **Neither `0` nor `-1`.**
    //
    // `0` because ONNX never says what integer division by zero produces (`SPECS.md` §2.2b).
    //
    // `-1` because of a *correlated* case the per-value exclusions cannot express: `MIN / -1`
    // overflows, since the true result is `2^31` and does not fit. ONNX specifies truncating
    // division and is **silent on overflow** (`SPECS.md` §2.11), so the answer is undetermined —
    // `onnx.reference` and ONNX Runtime wrap, `tract` panics with "attempt to divide with
    // overflow". Excluding `-1` from the divisor is the cheapest way to make the pair
    // unreachable; excluding `MIN` from the dividend would cost a boundary value that matters
    // elsewhere, while `-1` as a divisor is only a sign flip.
    //
    // The behaviour itself is preserved as a candidate finding (F-006) rather than discarded —
    // declining to generate it and reporting it are not in conflict.
    let avoid = |value: i64| value == 0 || value == -1;
    match elem {
        ElemType::I32 => TensorData::I32(
            (0..count)
                .map(|_| {
                    let mut value = rng.random_range(-100..100);
                    while avoid(i64::from(value)) {
                        value = rng.random_range(-100..100);
                    }
                    value
                })
                .collect(),
        ),
        ElemType::I64 => TensorData::I64(
            (0..count)
                .map(|_| {
                    let mut value = rng.random_range(-100i64..100);
                    while avoid(value) {
                        value = rng.random_range(-100i64..100);
                    }
                    value
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

    /// **The test that was missing.** Every float source, every integer target, in range.
    ///
    /// The old version of this function drew `-100.0..100.0` for all targets, which is in range
    /// for `int32` and out of range for `uint8` on every negative draw. There was no test that
    /// looked at the *target*, so a campaign found it instead — ten signatures, all ours
    /// (`SPECS.md` §2.5b). Iterating `ElemType::ALL` rather than naming the targets is the
    /// point: a new integer type joins this test by existing.
    #[test]
    fn cast_values_stay_inside_the_target_range() {
        for source in [ElemType::F32, ElemType::F64] {
            for target in ElemType::ALL {
                let Some((low, high)) = target.cast_target_range() else {
                    continue;
                };
                // A high special rate, so the special-value pool is exercised too — that pool
                // is where `-1.0` and `-0.0` live, and `-1.0` is the out-of-range one.
                let mut rng = SeededRng::from_seed(11);
                let data = cast_safe(source, target, 512, 0.5, &mut rng);
                // `as_f32` only unwraps the `F32` variant, so both are read as `f64` here.
                let values: Vec<f64> = match &data {
                    TensorData::F32(v) => v.iter().map(|x| f64::from(*x)).collect(),
                    TensorData::F64(v) => v.clone(),
                    other => panic!("{source:?} produced non-float data: {other:?}"),
                };
                for v in values {
                    assert!(
                        v.is_finite() && v >= low && v <= high,
                        "{source:?} -> {target:?} produced {v}, outside [{low}, {high}]"
                    );
                }
            }
        }
    }

    /// Narrowing the range must not narrow it to nothing.
    ///
    /// `uint8` loses `-0.0`, `-1.0` and every negative draw. If the filtering had gone one step
    /// further and left an empty pool, the generator would still pass the range test above
    /// while producing a constant — coverage lost silently, which is the failure mode that
    /// makes "safe" values worthless.
    #[test]
    fn a_narrow_cast_target_still_gets_varied_values() {
        let mut rng = SeededRng::from_seed(3);
        let data = cast_safe(ElemType::F32, ElemType::U8, 256, 0.25, &mut rng);
        let values = data.as_f32().expect("float data");
        let distinct = values
            .iter()
            .map(|v| v.to_bits())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert!(
            distinct > 32,
            "uint8 target collapsed to {distinct} distinct values"
        );
        assert!(
            values.contains(&0.0),
            "the in-range specials should still appear"
        );
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
