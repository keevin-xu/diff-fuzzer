//! The quantization round trip: `DequantizeLinear(QuantizeLinear(x))` must stay within half a
//! quantization step of `x`.
//!
//! # Why this oracle exists, beyond being a nice property
//!
//! Two reasons, and the second is the one that made it necessary rather than optional.
//!
//! **It is single-runtime.** A differential oracle compares implementations against each other,
//! so a bug they *share* is invisible — they agree, and agreement is what the oracle reports. A
//! metamorphic relation compares an implementation against **the specification's own arithmetic**,
//! and catches exactly that class.
//!
//! **And half the quantized surface has only one participant.** The N9 census measured it:
//! `tract` rejects both `QuantizeLinear` and `DequantizeLinear`, and candle has no `int8` type at
//! all, so ONNX Runtime is the only implementation that runs them. A differential oracle over one
//! participant is not an oracle. Without this relation those two operators would be generated,
//! executed, and then **skipped as `TooFewResults`** — measured and unjudged.
//!
//! # The bound is derived, never fitted
//!
//! From `SPECS.md` §2q.1 and §2q.2, quoted:
//!
//! - `QuantizeLinear` computes `saturate(round_half_even(x / scale) + zero_point)`
//! - `DequantizeLinear` computes `(q - zero_point) * scale`
//!
//! Compose them, and when no saturation occurs the zero-point cancels exactly:
//!
//! ```text
//! dequantize(quantize(x)) = (round_half_even(x / scale) + zp - zp) * scale
//!                         = round_half_even(x / scale) * scale
//! ```
//!
//! Rounding to the nearest representable multiple moves a value by at most **half a step**, so:
//!
//! ```text
//! |dequantize(quantize(x)) - x| <= scale / 2
//! ```
//!
//! **That is a derivation, not a measurement.** `02-METHODOLOGY.md` is explicit that a threshold
//! fitted to observed behaviour encodes today's implementations as the standard and then passes
//! forever — 235 false positives, once. Nothing here was tuned by running anything.
//!
//! # Where the bound does not hold, and why that is a generator rule
//!
//! **Saturation breaks it.** If `x / scale + zero_point` falls outside the target type's range,
//! the result clamps and the error is bounded by nothing at all. The relation is then simply
//! false, and a case that violates it would be a false finding.
//!
//! So saturation is excluded **by construction** rather than forgiven afterwards — the same
//! choice this domain made five times over in `known.rs`. [`representable`] decides it, and only
//! values that survive it are checked.

use crate::case::ElemType;

/// Half a quantization step: the largest error the round trip may introduce.
///
/// Derived in the module comment. Takes the scale rather than measuring anything.
pub fn tolerance(scale: f32) -> f32 {
    scale.abs() / 2.0
}

/// Whether `x` survives the round trip without saturating.
///
/// The quantized value is `round(x / scale) + zero_point`, and it must land inside the target
/// type's range. A margin of one step is kept on each side: a value exactly at the boundary
/// round-trips correctly but leaves no room for the rounding itself to move it outward, and a
/// relation that holds "except sometimes at the edge" is not one worth having.
pub fn representable(x: f32, scale: f32, zero_point: i64, target: ElemType) -> bool {
    if !x.is_finite() || !scale.is_finite() || scale <= 0.0 {
        return false;
    }
    let Some((low, high)) = target.saturation_range() else {
        return false;
    };
    let quantized = (x / scale).round() as i64 + zero_point;
    quantized > low && quantized < high
}

/// Did the round trip stay inside the derived bound?
///
/// Returns `None` when the input was not representable, which is *not* a pass — it is a case the
/// relation says nothing about, and the caller must count it separately rather than folding it in.
/// Counting unjudgeable cases as passes is how a bound that discriminates nothing looks like a
/// bound that everything satisfies.
pub fn holds(
    x: f32,
    round_tripped: f32,
    scale: f32,
    zero_point: i64,
    target: ElemType,
) -> Option<bool> {
    if !representable(x, scale, zero_point, target) {
        return None;
    }
    if !round_tripped.is_finite() {
        return Some(false);
    }
    Some((round_tripped - x).abs() <= tolerance(scale))
}

/// What a run of the relation found.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    /// Values the relation applied to and that satisfied it.
    pub held: usize,
    /// Values the relation applied to and that **violated** it. Findings.
    pub violated: usize,
    /// Values excluded because they would saturate. Neither pass nor fail.
    pub not_representable: usize,
}

impl Outcome {
    /// Values the relation was actually able to judge.
    ///
    /// The denominator any rate must use. `05-MEASUREMENT-AND-CAMPAIGNS.md` calls this the
    /// effective rather than the nominal bound, and the difference is the excluded values.
    pub fn judged(&self) -> usize {
        self.held + self.violated
    }

    /// Fold one value's verdict in.
    pub fn record(&mut self, verdict: Option<bool>) {
        match verdict {
            Some(true) => self.held += 1,
            Some(false) => self.violated += 1,
            None => self.not_representable += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound is exactly half a step, and it comes from the scale alone.
    #[test]
    fn the_bound_is_half_a_quantization_step() {
        assert_eq!(tolerance(1.0), 0.5);
        assert_eq!(tolerance(0.02), 0.01);
        // Sign of the scale cannot widen it; a negative scale is invalid anyway and is caught
        // by `representable`.
        assert_eq!(tolerance(-1.0), 0.5);
    }

    /// **The relation must actually hold on correct arithmetic.** Simulated here rather than
    /// run against a runtime, so the property is tested independently of any implementation.
    #[test]
    fn correct_round_trips_satisfy_the_bound() {
        let scale = 0.25_f32;
        for step in -200..200 {
            let x = step as f32 * 0.037; // deliberately not a multiple of the scale
            if !representable(x, scale, 0, ElemType::I8) {
                continue;
            }
            // Exactly what the two operators compose to, per the derivation.
            let q = (x / scale).round_ties_even();
            let back = q * scale;
            assert_eq!(
                holds(x, back, scale, 0, ElemType::I8),
                Some(true),
                "x = {x} round-tripped to {back} outside +/- {}",
                tolerance(scale)
            );
        }
    }

    /// **And it must reject an answer that is wrong.** A relation that accepts everything would
    /// pass the test above and be worthless — the same failure the tensor domain measured, where
    /// 96% of `matmul` cases carried a bound nothing could fail.
    #[test]
    fn an_error_larger_than_half_a_step_is_caught() {
        let scale = 0.25_f32;
        let x = 1.0_f32;
        // One full step out: double what the rounding rule permits.
        assert_eq!(holds(x, x + scale, scale, 0, ElemType::I8), Some(false));
        assert_eq!(holds(x, x - scale, scale, 0, ElemType::I8), Some(false));
        // And just inside is accepted, so the boundary is where it claims to be.
        assert_eq!(
            holds(x, x + scale / 2.0 - 1e-6, scale, 0, ElemType::I8),
            Some(true)
        );
    }

    /// A value that would saturate is **not judged**, rather than counted as a pass.
    #[test]
    fn saturating_values_are_excluded_not_passed() {
        let scale = 0.1_f32;
        // 127 * 0.1 = 12.7 is the top of int8's range at this scale.
        assert!(representable(5.0, scale, 0, ElemType::I8));
        assert!(!representable(100.0, scale, 0, ElemType::I8));
        assert_eq!(holds(100.0, 12.7, scale, 0, ElemType::I8), None);

        // The zero-point shifts the window, and the check must follow it.
        assert!(!representable(5.0, scale, 120, ElemType::I8));
    }

    /// `uint8` and `int8` have different windows, and the relation must use the right one.
    #[test]
    fn the_target_type_decides_the_window() {
        let scale = 1.0_f32;
        // -5 is fine for int8 and impossible for uint8 at zero-point 0.
        assert!(representable(-5.0, scale, 0, ElemType::I8));
        assert!(!representable(-5.0, scale, 0, ElemType::U8));
        // With the zero-point in the middle, uint8 handles it.
        assert!(representable(-5.0, scale, 128, ElemType::U8));
    }

    /// Non-finite inputs and invalid scales are excluded rather than crashing or silently
    /// passing. A zero scale would divide by zero; a negative one is not a valid scale at all.
    #[test]
    fn invalid_parameters_are_not_judgeable() {
        assert_eq!(holds(f32::NAN, 0.0, 1.0, 0, ElemType::I8), None);
        assert_eq!(holds(f32::INFINITY, 0.0, 1.0, 0, ElemType::I8), None);
        assert_eq!(holds(1.0, 1.0, 0.0, 0, ElemType::I8), None);
        assert_eq!(holds(1.0, 1.0, -1.0, 0, ElemType::I8), None);
    }

    /// A `NaN` coming *back* from a round trip is a violation, not an exclusion — the input was
    /// judgeable, so the answer must be too.
    #[test]
    fn a_nan_result_from_a_valid_input_is_a_violation() {
        assert_eq!(holds(1.0, f32::NAN, 0.25, 0, ElemType::I8), Some(false));
    }

    /// The tally must keep the three states apart.
    #[test]
    fn the_outcome_separates_unjudged_from_passed() {
        let mut outcome = Outcome::default();
        outcome.record(Some(true));
        outcome.record(Some(false));
        outcome.record(None);
        assert_eq!(outcome.held, 1);
        assert_eq!(outcome.violated, 1);
        assert_eq!(outcome.not_representable, 1);
        assert_eq!(outcome.judged(), 2, "an excluded value is not a judged one");
    }
}
