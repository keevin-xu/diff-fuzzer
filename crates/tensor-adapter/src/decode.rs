//! Turning fuzzer bytes into valid tensor cases.
//!
//! This is the piece that makes coverage-guided fuzzing worth doing, and it is easy to
//! get subtly wrong in a way that leaves the fuzzer no better than random sampling.
//!
//! # The property that matters: locality
//!
//! libFuzzer works by mutating inputs and keeping the mutations that reach new code. For
//! that to mean anything, **a small change to the bytes must produce a small change to
//! the case.** If flipping one bit produced an unrelated case, the fuzzer could never
//! learn "that input was close, try something like it" — the feedback would be noise, and
//! the whole apparatus would reduce to an expensive random generator.
//!
//! The decoding below therefore uses a **fixed layout**, consuming bytes in a fixed
//! order for fixed purposes:
//!
//! ```text
//!   byte 0      which operation
//!   byte 1      rank
//!   bytes 2..   one dimension each
//!   then        one byte per value
//! ```
//!
//! So a mutation late in the input perturbs a *value* while leaving the shape alone, and
//! a mutation at byte 1 changes the rank. Both are useful moves; neither destroys
//! everything the fuzzer had learned.
//!
//! # Clamping, never rejecting
//!
//! Every byte is mapped *into* the valid range rather than checked against it. Rejecting
//! an input costs the fuzzer an execution and teaches it nothing, and with constraints as
//! tight as a matrix multiplication's shared dimension, rejection would discard nearly
//! everything. The same correct-by-construction principle as the seeded generator, from a
//! different source of randomness.
//!
//! Running out of bytes is handled the same way: exhausted data reads as zero rather than
//! failing, so a short input still yields a valid, if minimal, case.

use crate::input::{ActivationOp, BinaryOp, ReduceOp, TensorOp, TensorValue, UnaryOp};
use crate::ops::{Bounds, DIVISOR_FLOOR, Domain, SPECIAL_VALUES};
use arbitrary::{Arbitrary, Result, Unstructured};

/// Bounds used when decoding. Fixed rather than configurable, because the fuzzer's
/// corpus is only meaningful for the layout that produced it: changing these would
/// silently reinterpret every saved input.
///
/// **`max_dim` widened at PHASE-8 from 8 to 64, and nothing else changed.** A sweep varying
/// one axis at a time settled which one mattered:
///
/// | bounds | diverged / 2,000 | sec | diverg/sec |
/// |---|---|---|---|
/// | `max_dim: 8` (historical) | 0 | 0.0 | 0 |
/// | `magnitude: 1000.0` | 0 | 0.0 | 0 |
/// | `max_dim: 64`, budget 4k | 0 | 23.7 | 0 |
/// | `max_dim: 64`, budget 64k | 3 | 22.7 | 0.132 |
/// | **`max_dim: 64`, budget 1M** | **5** | 24.3 | **0.206** |
///
/// Dimension is the whole effect; magnitude contributes nothing. That fits the mechanism
/// behind the one real finding — a tile-remainder effect governed by `(m mod 4) * (n mod 8)`
/// — which is a property of **shape**, not of the values.
///
/// **The cost is real and large:** roughly 80 cases/second against tens of thousands at the
/// old bounds, because matmul costs `m × k × n`. It is accepted because the alternative rate
/// is zero, and 0.206/sec beats any multiple of nothing — but it is a campaign-shaping
/// trade, not a free win.
///
/// **Changing these invalidates the corpus and every negative recorded under them.** The
/// corpus must be started fresh, and old negatives stop matching because
/// [`FUZZER_GENERATOR`](crate::negatives::FUZZER_GENERATOR) names the bounds — which is the
/// point: they were drawn from a different distribution.
pub const DECODE_BOUNDS: Bounds = Bounds {
    max_rank: crate::backends::MAX_RANK,
    max_dim: 64,
    magnitude: 10.0,
    // **Measured, not guessed.** At `max_dim: 64`, budgets of 4k / 64k / 1M cost 23.7s /
    // 22.7s / 24.3s per 2,000 cases and found 0 / 3 / 5 divergences. The budget is very
    // nearly free — the time goes to matmul's `m × k × n`, which no element cap touches —
    // so a tight budget is the worst of both worlds: full price, nothing found.
    max_elements: 1_048_576,
    // Unused here — special values are selected by the byte layout below rather than by
    // a probability — but the struct requires them.
    special_value_rate: 0.0,
    // **Unrestricted**, so `sqrt` receives negatives and divisors may be zero. Those
    // produce `NaN` and infinity, which the comparison has an explicit policy for since
    // PHASE-4 — and measurement showed the cost is about 0.4% of executions spent on
    // cases that verify nothing. In exchange the fuzzer explores the overflow region,
    // which is where the one real finding so far came from.
    restrict_domains: false,
};

/// One byte, or zero if the input is exhausted.
///
/// Never fails. A short input should still produce a case rather than being thrown away,
/// since a rejected execution teaches the fuzzer nothing.
fn byte(u: &mut Unstructured<'_>) -> u8 {
    u.arbitrary::<u8>().unwrap_or(0)
}

/// A number in `1..=max`, folded from one byte.
fn size(u: &mut Unstructured<'_>, max: usize) -> usize {
    1 + (byte(u) as usize) % max
}

/// One value, from one byte.
///
/// Exactly one byte per value keeps positions stable: mutating the byte for element 7
/// changes element 7 and nothing else.
///
/// The low range of the byte selects a deliberately interesting value, and the rest maps
/// onto a grid across the magnitude range. The grid is coarse — 224 distinct magnitudes —
/// and that is not a defect: fewer distinct values means fewer inputs that are equivalent
/// for our purposes, so the fuzzer wastes less time distinguishing between them.
fn value(u: &mut Unstructured<'_>, domain: Domain) -> f32 {
    let selector = byte(u);

    // Roughly one in eight, matching the seeded generator's rate.
    if selector < 32 {
        let allowed: Vec<f32> = SPECIAL_VALUES
            .iter()
            .copied()
            .filter(|v| match domain {
                Domain::Any => true,
                Domain::NonNegative => *v >= 0.0,
                Domain::NonZero => v.abs() >= DIVISOR_FLOOR,
            })
            .collect();

        if !allowed.is_empty() {
            return allowed[(selector as usize) % allowed.len()];
        }
    }

    // Map the remaining range onto the magnitude interval.
    let fraction = (selector as f32 - 32.0) / 224.0;
    let magnitude = DECODE_BOUNDS.magnitude;

    match domain {
        Domain::Any => -magnitude + fraction * 2.0 * magnitude,
        Domain::NonNegative => fraction * magnitude,
        Domain::NonZero => {
            let scaled = DIVISOR_FLOOR + fraction * (magnitude - DIVISOR_FLOOR);
            // The selector's lowest bit picks the sign, so both are reachable.
            if selector & 1 == 0 { scaled } else { -scaled }
        }
    }
}

/// A tensor of the given shape, one byte per element.
fn tensor(u: &mut Unstructured<'_>, shape: Vec<usize>, domain: Domain) -> TensorValue {
    let count: usize = shape.iter().product();
    let data = (0..count).map(|_| value(u, domain)).collect();
    TensorValue::new(shape, data)
}

/// What a unary operation's argument is allowed to be.
///
/// Consults `DECODE_BOUNDS` rather than hardcoding the restriction — which it previously
/// did, so flipping the setting would have had no effect on `sqrt` and quietly done
/// nothing.
fn unary_domain(kind: UnaryOp) -> Domain {
    match kind {
        UnaryOp::Sqrt if DECODE_BOUNDS.restrict_domains => Domain::NonNegative,
        _ => Domain::Any,
    }
}

/// What a binary operation's right operand is allowed to be.
fn binary_right_domain(kind: BinaryOp) -> Domain {
    match kind {
        BinaryOp::Div if DECODE_BOUNDS.restrict_domains => Domain::NonZero,
        _ => Domain::Any,
    }
}

/// A shape of `rank` dimensions.
/// A shape whose dimensions come from the input, clamped to [`MAX_ELEMENTS`].
///
/// Clamping rather than rejecting: a rejected input teaches the fuzzer nothing, and the
/// byte layout must keep meaning the same thing so that a mutation stays local.
fn shape(u: &mut Unstructured<'_>, rank: usize) -> Vec<usize> {
    let raw: Vec<usize> = (0..rank).map(|_| size(u, DECODE_BOUNDS.max_dim)).collect();
    crate::ops::clamp_to(raw, element_budget(u))
}

/// Split a result shape into two operands that broadcast to it, one byte per axis.
///
/// Setting an axis to 1 on one side makes that side stretch along it. **Never both sides at
/// once** — that would shrink the result below the shape already drawn and budgeted.
///
/// One byte per axis keeps the layout positional: flipping a shape byte changes an extent,
/// flipping a stretch byte changes which operand broadcasts, and neither disturbs the values
/// that follow.
fn broadcast_operands(u: &mut Unstructured<'_>, result: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut lhs = result.to_vec();
    let mut rhs = result.to_vec();

    for i in 0..result.len() {
        match byte(u) % 4 {
            0 => lhs[i] = 1,
            1 => rhs[i] = 1,
            // Two of four outcomes leave the axis alone, so equal-shaped cases stay common.
            // They are the ordinary path in real use and are where every finding so far came
            // from; a decoder that always broadcast would have stopped testing them.
            _ => {}
        }
    }

    (lhs, rhs)
}

/// How many elements the remaining input can meaningfully describe.
///
/// **The layout is one byte per value.** Once the input is exhausted `byte` returns 0, so a
/// 200-byte input describing a million-element tensor produces 999,800 zeros — a case far
/// larger than anything the fuzzer said, costing time proportional to its size while
/// carrying no more information than the 200 bytes did.
///
/// Tying the budget to the input length fixes that and hands size control to libFuzzer,
/// where it belongs: `-max_len` now genuinely bounds the work per execution, and a mutation
/// that lengthens an input is what reaches the larger shapes.
///
/// `DECODE_BOUNDS.max_elements` remains an absolute ceiling on top of this.
fn element_budget(u: &Unstructured<'_>) -> usize {
    // At least one, so an exhausted input still yields a valid (tiny) case rather than a
    // degenerate one — a rejected input teaches the fuzzer nothing.
    u.len().max(1).min(DECODE_BOUNDS.max_elements)
}

impl<'a> Arbitrary<'a> for TensorOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Byte 0 chooses the operation. Weighted by class the same way the seeded
        // generator is, so no operation is starved.
        //
        // **11 rather than 10 since PHASE-7D**, making room for `softmax`. Changing this
        // divisor re-interprets every byte string in a corpus, which is why
        // `FUZZER_GENERATOR` names the layout version.
        let choice = byte(u) as usize % 11;

        Ok(match choice {
            0..=3 => {
                let kind = [UnaryOp::Neg, UnaryOp::Abs, UnaryOp::Exp, UnaryOp::Sqrt][choice];
                let domain = unary_domain(kind);
                // Sequential bindings rather than nesting: the order bytes are consumed
                // in *is* the layout, so making it explicit keeps it stable.
                let rank = size(u, DECODE_BOUNDS.max_rank);
                let shape = shape(u, rank);
                TensorOp::unary(kind, tensor(u, shape, domain))
            }
            4..=7 => {
                let kind = [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul, BinaryOp::Div][choice - 4];
                let rank = size(u, DECODE_BOUNDS.max_rank);
                // The **result** shape, from which both operands are derived — the same
                // correct-by-construction order the seeded generator uses. Deriving operands
                // from the answer means the fuzzer cannot produce an incompatible pair, so no
                // mutation is ever wasted on a case that would panic before running.
                //
                // It also puts the element budget where it belongs: `shape` clamps what it
                // returns, and a broadcast result is larger than either operand.
                let result = shape(u, rank);
                let (lhs_shape, rhs_shape) = broadcast_operands(u, &result);
                let right_domain = binary_right_domain(kind);
                TensorOp::binary(
                    kind,
                    tensor(u, lhs_shape, Domain::Any),
                    tensor(u, rhs_shape, right_domain),
                )
            }
            10 => {
                // `softmax`, whose dimension is drawn *after* the shape so that the shape
                // bytes keep their positions and a mutation of one does not shift the other.
                let rank = size(u, DECODE_BOUNDS.max_rank);
                let shape = shape(u, rank);
                // Folded into range, so no byte value can produce a case burn would panic on.
                let dim = (byte(u) as usize) % shape.len();
                TensorOp::activation(ActivationOp::Softmax, tensor(u, shape, Domain::Any), dim)
            }
            8 => {
                let rank = size(u, DECODE_BOUNDS.max_rank);
                let shape = shape(u, rank);
                // Drawn from the range the shape defines, so it cannot be out of range.
                let axis = (byte(u) as usize) % shape.len();
                TensorOp::reduce(ReduceOp::Sum, tensor(u, shape, Domain::Any), axis)
            }
            _ => {
                // At least rank 2, since a matrix multiplication of vectors is undefined.
                let rank = 2 + (byte(u) as usize) % (DECODE_BOUNDS.max_rank - 1);
                let batch = shape(u, rank - 2);

                // Drawn once each and placed into both operands, so the shared inner
                // dimension cannot disagree.
                let m = size(u, DECODE_BOUNDS.max_dim);
                let k = size(u, DECODE_BOUNDS.max_dim);
                let n = size(u, DECODE_BOUNDS.max_dim);
                // A matmul operand is `batch × m × k`, so the batch gets whatever the
                // matrix dimensions have not already spent.
                let budget = element_budget(u);
                let per_matrix = (m * k).max(k * n).max(1);
                let batch = crate::ops::clamp_to(batch, budget / per_matrix.max(1));

                let mut lhs_shape = batch.clone();
                lhs_shape.extend([m, k]);
                let mut rhs_shape = batch;
                rhs_shape.extend([k, n]);

                TensorOp::matmul(
                    tensor(u, lhs_shape, Domain::Any),
                    tensor(u, rhs_shape, Domain::Any),
                )
            }
        })
    }
}

#[cfg(test)]
mod tests {
    /// **The decoder must never emit an operand pair that cannot run.** A case that panics
    /// in `TensorOp::binary` would take the whole fuzz process down, and libFuzzer would
    /// report it as a crash in the target rather than an invalid input.
    #[test]
    fn every_decoded_binary_case_has_operands_that_combine() {
        for bytes in byte_strings(3_000) {
            let mut u = Unstructured::new(&bytes);
            let Ok(case) = TensorOp::arbitrary(&mut u) else {
                continue;
            };
            if let TensorOp::Binary { lhs, rhs, .. } = case {
                assert!(
                    crate::ops::broadcast::compatible(lhs.shape(), rhs.shape()),
                    "decoded an incompatible pair: {:?} and {:?}",
                    lhs.shape(),
                    rhs.shape()
                );
            }
        }
    }

    /// **Every broadcast shape must be reachable, including both operands stretching.**
    ///
    /// A campaign corpus showed 28 broadcast cases of which *all 28* stretched a whole
    /// operand and **none** stretched both — which could mean the decoder cannot produce it,
    /// or merely that a two-minute run did not. That distinction matters: an unreachable
    /// feature can never be validated, and would sit in the vocabulary looking useful while
    /// scoring `NeverSampled` forever. This settles it directly.
    #[test]
    fn both_operands_stretching_is_reachable_from_bytes() {
        let mut both = 0;
        for bytes in byte_strings(5_000) {
            let mut u = Unstructured::new(&bytes);
            let Ok(TensorOp::Binary { lhs, rhs, .. }) = TensorOp::arbitrary(&mut u) else {
                continue;
            };
            let Some(result) = crate::ops::broadcast::result_shape(lhs.shape(), rhs.shape()) else {
                continue;
            };
            if lhs.shape() != result.as_slice() && rhs.shape() != result.as_slice() {
                both += 1;
            }
        }
        assert!(
            both > 0,
            "the decoder cannot produce a case where both operands stretch"
        );
    }

    /// **Broadcasting must be reachable from bytes**, not merely from the seeded generator.
    /// The fuzzer is what runs for hours; if its decoder never stretches an axis, PHASE-7C
    /// added nothing to a campaign however good the generator is.
    #[test]
    fn broadcasting_is_reachable_from_fuzzer_bytes() {
        let mut broadcasting = 0;
        let mut equal = 0;

        for bytes in byte_strings(3_000) {
            let mut u = Unstructured::new(&bytes);
            let Ok(TensorOp::Binary { lhs, rhs, .. }) = TensorOp::arbitrary(&mut u) else {
                continue;
            };
            if lhs.shape() == rhs.shape() {
                equal += 1;
            } else {
                broadcasting += 1;
            }
        }

        assert!(
            broadcasting > 50,
            "bytes rarely decode to a broadcast: {broadcasting}"
        );
        assert!(equal > 50, "equal shapes became rare: {equal}");
    }

    /// **The tie between the decode bounds and the string that identifies them.**
    ///
    /// The pool matches negatives on the generator description verbatim. If the bounds are
    /// widened and the description is not updated, negatives from two different
    /// distributions become indistinguishable and the guard silently stops guarding.
    #[test]
    fn bounds_are_named_in_the_generator_description() {
        let description = crate::negatives::FUZZER_GENERATOR;

        assert!(
            description.contains(&format!("max_dim {}", DECODE_BOUNDS.max_dim)),
            "DECODE_BOUNDS.max_dim is {} but the generator description says {description:?}",
            DECODE_BOUNDS.max_dim
        );
        assert!(
            description.contains(&format!("magnitude {}", DECODE_BOUNDS.magnitude as u64)),
            "DECODE_BOUNDS.magnitude is {} but the generator description says {description:?}",
            DECODE_BOUNDS.magnitude
        );
        assert!(
            description.contains(&format!("budget {}", DECODE_BOUNDS.max_elements)),
            "DECODE_BOUNDS.max_elements is {} but the generator description says \
             {description:?}",
            DECODE_BOUNDS.max_elements
        );
    }

    use super::*;
    use std::collections::HashSet;

    fn decode(bytes: &[u8]) -> TensorOp {
        TensorOp::arbitrary(&mut Unstructured::new(bytes)).expect("decoding never fails")
    }

    /// Pseudo-random byte strings, to stand in for what a fuzzer produces.
    fn byte_strings(count: usize) -> Vec<Vec<u8>> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..count)
            .map(|i| {
                (0..(8 + i % 200))
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        (state >> 24) as u8
                    })
                    .collect()
            })
            .collect()
    }

    /// **Every byte string must decode to a valid case.** The constructors assert their
    /// constraints, so merely decoding without panicking proves validity — the same
    /// correct-by-construction guarantee the seeded generator makes, from a different
    /// source of bytes.
    #[test]
    fn any_bytes_decode_to_a_valid_case() {
        for bytes in byte_strings(2_000) {
            decode(&bytes);
        }
    }

    /// Including degenerate inputs. Rejecting these would cost the fuzzer executions and
    /// teach it nothing.
    #[test]
    fn short_and_empty_inputs_still_decode() {
        for bytes in [vec![], vec![0], vec![255], vec![7, 3], vec![0; 3]] {
            decode(&bytes);
        }
    }

    /// **The property that makes coverage guidance meaningful.** Mutating a byte late in
    /// the input must perturb a *value* while leaving the operation and shape intact — so
    /// the fuzzer can explore around an interesting input rather than being thrown to an
    /// unrelated one.
    #[test]
    fn a_late_mutation_changes_values_but_not_structure() {
        let mut checked = 0;

        for bytes in byte_strings(500) {
            if bytes.len() < 24 {
                continue;
            }
            let original = decode(&bytes);

            let mut mutated_bytes = bytes.clone();
            let last = mutated_bytes.len() - 1;
            mutated_bytes[last] = mutated_bytes[last].wrapping_add(1);
            let mutated = decode(&mutated_bytes);

            assert_eq!(
                original.name(),
                mutated.name(),
                "a one-byte change altered the operation"
            );
            assert_eq!(
                original.rank(),
                mutated.rank(),
                "a one-byte change altered the rank"
            );
            checked += 1;
        }

        assert!(checked > 100, "only checked {checked} cases");
    }

    /// Decoding is a pure function of the bytes, which is what lets a saved corpus entry
    /// or a crashing input be replayed exactly.
    #[test]
    fn decoding_is_deterministic() {
        for bytes in byte_strings(200) {
            assert_eq!(decode(&bytes), decode(&bytes));
        }
    }

    /// Every operation must be reachable, or some would simply never be fuzzed and
    /// nothing would say so.
    #[test]
    fn every_operation_is_reachable() {
        let mut seen: HashSet<&str> = HashSet::new();
        for bytes in byte_strings(2_000) {
            seen.insert(decode(&bytes).name());
        }

        // **Asserted exhaustively, not just inclusively.** This test previously listed the
        // operations it wanted and checked each was present, so adding `softmax` to the
        // decoder left it passing while saying nothing about the new operation. The
        // generator's equivalent asserts the *count* matches and caught the same omission
        // immediately — so this one now does the same.
        let expected = [
            "add", "sub", "mul", "div", "neg", "abs", "exp", "sqrt", "sum", "matmul", "softmax",
        ];
        for name in expected {
            assert!(seen.contains(name), "{name} was never decoded");
        }
        assert_eq!(
            seen.len(),
            expected.len(),
            "an operation is decodable but unlisted: {seen:?}"
        );
    }

    /// Every supported rank too, since rank-specific paths are where shape-handling bugs
    /// live.
    #[test]
    fn every_rank_is_reachable() {
        let ranks: HashSet<usize> = byte_strings(2_000)
            .iter()
            .map(|bytes| decode(bytes).rank())
            .collect();

        for rank in 1..=crate::backends::MAX_RANK {
            assert!(ranks.contains(&rank), "rank {rank} was never decoded");
        }
    }

    /// Domain restrictions must hold **as configured**. Asserting a fixed expectation
    /// would pass whether or not the decoder actually consulted its setting — which it
    /// previously did not, so flipping `restrict_domains` had no effect and nothing said
    /// so.
    #[test]
    fn decoded_cases_respect_the_configured_domains() {
        if !DECODE_BOUNDS.restrict_domains {
            // Unrestricted: undefined results are the point, so there is nothing to
            // assert beyond validity, which other tests already cover.
            return;
        }

        for bytes in byte_strings(2_000) {
            match decode(&bytes) {
                TensorOp::Unary {
                    kind: UnaryOp::Sqrt,
                    arg,
                } => assert!(arg.data().iter().all(|v| *v >= 0.0), "sqrt got a negative"),
                TensorOp::Binary {
                    kind: BinaryOp::Div,
                    rhs,
                    ..
                } => assert!(
                    rhs.data().iter().all(|v| v.abs() >= DIVISOR_FLOOR),
                    "div got a divisor at or near zero"
                ),
                _ => {}
            }
        }
    }

    /// With restrictions lifted, the undefined region must actually be reached —
    /// otherwise the setting would be nominal.
    #[test]
    fn unrestricted_decoding_reaches_the_undefined_region() {
        if DECODE_BOUNDS.restrict_domains {
            return;
        }

        let mut saw_negative_sqrt = false;
        let mut saw_zero_divisor = false;

        for bytes in byte_strings(3_000) {
            match decode(&bytes) {
                TensorOp::Unary {
                    kind: UnaryOp::Sqrt,
                    arg,
                } => saw_negative_sqrt |= arg.data().iter().any(|v| *v < 0.0),
                TensorOp::Binary {
                    kind: BinaryOp::Div,
                    rhs,
                    ..
                } => saw_zero_divisor |= rhs.data().contains(&0.0),
                _ => {}
            }
        }

        assert!(saw_negative_sqrt, "sqrt never received a negative");
        assert!(saw_zero_divisor, "div never received a zero divisor");
    }

    /// The deliberately interesting values must actually appear, since a fuzzer mapping
    /// bytes onto a plain numeric range would never produce exactly zero or one.
    #[test]
    fn special_values_are_reachable() {
        let mut seen_zero = false;
        let mut seen_one = false;

        for bytes in byte_strings(2_000) {
            let case = decode(&bytes);
            let TensorOp::Unary { arg, .. } = &case else {
                continue;
            };
            seen_zero |= arg.data().contains(&0.0);
            seen_one |= arg.data().contains(&1.0);
        }

        assert!(seen_zero, "zero was never decoded");
        assert!(seen_one, "one was never decoded");
    }
}
