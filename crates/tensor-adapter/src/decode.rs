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

use crate::input::{BinaryOp, ReduceOp, TensorOp, TensorValue, UnaryOp};
use crate::ops::{Bounds, DIVISOR_FLOOR, Domain, SPECIAL_VALUES};
use arbitrary::{Arbitrary, Result, Unstructured};

/// Bounds used when decoding. Fixed rather than configurable, because the fuzzer's
/// corpus is only meaningful for the layout that produced it: changing these would
/// silently reinterpret every saved input.
const DECODE_BOUNDS: Bounds = Bounds {
    max_rank: crate::backends::MAX_RANK,
    max_dim: 8,
    magnitude: 10.0,
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
fn shape(u: &mut Unstructured<'_>, rank: usize) -> Vec<usize> {
    (0..rank).map(|_| size(u, DECODE_BOUNDS.max_dim)).collect()
}

impl<'a> Arbitrary<'a> for TensorOp {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Byte 0 chooses the operation. Weighted by class the same way the seeded
        // generator is, so no operation is starved.
        let choice = byte(u) as usize % 10;

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
                // One shape, used twice — the constraint cannot be violated because
                // there is only one shape to violate it with.
                let shape = shape(u, rank);
                let right_domain = binary_right_domain(kind);
                TensorOp::binary(
                    kind,
                    tensor(u, shape.clone(), Domain::Any),
                    tensor(u, shape, right_domain),
                )
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

        for expected in [
            "add", "sub", "mul", "div", "neg", "abs", "exp", "sqrt", "sum", "matmul",
        ] {
            assert!(seen.contains(expected), "{expected} was never decoded");
        }
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
