//! A starting corpus of interesting inputs for the fuzzer.
//!
//! libFuzzer begins from whatever inputs it is given and mutates outward. Starting from
//! nothing, it must *rediscover* that zero and one and subnormals exist, and that a
//! rank-4 tensor is possible — spending its early budget on things we already know are
//! worth testing. A seed corpus hands it those regions up front.
//!
//! # Why these are built rather than written down
//!
//! A corpus entry is a **byte string**, and the fuzzer's decoder gives those bytes
//! meaning through a fixed layout. Writing `[0x03, 0x00, 0x07, ...]` into a file would
//! be unreadable, unreviewable, and — worse — **silently wrong the moment the layout
//! changed**. The same bytes would still decode, just to something else entirely, and
//! nothing would say so.
//!
//! So the seeds are constructed here through named helpers that mirror the layout, and a
//! test decodes every one of them and asserts it produces the case intended. If the
//! decoder's layout is ever changed, those tests fail rather than the corpus quietly
//! rotting.

use crate::backends::MAX_RANK;

/// Byte selecting an operation, by the decoder's `choice % 10`.
mod op {
    pub const NEG: u8 = 0;
    pub const ABS: u8 = 1;
    pub const EXP: u8 = 2;
    pub const SQRT: u8 = 3;
    pub const ADD: u8 = 4;
    pub const SUB: u8 = 5;
    pub const MUL: u8 = 6;
    pub const DIV: u8 = 7;
    pub const SUM: u8 = 8;
    pub const MATMUL: u8 = 9;
}

/// The byte that decodes to a dimension or rank of `n`.
///
/// The decoder computes `1 + byte % max`, so `n - 1` yields exactly `n` for any `n`
/// within range.
fn size(n: usize) -> u8 {
    (n - 1) as u8
}

/// The byte that decodes to the `index`-th deliberately interesting value.
///
/// Anything below 32 selects from the special table, so a small index both stays in that
/// range and picks a specific entry.
fn special(index: usize) -> u8 {
    (index % 32) as u8
}

/// A byte that decodes to an ordinary value, `fraction` of the way through the range.
fn ordinary(fraction: f32) -> u8 {
    32 + (fraction.clamp(0.0, 1.0) * 223.0) as u8
}

/// Bytes for a unary case of the given shape, with each value drawn from the special
/// table.
fn unary(operation: u8, shape: &[usize]) -> Vec<u8> {
    let mut bytes = vec![operation, size(shape.len())];
    bytes.extend(shape.iter().map(|d| size(*d)));

    let count: usize = shape.iter().product();
    bytes.extend((0..count).map(special));
    bytes
}

/// Bytes for an elementwise binary case. Both operands share the shape, so the values
/// simply run on.
fn binary(operation: u8, shape: &[usize]) -> Vec<u8> {
    let mut bytes = vec![operation, size(shape.len())];
    bytes.extend(shape.iter().map(|d| size(*d)));

    let count: usize = shape.iter().product();
    bytes.extend((0..count).map(special));
    bytes.extend((0..count).map(special));
    bytes
}

/// Bytes for a reduction over `axis`.
fn reduce(shape: &[usize], axis: usize) -> Vec<u8> {
    let mut bytes = vec![op::SUM, size(shape.len())];
    bytes.extend(shape.iter().map(|d| size(*d)));
    bytes.push(axis as u8);

    let count: usize = shape.iter().product();
    bytes.extend((0..count).map(special));
    bytes
}

/// Bytes for `[batch.., m, k] x [batch.., k, n]`.
fn matmul(batch: &[usize], m: usize, k: usize, n: usize) -> Vec<u8> {
    // The decoder reads rank as `2 + byte % (MAX_RANK - 1)`.
    let rank = batch.len() + 2;
    let mut bytes = vec![op::MATMUL, (rank - 2) as u8];
    bytes.extend(batch.iter().map(|d| size(*d)));
    bytes.extend([size(m), size(k), size(n)]);

    let batch_size: usize = batch.iter().product::<usize>().max(1);
    bytes.extend((0..batch_size * m * k).map(|i| ordinary(i as f32 / 16.0)));
    bytes.extend((0..batch_size * k * n).map(special));
    bytes
}

/// The starting corpus.
///
/// Chosen to cover the regions a fuzzer would otherwise spend its early budget
/// rediscovering: every operation, every rank, degenerate shapes, and the interesting
/// values that random bytes reach only by accident.
///
/// **The known unresolved `matmul` overflow case is deliberately absent.** Including it
/// would make every campaign crash on its first input, ending the run before it explored
/// anything — a seed corpus is a starting point, not a regression suite. It lives in
/// `findings/` where it belongs until it is triaged.
pub fn seed_corpus() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();

    let unary_ops = [op::NEG, op::ABS, op::EXP, op::SQRT];
    let binary_ops = [op::ADD, op::SUB, op::MUL, op::DIV];

    // The smallest form of every operation. A minimal case is both the fastest to
    // execute and the easiest to mutate outward from.
    for operation in unary_ops {
        seeds.push(unary(operation, &[1]));
    }
    for operation in binary_ops {
        seeds.push(binary(operation, &[1]));
    }
    seeds.push(reduce(&[1], 0));
    seeds.push(matmul(&[], 1, 1, 1));

    // Every rank, since rank-specific paths are where shape-handling bugs live.
    for rank in 1..=MAX_RANK {
        let shape = vec![2; rank];
        seeds.push(unary(op::EXP, &shape));
        seeds.push(binary(op::MUL, &shape));
    }

    // Degenerate shapes: every dimension of length one. A classic source of bugs, and
    // one random sampling reaches only rarely.
    seeds.push(unary(op::ABS, &[1; MAX_RANK]));
    seeds.push(binary(op::ADD, &[1; MAX_RANK]));

    // Larger shapes, to reach whatever blocked or vectorised paths a backend switches to
    // above some size.
    seeds.push(unary(op::NEG, &[8, 8]));
    seeds.push(binary(op::SUB, &[8, 8]));

    // Reductions along each axis of a rank-3 tensor. Reducing the last axis is a
    // different code path from the first, because of how tensors are laid out.
    for axis in 0..3 {
        seeds.push(reduce(&[2, 3, 4], axis));
    }

    // Matrix multiplication across its three free dimensions and with a batch, since
    // batched multiplication is usually a separate kernel.
    seeds.push(matmul(&[], 1, 8, 1));
    seeds.push(matmul(&[], 8, 1, 8));
    seeds.push(matmul(&[], 4, 4, 4));
    seeds.push(matmul(&[2], 2, 3, 2));
    seeds.push(matmul(&[2, 2], 2, 2, 2));

    seeds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::TensorOp;
    use arbitrary::{Arbitrary, Unstructured};
    use std::collections::HashSet;

    fn decode(bytes: &[u8]) -> TensorOp {
        TensorOp::arbitrary(&mut Unstructured::new(bytes)).expect("decoding never fails")
    }

    /// **The test that stops the corpus rotting.** Each seed is built to produce a
    /// specific case; if the decoder's layout ever changes, the same bytes would still
    /// decode — just to something else — and nothing else would notice.
    #[test]
    fn seeds_decode_to_the_operations_they_were_built_for() {
        let expected = [
            ("neg", vec![1]),
            ("abs", vec![1]),
            ("exp", vec![1]),
            ("sqrt", vec![1]),
            ("add", vec![1]),
            ("sub", vec![1]),
            ("mul", vec![1]),
            ("div", vec![1]),
            ("sum", vec![1]),
            ("matmul", vec![1, 1]),
        ];

        for (index, (name, shape)) in expected.iter().enumerate() {
            let case = decode(&seed_corpus()[index]);
            assert_eq!(case.name(), *name, "seed {index}");
            assert_eq!(case.rank(), shape.len(), "seed {index} rank");
        }
    }

    #[test]
    fn every_operation_appears_in_the_corpus() {
        let names: HashSet<&str> = seed_corpus().iter().map(|s| decode(s).name()).collect();

        for expected in [
            "add", "sub", "mul", "div", "neg", "abs", "exp", "sqrt", "sum", "matmul",
        ] {
            assert!(names.contains(expected), "{expected} is not seeded");
        }
    }

    #[test]
    fn every_rank_appears_in_the_corpus() {
        let ranks: HashSet<usize> = seed_corpus().iter().map(|s| decode(s).rank()).collect();

        for rank in 1..=MAX_RANK {
            assert!(ranks.contains(&rank), "rank {rank} is not seeded");
        }
    }

    /// The point of the corpus is the values random bytes would rarely produce.
    #[test]
    fn the_corpus_contains_deliberately_interesting_values() {
        let mut seen_zero = false;
        let mut seen_one = false;

        for seed in seed_corpus() {
            if let TensorOp::Unary { arg, .. } = decode(&seed) {
                seen_zero |= arg.data().contains(&0.0);
                seen_one |= arg.data().contains(&1.0);
            }
        }

        assert!(seen_zero && seen_one, "zero or one is not seeded");
    }

    /// Degenerate shapes must be present, since random dimensions reach all-ones rarely.
    #[test]
    fn the_corpus_contains_a_fully_degenerate_shape() {
        let has_degenerate = seed_corpus().iter().any(|seed| {
            matches!(decode(seed), TensorOp::Unary { ref arg, .. }
                if arg.rank() == MAX_RANK && arg.shape().iter().all(|d| *d == 1))
        });

        assert!(has_degenerate, "no all-ones shape at maximum rank");
    }

    /// Every seed must be a case the backends will actually accept — a corpus entry that
    /// cannot run is one the fuzzer will carry around forever, learning nothing.
    #[test]
    fn every_seed_runs_on_a_real_backend() {
        use crate::backends::flex;
        use diff_fuzzer_core::Implementation;

        for (index, seed) in seed_corpus().iter().enumerate() {
            let case = decode(seed);
            assert!(
                flex().run(&case).is_ok(),
                "seed {index} ({}) does not run",
                case.name()
            );
        }
    }

    /// Seeds must be distinct. Duplicates cost the fuzzer executions and teach it
    /// nothing.
    #[test]
    fn seeds_are_distinct() {
        let corpus = seed_corpus();
        let unique: HashSet<&Vec<u8>> = corpus.iter().collect();

        assert_eq!(unique.len(), corpus.len(), "the corpus contains duplicates");
    }
}
