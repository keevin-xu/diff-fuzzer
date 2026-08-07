//! Producing tensor test cases.
//!
//! [`TensorOpGenerator`] is the real one: it picks an operation and then builds
//! arguments satisfying that operation's rules, entirely from a seed.
//!
//! [`FixedAddGenerator`] is kept alongside it, always returning the same small case.
//! Varied input is what finds bugs, but a fixed case is what makes a failing test
//! readable — so the tests that check the detector itself use the fixed one, and only
//! the search for real divergences uses the varied one.

use crate::input::{BinaryOp, TensorOp, TensorValue};
use crate::ops::{Bounds, activation, binary, matmul, reduce, unary};
use diff_fuzzer_core::{Generator, SeededRng};
use rand::RngExt;

/// The four kinds of case, and how many operations each covers.
///
/// The counts are what make the choice fair *per operation* rather than per class.
/// Choosing uniformly between the four classes would hand `matmul` — one operation —
/// as many cases as all four elementwise binary operations put together, so a quarter
/// of the entire budget would go to one kernel while `sub` got a sixteenth.
const CLASS_WEIGHTS: [(Class, usize); 5] = [
    (Class::Unary, unary::ALL.len()),
    (Class::Binary, binary::ALL.len()),
    (Class::Reduce, reduce::ALL.len()),
    (Class::Matmul, 1),
    (Class::Activation, activation::ALL.len()),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Unary,
    Binary,
    Reduce,
    Matmul,
    Activation,
}

/// Builds a random valid tensor case from a seed.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorOpGenerator {
    pub bounds: Bounds,
}

impl TensorOpGenerator {
    pub fn new(bounds: Bounds) -> Self {
        Self { bounds }
    }

    /// Pick a class, weighted so that every *operation* is equally likely.
    ///
    /// Walks the weights subtracting as it goes: a number is drawn from the total, and
    /// whichever class's share that number falls inside is the one chosen.
    fn choose_class(rng: &mut SeededRng) -> Class {
        let total: usize = CLASS_WEIGHTS.iter().map(|(_, weight)| weight).sum();
        let mut pick = rng.random_range(0..total);

        for (class, weight) in CLASS_WEIGHTS {
            if pick < weight {
                return class;
            }
            pick -= weight;
        }

        // Unreachable: `pick` starts below the total and the weights sum to it.
        unreachable!("weighted choice fell through, which the arithmetic forbids")
    }
}

impl Generator for TensorOpGenerator {
    type In = TensorOp;

    fn generate(&self, rng: &mut SeededRng) -> TensorOp {
        match Self::choose_class(rng) {
            Class::Unary => unary::generate(rng, &self.bounds),
            Class::Binary => binary::generate(rng, &self.bounds),
            Class::Reduce => reduce::generate(rng, &self.bounds),
            Class::Matmul => matmul::generate(rng, &self.bounds),
            Class::Activation => activation::generate(rng, &self.bounds),
        }
    }
}

/// Always produces the same small elementwise `add`.
///
/// Values are chosen to be exactly representable in `f32` and easy to add in your
/// head, so that a wrong result is obvious rather than something to be squinted at:
///
/// ```text
///   [1 2]     [10 20]     [11 22]
///   [3 4]  +  [30 40]  =  [33 44]
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct FixedAddGenerator;

impl Generator for FixedAddGenerator {
    type In = TensorOp;

    /// The `_rng` parameter is unused, hence the leading underscore — without it the
    /// compiler warns about an unused variable. The parameter stays in the signature
    /// because it is part of the trait's contract.
    fn generate(&self, _rng: &mut SeededRng) -> TensorOp {
        TensorOp::binary(
            BinaryOp::Add,
            TensorValue::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            TensorValue::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn cases(count: u64) -> Vec<TensorOp> {
        let generator = TensorOpGenerator::default();
        (0..count)
            .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
            .collect()
    }

    /// Every operation must be reachable. One that is never generated is one that is
    /// never tested, and nothing else in the suite would notice.
    #[test]
    fn every_operation_gets_generated() {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for case in cases(2000) {
            *counts.entry(case.name()).or_default() += 1;
        }

        let expected = [
            "add", "sub", "mul", "div", "neg", "abs", "exp", "sqrt", "sum", "matmul", "softmax",
        ];
        for name in expected {
            assert!(counts.contains_key(name), "{name} was never generated");
        }
        assert_eq!(counts.len(), expected.len(), "unexpected: {counts:?}");
    }

    /// Weighting by operation count is meant to give each operation a roughly equal
    /// share. This checks the intent rather than exact proportions: no operation should
    /// be starved, which is what would happen if classes were picked uniformly.
    #[test]
    fn no_operation_is_starved() {
        let total = 2000;
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for case in cases(total) {
            *counts.entry(case.name()).or_default() += 1;
        }

        // A tenth of the budget each would be exactly even; require at least half that.
        let floor = total as usize / 10 / 2;
        for (name, count) in &counts {
            assert!(*count >= floor, "{name} got only {count} of {total}");
        }
    }

    /// Every rank the backends support must be produced, since rank-specific paths are
    /// where shape-handling bugs live.
    #[test]
    fn every_supported_rank_gets_generated() {
        let ranks: std::collections::HashSet<usize> =
            cases(2000).iter().map(|case| case.rank()).collect();
        for rank in 1..=crate::backends::MAX_RANK {
            assert!(ranks.contains(&rank), "rank {rank} was never generated");
        }
    }

    /// The property every finding depends on: a seed determines its case exactly.
    #[test]
    fn the_same_seed_produces_the_same_case() {
        let generator = TensorOpGenerator::default();
        let from = |seed| generator.generate(&mut SeededRng::from_seed(seed));
        assert_eq!(from(1234), from(1234));
    }

    /// Determinism over a *sequence*, which is the form that actually matters.
    ///
    /// The test above draws one case from a fresh generator. A campaign instead draws
    /// thousands from a single advancing generator, so what must be reproducible is the
    /// whole stream: case 900 of a run has to be the same case every time, or replaying
    /// a run to reach a finding would not work.
    #[test]
    fn one_seed_produces_the_same_sequence_of_cases() {
        let sequence = |seed| {
            let generator = TensorOpGenerator::default();
            let mut rng = SeededRng::from_seed(seed);
            (0..200)
                .map(|_| generator.generate(&mut rng))
                .collect::<Vec<_>>()
        };

        assert_eq!(sequence(9), sequence(9));
        assert_ne!(sequence(9), sequence(10));
    }

    #[test]
    fn different_seeds_produce_different_cases() {
        let generator = TensorOpGenerator::default();
        let from = |seed| generator.generate(&mut SeededRng::from_seed(seed));
        // Not every pair need differ, but a whole run of identical cases would mean the
        // seed is not actually reaching the generator.
        let distinct = (0..50).map(from).collect::<Vec<_>>();
        assert!(distinct.windows(2).any(|w| w[0] != w[1]));
    }

    /// Bounds must actually bound. Narrowing them should visibly narrow the output —
    /// otherwise the knob is decorative.
    #[test]
    fn bounds_are_respected() {
        let generator = TensorOpGenerator::new(Bounds {
            max_rank: 2,
            max_dim: 3,
            magnitude: 1.0,
            ..Bounds::default()
        });
        for seed in 0..500 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            assert!(case.rank() <= 2, "{case:?}");
        }
    }

    #[test]
    fn the_fixed_generator_produces_the_documented_case() {
        let case = FixedAddGenerator.generate(&mut SeededRng::from_seed(0));

        assert_eq!(case.name(), "add");
        assert_eq!(case.rank(), 2);

        let TensorOp::Binary { lhs, rhs, .. } = &case else {
            panic!("expected a binary operation, got {case:?}");
        };
        assert_eq!(lhs.shape(), &[2, 2]);
        assert_eq!(lhs.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rhs.data(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn the_fixed_generator_ignores_its_seed() {
        let from = |seed| FixedAddGenerator.generate(&mut SeededRng::from_seed(seed));
        assert_eq!(from(1), from(2));
    }
}
