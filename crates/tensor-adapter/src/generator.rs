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
use crate::ops::{Bounds, activation, binary, conv, matmul, reduce, scan, unary};
use diff_fuzzer_core::{Generator, SeededRng};
use rand::RngExt;

/// The four kinds of case, and how many operations each covers.
///
/// The counts are what make the choice fair *per operation* rather than per class.
/// Choosing uniformly between the four classes would hand `matmul` — one operation —
/// as many cases as all four elementwise binary operations put together, so a quarter
/// of the entire budget would go to one kernel while `sub` got a sixteenth.
const CLASS_WEIGHTS: [(Class, usize); 7] = [
    (Class::Unary, unary::ALL.len()),
    (Class::Binary, binary::ALL.len()),
    (Class::Reduce, reduce::ALL.len()),
    (Class::Matmul, 1),
    (Class::Activation, activation::ALL.len()),
    (Class::Scan, scan::ALL.len()),
    // One operation, but it selects among five backend algorithms — so it is weighted like
    // five, matching how the other classes are weighted by what they actually reach.
    (Class::Conv2d, conv::ALL_PROFILES.len()),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    Unary,
    Binary,
    Reduce,
    Matmul,
    Activation,
    Scan,
    Conv2d,
}

impl Class {
    /// Whether this class is enabled by a configuration.
    ///
    /// `Reduce` splits across two axes: `sum`/`mean` accumulate and are numerically
    /// interesting, while `max`/`min` select an input unchanged and are the pair most worth
    /// switching off once their disagreement is understood.
    fn enabled_by(self, bounds: &Bounds) -> bool {
        match self {
            Class::Unary => bounds.unary_ops,
            Class::Binary => bounds.binary_ops,
            Class::Reduce => bounds.accumulating_reductions || bounds.selecting_reductions,
            Class::Matmul => bounds.matmul,
            Class::Activation => bounds.activations,
            Class::Scan => bounds.scans,
            // **A convolution is always rank 4, so it cannot honour a lower `max_rank`.**
            // Excluding it is the honest reading of a bound that says "highest rank to
            // generate" — the alternative is emitting rank-4 cases under a configuration
            // that declared rank 2, which would make the bound decorative. The exclusion is
            // visible because `max_rank` is part of the configuration's derived identity.
            Class::Conv2d => bounds.convolution && bounds.max_rank >= 4,
        }
    }
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
    fn choose_class(rng: &mut SeededRng, bounds: &Bounds) -> Class {
        // **Disabled classes are removed before the draw, not filtered after it.**
        // Rejecting afterwards would spend the budget generating cases nobody wants, and
        // would make the weighting depend on how often a rejection happened.
        let enabled: Vec<(Class, usize)> = CLASS_WEIGHTS
            .into_iter()
            .filter(|(class, _)| class.enabled_by(bounds))
            .collect();

        assert!(
            !enabled.is_empty(),
            "every operation class is disabled; a generator with nothing to generate is a \
             configuration error, not an empty campaign"
        );

        let total: usize = enabled.iter().map(|(_, weight)| weight).sum();
        let mut pick = rng.random_range(0..total);

        for (class, weight) in enabled {
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
        match Self::choose_class(rng, &self.bounds) {
            Class::Unary => unary::generate(rng, &self.bounds),
            Class::Binary => binary::generate(rng, &self.bounds),
            Class::Reduce => reduce::generate(rng, &self.bounds),
            Class::Matmul => matmul::generate(rng, &self.bounds),
            Class::Activation => activation::generate(rng, &self.bounds),
            Class::Scan => scan::generate(rng, &self.bounds),
            Class::Conv2d => conv::generate(rng, &self.bounds),
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

    /// **A disabled axis produces nothing at all**, which is the whole point — a campaign
    /// narrows to stop a known disagreement crowding out the rest.
    #[test]
    fn a_disabled_operation_class_is_never_generated() {
        let bounds = Bounds::WITHOUT_SELECTING_REDUCTIONS;
        let generator = TensorOpGenerator::new(bounds);

        let mut seen: HashMap<&str, usize> = HashMap::new();
        for seed in 0..3_000u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            *seen.entry(case.name()).or_default() += 1;
        }

        assert!(!seen.contains_key("max"), "max was generated: {seen:?}");
        assert!(!seen.contains_key("min"), "min was generated: {seen:?}");
        // And the rest still are — narrowing must remove one thing, not everything.
        for still_there in ["sum", "mean", "softmax", "add", "matmul", "exp"] {
            assert!(
                seen.contains_key(still_there),
                "{still_there} vanished: {seen:?}"
            );
        }
    }

    /// **The narrowest preset still generates the operations it claims to.**
    #[test]
    fn the_numerically_interesting_preset_keeps_what_it_names() {
        let generator = TensorOpGenerator::new(Bounds::NUMERICALLY_INTERESTING);

        let mut seen: HashMap<&str, usize> = HashMap::new();
        for seed in 0..3_000u64 {
            seen.entry(generator.generate(&mut SeededRng::from_seed(seed)).name())
                .and_modify(|n| *n += 1)
                .or_insert(1);
        }

        for kept in ["softmax", "exp", "log", "sum", "mean"] {
            assert!(seen.contains_key(kept), "{kept} was excluded: {seen:?}");
        }
        for excluded in ["max", "min", "matmul", "add"] {
            assert!(
                !seen.contains_key(excluded),
                "{excluded} leaked in: {seen:?}"
            );
        }
    }

    /// **A configuration is identified by its axes**, so a narrowed run cannot be confused
    /// with a full one — which is what keeps their corpora and negatives apart.
    #[test]
    fn narrowing_the_operation_set_changes_the_configuration_identity() {
        use diff_fuzzer_core::GenerationAxes;

        let all = Bounds::ALL_OPERATIONS.description();
        let narrowed = Bounds::WITHOUT_SELECTING_REDUCTIONS.description();

        assert_ne!(all, narrowed);
        assert!(all.contains("selecting_reductions=on"), "{all}");
        assert!(narrowed.contains("selecting_reductions=off"), "{narrowed}");
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
            "add", "sub", "mul", "div", "neg", "abs", "exp", "sqrt", "log", "erf", "sum", "mean",
            "max", "min", "prod", "matmul", "softmax", "cumsum", "cumprod", "conv2d",
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

        // **The even share is derived from how many operations there are**, not from a
        // hardcoded fraction. This said "a tenth of the budget each", which was right when
        // there were ten operations and wrong the moment there were seventeen — a test that
        // fails on growth rather than on regression.
        let even_share = total as usize / counts.len();
        let floor = even_share / 2;
        for (name, count) in &counts {
            assert!(
                *count >= floor,
                "{name} got only {count} of {total}; an even share across {} operations \
                 would be {even_share}",
                counts.len()
            );
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
    ///
    /// **`conv2d` is the one operation with a fixed rank**, so a `max_rank` below 4 excludes
    /// it entirely rather than being ignored. This test is what says so: it was failing on
    /// generated rank-4 convolutions until `enabled_by` was taught the rule.
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
