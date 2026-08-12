//! Producing valid cases from a seed.
//!
//! **This is the N1 skeleton, not the real generator.** PHASE-N3 builds that: shape-then-
//! value generation, adversarial values at a controlled rate, a `GenerationAxes`
//! description, and the corpus-shape measurements that say what a campaign actually
//! covered. What is here is the minimum needed to prove the loop is deterministic and that
//! the seams connect.
//!
//! Two properties are real from the start, though, because retrofitting either is painful:
//!
//! **Correct-by-construction.** Every case this produces satisfies `validate`. Generating
//! something invalid and filtering it later tests the validator, not the operator — and a
//! model that is invalid *and* crashes a runtime is our bug, not theirs. A test asserts
//! this over thousands of seeds.
//!
//! **Determinism.** All randomness comes from the engine's `SeededRng`, which is passed in.
//! A second source of randomness would mean a finding that cannot be replayed from its
//! seed, and a divergence that cannot be reproduced is a bug in *this tool* rather than a
//! discovery about anything else.

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Generator;
use rand::RngExt;

use crate::case::{OnnxCase, OpKind, TensorValue};
use crate::model::DEFAULT_OPSET;
use crate::validation::input_name;

/// The trivial N1 generator.
#[derive(Debug, Clone)]
pub struct OnnxGenerator {
    /// Which operators may be produced.
    pub operators: Vec<OpKind>,
    /// Inclusive bounds on rank.
    pub max_rank: usize,
    /// Inclusive bound on any single dimension.
    pub max_dim: i64,
    /// Fraction of elements drawn from the special-value pool, 0.0 to 1.0.
    ///
    /// Uniform sampling essentially never produces `0.0`, `±inf`, `NaN`, a subnormal or
    /// `f32::MAX`, and **both of this project's real findings were special-value bugs**. So
    /// they are injected deliberately. The *rate* is an axis whose effect on yield is a
    /// measurement for N3, not a guess — this default is a placeholder, not a finding.
    pub special_value_rate: f64,
    pub opset: i64,
}

impl Default for OnnxGenerator {
    fn default() -> Self {
        Self {
            operators: OpKind::ALL.to_vec(),
            max_rank: 3,
            max_dim: 4,
            special_value_rate: 0.25,
            opset: DEFAULT_OPSET,
        }
    }
}

/// The values worth injecting deliberately.
///
/// Not exhaustive, and not yet justified by measurement — N3 decides the pool and the rate
/// on evidence. Every entry is here because it has broken something somewhere: overflow to
/// infinity, the sign of zero, the subnormal boundary, and the largest finite magnitudes.
const SPECIAL_VALUES: [f32; 10] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    f32::MIN_POSITIVE, // smallest normal; the subnormal boundary
    f32::MAX,
    f32::MIN,
];

impl OnnxGenerator {
    /// A configuration that produces no special values — the control for measuring what
    /// they buy. A rate without a baseline is not a measurement.
    pub fn without_special_values() -> Self {
        Self {
            special_value_rate: 0.0,
            ..Self::default()
        }
    }

    /// A one-line description of the configuration.
    ///
    /// Recorded in every finding: without it a seed is unusable, because a seed identifies
    /// a case only in combination with the configuration that produced it. A reader with
    /// the log alone would otherwise have to be told the bounds out of band, which in
    /// practice means guessing.
    pub fn describe(&self) -> String {
        let operators: Vec<&str> = self.operators.iter().map(|o| o.onnx_name()).collect();
        format!(
            "ops=[{}] max_rank={} max_dim={} special_rate={:.2} opset={}",
            operators.join(","),
            self.max_rank,
            self.max_dim,
            self.special_value_rate,
            self.opset
        )
    }

    fn value(&self, rng: &mut SeededRng) -> f32 {
        if rng.random_bool(self.special_value_rate) {
            SPECIAL_VALUES[rng.random_range(0..SPECIAL_VALUES.len())]
        } else {
            // A modest range: large enough to be arithmetic, small enough that `Mul` does
            // not overflow to infinity on every case and drown the special-value signal in
            // ordinary overflow.
            rng.random_range(-100.0..100.0)
        }
    }
}

impl Generator for OnnxGenerator {
    type In = OnnxCase;

    fn generate(&self, rng: &mut SeededRng) -> OnnxCase {
        let op = self.operators[rng.random_range(0..self.operators.len())];

        // Shape first, then values — the analogue of the SQL domain's state-then-query
        // split. Deciding the world before the numbers keeps constraint logic in one place
        // and value strategy in another, and lets the special-value rate vary independently
        // of shape.
        //
        // Rank 0 is included: a rank-0 tensor is a legal ONNX scalar, and degenerate shapes
        // are exactly where implementations differ.
        let rank = rng.random_range(0..=self.max_rank);
        let dims: Vec<i64> = (0..rank)
            .map(|_| rng.random_range(0..=self.max_dim))
            .collect();
        let count = dims.iter().product::<i64>().max(0) as usize;

        // Every input shares the shape. Broadcasting is an N3 decision with its own shape
        // rule and its own tests, not something to slip in here.
        let inputs = (0..op.arity())
            .map(|index| {
                let values: Vec<f32> = (0..count).map(|_| self.value(rng)).collect();
                TensorValue::f32(&input_name(index), dims.clone(), values)
            })
            .collect();

        OnnxCase::new(op, self.opset, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::{is_valid, validate};

    /// Correct-by-construction, over enough seeds that a rare path is reached.
    ///
    /// Not a formality: a generator that occasionally emits an invalid case produces
    /// divergences that are *ours*, and both prior domains lost hours to exactly that —
    /// one SQL sweep produced 825 findings from its own invalid queries.
    #[test]
    fn every_generated_case_is_valid() {
        let generator = OnnxGenerator::default();
        for seed in 0..5_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let problems = validate(&case);
            assert!(
                problems.is_empty(),
                "seed {seed} produced an invalid case: {problems:?}\n{case:?}"
            );
        }
    }

    /// The property every finding depends on.
    #[test]
    fn the_same_seed_produces_the_same_case() {
        let generator = OnnxGenerator::default();
        for seed in [0, 1, 42, 9_999, u64::MAX] {
            let first = generator.generate(&mut SeededRng::from_seed(seed));
            let second = generator.generate(&mut SeededRng::from_seed(seed));
            // Compared through serialization so the check covers the *stored* form too —
            // `PartialEq` on `f32` would make two NaN cases unequal.
            assert_eq!(
                serde_json::to_string(&first).unwrap(),
                serde_json::to_string(&second).unwrap(),
                "seed {seed} was not reproducible"
            );
        }
    }

    /// Different seeds must actually explore. A generator that ignored its rng would pass
    /// the determinism test perfectly.
    #[test]
    fn different_seeds_produce_different_cases() {
        let generator = OnnxGenerator::default();
        let cases: Vec<String> = (0..200)
            .map(|seed| {
                serde_json::to_string(&generator.generate(&mut SeededRng::from_seed(seed))).unwrap()
            })
            .collect();
        let mut unique = cases.clone();
        unique.sort();
        unique.dedup();
        assert!(
            unique.len() > 150,
            "only {} distinct cases from 200 seeds — the generator is barely using its rng",
            unique.len()
        );
    }

    /// Special values must actually appear. A pool that is never drawn from is a feature
    /// nothing tests, and the corpus would look healthy while covering none of the thesis.
    #[test]
    fn special_values_actually_reach_the_corpus() {
        let generator = OnnxGenerator::default();
        let mut seen_nan = false;
        let mut seen_inf = false;
        let mut seen_negative_zero = false;

        for seed in 0..2_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                for value in &input.values {
                    seen_nan |= value.is_nan();
                    seen_inf |= value.is_infinite();
                    seen_negative_zero |= value.to_bits() == (-0.0f32).to_bits();
                }
            }
        }
        assert!(seen_nan, "no NaN in 2,000 cases");
        assert!(seen_inf, "no infinity in 2,000 cases");
        assert!(seen_negative_zero, "no negative zero in 2,000 cases");
    }

    /// The baseline configuration must genuinely produce none, or it is not a control.
    #[test]
    fn the_baseline_configuration_produces_no_special_values() {
        let generator = OnnxGenerator::without_special_values();
        for seed in 0..2_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                for value in &input.values {
                    assert!(
                        value.is_finite(),
                        "seed {seed} produced {value} with the special-value rate at zero"
                    );
                }
            }
        }
    }

    /// Every operator must be reachable. An operator in the list that the generator never
    /// emits is an operator nothing tests, and the axis table would still count it.
    #[test]
    fn every_configured_operator_is_reachable() {
        let generator = OnnxGenerator::default();
        let mut seen: Vec<OpKind> = Vec::new();
        for seed in 0..1_000 {
            let op = generator.generate(&mut SeededRng::from_seed(seed)).op;
            if !seen.contains(&op) {
                seen.push(op);
            }
        }
        for op in &generator.operators {
            assert!(
                seen.contains(op),
                "{op:?} was never generated in 1,000 seeds"
            );
        }
    }

    /// Degenerate shapes — rank 0 and zero-length dimensions — must be reachable. They are
    /// where implementations differ, and a generator that never produced them would report
    /// a confident zero over a surface it never touched.
    #[test]
    fn degenerate_shapes_are_reachable() {
        let generator = OnnxGenerator::default();
        let mut seen_scalar = false;
        let mut seen_empty = false;

        for seed in 0..1_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let dims = &case.inputs[0].dims;
            seen_scalar |= dims.is_empty();
            seen_empty |= dims.contains(&0);
        }
        assert!(seen_scalar, "no rank-0 scalar in 1,000 cases");
        assert!(seen_empty, "no zero-length dimension in 1,000 cases");
    }

    #[test]
    fn the_description_records_the_configuration() {
        let description = OnnxGenerator::default().describe();
        for fragment in ["ops=[", "max_rank=", "max_dim=", "special_rate=", "opset="] {
            assert!(
                description.contains(fragment),
                "{fragment} missing from {description}"
            );
        }
    }

    /// A case with a shape whose element count is zero still validates — the empty tensor
    /// is legal, and `is_valid` must not accidentally require data.
    #[test]
    fn empty_tensors_are_valid() {
        let generator = OnnxGenerator::default();
        let empty: Vec<OnnxCase> = (0..1_000)
            .map(|s| generator.generate(&mut SeededRng::from_seed(s)))
            .filter(|c| c.total_elements() == 0)
            .collect();
        assert!(!empty.is_empty(), "no empty-tensor cases were produced");
        for case in empty {
            assert!(is_valid(&case));
        }
    }
}
