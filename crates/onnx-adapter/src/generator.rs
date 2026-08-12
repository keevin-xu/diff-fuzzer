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

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Generator;
use rand::RngExt;

use crate::case::OnnxCase;
use crate::gen_shape::{self, Bounds};

/// Produces valid cases from a seed, according to a [`Bounds`] configuration.
#[derive(Debug, Clone, Default)]
pub struct OnnxGenerator {
    pub bounds: Bounds,
}

impl OnnxGenerator {
    pub fn new(bounds: Bounds) -> Self {
        Self { bounds }
    }

    /// The baseline control for measuring what special values buy.
    ///
    /// *A rate without a baseline is not a measurement* — a run at 0% is equally consistent
    /// with "the rules are wrong" and "this pair never disagrees".
    pub fn without_special_values() -> Self {
        Self::new(Bounds::default().without_special_values())
    }

    /// The configuration's identity, recorded with every finding.
    ///
    /// Without it a seed is unusable: a seed identifies a case only in combination with the
    /// configuration that produced it, so a reader with the log alone would have to be told
    /// the bounds out of band — which in practice means guessing.
    pub fn describe(&self) -> String {
        self.bounds.description()
    }
}

impl Generator for OnnxGenerator {
    type In = OnnxCase;

    /// Choose an operator, then an element type it accepts, then build.
    ///
    /// # Why the retry loop, and why it is bounded
    ///
    /// Operator and element type are drawn independently, and not every pair is buildable —
    /// `Sqrt` takes no integers, `And` takes only booleans. Rather than pre-computing the
    /// joint distribution, an unbuildable pair is redrawn.
    ///
    /// **The loop is bounded and its exhaustion is a panic, not a silent fallback.** A
    /// generator that quietly returned a default case on exhaustion would keep producing
    /// output while testing something other than what its configuration claims — which is
    /// precisely the failure `08-RISKS.md` §5 describes, where a confident zero was reported
    /// over a surface the generator could not reach. `Bounds::operators()` already excludes
    /// operators with no buildable type, so exhaustion means a real inconsistency.
    fn generate(&self, rng: &mut SeededRng) -> OnnxCase {
        let operators = self.bounds.operators();
        assert!(
            !operators.is_empty(),
            "this configuration admits no operators at all: {}",
            self.describe()
        );
        let types = self.bounds.element_types();

        for _ in 0..64 {
            let op = operators[rng.random_range(0..operators.len())];
            let elem = types[rng.random_range(0..types.len())];
            if let Some(case) = gen_shape::generate_case(op, elem, &self.bounds, rng) {
                return case;
            }
        }
        panic!(
            "could not build a case in 64 attempts under {} — every admitted operator should \
             have at least one buildable element type",
            self.describe()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{ElemType, OpKind};
    use crate::validation::validate;
    use std::collections::BTreeSet;

    /// Seeds spread across the whole `u64` range, not a tidy `0..n`.
    ///
    /// **N3.6 asks for this explicitly** and the reason is specific: a generator with a hidden
    /// dependence on seed magnitude — a modulo, a cast that truncates, a bound derived from the
    /// seed — passes a sequential run and fails in a campaign that uses a resumable range. The
    /// large values below are the ones a `0..n` test never reaches.
    fn wide_seeds(count: u64) -> impl Iterator<Item = u64> {
        // An odd stride coprime with 2^64, so the sequence walks the whole space rather than
        // clustering in one region.
        (0..count).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }

    /// **The validity stress test.** Every generated case must satisfy `validate`.
    ///
    /// A generator that occasionally emits an invalid case produces divergences that are
    /// *ours*: both prior domains lost hours to exactly that, and one SQL sweep produced 825
    /// findings from its own invalid queries.
    #[test]
    fn every_generated_case_is_valid_across_widely_separated_seeds() {
        let generator = OnnxGenerator::default();
        for seed in wide_seeds(5_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let problems = validate(&case);
            assert!(
                problems.is_empty(),
                "seed {seed} produced an invalid case: {problems:?}\n{case:?}"
            );
        }
    }

    /// The element budget must actually bind. A budget nothing exceeds is not a bound, and the
    /// knobs permit 4,096 elements against a budget of 256.
    #[test]
    fn no_generated_tensor_exceeds_the_element_budget() {
        let generator = OnnxGenerator::default();
        let budget = generator.bounds.element_budget;
        let mut largest = 0;

        for seed in wide_seeds(3_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                let count = input.element_count();
                largest = largest.max(count);
                assert!(
                    count <= budget,
                    "seed {seed}: a {count}-element tensor exceeds the {budget} budget"
                );
            }
        }
        // ...and it must be reachable, or the budget is as inert as the one this replaced.
        assert!(
            largest > budget / 4,
            "the largest tensor in 3,000 cases was {largest} against a {budget} budget — the \
             generator is not exercising anything near it"
        );
    }

    /// The property every finding depends on.
    #[test]
    fn the_same_seed_produces_the_same_case() {
        let generator = OnnxGenerator::default();
        for seed in wide_seeds(200) {
            let first = generator.generate(&mut SeededRng::from_seed(seed));
            let second = generator.generate(&mut SeededRng::from_seed(seed));
            // Compared through serialization so the check covers the *stored* form too:
            // `PartialEq` on `f32` makes two NaN cases unequal.
            assert_eq!(
                serde_json::to_string(&first).unwrap(),
                serde_json::to_string(&second).unwrap(),
                "seed {seed} was not reproducible"
            );
        }
    }

    /// A long run must be reproducible *at every position*, not only at its start.
    ///
    /// N3.10 asks for case *N* of a long run specifically. A generator that reset state, or
    /// drew from anything but the passed rng, would pass a first-case check and diverge later.
    #[test]
    fn a_long_run_is_reproducible_at_every_position() {
        let generator = OnnxGenerator::default();
        let sequence = |from: u64, count: u64| -> Vec<String> {
            (from..from + count)
                .map(|seed| {
                    serde_json::to_string(&generator.generate(&mut SeededRng::from_seed(seed)))
                        .unwrap()
                })
                .collect()
        };
        // The same window, reached twice — once as part of a longer run.
        assert_eq!(sequence(900, 100), sequence(900, 100));
        let long = sequence(0, 1_000);
        assert_eq!(long[900..], sequence(900, 100)[..]);
    }

    /// Different seeds must actually explore. A generator ignoring its rng would pass the
    /// determinism tests perfectly.
    #[test]
    fn different_seeds_produce_different_cases() {
        let generator = OnnxGenerator::default();
        let distinct: BTreeSet<String> = wide_seeds(300)
            .map(|seed| {
                serde_json::to_string(&generator.generate(&mut SeededRng::from_seed(seed))).unwrap()
            })
            .collect();
        assert!(
            distinct.len() > 250,
            "only {} distinct cases from 300 seeds",
            distinct.len()
        );
    }

    /// **Every admitted operator must actually be produced.**
    ///
    /// An operator the generator never emits is an operator nothing tests, while the axis
    /// table still counts it — `08-RISKS.md` §5, where a confident zero was reported over a
    /// surface the generator could not reach.
    #[test]
    fn every_admitted_operator_is_reachable() {
        let generator = OnnxGenerator::default();
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for seed in wide_seeds(4_000) {
            seen.insert(
                generator
                    .generate(&mut SeededRng::from_seed(seed))
                    .op
                    .onnx_name(),
            );
        }
        for op in generator.bounds.operators() {
            assert!(
                seen.contains(op.onnx_name()),
                "{op:?} is admitted but was never generated in 4,000 cases"
            );
        }
    }

    /// Every permitted element type must be produced, for the same reason.
    #[test]
    fn every_permitted_element_type_is_reachable() {
        let generator = OnnxGenerator::default();
        let mut seen: BTreeSet<ElemType> = BTreeSet::new();
        for seed in wide_seeds(4_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                seen.insert(input.elem_type());
            }
        }
        for elem in generator.bounds.element_types() {
            assert!(
                seen.contains(&elem),
                "{elem:?} is permitted but never generated"
            );
        }
    }

    /// Degenerate shapes — rank-0 scalars and zero-length dimensions — must be reachable when
    /// the axis is on. They are legal ONNX and are where implementations differ.
    #[test]
    fn degenerate_shapes_are_reachable_when_enabled() {
        let generator = OnnxGenerator::default();
        assert!(generator.bounds.degenerate_shapes);

        let mut scalar = false;
        let mut empty = false;
        for seed in wide_seeds(2_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                scalar |= input.dims.is_empty();
                empty |= input.dims.contains(&0);
            }
        }
        assert!(scalar, "no rank-0 scalar in 2,000 cases");
        assert!(empty, "no zero-length dimension in 2,000 cases");
    }

    /// ...and must **not** appear when the axis is off, or the axis is decoration.
    #[test]
    fn degenerate_shapes_are_absent_when_disabled() {
        let generator = OnnxGenerator::new(Bounds {
            degenerate_shapes: false,
            ..Bounds::default()
        });
        for seed in wide_seeds(2_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            for input in &case.inputs {
                assert!(
                    !input.dims.contains(&0),
                    "seed {seed} produced a zero-length dimension with the axis off"
                );
            }
        }
    }

    /// The narrow preset must stay narrow — it is the "one axis first" configuration, and a
    /// preset that quietly widened would make its measurements incomparable.
    #[test]
    fn the_one_axis_preset_produces_only_what_it_claims() {
        let generator = OnnxGenerator::new(Bounds::one_axis());
        for seed in wide_seeds(1_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            assert_eq!(
                case.inputs[0].elem_type(),
                ElemType::F32,
                "the one-axis preset is f32 only"
            );
            assert!(
                crate::ops::spec(case.op).tier == crate::ops::Tier::B,
                "{:?} is not Tier B",
                case.op
            );
        }
    }

    /// The configuration description must reach the generator's own report.
    #[test]
    fn the_description_records_the_configuration() {
        let described = OnnxGenerator::default().describe();
        for fragment in [
            "float-elementwise=",
            "max-rank=",
            "element-budget=",
            "logic=",
        ] {
            assert!(
                described.contains(fragment),
                "{fragment} missing from {described}"
            );
        }
    }

    /// `Reshape`'s target must always have exactly the input's element count — the one rule it
    /// has, and one a generator can get wrong invisibly, since a wrong target is rejected by
    /// the runtime and would read as non-support.
    #[test]
    fn reshape_targets_always_preserve_the_element_count() {
        let generator = OnnxGenerator::default();
        let mut checked = 0;
        for seed in wide_seeds(4_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if case.op != OpKind::Reshape {
                continue;
            }
            checked += 1;
            let source: i64 = case.inputs[0].dims.iter().product::<i64>().max(0);
            let crate::case::TensorData::I64(target) = &case.inputs[1].data else {
                panic!("Reshape's target must be an i64 vector");
            };
            assert_eq!(
                target.iter().product::<i64>().max(0),
                source,
                "seed {seed}: Reshape target {target:?} does not preserve {source} elements"
            );
        }
        assert!(
            checked > 20,
            "only {checked} Reshape cases — the check is nearly vacuous"
        );
    }

    /// `Gather`'s indices must be inside the axis extent. An out-of-range index is undefined in
    /// ONNX, and a case whose answer is undetermined is a false finding waiting to be triaged.
    #[test]
    fn gather_indices_are_always_in_range() {
        let generator = OnnxGenerator::default();
        let mut checked = 0;
        for seed in wide_seeds(4_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if case.op != OpKind::Gather {
                continue;
            }
            checked += 1;
            let axis = match case.attrs.get("axis") {
                Some(crate::attrs::AttrValue::Int(a)) => *a as usize,
                _ => 0,
            };
            let extent = case.inputs[0].dims[axis];
            let crate::case::TensorData::I64(indices) = &case.inputs[1].data else {
                panic!("Gather's indices must be an i64 vector");
            };
            for index in indices {
                assert!(
                    *index >= 0 && *index < extent,
                    "seed {seed}: index {index} outside axis extent {extent}"
                );
            }
        }
        assert!(
            checked > 20,
            "only {checked} Gather cases — the check is nearly vacuous"
        );
    }

    /// `Squeeze` may only remove a dimension of extent 1.
    #[test]
    fn squeeze_only_targets_unit_dimensions() {
        let generator = OnnxGenerator::default();
        let mut checked = 0;
        for seed in wide_seeds(4_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if case.op != OpKind::Squeeze {
                continue;
            }
            checked += 1;
            let crate::case::TensorData::I64(axes) = &case.inputs[1].data else {
                panic!("Squeeze's axes must be an i64 vector");
            };
            for axis in axes {
                assert_eq!(
                    case.inputs[0].dims[*axis as usize], 1,
                    "seed {seed}: squeezing a non-unit dimension"
                );
            }
        }
        assert!(
            checked > 20,
            "only {checked} Squeeze cases — the check is nearly vacuous"
        );
    }

    /// Configuration inputs are initializers; data inputs are fed. The role split, asserted
    /// over the whole corpus rather than on one hand-built probe.
    #[test]
    fn configuration_inputs_are_always_initializers() {
        let generator = OnnxGenerator::default();
        let mut with_initializers = 0;
        for seed in wide_seeds(3_000) {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if case.initializers().count() > 0 {
                with_initializers += 1;
            }
            // Whatever the operator, at least one input must still be fed — otherwise the
            // whole graph is constant and the kernel never runs.
            assert!(
                case.fed_inputs().count() > 0,
                "seed {seed}: every input is an initializer, so nothing is computed"
            );
        }
        assert!(
            with_initializers > 100,
            "only {with_initializers} cases used an initializer — the role split is barely covered"
        );
    }
}
