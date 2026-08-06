//! Producing cases to test.
//!
//! The two halves live in [`crate::gen_schema`] and [`crate::gen_query`]; this binds them
//! to the engine's [`Generator`] seam and fixes the order they run in — **state first, then
//! a query against that state**. The order is the whole reason cases are valid by
//! construction: a query built with the schema and data in hand can only reference columns
//! that exist, at types that match, and can only carry a `LIMIT` when the rows it will
//! order are actually distinct.
//!
//! `generate` receives `&mut SeededRng` and nothing else, so every choice traces back to
//! one 64-bit seed. That is what makes a run replayable — though a finding still records
//! the whole case rather than the seed, because a seed only reproduces a case for the exact
//! generator that produced it, and generators change.

use crate::ast::SqlCase;
use crate::gen_query::generate_query;
use crate::gen_schema::{Bounds, generate_data, generate_schema};
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Generator;

/// Generates valid `SqlCase`s within [`Bounds`].
#[derive(Debug, Clone, Copy)]
pub struct SqlGenerator {
    bounds: Bounds,
}

impl SqlGenerator {
    pub fn new(bounds: Bounds) -> Self {
        Self { bounds }
    }

    /// The bounds in force, which a report must record beside any finding.
    ///
    /// Two cases drawn under different bounds come from different distributions, even
    /// though both would describe themselves as "generated" — which is how the tensor
    /// domain ended up scoring findings against a pool that was not comparable to them.
    pub fn description(&self) -> String {
        self.bounds.description()
    }
}

impl Default for SqlGenerator {
    fn default() -> Self {
        Self::new(Bounds::V1)
    }
}

impl Generator for SqlGenerator {
    type In = SqlCase;

    fn generate(&self, rng: &mut SeededRng) -> SqlCase {
        let schema = generate_schema(rng, self.bounds);
        let data = generate_data(rng, &schema, self.bounds);
        let query = generate_query(rng, &schema, &data, self.bounds);

        SqlCase {
            schema,
            data,
            query,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate(seed: u64) -> SqlCase {
        SqlGenerator::default().generate(&mut SeededRng::from_seed(seed))
    }

    #[test]
    fn the_same_seed_gives_the_same_case() {
        // The real determinism test, which S1's placeholder could not be: different seeds
        // now genuinely produce different cases, so this asserts something.
        for seed in [0, 1, 42, 9_999] {
            assert_eq!(generate(seed), generate(seed));
        }
    }

    #[test]
    fn different_seeds_give_different_cases() {
        let distinct = (0..50)
            .map(|seed| serde_json::to_string(&generate(seed)).expect("serializes"))
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > 40,
            "only {distinct} distinct cases from 50 seeds"
        );
    }

    #[test]
    fn every_generated_case_validates() {
        for seed in 0..500 {
            generate(seed)
                .validate()
                .unwrap_or_else(|problem| panic!("seed {seed}: {problem}"));
        }
    }

    #[test]
    fn the_description_names_the_bounds_in_force() {
        assert_eq!(
            SqlGenerator::default().description(),
            Bounds::V1.description()
        );
    }
}
