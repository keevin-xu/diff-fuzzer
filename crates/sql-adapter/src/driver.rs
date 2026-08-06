//! Running one case, end to end.
//!
//! This is where the five seams meet, and it is deliberately thin — the loop itself lives
//! in `diff_fuzzer_core::driver::run_once`, which knows nothing about SQL. Everything here
//! is assembly: pick the generator, pair each engine with the normalizer for its output,
//! hand the lot to the engine's driver, and get back a verdict.
//!
//! The assembly is the interesting part, because of what it costs: **nothing in
//! `diff-fuzzer-core` changed to make this work.** A `Runner` is an implementation plus a
//! normalizer, and once both engines are behind that pairing they are the same type from
//! the driver's point of view, despite producing values through two unrelated database
//! drivers. That is the whole claim of the architecture, and this file is where it either
//! compiles or does not.

use crate::ast::SqlCase;
use crate::backends::{DuckDbImpl, SqliteImpl};
use crate::generator::SqlGenerator;
use crate::normalize::{CanonicalResult, SqlNormalizer};
use crate::oracle::SqlDifferentialOracle;
use diff_fuzzer_core::driver::{RunOutcome, run_once};
use diff_fuzzer_core::runner::{NormalizedRunner, Runner};

/// Generate one case from `seed`, run it on both engines, and judge the results.
///
/// The seed travels back inside [`RunOutcome`] because a verdict nobody can reproduce is
/// not actionable. Note the honest limit at this stage: the generator ignores the seed, so
/// every seed yields the same case. That changes at S2; the plumbing is what is being
/// proven here.
pub fn check_case(seed: u64) -> RunOutcome {
    // Each engine is paired with the normalizer for its output. The pairing is checked at
    // compile time — `NormalizedRunner` requires `N: Normalizer<Out = I::Out>`, so pairing
    // an engine with a normalizer that cannot accept what it produces would not build.
    let sqlite = NormalizedRunner::new(SqliteImpl, SqlNormalizer);
    let duckdb = NormalizedRunner::new(DuckDbImpl, SqlNormalizer);

    // `&dyn Runner<..>` is what lets two concrete, unrelated types sit in one slice. The
    // driver iterates over this without knowing either engine exists.
    let runners: [&dyn Runner<In = SqlCase, Canon = CanonicalResult>; 2] = [&sqlite, &duckdb];

    run_once(
        seed,
        &SqlGenerator::default(),
        &runners,
        &SqlDifferentialOracle,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use diff_fuzzer_core::traits::Verdict;

    #[test]
    fn the_fixed_case_flows_through_every_seam_and_the_engines_agree() {
        // The walking skeleton, in one assertion: generate, run on two engines, normalize
        // both, judge. `Agree` here is a real verdict — two answers were compared — not the
        // absence of one.
        let outcome = check_case(0);
        assert_eq!(outcome.seed, 0);
        assert_eq!(outcome.verdict, Verdict::Agree);
    }

    #[test]
    fn the_same_seed_gives_the_same_verdict() {
        // True, but weak, and the weakness is the point: the generator ignores the seed, so
        // this proves the *pipeline* is deterministic rather than the generation. It
        // becomes a real test at S2, when different seeds produce different cases.
        assert_eq!(check_case(7).verdict, check_case(7).verdict);
    }

    #[test]
    fn the_seed_is_carried_back_with_the_verdict() {
        for seed in [0, 1, 42, u64::MAX] {
            assert_eq!(check_case(seed).seed, seed);
        }
    }
}
