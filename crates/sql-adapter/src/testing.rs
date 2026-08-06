//! An engine that is wrong on purpose.
//!
//! # Why a tool needs one
//!
//! A campaign that finds nothing has two explanations: the software under test is fine, or
//! **the detector is broken**. From the outside those look identical — a quiet log either
//! way — and only one of them is good news. Wrapping a real engine in a known fault and
//! checking that the oracle catches it is what separates them. Without this, "we ran for
//! five hours and found nothing" is not a result; it is an absence of information.
//!
//! The same wrapper is what a campaign uses for its fault-injection check before trusting a
//! long quiet run (S6.6).
//!
//! # What the faults are chosen to be
//!
//! Not arbitrary corruption. Each fault produces a *different kind* of disagreement, so
//! together they exercise the oracle's comparison paths rather than one of them three
//! times:
//!
//! - [`Fault::DropLastRow`] — the grids differ in **height**.
//! - [`Fault::ChangeFirstCell`] — same shape, differing **content**.
//! - [`Fault::AlwaysRefuse`] — rows against a **refusal**, the pairing this domain exists
//!   to notice.

use crate::ast::SqlCase;
use crate::outcome::{Cell, ErrorClass, SqlOutcome};
use diff_fuzzer_core::traits::{Implementation, RunError};

/// A way of being wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// Return one row fewer than the engine actually produced.
    ///
    /// The subtlest of the three: everything the engine returned is correct, there is
    /// simply less of it. A comparison that only checked cell values pairwise without
    /// checking length could miss this.
    DropLastRow,
    /// Replace the first cell of the first row with a value nothing would produce.
    ChangeFirstCell,
    /// Refuse every query the engine accepted.
    AlwaysRefuse,
}

/// Wraps a real engine and corrupts what it returns.
///
/// Generic over the engine so the fault can be applied to either side — which matters,
/// because "the oracle catches SQLite being wrong" and "the oracle catches DuckDB being
/// wrong" are two claims, and an oracle written with an accidental bias toward its first
/// argument would satisfy only one of them.
#[derive(Debug, Clone, Copy)]
pub struct FaultyEngine<I> {
    inner: I,
    fault: Fault,
    name: &'static str,
}

impl<I> FaultyEngine<I> {
    /// Wrap `inner`, applying `fault` to every result.
    ///
    /// The name is given explicitly rather than derived from the wrapped engine, so a
    /// faulty engine can never be mistaken for the real one in a report. A finding that
    /// says `sqlite` when the case ran against a deliberately-broken `sqlite` would be a
    /// fabricated result — the exact thing this project refuses to produce.
    pub fn new(inner: I, fault: Fault, name: &'static str) -> Self {
        Self { inner, fault, name }
    }
}

impl<I> Implementation for FaultyEngine<I>
where
    I: Implementation<In = SqlCase, Out = SqlOutcome>,
{
    type In = SqlCase;
    type Out = SqlOutcome;

    fn name(&self) -> &str {
        self.name
    }

    fn run(&self, case: &SqlCase) -> Result<SqlOutcome, RunError> {
        // Run the real engine first. A fault that skipped execution would also skip every
        // way execution can fail, and the test would stop covering the paths it claims to.
        let honest = self.inner.run(case)?;

        Ok(match (self.fault, honest) {
            (Fault::AlwaysRefuse, _) => SqlOutcome::Error(ErrorClass::Other),

            (Fault::DropLastRow, SqlOutcome::Rows(mut rows)) => {
                rows.pop();
                SqlOutcome::Rows(rows)
            }

            (Fault::ChangeFirstCell, SqlOutcome::Rows(mut rows)) => {
                if let Some(first_row) = rows.first_mut()
                    && let Some(first_cell) = first_row.first_mut()
                {
                    *first_cell = Cell::Text("corrupted-by-fault-injection".to_string());
                }
                SqlOutcome::Rows(rows)
            }

            // A fault that alters rows has nothing to alter when the engine already
            // refused the query. Returning the refusal unchanged is honest: this run
            // injected no fault, and a test relying on one would fail rather than pass for
            // the wrong reason.
            (_, refusal @ SqlOutcome::Error(_)) => refusal,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{DuckDbImpl, SqliteImpl};
    use crate::normalize::{CanonicalResult, SqlNormalizer};
    use crate::oracle::SqlDifferentialOracle;
    use diff_fuzzer_core::runner::{NormalizedRunner, Runner};
    use diff_fuzzer_core::traits::{NamedOutput, Oracle, Verdict};

    /// Run one case through the whole pipeline against an arbitrary pair of engines.
    fn verdict_for(
        case: &SqlCase,
        left: &dyn Runner<In = SqlCase, Canon = CanonicalResult>,
        right: &dyn Runner<In = SqlCase, Canon = CanonicalResult>,
    ) -> Verdict {
        let outputs: Vec<NamedOutput<CanonicalResult>> = [left, right]
            .iter()
            .map(|runner| NamedOutput {
                implementation: runner.name().to_string(),
                output: runner
                    .run_and_normalize(case)
                    .expect("both engines run the fixed case"),
            })
            .collect();

        SqlDifferentialOracle.check(case, &outputs)
    }

    /// The claim this module exists to support: a known-wrong engine is caught.
    ///
    /// Every fault, against both engines — because "the oracle catches a fault on the
    /// left" and "on the right" are separate claims, and an oracle biased toward its first
    /// argument would pass only half of these.
    #[test]
    fn every_fault_is_caught_on_either_side() {
        let case = SqlCase::fixed_example();
        let honest_sqlite = NormalizedRunner::new(SqliteImpl, SqlNormalizer);
        let honest_duckdb = NormalizedRunner::new(DuckDbImpl, SqlNormalizer);

        for fault in [
            Fault::DropLastRow,
            Fault::ChangeFirstCell,
            Fault::AlwaysRefuse,
        ] {
            let faulty_sqlite = NormalizedRunner::new(
                FaultyEngine::new(SqliteImpl, fault, "sqlite-with-injected-fault"),
                SqlNormalizer,
            );
            let faulty_duckdb = NormalizedRunner::new(
                FaultyEngine::new(DuckDbImpl, fault, "duckdb-with-injected-fault"),
                SqlNormalizer,
            );

            assert!(
                matches!(
                    verdict_for(&case, &faulty_sqlite, &honest_duckdb),
                    Verdict::Diverged(_)
                ),
                "{fault:?} on the left went undetected"
            );
            assert!(
                matches!(
                    verdict_for(&case, &honest_sqlite, &faulty_duckdb),
                    Verdict::Diverged(_)
                ),
                "{fault:?} on the right went undetected"
            );
        }
    }

    /// The control. Without it, the test above could pass because *everything* diverges.
    ///
    /// A detector that reports every case as a divergence catches all three faults and is
    /// worthless. This is the same reasoning as a baseline rate beside a measured one: a
    /// number without its control is not evidence.
    #[test]
    fn the_honest_pair_still_agrees() {
        let case = SqlCase::fixed_example();
        let sqlite = NormalizedRunner::new(SqliteImpl, SqlNormalizer);
        let duckdb = NormalizedRunner::new(DuckDbImpl, SqlNormalizer);

        assert_eq!(verdict_for(&case, &sqlite, &duckdb), Verdict::Agree);
    }

    /// Two engines wrong in the *same* way agree — the blind spot, stated as a test.
    ///
    /// This is not a defect in the oracle; it is what differential testing structurally
    /// cannot see, and the reason a single-engine metamorphic oracle is worth building
    /// later (S8). Writing it down as an executable claim keeps it from being rediscovered
    /// as a surprise.
    #[test]
    fn a_fault_shared_by_both_engines_is_invisible() {
        let case = SqlCase::fixed_example();
        let both_wrong_left = NormalizedRunner::new(
            FaultyEngine::new(SqliteImpl, Fault::DropLastRow, "sqlite-faulty"),
            SqlNormalizer,
        );
        let both_wrong_right = NormalizedRunner::new(
            FaultyEngine::new(DuckDbImpl, Fault::DropLastRow, "duckdb-faulty"),
            SqlNormalizer,
        );

        assert_eq!(
            verdict_for(&case, &both_wrong_left, &both_wrong_right),
            Verdict::Agree,
            "differential testing cannot see a fault both engines share"
        );
    }

    #[test]
    fn a_faulty_engine_never_reports_the_honest_engines_name() {
        // A finding attributed to `sqlite` when the case ran against a broken wrapper
        // would be a fabricated result.
        let faulty = FaultyEngine::new(SqliteImpl, Fault::DropLastRow, "sqlite-with-fault");
        assert_ne!(faulty.name(), SqliteImpl.name());
    }
}
