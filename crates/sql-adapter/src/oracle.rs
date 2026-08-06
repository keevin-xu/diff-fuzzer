//! Deciding whether two engines disagreed.
//!
//! The reasoning is deliberately narrow, and it is the whole trick of differential
//! testing: several engines were asked the same question and are supposed to give the same
//! answer, so if their answers differ, at least one is wrong. *Which* one, and why, are
//! separate questions this does not attempt — which is exactly what lets the technique work
//! on software whose correct answers nobody can cheaply compute.
//!
//! # Why this is the adapter's own type rather than the engine's
//!
//! `diff_fuzzer_core::oracle::DifferentialOracle` exists and is not usable here: it is
//! bounded `C: ApproxEq` and takes a `TolerancePolicy`, because it was built for a numeric
//! domain where two correct answers differ in the last bits. **SQL has no such notion.**
//! Results are text; they match or they do not. Rather than widen the shared engine to
//! accommodate a domain it was not built for, this adapter writes its own — the rule from
//! `09` §2b: copy, don't generalise, until a second instance shows what is actually shared.
//!
//! What is *not* copied is the reasoning, which carries verbatim from the tensor domain and
//! is written into [`SqlDifferentialOracle::check`]: compare **every pair**, never
//! everything against the first.

use crate::ast::SqlCase;
use crate::normalize::CanonicalResult;
use diff_fuzzer_core::report::Divergence;
use diff_fuzzer_core::traits::{NamedOutput, Oracle, SkipReason, Verdict};

/// Flags disagreement between two or more engines on the same case.
///
/// No configuration and no state: everything it needs arrives in `check`. It gains one
/// thing at S4 — the `KnownLegal` catalog, which turns a *documented* legal difference into
/// a `Skipped` rather than a finding. Until then every difference is reported, which is the
/// right direction to be wrong in: a false positive is noisy and self-correcting, while a
/// difference wrongly filtered is silent and permanent.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlDifferentialOracle;

impl Oracle for SqlDifferentialOracle {
    type In = SqlCase;
    type Canon = CanonicalResult;

    fn check(&self, input: &SqlCase, outputs: &[NamedOutput<CanonicalResult>]) -> Verdict {
        // Fewer than two answers means there is nothing to compare against. A skip, not a
        // failure: an engine legitimately being unable to run a case is expected, and
        // comparing an answer against no answer would be meaningless.
        if outputs.len() < 2 {
            return Verdict::Skipped(SkipReason::TooFewResults {
                available: outputs.len(),
            });
        }

        // **Every pair, not everything against `outputs[0]`.**
        //
        // With two engines these are the same operation, which is precisely the danger:
        // the choice is untested until a third arrives, and choosing wrongly fails
        // silently — a missed disagreement is indistinguishable from agreement. The tensor
        // domain found this same bug in three separate places once it had a third backend.
        // Writing it correctly now costs one extra loop.
        let mut complaints = Vec::new();
        for (index, left) in outputs.iter().enumerate() {
            for right in &outputs[index + 1..] {
                if left.output != right.output {
                    complaints.push(format!(
                        "{} and {} disagree: {}",
                        left.implementation,
                        right.implementation,
                        disagreement(&left.output, &right.output)
                    ));
                }
            }
        }

        if complaints.is_empty() {
            return Verdict::Agree;
        }

        Verdict::Diverged(Divergence {
            input: format!("{input:?}"),
            outputs: outputs
                .iter()
                .map(|named| (named.implementation.clone(), render(&named.output)))
                .collect(),
            summary: complaints.join("; "),
        })
    }
}

/// Say what differs between two results, in one short phrase.
///
/// Deliberately a *description*, not a classification. Grouping findings by the kind of
/// disagreement is S5's job and needs a stable key; this only has to make a log line
/// readable enough to decide whether the case is worth opening.
fn disagreement(left: &CanonicalResult, right: &CanonicalResult) -> String {
    match (left, right) {
        (CanonicalResult::Rows(left_rows), CanonicalResult::Rows(right_rows)) => {
            if left_rows.len() != right_rows.len() {
                return format!("{} rows versus {}", left_rows.len(), right_rows.len());
            }
            // Same height: find the first cell that differs, which is almost always the
            // one worth looking at.
            for (row_index, (left_row, right_row)) in
                left_rows.iter().zip(right_rows.iter()).enumerate()
            {
                if left_row.len() != right_row.len() {
                    return format!(
                        "row {row_index} has {} columns versus {}",
                        left_row.len(),
                        right_row.len()
                    );
                }
                for (column, (left_cell, right_cell)) in
                    left_row.iter().zip(right_row.iter()).enumerate()
                {
                    if left_cell != right_cell {
                        return format!(
                            "row {row_index} column {column}: {left_cell} versus {right_cell}"
                        );
                    }
                }
            }
            // Unreachable while `CanonicalResult` compares by value, but stated rather
            // than assumed: if the grids are equal cell by cell, they were equal.
            "the same rows in a different order".to_string()
        }
        (CanonicalResult::Rows(rows), CanonicalResult::Error(class)) => {
            format!("{} rows versus a {class:?} error", rows.len())
        }
        (CanonicalResult::Error(class), CanonicalResult::Rows(rows)) => {
            format!("a {class:?} error versus {} rows", rows.len())
        }
        (CanonicalResult::Error(left_class), CanonicalResult::Error(right_class)) => {
            format!("{left_class:?} versus {right_class:?}")
        }
    }
}

/// Render a result compactly enough to sit in a report.
fn render(result: &CanonicalResult) -> String {
    match result {
        CanonicalResult::Rows(rows) => {
            let body: Vec<String> = rows.iter().map(|row| row.join(", ")).collect();
            format!("[{}]", body.join(" | "))
        }
        CanonicalResult::Error(class) => format!("error: {class:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::ErrorClass;

    /// Build a labelled output without going near a database — the reason the oracle takes
    /// canonical results rather than running anything itself. It can be tested exhaustively
    /// against fabricated answers, with no engines involved.
    fn output(name: &str, rows: &[&[&str]]) -> NamedOutput<CanonicalResult> {
        NamedOutput {
            implementation: name.to_string(),
            output: CanonicalResult::Rows(
                rows.iter()
                    .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                    .collect(),
            ),
        }
    }

    fn error_output(name: &str, class: ErrorClass) -> NamedOutput<CanonicalResult> {
        NamedOutput {
            implementation: name.to_string(),
            output: CanonicalResult::Error(class),
        }
    }

    #[test]
    fn identical_results_agree() {
        let outputs = [
            output("sqlite", &[&["1", "'one'"], &["2", "NULL"]]),
            output("duckdb", &[&["1", "'one'"], &["2", "NULL"]]),
        ];
        assert_eq!(
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs),
            Verdict::Agree
        );
    }

    #[test]
    fn a_differing_cell_diverges_and_the_summary_names_it() {
        let outputs = [
            output("sqlite", &[&["1", "'one'"]]),
            output("duckdb", &[&["1", "'ONE'"]]),
        ];

        let Verdict::Diverged(divergence) =
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs)
        else {
            panic!("differing cells must diverge");
        };
        assert!(divergence.summary.contains("row 0 column 1"));
        assert!(divergence.summary.contains("'one'"));
        assert!(divergence.summary.contains("'ONE'"));
    }

    #[test]
    fn a_differing_row_count_diverges() {
        let outputs = [
            output("sqlite", &[&["1"], &["2"]]),
            output("duckdb", &[&["1"]]),
        ];
        let Verdict::Diverged(divergence) =
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs)
        else {
            panic!("differing row counts must diverge");
        };
        assert!(divergence.summary.contains("2 rows versus 1"));
    }

    #[test]
    fn rows_against_an_error_diverges() {
        // The pairing this domain exists to notice: one engine answered, the other refused.
        let outputs = [
            output("sqlite", &[&["1"]]),
            error_output("duckdb", ErrorClass::DivideByZero),
        ];
        let Verdict::Diverged(divergence) =
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs)
        else {
            panic!("rows versus an error must diverge");
        };
        assert!(divergence.summary.contains("DivideByZero"));
    }

    #[test]
    fn the_same_error_class_agrees() {
        // Both engines refused the query for the same reason. They agree — and note this
        // is agreement about a *refusal*, which is a real answer, not an absence of one.
        let outputs = [
            error_output("sqlite", ErrorClass::DivideByZero),
            error_output("duckdb", ErrorClass::DivideByZero),
        ];
        assert_eq!(
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs),
            Verdict::Agree
        );
    }

    #[test]
    fn different_error_classes_diverge() {
        let outputs = [
            error_output("sqlite", ErrorClass::DivideByZero),
            error_output("duckdb", ErrorClass::TypeMismatch),
        ];
        assert!(matches!(
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs),
            Verdict::Diverged(_)
        ));
    }

    #[test]
    fn one_result_is_skipped_not_agreed() {
        // Nothing was compared, so nothing was learned. Calling this `Agree` would inflate
        // the evidence a campaign appears to provide.
        let outputs = [output("sqlite", &[&["1"]])];
        assert_eq!(
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs),
            Verdict::Skipped(SkipReason::TooFewResults { available: 1 })
        );
    }

    /// The reason for comparing every pair, tested before a third engine exists.
    ///
    /// Two agreeing results followed by a dissenter: comparing everything against
    /// `outputs[0]` would still catch this one. The next test is the one that would not be.
    #[test]
    fn a_third_engine_disagreeing_with_the_first_two_diverges() {
        let outputs = [
            output("sqlite", &[&["1"]]),
            output("duckdb", &[&["1"]]),
            output("third", &[&["2"]]),
        ];
        assert!(matches!(
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs),
            Verdict::Diverged(_)
        ));
    }

    /// The case that "compare against the first" gets **wrong**.
    ///
    /// Here the first engine agrees with the second and with the third, but the second and
    /// third disagree with *each other*. Comparing everything against `outputs[0]` reports
    /// agreement; comparing all pairs reports the divergence. With text equality this
    /// cannot arise from rounding, but it can arise the moment any normalization is not
    /// transitive — and the point is that the code does not depend on that argument.
    #[test]
    fn a_disagreement_that_does_not_involve_the_first_result_is_still_caught() {
        let outputs = [
            NamedOutput {
                implementation: "first".to_string(),
                output: CanonicalResult::Rows(vec![vec!["1".to_string()]]),
            },
            NamedOutput {
                implementation: "second".to_string(),
                output: CanonicalResult::Rows(vec![vec!["1".to_string()]]),
            },
            NamedOutput {
                implementation: "third".to_string(),
                output: CanonicalResult::Rows(vec![vec!["2".to_string()]]),
            },
        ];

        let Verdict::Diverged(divergence) =
            SqlDifferentialOracle.check(&SqlCase::fixed_example(), &outputs)
        else {
            panic!("a disagreement between the second and third must be caught");
        };
        // Both pairs involving `third` are reported, not just the one with `first`.
        assert!(divergence.summary.contains("first and third"));
        assert!(divergence.summary.contains("second and third"));
    }
}
