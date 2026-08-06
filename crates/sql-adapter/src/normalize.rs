//! Turning what an engine returned into something comparable.
//!
//! Two engines can be equally correct and still hand back results that differ in ways that
//! carry no meaning — a different integer width, a different row order where SQL promises
//! none. Comparing before canonicalizing is the standard way a differential tester drowns
//! in false alarms, so both sides are converted to one form first.
//!
//! # What this stage may and may not do
//!
//! Canonicalizing is **lossy on purpose**: it deletes differences. Every difference it
//! deletes is one the oracle can no longer see, so the rule is narrow — *delete only what
//! is genuinely meaningless.* A canonical form that maps two different answers onto the
//! same text does not reduce noise; it hides a divergence, and it does so silently.
//!
//! That is the reasoning behind the one place this deviates from the plan's shorthand:
//! see [`cell_to_text`].
//!
//! # Not yet done here: sorting
//!
//! Row order is **not** normalized at this stage — the grids are compared in the order the
//! engines produced them. That is correct for the fixed case, which carries a total
//! `ORDER BY`, and it would be *wrong* for a query without one, where SQL promises no order
//! and two engines may legally differ. S3 adds the sort mode, decided from the whole case
//! (query *and* data), because whether an `ORDER BY` totally orders the answer depends on
//! whether the seeded rows tie on its columns.

use crate::outcome::{Cell, ErrorClass, SqlOutcome};
use diff_fuzzer_core::traits::Normalizer;

/// The comparable form of a result: text rows, or a normalized error class.
///
/// Text, rather than the typed cells, because the engines' own types do not line up — the
/// same `INTEGER` column is `i64` on one side and `i32` on the other. Rendering to a
/// canonical string is what makes "did these two produce the same answer?" a single
/// question rather than a per-type negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalResult {
    /// Rows of rendered cells, outer vector rows and inner columns.
    Rows(Vec<Vec<String>>),
    /// The engine refused the query, with this class of complaint.
    ///
    /// Kept as the class, never the message: engines phrase the same complaint
    /// differently, and comparing wording would report prose as a bug.
    Error(ErrorClass),
}

/// Renders an engine's outcome into [`CanonicalResult`].
///
/// A unit struct with no configuration — everything it needs is in the outcome. It gains
/// the sort mode at S3, at which point it will need the case too.
#[derive(Debug, Clone, Copy, Default)]
pub struct SqlNormalizer;

impl Normalizer for SqlNormalizer {
    type Out = SqlOutcome;
    type Canon = CanonicalResult;

    /// Note this **takes ownership** of the outcome rather than borrowing it. Rendering
    /// consumes the cells, and consuming lets the strings inside them be moved into the
    /// canonical form instead of copied.
    fn normalize(&self, out: SqlOutcome) -> CanonicalResult {
        match out {
            SqlOutcome::Rows(rows) => CanonicalResult::Rows(
                rows.into_iter()
                    .map(|row| row.into_iter().map(cell_to_text).collect())
                    .collect(),
            ),
            SqlOutcome::Error(class) => CanonicalResult::Error(class),
        }
    }
}

/// Render one cell to canonical text.
///
/// | Cell | Text |
/// |---|---|
/// | `Null` | `NULL` |
/// | `Integer(42)` | `42` |
/// | `Text("hi")` | `'hi'` |
/// | `Text("")` | `''` |
/// | `Text("NULL")` | `'NULL'` |
///
/// # Why text is quoted, when the plan said `NULL` → `"NULL"` and nothing about quotes
///
/// Because the unquoted version **collides**. A cell containing the four-character string
/// `NULL` would render identically to an actual `NULL`, so a case where one engine returned
/// the string and the other returned the absent value would canonicalize to the same text
/// and be judged **`Agree`**. That is a false negative: a real divergence, silently erased,
/// with nothing in any output to show it happened.
///
/// Quoting costs nothing and removes the collision — and it removes the same collision for
/// the empty string, which would otherwise render as nothing at all and be impossible to
/// tell from a missing value by eye.
///
/// This deviates from `sqllogictest`, which renders `NULL` as bare `NULL` and text as-is.
/// That is a defensible choice *for sqllogictest*, whose expected output lives in a file
/// that a human wrote and reads. It is not defensible here, where both sides are produced
/// by machines and nobody would ever look at the collision. **The sqllogictest rules have
/// not been retrieved yet** (`SPECS.md` §5.1), so no rule here cites them; when they are
/// retrieved at S3, this decision gets revisited with the actual text in hand
/// (`PENDING` 2.9).
///
/// The asymmetry that settles it: keeping a distinction the engines might not care about
/// costs at worst a false positive — visible, and self-correcting the moment someone looks.
/// Erasing a distinction they do care about costs a bug nobody will ever see.
fn cell_to_text(cell: Cell) -> String {
    match cell {
        Cell::Null => "NULL".to_string(),
        Cell::Integer(number) => number.to_string(),
        Cell::Text(text) => format!("'{text}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_null_and_the_string_null_do_not_collide() {
        // The reason for quoting, as an executable claim. If this ever fails, the
        // canonical form has started hiding divergences rather than removing noise.
        assert_ne!(
            cell_to_text(Cell::Null),
            cell_to_text(Cell::Text("NULL".to_string()))
        );
    }

    #[test]
    fn an_empty_string_and_a_null_do_not_collide() {
        assert_ne!(
            cell_to_text(Cell::Null),
            cell_to_text(Cell::Text(String::new()))
        );
        // And the empty string renders as something visible, rather than as nothing.
        assert_eq!(cell_to_text(Cell::Text(String::new())), "''");
    }

    #[test]
    fn integers_render_as_their_digits() {
        assert_eq!(cell_to_text(Cell::Integer(42)), "42");
        assert_eq!(cell_to_text(Cell::Integer(-1)), "-1");
        assert_eq!(cell_to_text(Cell::Integer(0)), "0");
    }

    #[test]
    fn normalizing_preserves_the_shape_of_the_grid() {
        let outcome = SqlOutcome::Rows(vec![
            vec![Cell::Integer(1), Cell::Text("one".to_string())],
            vec![Cell::Integer(2), Cell::Null],
        ]);

        let CanonicalResult::Rows(rows) = SqlNormalizer.normalize(outcome) else {
            panic!("rows normalize to rows");
        };
        assert_eq!(rows, vec![vec!["1", "'one'"], vec!["2", "NULL"]]);
    }

    #[test]
    fn an_error_normalizes_to_its_class_and_stays_an_error() {
        let normalized = SqlNormalizer.normalize(SqlOutcome::Error(ErrorClass::DivideByZero));
        assert_eq!(normalized, CanonicalResult::Error(ErrorClass::DivideByZero));
        // An error must never canonicalize into "no rows" — that would turn a refusal
        // into an empty answer, and make a refusing engine agree with an empty result.
        assert_ne!(normalized, CanonicalResult::Rows(vec![]));
    }

    #[test]
    fn row_order_is_not_yet_normalized() {
        // Documenting the current limitation as a test rather than as a comment: these two
        // grids hold the same rows in a different order and are *not* equal today. That is
        // correct only because every case at this stage has a total `ORDER BY`. When S3
        // adds unordered queries, this test should change — and it will fail loudly first,
        // rather than letting an unsorted comparison quietly invent divergences.
        let ascending = SqlNormalizer.normalize(SqlOutcome::Rows(vec![
            vec![Cell::Integer(1)],
            vec![Cell::Integer(2)],
        ]));
        let descending = SqlNormalizer.normalize(SqlOutcome::Rows(vec![
            vec![Cell::Integer(2)],
            vec![Cell::Integer(1)],
        ]));
        assert_ne!(ascending, descending);
    }
}
