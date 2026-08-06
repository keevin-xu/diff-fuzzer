//! What running a case produces — which, in SQL, is one of *two different kinds of
//! thing*.
//!
//! A tensor operation returns a tensor. A SQL query either returns rows **or raises an
//! error**, and "these two engines disagree about whether this query is even legal" is a
//! finding in its own right, not a failure to get a result. That is the reason this is an
//! enum rather than a row grid: making the error case a peer of the rows case means the
//! oracle has to decide what a rows-versus-error pairing means, instead of one engine's
//! error quietly ending up in the same bucket as "could not run".
//!
//! Note the distinction that costs people bugs: [`SqlOutcome::Error`] means the engine
//! **ran the query and refused it** — a defined answer. It is not
//! `diff_fuzzer_core::traits::RunError`, which means *we* could not get an answer at all
//! (a connection failed, a feature is unsupported). The first is evidence; the second is
//! a reason to skip the case.

use serde::{Deserialize, Serialize};

/// One value in one cell of a result.
///
/// The variants are exactly the v1 generated types plus `Null` — `INTEGER`/`BIGINT` both
/// land in `Integer`, since SQL's integer types differ in declared width but not in the
/// value we read back.
///
/// **What is deliberately absent: floating point, `BOOLEAN` and `DECIMAL`.** Not an
/// oversight and not laziness — retrieved evidence says SQLite has no separate Boolean
/// storage class (it stores `0`/`1`) and no fixed-point decimal (a `DECIMAL` column lands
/// on IEEE 754 binary64 by affinity), while DuckDB has both natively. Generating either
/// would produce differences on *every* such cell that are about representation rather
/// than about correctness. See `SPECS.md` §4.1–4.2 and `POLICY.md` §3.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cell {
    /// SQL's `NULL` — distinct from every other value *and from itself* in SQL's own
    /// three-valued logic. Here it is an ordinary variant that compares equal to another
    /// `Null`, which is the right choice for *comparing results*: two engines both
    /// returning `NULL` in a cell agree about that cell. SQL's `NULL = NULL` being unknown
    /// is a fact about queries, not about result comparison, and conflating the two would
    /// make every `NULL` a divergence.
    Null,
    /// `INTEGER` and `BIGINT`. `i64` covers both engines' 64-bit signed range.
    Integer(i64),
    /// `TEXT`. Held as a Rust `String`, so it is valid UTF-8 by construction.
    Text(String),
}

/// A normalized category of query error.
///
/// Engines phrase the same complaint differently — one says `division by zero`, another
/// `Division by Zero!` — and comparing message text would report wording as a bug. So an
/// error is compared by *class*, never by message.
///
/// **Only [`ErrorClass::Other`] is produced at this stage.** The classification that
/// makes the other three reachable arrives at S3.4, together with a test that every
/// variant *can* be returned. That test is not ceremony: the tensor domain shipped a
/// classifier with a variant it could never produce, so those cases silently became
/// "unknown" and were discarded — and the loss looked like missing data rather than like
/// a bug. Until S3.4, treat the three specific variants as declared-but-unreachable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorClass {
    /// Division (or modulo) by zero.
    DivideByZero,
    /// A value could not be interpreted as the type the query required.
    TypeMismatch,
    /// Arithmetic overflow, or a value outside a column's range.
    OutOfRange,
    /// Anything not yet classified. Every engine error maps here until S3.4.
    Other,
}

/// What one engine produced for one case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlOutcome {
    /// Rows, as a grid: outer vector is rows, inner is columns.
    ///
    /// The grid keeps the engine's own row order. Deciding whether that order is part of
    /// the answer is the normalizer's job (S3), and it is a decision about the *case*, not
    /// about this value — an `ORDER BY` only orders the answer totally if the data has no
    /// ties on its columns.
    Rows(Vec<Vec<Cell>>),
    /// The engine ran the query and refused it, with this class of complaint.
    Error(ErrorClass),
}

impl SqlOutcome {
    /// How many rows came back, or `None` for an error.
    ///
    /// Useful in tests and reports; deliberately not used for judging, since "both
    /// returned three rows" is not agreement.
    pub fn row_count(&self) -> Option<usize> {
        match self {
            SqlOutcome::Rows(rows) => Some(rows.len()),
            SqlOutcome::Error(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_and_errors_are_different_outcomes() {
        // The point of the enum: an engine returning zero rows and an engine refusing the
        // query are not the same event, and must never compare equal.
        assert_ne!(
            SqlOutcome::Rows(vec![]),
            SqlOutcome::Error(ErrorClass::Other)
        );
    }

    #[test]
    fn null_equals_null_when_comparing_results() {
        // Comparing *results*, not evaluating SQL: two engines that both returned NULL
        // here agree. If this were SQL's own `=`, the answer would be unknown.
        assert_eq!(Cell::Null, Cell::Null);
        assert_ne!(Cell::Null, Cell::Integer(0));
        assert_ne!(Cell::Null, Cell::Text(String::new()));
    }

    #[test]
    fn empty_text_is_not_null() {
        // The classic result-comparison trap: an empty string and a NULL print the same
        // way unless you make them print differently. They are different values here, and
        // the normalizer has to keep them different (S3).
        assert_ne!(Cell::Text(String::new()), Cell::Null);
    }

    #[test]
    fn row_count_distinguishes_no_rows_from_an_error() {
        assert_eq!(SqlOutcome::Rows(vec![]).row_count(), Some(0));
        assert_eq!(
            SqlOutcome::Error(ErrorClass::DivideByZero).row_count(),
            None
        );
    }
}
