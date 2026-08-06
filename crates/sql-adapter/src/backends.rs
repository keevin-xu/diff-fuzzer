//! The two engines, as implementations of the shared [`Implementation`] seam.
//!
//! Each one takes a [`SqlCase`], opens a **fresh in-memory database**, applies the schema
//! and the data, runs the query, and hands back a [`SqlOutcome`]. The database is created
//! and dropped inside the call, which is the concrete reason the shared trait still takes
//! `&self`: there is no state to keep, so there is nothing for `self` to hold.
//!
//! # Three failures that are not the same failure
//!
//! Getting this wrong is how a differential tester reports its own bugs as findings, so
//! the distinction is worth stating plainly:
//!
//! | What happened | What it means | Result |
//! |---|---|---|
//! | The connection or the **setup** (`CREATE`/`INSERT`) failed | we never reached the query on this engine | [`RunError`] → the case is *skipped* |
//! | The engine ran the **query** and refused it | a defined answer, and disagreeing about it is a finding | [`SqlOutcome::Error`] |
//! | A returned value cannot be represented as a [`Cell`] | *our* gap, not the engine's | [`RunError`] → skipped, visibly |
//!
//! The middle row is the one with teeth. "SQLite returned rows and DuckDB refused the
//! query" is exactly the kind of disagreement this domain exists to notice, so a refusal
//! must not be quietly filed alongside "could not run".

use crate::ast::SqlCase;
use crate::outcome::{Cell, ErrorClass, SqlOutcome};
use diff_fuzzer_core::traits::{Implementation, RunError};

/// How SQLite identifies itself in findings, negatives, and reports.
///
/// **One definition, because these are matched by string equality.** The tensor domain
/// paid for this lesson: examples hardcoded backend names that did not match what the
/// runners reported, so every comparison silently found nothing — and the failure read as
/// *missing data* rather than as a typo, which is how it survived review. The tests below
/// tie each constant to what `name()` actually returns.
pub const SQLITE_NAME: &str = "sqlite";

/// How DuckDB identifies itself. See [`SQLITE_NAME`].
pub const DUCKDB_NAME: &str = "duckdb";

/// SQLite, via `rusqlite`, with the engine compiled from source (`bundled`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SqliteImpl;

/// DuckDB, via the `duckdb` crate, likewise compiled from source.
#[derive(Debug, Clone, Copy, Default)]
pub struct DuckDbImpl;

impl Implementation for SqliteImpl {
    type In = SqlCase;
    type Out = SqlOutcome;

    fn name(&self) -> &str {
        SQLITE_NAME
    }

    fn run(&self, case: &SqlCase) -> Result<SqlOutcome, RunError> {
        // `:memory:` — this database exists only for this call. Nothing to reset, nothing
        // to leak into the next case, and no file for a crashed run to leave behind.
        let conn = rusqlite::Connection::open_in_memory()
            .map_err(|error| setup_failed(SQLITE_NAME, error))?;

        for statement in case.schema.iter().chain(case.data.iter()) {
            conn.execute(statement, [])
                .map_err(|error| setup_failed(SQLITE_NAME, error))?;
        }

        // From here on, a complaint from the engine is an *answer*: the query was put to
        // it and refused.
        let mut prepared = match conn.prepare(&case.query) {
            Ok(prepared) => prepared,
            Err(_) => return Ok(SqlOutcome::Error(ErrorClass::Other)),
        };

        let mut rows = match prepared.query([]) {
            Ok(rows) => rows,
            Err(_) => return Ok(SqlOutcome::Error(ErrorClass::Other)),
        };

        // Taken *after* running the query, not before — see the note on the DuckDB
        // implementation, which panics if asked earlier. Doing it identically on both
        // sides keeps the two functions comparable line for line.
        let column_count = column_count_of(&rows);

        let mut grid = Vec::new();
        loop {
            // `Rows` is a cursor, not an iterator: each `next()` may fail, so it cannot
            // implement `Iterator` (whose `next` cannot report an error). That is why this
            // is a `loop` with a `match` rather than a `for`.
            match rows.next() {
                Ok(Some(row)) => {
                    let mut cells = Vec::with_capacity(column_count);
                    for index in 0..column_count {
                        let value: rusqlite::types::Value = row
                            .get(index)
                            .map_err(|error| setup_failed(SQLITE_NAME, error))?;
                        cells.push(sqlite_cell(value)?);
                    }
                    grid.push(cells);
                }
                Ok(None) => break,
                // Failing *part way through* reading rows is not a refusal of the query —
                // the engine already accepted it. Treat it as our inability to get an
                // answer, so the case is skipped rather than half-reported.
                Err(error) => return Err(setup_failed(SQLITE_NAME, error)),
            }
        }

        Ok(SqlOutcome::Rows(grid))
    }
}

impl Implementation for DuckDbImpl {
    type In = SqlCase;
    type Out = SqlOutcome;

    fn name(&self) -> &str {
        DUCKDB_NAME
    }

    fn run(&self, case: &SqlCase) -> Result<SqlOutcome, RunError> {
        let conn = duckdb::Connection::open_in_memory()
            .map_err(|error| setup_failed(DUCKDB_NAME, error))?;

        for statement in case.schema.iter().chain(case.data.iter()) {
            conn.execute(statement, [])
                .map_err(|error| setup_failed(DUCKDB_NAME, error))?;
        }

        let mut prepared = match conn.prepare(&case.query) {
            Ok(prepared) => prepared,
            Err(_) => return Ok(SqlOutcome::Error(ErrorClass::Other)),
        };

        let mut rows = match prepared.query([]) {
            Ok(rows) => rows,
            Err(_) => return Ok(SqlOutcome::Error(ErrorClass::Other)),
        };

        // **After** the query, not before. DuckDB's `Statement::column_count()` panics
        // with "The statement was not executed yet" if asked earlier, where SQLite's
        // answers happily — a difference between the two *drivers* that has nothing to do
        // with either engine's SQL, and exactly the sort of thing that would otherwise
        // surface later disguised as a divergence.
        let column_count = duckdb_column_count(&rows);

        let mut grid = Vec::new();
        loop {
            match rows.next() {
                Ok(Some(row)) => {
                    let mut cells = Vec::with_capacity(column_count);
                    for index in 0..column_count {
                        let value: duckdb::types::Value = row
                            .get(index)
                            .map_err(|error| setup_failed(DUCKDB_NAME, error))?;
                        cells.push(duckdb_cell(value)?);
                    }
                    grid.push(cells);
                }
                Ok(None) => break,
                Err(error) => return Err(setup_failed(DUCKDB_NAME, error)),
            }
        }

        Ok(SqlOutcome::Rows(grid))
    }
}

/// How many columns the executed query returned, per driver.
///
/// Both drivers hand back the executed statement through `Rows::as_ref()`, which is the
/// only place a column count is available *after* execution — and after is the only time
/// DuckDB will answer. `None` means no statement, which can only happen if nothing ran, so
/// zero columns is the truthful answer.
fn column_count_of(rows: &rusqlite::Rows<'_>) -> usize {
    rows.as_ref()
        .map_or(0, |statement| statement.column_count())
}

/// The DuckDB half of [`column_count_of`]. Two functions rather than one generic: the two
/// drivers share no trait, and a trait invented here to unify them would be more machinery
/// than the four lines it saves.
fn duckdb_column_count(rows: &duckdb::Rows<'_>) -> usize {
    rows.as_ref()
        .map_or(0, |statement| statement.column_count())
}

/// "We could not get an answer from this engine" — not "this engine is wrong".
fn setup_failed(implementation: &str, error: impl std::fmt::Display) -> RunError {
    RunError::Failed {
        implementation: implementation.to_string(),
        message: error.to_string(),
    }
}

/// "This engine returned something we cannot represent" — our gap, and a visible one.
///
/// Deliberately an error rather than a lossy conversion. Truncating a value we did not
/// expect would make two engines agree on a number neither of them returned, which is a
/// false *negative*: silent, and the kind that hides bugs rather than inventing them.
fn unrepresentable(implementation: &str, value: impl std::fmt::Debug) -> RunError {
    RunError::Failed {
        implementation: implementation.to_string(),
        message: format!(
            "cannot represent {value:?} as a cell; v1 handles NULL, integers and text"
        ),
    }
}

/// SQLite's value model is small: five storage classes, one of them `INTEGER`.
fn sqlite_cell(value: rusqlite::types::Value) -> Result<Cell, RunError> {
    use rusqlite::types::Value;
    match value {
        Value::Null => Ok(Cell::Null),
        Value::Integer(number) => Ok(Cell::Integer(number)),
        Value::Text(text) => Ok(Cell::Text(text)),
        // `Real` and `Blob` are reachable in SQLite but outside the generated subset.
        other => Err(unrepresentable(SQLITE_NAME, other)),
    }
}

/// DuckDB's value model is wide, and this is where the two engines stop looking alike.
///
/// SQLite has **one** integer storage class; DuckDB has eight, and an `INTEGER` column
/// comes back as `Int(i32)` rather than `BigInt(i64)`. Handling only `BigInt` would have
/// failed on every integer this domain generates — which is why the variants were read
/// off the crate rather than assumed.
///
/// Widening the signed widths into `i64` is safe: every value fits, so nothing is lost and
/// no difference is hidden. The unsigned and 128-bit variants are *not* widened — they can
/// exceed `i64`, so they are refused rather than truncated.
fn duckdb_cell(value: duckdb::types::Value) -> Result<Cell, RunError> {
    use duckdb::types::Value;
    match value {
        Value::Null => Ok(Cell::Null),
        Value::TinyInt(number) => Ok(Cell::Integer(number.into())),
        Value::SmallInt(number) => Ok(Cell::Integer(number.into())),
        Value::Int(number) => Ok(Cell::Integer(number.into())),
        Value::BigInt(number) => Ok(Cell::Integer(number)),
        Value::Text(text) => Ok(Cell::Text(text)),
        other => Err(unrepresentable(DUCKDB_NAME, other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The constants must equal what the implementations actually report.
    ///
    /// The whole point of the tensor domain's naming bug: a mismatch here shows up later
    /// as an empty result set, which reads as "nothing found" rather than as a typo.
    #[test]
    fn names_match_what_the_implementations_report() {
        assert_eq!(SqliteImpl.name(), SQLITE_NAME);
        assert_eq!(DuckDbImpl.name(), DUCKDB_NAME);
        assert_ne!(SQLITE_NAME, DUCKDB_NAME);
    }

    #[test]
    fn both_engines_run_the_fixed_case() {
        let case = SqlCase::fixed_example();

        let from_sqlite = SqliteImpl.run(&case).expect("sqlite runs the fixed case");
        let from_duckdb = DuckDbImpl.run(&case).expect("duckdb runs the fixed case");

        // Three of the four seeded rows survive `WHERE a > 0`.
        assert_eq!(from_sqlite.row_count(), Some(3));
        assert_eq!(from_duckdb.row_count(), Some(3));
    }

    /// The one assertion this step can honestly make about *agreement*.
    ///
    /// Comparing the two outcomes directly is not the oracle's job and does not prove the
    /// oracle works — but it does prove both drivers decode the same values the same way,
    /// including the `NULL` and the empty string, which is the thing most likely to differ
    /// between two drivers for reasons that have nothing to do with either engine.
    #[test]
    fn both_engines_decode_the_same_values() {
        let case = SqlCase::fixed_example();

        let from_sqlite = SqliteImpl.run(&case).expect("sqlite runs");
        let from_duckdb = DuckDbImpl.run(&case).expect("duckdb runs");

        assert_eq!(from_sqlite, from_duckdb);

        let SqlOutcome::Rows(rows) = from_sqlite else {
            panic!("the fixed case returns rows");
        };
        assert_eq!(rows[0], vec![Cell::Integer(1), Cell::Text("one".into())]);
        // An empty string, still an empty string and not a NULL.
        assert_eq!(rows[1], vec![Cell::Integer(2), Cell::Text(String::new())]);
        // And a NULL, still a NULL and not an empty string.
        assert_eq!(rows[2], vec![Cell::Integer(3), Cell::Null]);
    }

    /// A fresh database per call, proven rather than asserted.
    ///
    /// If any state leaked between calls, the second run would fail on `CREATE TABLE t`
    /// (the table would already exist). Running the same case twice is therefore a direct
    /// test of the property that lets the shared trait keep taking `&self`.
    #[test]
    fn each_run_gets_a_fresh_database() {
        let case = SqlCase::fixed_example();
        let engine = SqliteImpl;

        let first = engine.run(&case).expect("first run");
        let second = engine.run(&case).expect("second run — a fresh database");
        assert_eq!(first, second);

        let duck = DuckDbImpl;
        let first = duck.run(&case).expect("first run");
        let second = duck.run(&case).expect("second run — a fresh database");
        assert_eq!(first, second);
    }

    /// A query the engine refuses is an *answer*, not a failure to run.
    #[test]
    fn a_refused_query_is_an_outcome_not_a_run_error() {
        let case = SqlCase {
            schema: vec!["CREATE TABLE t (a INTEGER)".to_string()],
            data: vec![],
            // `no_such_column` resolves against nothing: both engines refuse it.
            query: "SELECT no_such_column FROM t".to_string(),
        };

        assert_eq!(
            SqliteImpl
                .run(&case)
                .expect("refusal is Ok(..), not Err(..)"),
            SqlOutcome::Error(ErrorClass::Other)
        );
        assert_eq!(
            DuckDbImpl
                .run(&case)
                .expect("refusal is Ok(..), not Err(..)"),
            SqlOutcome::Error(ErrorClass::Other)
        );
    }

    /// Setup failing is different: the query never ran, so there is nothing to compare.
    #[test]
    fn a_broken_schema_is_a_run_error() {
        let case = SqlCase {
            schema: vec!["CREATE TABLE (this is not sql".to_string()],
            data: vec![],
            query: "SELECT 1".to_string(),
        };

        assert!(matches!(
            SqliteImpl.run(&case),
            Err(RunError::Failed { .. })
        ));
        assert!(matches!(
            DuckDbImpl.run(&case),
            Err(RunError::Failed { .. })
        ));
    }
}
