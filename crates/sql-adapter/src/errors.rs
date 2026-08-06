//! Turning an engine's complaint into a class.
//!
//! Engines phrase the same objection differently. Overflowing an addition, DuckDB says
//! *"Out of Range Error: Overflow in addition of INT64 (9223372036854775807 + 1)!"*; a
//! hypothetical other engine might say *"integer overflow"*. Comparing those strings would
//! report **wording** as a divergence, which is both wrong and unfixable — there is no
//! canonical form for prose.
//!
//! So an error is compared by class. The classes are deliberately few: this is not an
//! attempt to model every way a query can fail, only to distinguish the failures that mean
//! genuinely different things.
//!
//! # The rule this module exists to obey
//!
//! **A classifier must be able to return every class it claims to have.** The tensor domain
//! shipped one that could never produce one of its five variants, so those cases silently
//! became "unknown" and were discarded — throwing away the strongest evidence in the set,
//! and looking like missing data rather than a bug.
//!
//! The tests below therefore obtain their messages **from the engines, at test time**,
//! rather than from hardcoded strings. A hardcoded message tests that the classifier matches
//! a string someone once saw; a live one tests that it matches what the engine says *today*,
//! and fails when a version bump changes the wording.

use crate::outcome::ErrorClass;

/// Classify an engine's error message.
///
/// Matching is on lowercased substrings, which is crude and appropriate: the alternative is
/// parsing two engines' error formats, and the classes are coarse enough that a substring
/// carries the distinction. Anything unrecognised is [`ErrorClass::Other`] — never a guess.
pub fn classify(message: &str) -> ErrorClass {
    let message = message.to_lowercase();

    // Order matters where messages could match more than one rule. Overflow is checked
    // before conversion because DuckDB's overflow message also contains a type name.
    if message.contains("out of range") || message.contains("overflow") {
        ErrorClass::OutOfRange
    } else if message.contains("division by zero") || message.contains("divide by zero") {
        ErrorClass::DivideByZero
    } else if message.contains("conversion error")
        || message.contains("could not convert")
        || message.contains("cannot be cast")
        || message.contains("mismatched types")
        || message.contains("datatype mismatch")
    {
        ErrorClass::TypeMismatch
    } else {
        ErrorClass::Other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ask an engine to do something it will refuse, and return what it says.
    fn sqlite_error(sql: &str) -> String {
        let conn = rusqlite::Connection::open_in_memory().expect("open");
        conn.query_row(sql, [], |row| row.get::<_, rusqlite::types::Value>(0))
            .expect_err("this SQL is meant to fail")
            .to_string()
    }

    fn duckdb_error(sql: &str) -> String {
        let conn = duckdb::Connection::open_in_memory().expect("open");
        conn.query_row(sql, [], |row| row.get::<_, duckdb::types::Value>(0))
            .expect_err("this SQL is meant to fail")
            .to_string()
    }

    /// `OutOfRange` is reachable, from a real message.
    #[test]
    fn overflow_is_classified_from_a_live_message() {
        let message = duckdb_error("SELECT 9223372036854775807 + 1");
        assert_eq!(
            classify(&message),
            ErrorClass::OutOfRange,
            "unclassified: {message}"
        );
    }

    /// `TypeMismatch` is reachable, from a real message.
    #[test]
    fn a_bad_cast_is_classified_from_a_live_message() {
        let message = duckdb_error("SELECT CAST('abc' AS INTEGER)");
        assert_eq!(
            classify(&message),
            ErrorClass::TypeMismatch,
            "unclassified: {message}"
        );
    }

    /// `Other` is reachable, and *should* be: an unresolvable column is a real failure with
    /// no more specific class, and inventing one would be worse than admitting it.
    #[test]
    fn an_unknown_column_falls_through_to_other() {
        assert_eq!(classify(&sqlite_error("SELECT nope")), ErrorClass::Other);
        assert_eq!(
            classify(&duckdb_error("SELECT nope FROM (SELECT 1)")),
            ErrorClass::Other
        );
    }

    /// `DivideByZero` is the awkward one, and the awkwardness is the finding.
    ///
    /// **Neither engine raises on `1/0`.** Measured: SQLite returns `NULL`, DuckDB returns
    /// `inf`. So this class is currently reachable only from a message no engine under test
    /// produces — which is exactly the situation the tensor domain's unreachable-variant bug
    /// was about, and it is recorded rather than hidden (`SPECS.md` §4.6, `PENDING` 2.12).
    ///
    /// The class is kept because the *rule* is what is being tested: if a future engine or
    /// version does raise, it must classify correctly rather than fall into `Other`.
    #[test]
    fn divide_by_zero_classifies_although_neither_engine_currently_raises() {
        assert_eq!(
            classify("Division by zero!"),
            ErrorClass::DivideByZero,
            "the rule must work even though no engine under test exercises it"
        );

        // The measured behaviour, asserted so that a version bump which starts raising is
        // noticed here rather than as a mysterious divergence in a campaign.
        let sqlite = rusqlite::Connection::open_in_memory().expect("open");
        let by_zero: rusqlite::types::Value = sqlite
            .query_row("SELECT 1/0", [], |row| row.get(0))
            .expect("sqlite does not raise on 1/0");
        assert_eq!(by_zero, rusqlite::types::Value::Null, "sqlite returns NULL");

        let duckdb = duckdb::Connection::open_in_memory().expect("open");
        let by_zero: duckdb::types::Value = duckdb
            .query_row("SELECT 1/0", [], |row| row.get(0))
            .expect("duckdb does not raise on 1/0");
        assert!(
            matches!(by_zero, duckdb::types::Value::Double(value) if value.is_infinite()),
            "duckdb returns inf, got {by_zero:?}"
        );
    }

    #[test]
    fn classification_ignores_case_and_surrounding_prose() {
        assert_eq!(classify("OUT OF RANGE"), ErrorClass::OutOfRange);
        assert_eq!(
            classify("something something Could Not Convert something"),
            ErrorClass::TypeMismatch
        );
        assert_eq!(classify(""), ErrorClass::Other);
    }
}
