//! What a test case *is* in this domain.
//!
//! Not one call — a whole small database program: a schema to create, rows to insert, and
//! one query to run. All three travel together in a [`SqlCase`], and that is the property
//! the entire design rests on. Because the case carries its own world, running it means
//! opening a database, using it, and dropping it *inside a single call*; nothing persists
//! between cases; so the shared engine's `Implementation::run(&self, ..)` never needs to
//! become `&mut self`. The statefulness that looked like this domain's hard problem
//! dissolves into the shape of the case.
//!
//! # This is deliberately text, and deliberately temporary
//!
//! At this stage the three parts are SQL **strings**, hand-written. That is enough to get
//! one case flowing through every seam, which is all a walking skeleton owes anyone.
//!
//! S2 replaces them with a typed tree, and the reason is worth stating now, because it is
//! the argument for owning an AST at all:
//!
//! - **Shrinking needs structure.** Minimizing a finding means dropping a predicate,
//!   a column, a row. On a `String` those are text edits that can produce SQL which no
//!   longer parses; on a tree they are node removals that cannot.
//! - **Two dialects need one meaning.** The same case must be rendered as SQLite spells it
//!   and as DuckDB spells it. From a tree that is two printers. From text it is search and
//!   replace, which is how a "translation" quietly changes what the query asks.
//! - **Features need to be read off the case.** Later phases ask questions like "does this
//!   case put a `NULL` in a comparison?" — cheap on a tree, string-matching on text.

use diff_fuzzer_core::traits::Input;
use serde::{Deserialize, Serialize};

/// One self-contained SQL test case: schema, seed data, and the query under test.
///
/// `Clone` because minimization repeatedly produces modified copies; `Debug` because a
/// case that cannot be printed cannot be reported; `Serialize`/`Deserialize` because the
/// *whole case* is what gets written to a finding. Not the seed — a seed only reproduces
/// a case for the exact generator that produced it, and generators change. The tensor
/// domain learned that when 810 of 814 recorded findings stopped reproducing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlCase {
    /// `CREATE TABLE` statements, run first and in order.
    pub schema: Vec<String>,
    /// `INSERT` statements, run after the schema and in order.
    pub data: Vec<String>,
    /// The single `SELECT` under test. Exactly one, so a divergence names one query.
    pub query: String,
}

/// Marks `SqlCase` as a test case for the engine.
///
/// The trait has no methods — it exists so the other seams can say `type In: Input` and be
/// sure whatever flows through them can be cloned, printed, and therefore reported.
impl Input for SqlCase {}

impl SqlCase {
    /// Every statement of the case, in execution order: schema, then data, then the query.
    ///
    /// One definition of "in order", used by both engines, so the two can never disagree
    /// because one of them applied the case differently. That would be a divergence
    /// manufactured by this crate — the worst kind, since it looks exactly like a finding.
    pub fn statements(&self) -> impl Iterator<Item = &str> {
        self.schema
            .iter()
            .chain(self.data.iter())
            .map(String::as_str)
            .chain(std::iter::once(self.query.as_str()))
    }

    /// A small fixed case, used until S2's generator exists.
    ///
    /// Chosen to exercise the things that break result comparison first: a `NULL` in the
    /// data, an empty string alongside it (the two must never render alike), a row the
    /// `WHERE` excludes, and a total `ORDER BY` so the row order is genuinely part of the
    /// answer rather than something each engine may choose for itself.
    pub fn fixed_example() -> Self {
        Self {
            schema: vec!["CREATE TABLE t (a INTEGER, b TEXT)".to_string()],
            data: vec![
                "INSERT INTO t VALUES (1, 'one'), (2, ''), (3, NULL), (-1, 'neg')".to_string(),
            ],
            query: "SELECT a, b FROM t WHERE a > 0 ORDER BY a".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statements_run_schema_then_data_then_query() {
        let case = SqlCase::fixed_example();
        let statements: Vec<&str> = case.statements().collect();

        assert_eq!(statements.len(), 3);
        assert!(statements[0].starts_with("CREATE TABLE"));
        assert!(statements[1].starts_with("INSERT INTO"));
        assert!(statements[2].starts_with("SELECT"));
        // The query is last: it must see the whole schema and all the data.
        assert_eq!(statements[2], case.query);
    }

    #[test]
    fn a_case_survives_a_round_trip_through_json() {
        // The property a finding depends on. If a case cannot be written and read back
        // unchanged, a saved divergence is a story rather than a reproduction.
        let case = SqlCase::fixed_example();
        let json = serde_json::to_string(&case).expect("a case serializes");
        let back: SqlCase = serde_json::from_str(&json).expect("and deserializes");
        assert_eq!(case, back);
    }

    #[test]
    fn the_fixed_case_carries_the_awkward_values() {
        // Guarding intent, not syntax: this case exists to contain a NULL and an empty
        // string. If someone "tidies" them away, the first thing the pipeline stops
        // testing is the thing most likely to break.
        let data = SqlCase::fixed_example().data.join(" ");
        assert!(data.contains("NULL"), "the fixed case must contain a NULL");
        assert!(
            data.contains("''"),
            "and an empty string, which is not NULL"
        );
    }
}
