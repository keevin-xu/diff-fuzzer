//! Differential testing of SQL query engines: **SQLite** against **DuckDB**.
//!
//! This is the project's second domain. The first (tensor operations across `burn`
//! backends) lives in `tensor-adapter`, and the two are siblings: both implement the
//! five seams in [`diff_fuzzer_core::traits`], neither knows the other exists.
//!
//! # What a test case is here
//!
//! A whole small database program, not a single call:
//!
//! ```text
//! CREATE TABLE t (a INTEGER, b TEXT);   -- schema
//! INSERT INTO t VALUES (1, 'x'), ...;   -- seed data
//! SELECT a FROM t WHERE a > 0;          -- the query being tested
//! ```
//!
//! All three parts travel together in one `SqlCase`, which is why this domain needs no
//! change to the shared engine. The mutable thing — a database connection — is created
//! and dropped *inside* one `run()` call, so nothing carries between cases and
//! `Implementation::run(&self, ..)` keeps taking `&self`.
//!
//! # How two engines are compared
//!
//! Run the same case on both, render each result set to canonical text (sqllogictest's
//! rules), and compare. The comparison is **discrete** — equal or not — with no
//! tolerance anywhere, which is the main way this domain is simpler than tensors.
//! Differences that are legal rather than wrong (two correct engines are allowed to
//! disagree about some things) are handled in one of two ways, never by loosening the
//! comparison: either the generator never produces them, or a cited catalog entry marks
//! them as known-legal.
//!
//! # Where things are
//!
//! - `planning/sql-duckdb/` — the plan, its gates, and its open questions.
//! - `crates/sql-adapter/DECISIONS.md` — this domain's decision ledger.
//! - `POLICY.md` / `SPECS.md` (next to this crate's `Cargo.toml`) — the legal-difference
//!   decisions, and the cited evidence behind them, kept as separate documents on
//!   purpose.

/// What a test case is: schema, seed rows, and one query, carried together.
pub mod ast;
/// The two engines under comparison, each behind the shared `Implementation` seam.
pub mod backends;
/// Running one case end to end: generate, run both engines, normalize, judge.
pub mod driver;
/// Turning an engine's complaint into a class, so wording is never compared.
pub mod errors;
/// Generating the query, against a schema and data that already exist.
pub mod gen_query;
/// Generating the state a query runs against: tables, and the rows in them.
pub mod gen_schema;
/// Producing cases to test. A placeholder until S2.
pub mod generator;
/// Turning what an engine returned into a comparable canonical form.
pub mod normalize;
/// Deciding whether the engines disagreed.
pub mod oracle;
/// Whether a query's `ORDER BY` actually orders the rows of this case.
pub mod ordering;
/// What running a case produces: rows, or a refusal.
pub mod outcome;
/// Turning the tree into SQL text, once per engine.
pub mod render;
/// The typed tree a case is made of: types, tables, values, expressions, one query.
pub mod schema;
/// An engine that is wrong on purpose, so "found nothing" can mean something.
pub mod testing;

/// Where this domain's outputs live, relative to the repository root.
///
/// **One constant, so a domain is a constant rather than a search-and-replace.** Findings,
/// negatives and their archives all hang off it, and nothing else in this crate spells a
/// path prefix — the same rule the tensor adapter follows with `findings/tensor`.
///
/// Scoped by domain because the file *names* would otherwise collide while the file
/// *contents* are unrelated: a `join-3f2a.json` and a `matmul-3f2a.json` share nothing but
/// a hash.
pub const FINDINGS_ROOT: &str = "findings/sql";

/// Where this domain's non-diverging cases live.
///
/// A case that was judged and **agreed** is evidence, not absence of evidence: a later
/// claim about what triggers a divergence is only worth something if it can be scored
/// against cases that did *not* diverge.
pub const NEGATIVES_ROOT: &str = "findings/sql/negatives";

#[cfg(test)]
mod tests {
    use super::*;

    /// The negatives directory must sit *under* the findings root.
    ///
    /// Trivial to state and trivial to break: these are two separate string literals, and
    /// nothing but this test ties them together. The tensor domain learned the general
    /// lesson the expensive way — a value matched by string equality needs a single
    /// definition, or the mismatch shows up as *missing data* rather than as a typo, which
    /// is how it survives review.
    #[test]
    fn negatives_live_under_findings_root() {
        assert!(
            NEGATIVES_ROOT.starts_with(FINDINGS_ROOT),
            "negatives root {NEGATIVES_ROOT} must be under findings root {FINDINGS_ROOT}"
        );
    }

    /// This domain must not write where the tensor domain writes.
    #[test]
    fn findings_root_is_domain_scoped() {
        assert_eq!(FINDINGS_ROOT, "findings/sql");
        assert_ne!(FINDINGS_ROOT, "findings/tensor");
    }
}
