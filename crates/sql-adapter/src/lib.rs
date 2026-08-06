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

// Nothing is implemented yet: this step exists to prove the crate joins the workspace
// and builds without disturbing anything already here. The types arrive next.
