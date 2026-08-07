//! Does crossing DuckDB's execution chunk boundary change anything?
//!
//! # The question
//!
//! DuckDB passes data between operators in fixed-size batches whose default is **2048 tuples**
//! (`SPECS.md` §3.9, retrieved 2026-08-07). Every campaign this project has run used
//! `max_rows = 8` — **256× below one chunk** — so every query ever executed here fit inside a
//! single vector. Chunk-boundary handling, multi-chunk operators, hash-table growth and the
//! compressed vector formats have never once been reached, on an engine whose architecture *is*
//! chunked vectorized execution.
//!
//! This sweeps `max_rows` across that boundary and asks whether yield changes.
//!
//! # Why the base configuration is not `V1_ALL`
//!
//! **Joins and correlated subqueries are excluded, and it is not an oversight.** Both are
//! superlinear in row count:
//!
//! - A join of two 4,096-row tables can produce ~16.8 million result rows, which this harness
//!   materializes into `Vec<Vec<Cell>>`. That exhausts memory rather than measuring anything.
//! - A correlated subquery is re-evaluated per outer row, so 4,096 rows means ~16.8 million
//!   inner evaluations.
//!
//! What is kept is what actually tests the hypothesis: **aggregates** (hash aggregation builds a
//! table that grows across chunks) and **set operations** (deduplication compares across the
//! whole input, not within one batch). Both are multi-chunk operators and both stay linear.
//!
//! The size×join interaction is real and still needs testing — with a cap on result size — but
//! that belongs in the combined-configuration check, not here, where it would confound the one
//! variable being swept.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example chunk_sweep -- [cases per setting]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::minimize::{Budget, minimize_within};
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Normalizer, Oracle, Verdict,
};
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::metamorphic::{Relation, check, check_aggregate, partition, partition_aggregate};
use sql_adapter::normalize::SqlNormalizer;
use sql_adapter::oracle::{SortMode, SqlDifferentialOracle};
use sql_adapter::outcome::SqlOutcome;
use std::time::Instant;

/// One row of the sweep table.
struct Measurement {
    rows: usize,
    agreed: usize,
    diverged: usize,
    skipped: usize,
    tlp_checked: usize,
    tlp_violations: usize,
    ordered: usize,
    /// Total wall time for the differential half.
    seconds: f64,
    /// Mean time to open a fresh DuckDB connection *and* load the data, in milliseconds.
    /// `PENDING` 1.3 measured 3.98 ms at 8 rows and found it was 86% of per-case cost; this
    /// says whether that still holds when the data is real.
    duckdb_ms: f64,
    /// Time to minimize one diverging-shaped case, in milliseconds. Shrinking has only ever
    /// been exercised on 8 rows, and its cost scales with the data.
    shrink_ms: f64,
    /// Candidates the minimizer evaluated, so the millisecond figure can be read per candidate
    /// rather than as one opaque number.
    shrink_candidates: usize,
}

fn main() {
    let cases: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);

    // Aggregates and set operations only — see the module docs for why joins and correlated
    // subqueries are excluded from a *size* sweep.
    let base = Bounds {
        aggregates: true,
        set_ops: true,
        ..Bounds::V1
    };

    // Spanning the 2048 boundary in both directions, so "did crossing it matter?" is answerable
    // rather than inferred from one side.
    let settings = [8usize, 100, 1_000, 2_048, 4_096];

    println!("chunk boundary: DuckDB STANDARD_VECTOR_SIZE = 2048 tuples (SPECS.md §3.9)");
    println!("base: {}", SqlGenerator::new(base).description());
    println!("cases per setting: {cases}\n");

    let mut table = Vec::new();
    for rows in settings {
        let bounds = Bounds {
            max_rows: rows,
            ..base
        };
        eprintln!("  running max_rows={rows} …");
        table.push(measure(bounds, rows, cases));
    }

    println!(
        "  {:>7} {:>9} {:>9} {:>8} {:>9} {:>10} {:>8} {:>10} {:>10} {:>10} {:>9}",
        "rows", "agreed", "diverged", "skipped", "tlp-chk", "tlp-viol", "ordered", "cases/s",
        "duckdb ms", "shrink ms", "shr cand"
    );
    for m in &table {
        println!(
            "  {:>7} {:>9} {:>9} {:>8} {:>9} {:>10} {:>7.0}% {:>10.1} {:>10.2} {:>10.1} {:>9}",
            m.rows,
            m.agreed,
            m.diverged,
            m.skipped,
            m.tlp_checked,
            m.tlp_violations,
            100.0 * m.ordered as f64 / (m.agreed + m.diverged + m.skipped).max(1) as f64,
            (m.agreed + m.diverged + m.skipped) as f64 / m.seconds,
            m.duckdb_ms,
            m.shrink_ms,
            m.shrink_candidates,
        );
    }

    println!("\nreading");
    let below: usize = table.iter().filter(|m| m.rows < 2048).map(|m| m.diverged + m.tlp_violations).sum();
    let at_or_above: usize = table.iter().filter(|m| m.rows >= 2048).map(|m| m.diverged + m.tlp_violations).sum();
    if below == 0 && at_or_above == 0 {
        println!(
            "  Nothing on either side of the boundary. Crossing 2048 did not change the yield,\n  \
             which is evidence the chunked-execution hypothesis is wrong *for this subset* —\n  \
             not that larger data is pointless, since joins and subqueries were excluded."
        );
    } else {
        println!("  below 2048: {below} events. at/above 2048: {at_or_above} events. Investigate.");
    }
    let slowest = table.last().expect("settings is non-empty");
    let fastest = table.first().expect("settings is non-empty");
    println!(
        "  Throughput cost of the largest setting: {:.0}x slower than 8 rows.",
        (fastest.agreed + fastest.diverged + fastest.skipped) as f64 / fastest.seconds
            / (((slowest.agreed + slowest.diverged + slowest.skipped) as f64 / slowest.seconds).max(0.001))
    );
    println!(
        "  Findings per second is the number that decides, and a multiple of zero is zero —\n  \
         so with no yield at any setting, the cheapest setting wins by default."
    );
}

fn measure(bounds: Bounds, rows: usize, cases: usize) -> Measurement {
    let generator = SqlGenerator::new(bounds);
    let oracle = SqlDifferentialOracle;

    let (mut agreed, mut diverged, mut skipped, mut ordered) = (0, 0, 0, 0);
    let (mut tlp_checked, mut tlp_violations) = (0, 0);
    let mut duckdb_total = 0.0f64;

    let started = Instant::now();
    for seed in 0..cases as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if SortMode::for_case(&case) == SortMode::Ordered {
            ordered += 1;
        }

        let duck_started = Instant::now();
        let right = DuckDbImpl.run(&case);
        duckdb_total += duck_started.elapsed().as_secs_f64() * 1000.0;

        let (Ok(left), Ok(right)) = (SqliteImpl.run(&case), right) else {
            skipped += 1;
            continue;
        };

        let outputs = [
            NamedOutput {
                implementation: "sqlite".to_string(),
                output: SqlNormalizer.normalize(left),
            },
            NamedOutput {
                implementation: "duckdb".to_string(),
                output: SqlNormalizer.normalize(right),
            },
        ];
        match oracle.check(&case, &outputs) {
            Verdict::Agree => agreed += 1,
            Verdict::Diverged { .. } => diverged += 1,
            Verdict::Skipped { .. } => skipped += 1,
        }

        // The metamorphic half, on DuckDB only — this sweep is about DuckDB's execution model,
        // and SQLite is not vectorized so it has no chunk boundary to cross.
        let run = |c: &_| -> Option<SqlOutcome> { DuckDbImpl.run(c).ok() };
        let relation = if let Some(parts) = partition(&case) {
            match (run(&parts.whole), run(&parts.is_true), run(&parts.is_false), run(&parts.is_unknown)) {
                (Some(w), Some(t), Some(f), Some(u)) => Some(check(&w, &t, &f, &u)),
                _ => None,
            }
        } else if let Some(parts) = partition_aggregate(&case) {
            match (run(&parts.whole), run(&parts.is_true), run(&parts.is_false), run(&parts.is_unknown)) {
                (Some(w), Some(t), Some(f), Some(u)) => Some(check_aggregate(parts.func, &w, &t, &f, &u)),
                _ => None,
            }
        } else {
            None
        };
        match relation {
            Some(Relation::Holds) => tlp_checked += 1,
            Some(Relation::Violated { .. }) => tlp_violations += 1,
            _ => {}
        }
    }
    let seconds = started.elapsed().as_secs_f64();

    // Shrinking cost at this size, on one representative case. Minimization walks candidate
    // reductions and re-runs a predicate on each, so its cost scales with the data — and it has
    // only ever been exercised on 8 rows. A campaign that finds something at 4,096 rows and then
    // cannot minimize it in reasonable time has a finding it cannot report.
    let case = generator.generate(&mut SeededRng::from_seed(0));
    let shrink_started = Instant::now();
    // The **real** minimizer, not a hand-rolled loop — measuring a stand-in would measure the
    // stand-in. "Still fails" is stubbed as "still runs on DuckDB", which is true of every
    // candidate and so drives the search to its budget: the worst case, which is the one worth
    // knowing. The budget is capped well below the default so one setting cannot run away.
    let budget = Budget {
        max_candidates: 200,
        ..Budget::default()
    };
    let minimized = minimize_within(case, budget, |candidate| DuckDbImpl.run(candidate).is_ok());
    let shrink_ms = shrink_started.elapsed().as_secs_f64() * 1000.0;
    let shrink_candidates = minimized.candidates_tried;

    Measurement {
        rows,
        agreed,
        diverged,
        skipped,
        tlp_checked,
        tlp_violations,
        ordered,
        seconds,
        duckdb_ms: duckdb_total / cases as f64,
        shrink_ms,
        shrink_candidates,
    }
}
