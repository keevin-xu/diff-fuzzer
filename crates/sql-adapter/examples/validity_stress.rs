//! Does the generator produce cases both engines can actually run?
//!
//! The claim being tested is **validity**, not agreement: every generated case should parse,
//! resolve, and execute on SQLite *and* DuckDB. A case that one engine refuses is not a
//! finding — it is a case that was never judged, and a generator producing them spends the
//! budget proving that parsers work.
//!
//! Reports, per engine:
//!
//! - **ran** — the engine executed the query and returned rows.
//! - **refused** — the engine ran the query and rejected it (`SqlOutcome::Error`). Legitimate
//!   for arithmetic overflow; suspicious for anything else, so a sample is printed.
//! - **could not run** — setup failed, so the case never reached the query. **This is the
//!   number that must be zero**: it means the generator emitted something an engine cannot
//!   even accept.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example validity_stress -- [cases]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::generator::SqlGenerator;
use sql_adapter::outcome::SqlOutcome;
use std::time::Instant;

#[derive(Default)]
struct Tally {
    ran: usize,
    refused: usize,
    could_not_run: usize,
    examples: Vec<String>,
}

impl Tally {
    fn record(
        &mut self,
        outcome: Result<SqlOutcome, diff_fuzzer_core::traits::RunError>,
        seed: u64,
    ) {
        match outcome {
            Ok(SqlOutcome::Rows(_)) => self.ran += 1,
            Ok(SqlOutcome::Error(_)) => {
                self.refused += 1;
                if self.examples.len() < 5 {
                    self.examples.push(format!("seed {seed}: query refused"));
                }
            }
            Err(error) => {
                self.could_not_run += 1;
                if self.examples.len() < 5 {
                    self.examples.push(format!("seed {seed}: {error}"));
                }
            }
        }
    }

    fn report(&self, engine: &str, total: usize) {
        let percent = |count: usize| 100.0 * count as f64 / total as f64;
        println!(
            "  {engine:<8} ran {:>6} ({:>5.1}%)   refused {:>5} ({:>4.1}%)   could-not-run {:>5} ({:>4.1}%)",
            self.ran,
            percent(self.ran),
            self.refused,
            percent(self.refused),
            self.could_not_run,
            percent(self.could_not_run),
        );
        for example in &self.examples {
            println!("      {example}");
        }
    }
}

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse().ok())
        .unwrap_or(10_000);

    let bounds = match std::env::args().nth(2).as_deref() {
        Some("aggregates") => sql_adapter::gen_schema::Bounds::V1_AGGREGATES,
        Some("setops") => sql_adapter::gen_schema::Bounds::V1_SET_OPS,
        Some("chained") => sql_adapter::gen_schema::Bounds::V1_CHAINED_SET_OPS,
        Some("joins") => sql_adapter::gen_schema::Bounds::V1_JOINS,
        Some("wide") => sql_adapter::gen_schema::Bounds::V1_WIDE_ARITHMETIC,
        _ => sql_adapter::gen_schema::Bounds::V1,
    };
    let generator = SqlGenerator::new(bounds);
    println!("generator: {}", generator.description());
    println!("cases:     {total}");

    let mut sqlite = Tally::default();
    let mut duckdb = Tally::default();
    let mut invalid = 0usize;

    let started = Instant::now();
    for seed in 0..total as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        // The generator's own claim, checked before either engine sees the case. A failure
        // here is a bug in generation, not in an engine.
        if let Err(problem) = case.validate() {
            invalid += 1;
            if invalid <= 5 {
                println!("  INVALID seed {seed}: {problem}");
            }
        }

        sqlite.record(SqliteImpl.run(&case), seed);
        duckdb.record(DuckDbImpl.run(&case), seed);
    }
    let elapsed = started.elapsed();

    println!("\nvalidity");
    sqlite.report("sqlite", total);
    duckdb.report("duckdb", total);
    println!("  cases failing their own validate(): {invalid}");

    println!("\nthroughput");
    println!("  {:.1}s total", elapsed.as_secs_f64());
    println!(
        "  {:.0} cases/sec (both engines, fresh in-memory database each)",
        total as f64 / elapsed.as_secs_f64()
    );
    println!(
        "  {:.2} ms per case",
        elapsed.as_secs_f64() * 1000.0 / total as f64
    );

    let could_not_run = sqlite.could_not_run + duckdb.could_not_run;
    if could_not_run == 0 && invalid == 0 {
        println!("\nevery case was valid and ran on both engines.");
    } else {
        println!(
            "\n{could_not_run} case-runs could not execute and {invalid} failed validate() — \
             the generator is producing cases an engine cannot accept."
        );
    }
}
