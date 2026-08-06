//! Where the time goes in one case, so the fresh-database decision is measured not argued.
//!
//! `PENDING` 1.3 asks whether a fresh in-memory database per case is affordable, or whether
//! a reused connection reset between cases is needed. Reuse is the riskier design — an
//! incomplete reset leaks state between cases and produces phantom divergences — so it
//! needs evidence, not a guess.

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::generator::SqlGenerator;
use std::time::Instant;

fn time<T>(label: &str, count: usize, mut work: impl FnMut(u64) -> T) {
    let started = Instant::now();
    for index in 0..count as u64 {
        std::hint::black_box(work(index));
    }
    let each = started.elapsed().as_secs_f64() * 1000.0 / count as f64;
    println!("  {label:<34} {each:>7.3} ms");
}

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2000);
    let generator = SqlGenerator::default();
    println!("per-case cost, averaged over {count} cases\n");

    time("generate a case", count, |seed| {
        generator.generate(&mut SeededRng::from_seed(seed))
    });
    time("open a sqlite database", count, |_| {
        rusqlite::Connection::open_in_memory().unwrap()
    });
    time("open a duckdb database", count, |_| {
        duckdb::Connection::open_in_memory().unwrap()
    });

    let cases: Vec<_> = (0..count as u64)
        .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
        .collect();
    time("run on sqlite (open + apply + query)", count, |i| {
        SqliteImpl.run(&cases[i as usize])
    });
    time("run on duckdb (open + apply + query)", count, |i| {
        DuckDbImpl.run(&cases[i as usize])
    });
}
