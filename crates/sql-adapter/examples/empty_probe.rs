//! How often does a configuration return **no rows at all**?
//!
//! Added at S9.13, which measured 61.8% empty on `V1_ALL` — two engines both returning nothing
//! always agree, so most of a campaign was testing essentially nothing. This is the measurement
//! that says whether a generator change fixed it.
use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use sql_adapter::backends::DuckDbImpl;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::outcome::SqlOutcome;

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000);
    println!(
        "  {:<24} {:>18} {:>12}",
        "configuration", "empty results", "mean rows"
    );
    for (name, bounds) in [
        ("V1", Bounds::V1),
        ("V1_ALL", Bounds::V1_ALL),
        ("V1_NOT_IN", Bounds::V1_NOT_IN),
        ("V1_NOT_IN_LIST", Bounds::V1_NOT_IN_LIST),
        ("V1_JOINS", Bounds::V1_JOINS),
        ("V1_AGGREGATES", Bounds::V1_AGGREGATES),
    ] {
        let generator = SqlGenerator::new(bounds);
        let (mut empty, mut ran, mut rows_total) = (0usize, 0usize, 0usize);
        for seed in 0..total as u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if let Ok(SqlOutcome::Rows(rows)) = DuckDbImpl.run(&case) {
                ran += 1;
                rows_total += rows.len();
                if rows.is_empty() {
                    empty += 1;
                }
            }
        }
        println!(
            "  {:<24} {:>9} ({:>5.1}%) {:>12.2}",
            name,
            empty,
            100.0 * empty as f64 / ran as f64,
            rows_total as f64 / ran as f64
        );
    }
}
