//! What do the size bounds actually buy?
//!
//! Every widening so far has been about **constructs** — grouping, joins, set operations,
//! subqueries. The *size* bounds have never been varied: `max_rows = 8`, `max_columns = 4`,
//! `max_tables = 2` were written as starting points "to be measured" and never were.
//!
//! Size is not cosmetic here. More rows change join cardinalities, group sizes, and how often
//! a `NULL` meets a match; more columns widen every result and make ties in an `ORDER BY`
//! rarer. Both change *what can go wrong*, not just how much data flows through.
//!
//! The number that decides is **findings per second**, never the divergence rate alone: a
//! wider setting usually raises the rate and lowers throughput at once, and only their product
//! says which wins. This is the tensor domain's rule applied to the one axis this domain never
//! applied it to — where a setting 800x slower still won, because any multiple of zero is zero.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example bounds_sweep -- [cases per setting]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::NamedOutput;
use diff_fuzzer_core::traits::{Generator, Implementation, Normalizer, Oracle, Verdict};
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::normalize::SqlNormalizer;
use sql_adapter::oracle::{SortMode, SqlDifferentialOracle};
use std::time::Instant;

struct Outcome {
    agreed: usize,
    diverged: usize,
    skipped: usize,
    ordered: usize,
    seconds: f64,
}

fn run_setting(bounds: Bounds, cases: usize) -> Outcome {
    let generator = SqlGenerator::new(bounds);
    let (mut agreed, mut diverged, mut skipped, mut ordered) = (0, 0, 0, 0);

    let started = Instant::now();
    for seed in 0..cases as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if SortMode::for_case(&case) == SortMode::Ordered {
            ordered += 1;
        }

        let (Ok(left), Ok(right)) = (SqliteImpl.run(&case), DuckDbImpl.run(&case)) else {
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
        match SqlDifferentialOracle.check(&case, &outputs) {
            Verdict::Agree => agreed += 1,
            Verdict::Diverged(_) => diverged += 1,
            Verdict::Skipped(_) => skipped += 1,
        }
    }

    Outcome {
        agreed,
        diverged,
        skipped,
        ordered,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn main() {
    let cases: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(2_000);

    // One axis varied at a time against the combined baseline, so any difference is
    // attributable. Varying two at once is how "wide bounds find bugs" became a claim the
    // tensor domain could not support until it swept them separately.
    let settings: Vec<(String, Bounds)> = vec![
        ("baseline (rows<=8, cols<=4)".to_string(), Bounds::V1_ALL),
        (
            "rows<=32".to_string(),
            Bounds {
                max_rows: 32,
                ..Bounds::V1_ALL
            },
        ),
        (
            "rows<=128".to_string(),
            Bounds {
                max_rows: 128,
                ..Bounds::V1_ALL
            },
        ),
        (
            "cols<=8".to_string(),
            Bounds {
                max_columns: 8,
                ..Bounds::V1_ALL
            },
        ),
        (
            "rows<=32, cols<=8".to_string(),
            Bounds {
                max_rows: 32,
                max_columns: 8,
                ..Bounds::V1_ALL
            },
        ),
        (
            "depth<=5".to_string(),
            Bounds {
                max_expr_depth: 5,
                ..Bounds::V1_ALL
            },
        ),
    ];

    println!("{cases} cases per setting, one axis varied at a time\n");
    println!(
        "  {:<28} {:>8} {:>9} {:>8} {:>9} {:>10} {:>12}",
        "setting", "agreed", "diverged", "skipped", "ordered", "cases/sec", "findings/sec"
    );

    for (name, bounds) in settings {
        let outcome = run_setting(bounds, cases);
        let per_second = cases as f64 / outcome.seconds;
        let findings_per_second = outcome.diverged as f64 / outcome.seconds;
        println!(
            "  {:<28} {:>8} {:>9} {:>8} {:>8.0}% {:>10.0} {:>12.4}",
            name,
            outcome.agreed,
            outcome.diverged,
            outcome.skipped,
            100.0 * outcome.ordered as f64 / cases as f64,
            per_second,
            findings_per_second
        );
    }

    println!(
        "\nThe decision metric is the last column, not the divergence count: a wider setting\n\
         usually raises the rate and lowers throughput at once, and only their product says\n\
         which one a campaign should run at."
    );
}
