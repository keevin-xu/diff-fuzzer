//! Hunt with the metamorphic oracle: check each engine against **itself**.
//!
//! The differential campaign asks "do these two engines agree?". This asks "is this engine
//! self-consistent?", which reaches the one class the other cannot — a bug **both** engines
//! share, where agreement is not evidence of correctness.
//!
//! For each case it builds four queries — the whole, and the TRUE/FALSE/UNKNOWN partitions of
//! its predicate — runs all four on **one** engine, and checks that the partitions reconstruct
//! the whole. Then it does the same for the other engine, separately. A violation names one
//! engine; no comparison between them is involved at any point.
//!
//! # Which engine to point it at
//!
//! **DuckDB by default.** A metamorphic oracle spends every execution on one target, so the
//! choice matters more than it does for a differential run. SQLite is the most-deployed
//! database in the world with a correspondingly heavy test suite; DuckDB is younger and far
//! less mined, which is the premise this whole pairing was chosen on (`planning/13` §0, from
//! Rigger's DBMS bug ledger).
//!
//! **The reason is yield, not speed** — a correction, because the first version of this comment
//! claimed dropping SQLite would "roughly double throughput". Measured: 76 → 83 cases/sec,
//! about 9%. A SQLite run costs 0.049 ms against DuckDB's 4.6 ms, so removing it saves almost
//! nothing. Running both is nearly free, which is a good argument for keeping the control.
//!
//! **SQLite stays available as a control, and that is not optional.** If TLP violates on
//! DuckDB and holds on SQLite, that is a strong signal. If it violates on *both*, the far
//! likelier explanation is that our transform is wrong — and without the ability to run the
//! comparison, there is no way to tell those apart.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example tlp_hunt -- [cases] [label] [setting] [engine]
//!
//! `engine` is `duckdb` (default), `sqlite`, or `both`.

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use sql_adapter::FINDINGS_ROOT;
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::metamorphic::{Partitioned, Relation, check, partition};
use sql_adapter::outcome::SqlOutcome;
use sql_adapter::render::Dialect;
use std::time::Instant;

/// Run the four variants on one engine and check the relation.
fn judge(engine: &str, parts: &Partitioned) -> Relation {
    let run = |case: &SqlCase| -> Option<SqlOutcome> {
        match engine {
            "sqlite" => SqliteImpl.run(case).ok(),
            _ => DuckDbImpl.run(case).ok(),
        }
    };

    let (Some(whole), Some(is_true), Some(is_false), Some(is_unknown)) = (
        run(&parts.whole),
        run(&parts.is_true),
        run(&parts.is_false),
        run(&parts.is_unknown),
    ) else {
        return Relation::NotChecked("a variant could not be run");
    };

    check(&whole, &is_true, &is_false, &is_unknown)
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let total: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20_000);
    let label = arguments.next().unwrap_or_else(|| "tlp".to_string());
    let bounds = match arguments.next().as_deref() {
        Some("all") => Bounds::V1_ALL,
        Some("joins") => Bounds::V1_JOINS,
        Some("subqueries") => Bounds::V1_SUBQUERIES,
        // **The default is V1, not V1_ALL, deliberately.** TLP refuses aggregates, grouping,
        // set operations and `LIMIT` because the relation does not hold for them, so the
        // combined configuration would leave most cases unjudged. Running a metamorphic
        // campaign on the differential campaign's settings would mean discarding ~84% of it.
        _ => Bounds::V1,
    };

    // The target under test. DuckDB alone by default — see the module docs.
    let engines: Vec<&str> = match arguments.next().as_deref() {
        Some("sqlite") => vec!["sqlite"],
        Some("both") => vec!["sqlite", "duckdb"],
        _ => vec!["duckdb"],
    };

    let generator = SqlGenerator::new(bounds);
    let directory = format!("{FINDINGS_ROOT}/runs/{label}");

    println!("generator: {}", generator.description());
    println!("cases:     {total}");
    println!("engines:   {}", engines.join(", "));
    println!("oracle:    TLP, each engine against itself\n");

    let (mut checked, mut skipped, mut not_partitionable) = (0usize, 0usize, 0usize);
    let mut violations = 0usize;

    let started = Instant::now();
    for seed in 0..total as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        let Some(parts) = partition(&case) else {
            not_partitionable += 1;
            continue;
        };

        for engine in &engines {
            match judge(engine, &parts) {
                Relation::Holds => checked += 1,
                Relation::NotChecked(_) => skipped += 1,
                Relation::Violated {
                    whole,
                    partitions,
                    only_in_whole,
                    only_in_partitions,
                } => {
                    violations += 1;

                    let record = serde_json::json!({
                        "oracle": "TLP",
                        "engine": engine,
                        "seed": seed,
                        "generator": generator.description(),
                        "whole_rows": whole,
                        "partition_rows": partitions,
                        "only_in_whole": only_in_whole,
                        "only_in_partitions": only_in_partitions,
                        "sql": {
                            "whole": parts.whole.statements(Dialect::Sqlite),
                            "is_true": parts.is_true.statements(Dialect::Sqlite),
                            "is_false": parts.is_false.statements(Dialect::Sqlite),
                            "is_unknown": parts.is_unknown.statements(Dialect::Sqlite),
                        },
                        "case": parts.whole,
                    });

                    std::fs::create_dir_all(&directory).expect("create findings directory");
                    let path = format!("{directory}/tlp-{engine}-{seed}.json");
                    std::fs::write(&path, serde_json::to_string_pretty(&record).expect("json"))
                        .expect("write finding");

                    if violations <= 5 {
                        println!("VIOLATION  {engine}, seed {seed}");
                        println!(
                            "  whole returned {whole} rows; partitions returned {partitions}"
                        );
                        for statement in parts.whole.statements(Dialect::Sqlite) {
                            println!("  {statement};");
                        }
                        println!("  saved {path}\n");
                    }
                }
            }
        }
    }
    let elapsed = started.elapsed();

    println!("over {total} cases");
    println!(
        "  not partitionable  {not_partitionable:>8} ({:>4.1}%) — no predicate, or an aggregate/set-op/LIMIT",
        100.0 * not_partitionable as f64 / total as f64
    );
    println!("  relation held      {checked:>8} engine-checks");
    println!("  unchecked          {skipped:>8}");
    println!("  **violations**     {violations:>8}");
    println!(
        "\n{:.1}s, {:.0} cases/sec",
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64()
    );

    if violations == 0 {
        println!(
            "\nNo self-inconsistency found. Unlike the differential zero, this one *can* speak \
             to a shared bug — it just did not find one on this surface."
        );
    }
}
