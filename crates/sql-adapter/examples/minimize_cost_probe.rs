//! How much does minimization cost as the data grows — and does shrinking rows first help?
//!
//! `PENDING` 2.17 measured a **424×** slowdown in full minimization between 8 rows and 4,096,
//! against only ~6× in raw throughput. That gap is the finding: it is *shrinking*, not
//! *executing*, that makes large data expensive, because the minimizer is greedy and spends its
//! trials on query structure while every trial pays full data cost.
//!
//! S10.7 raises the generator's row count, which is exactly the condition 2.17 was parked
//! against, so the fix — hoisting row reductions to the front of the candidate list above
//! `ROW_FIRST_THRESHOLD` — needs a number rather than an argument.
//!
//! # Reading the output
//!
//! The per-candidate figure matters more than the total. A total can fall simply because the
//! search took a different path and stopped earlier; cost *per candidate* is the thing the
//! reordering cannot fake, and comparing the two tells you which happened.
//!
//! To measure the un-hoisted behaviour for comparison, set `ROW_FIRST_THRESHOLD` in `shrink.rs`
//! to `usize::MAX` and re-run. That is a deliberate manual A/B rather than a runtime switch: a
//! knob on library code, added only so a probe can flip it, is a knob a campaign can get wrong.

use sql_adapter::ast::SqlCase;
use sql_adapter::backends::SqliteImpl;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::schema::{Literal, SqlType};
use std::cell::Cell;
use std::time::Instant;

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::minimize::{Shrink, minimize};
use diff_fuzzer_core::traits::{Generator, Implementation};

/// A generated case padded to `count` rows per table, every cell distinct.
///
/// Distinctness is not cosmetic: repeated rows tie, and a tie can invalidate an `ORDER BY` that
/// a `LIMIT` depends on — which would make the padded case fail `validate()` and yield no
/// candidates at all, quietly measuring nothing.
fn padded(seed: u64, count: usize) -> SqlCase {
    let mut case = SqlGenerator::new(Bounds::V1).generate(&mut SeededRng::from_seed(seed));
    let types: Vec<(String, Vec<SqlType>)> = case
        .schema
        .iter()
        .map(|table| {
            (
                table.name.clone(),
                table.columns.iter().map(|column| column.sql_type).collect(),
            )
        })
        .collect();

    let mut next = 1_000i64;
    for insert in &mut case.data {
        let Some(template) = insert.rows.first().cloned() else {
            continue;
        };
        let Some((_, columns)) = types.iter().find(|(name, _)| *name == insert.table) else {
            continue;
        };
        while insert.rows.len() < count {
            let mut row = template.clone();
            for (cell, sql_type) in row.iter_mut().zip(columns) {
                *cell = match sql_type {
                    SqlType::Text => Literal::Text(format!("v{next}")),
                    _ => Literal::Integer(next),
                };
                next += 1;
            }
            insert.rows.push(row);
        }
    }
    case
}

fn main() {
    println!("minimization cost against row count\n");
    println!(
        "{:>8}  {:>10}  {:>12}  {:>10}  {:>12}",
        "rows", "candidates", "total ms", "ms/cand", "final rows"
    );

    for count in [8usize, 64, 256, 512, 1_000, 2_000] {
        // Seeds are fixed so the comparison is between row counts and orderings, never between
        // different queries. Several of them, because one case's shape is not a measurement.
        let mut totals = (0u128, 0usize, 0usize);
        for seed in [3u64, 7, 11, 19] {
            let case = padded(seed, count);
            if case.validate().is_err() {
                continue;
            }

            // The predicate **executes the case**, because that is where the cost lives — a
            // structural predicate would measure the shrinker's bookkeeping and miss the point.
            //
            // "Runs without error" rather than "still returns rows": the first attempt used the
            // latter and measured nothing at all, because padding replaces the values the
            // generated `WHERE` was built against, so the padded case matches no rows, the
            // predicate is false at the start, and `minimize` correctly returns an unshrunk case
            // untouched. Four candidates and zero rows removed, at every size.
            //
            // What this predicate measures is therefore the **full descent** — the shrinker
            // takes its first offered candidate every round and runs to the floor. That is the
            // right thing to time for this question, since the comparison is about *which*
            // candidate is offered first and how much data each subsequent trial carries.
            let tried = Cell::new(0usize);
            let started = Instant::now();
            let result = minimize(case, |candidate| {
                tried.set(tried.get() + 1);
                SqliteImpl.run(candidate).is_ok()
            });
            let elapsed = started.elapsed().as_millis();

            totals.0 += elapsed;
            totals.1 += tried.get();
            totals.2 += result
                .input
                .data
                .iter()
                .map(|insert| insert.rows.len())
                .sum::<usize>();
        }

        let (millis, candidates, final_rows) = totals;
        let per = if candidates == 0 {
            0.0
        } else {
            millis as f64 / candidates as f64
        };
        println!("{count:>8}  {candidates:>10}  {millis:>12}  {per:>10.2}  {final_rows:>12}");
    }

    println!(
        "\nper-candidate cost is the figure to compare across orderings; a total can fall \
         because the search stopped earlier rather than because each step got cheaper"
    );

    // **Where does the time actually go?** Splitting list-construction from execution, because
    // the reordering only helps if execution dominates. If building the candidate list is the
    // cost, hoisting cannot help at all — the list is fully built before anything is reordered.
    println!("\ncandidate-list construction vs one execution, per call\n");
    println!(
        "{:>8}  {:>12}  {:>14}  {:>12}",
        "rows", "candidates", "build ms", "exec ms"
    );
    for count in [8usize, 256, 1_000, 2_000] {
        let case = padded(3, count);
        let started = Instant::now();
        let candidates = case.candidates();
        let build = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        let _ = SqliteImpl.run(&case);
        let exec = started.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{count:>8}  {:>12}  {build:>14.1}  {exec:>12.1}",
            candidates.len()
        );
    }
}
