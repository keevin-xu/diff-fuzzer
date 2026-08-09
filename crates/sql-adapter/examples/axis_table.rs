//! Every axis, swept alone, at one uniform scale — the table that decides the campaign.
//!
//! # Why this exists rather than a hand-assembled summary
//!
//! The axes were measured as they were built, over several sessions, at **different scales**:
//! the early ones at 3,000–5,000 cases, the recent ones at 30,000. By the rule of three those
//! bound the divergence rate at ~10⁻³ and ~10⁻⁴ respectively — an order of magnitude apart. A
//! table mixing them would put "0" in every row and invite the reader to treat the rows as
//! equally strong evidence. They are not, and S9.2 already produced one wrong reading from
//! exactly that mistake: five zeros at 200 cases each, which bounded nothing useful and were
//! nearly read as a refutation.
//!
//! So every axis is re-swept here at the same case count, in one run, with the bound printed
//! per row. Reproducible beats remembered.
//!
//! # What each column means
//!
//! - **diverged** — the differential oracle's verdict: the two engines disagreed.
//! - **tlp-viol** — the metamorphic oracle's: one engine contradicted itself.
//! - **judged** — metamorphic engine-checks actually performed. A low number with zero
//!   violations is much weaker evidence than a high one, and the difference is invisible in the
//!   violation count alone.
//! - **ordered** — the share of totally-ordered queries. **Read this column.** Three widenings
//!   have silently suppressed ordering while reporting clean agreement, and a fourth did it from
//!   a size bound. A row whose ordered share collapses is a row whose zero means less.
//! - **bound** — rule of three, 3/n, on the differential count.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example axis_table -- [diff cases] [tlp cases]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Normalizer, Oracle, Verdict,
};
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::metamorphic::{
    Relation, check, check_aggregate, check_distinct, check_grouped, partition,
    partition_aggregate, partition_grouped, partition_having,
};
use sql_adapter::normalize::SqlNormalizer;
use sql_adapter::oracle::{SortMode, SqlDifferentialOracle};
use sql_adapter::outcome::SqlOutcome;
use std::time::Instant;

struct Row {
    axis: &'static str,
    diverged: usize,
    skipped: usize,
    ordered: usize,
    cases: usize,
    per_second: f64,
    tlp_judged: usize,
    tlp_violations: usize,
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let diff_cases: usize = arguments
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000);
    let tlp_cases: usize = arguments
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);

    // Every axis this domain has, each **alone** against the same baseline. Two of them cannot
    // be varied entirely alone and are labelled so: `having` and `multi-group-by` need
    // aggregates to attach to, so their yield is "given aggregates" and the honest comparison
    // is against the `aggregates` row, not against `V1`.
    let axes: Vec<(&'static str, Bounds)> = vec![
        ("(baseline V1)", Bounds::V1),
        ("wide-arithmetic", Bounds::V1_WIDE_ARITHMETIC),
        ("aggregates", Bounds::V1_AGGREGATES),
        ("set-ops", Bounds::V1_SET_OPS),
        ("chained-set-ops", Bounds::V1_CHAINED_SET_OPS),
        ("joins", Bounds::V1_JOINS),
        ("subqueries", Bounds::V1_SUBQUERIES),
        ("not-in", Bounds::V1_NOT_IN),
        ("not-in-list", Bounds::V1_NOT_IN_LIST),
        ("not-in-correlated", Bounds::V1_NOT_IN_CORRELATED),
        ("distinct", Bounds::V1_DISTINCT),
        ("having*", Bounds::V1_HAVING),
        ("multi-group-by*", Bounds::V1_MULTI_GROUP_BY),
        ("case", Bounds::V1_CASE),
    ];

    println!("differential: {diff_cases} cases per axis · metamorphic: {tlp_cases} cases per axis");
    println!("* needs aggregates enabled to attach to; compare against the `aggregates` row\n");

    // **Each row prints as it completes, and the stream is flushed.** The first version of
    // this example accumulated everything and printed at the end; a run that was interrupted
    // an hour in therefore lost every axis it had already finished. A long measurement should
    // be resumable-by-inspection: whatever it got through is on screen.
    println!(
        "  {:<20} {:>9} {:>8} {:>8} {:>9} {:>10} {:>9} {:>10}",
        "axis", "diverged", "skipped", "ordered", "cases/s", "tlp-judged", "tlp-viol", "bound"
    );

    let mut table = Vec::new();
    for (axis, bounds) in axes {
        let row = measure(axis, bounds, diff_cases, tlp_cases);
        println!(
            "  {:<20} {:>9} {:>8} {:>7.0}% {:>9.0} {:>10} {:>9} {:>10.1e}",
            row.axis,
            row.diverged,
            row.skipped,
            100.0 * row.ordered as f64 / row.cases as f64,
            row.per_second,
            row.tlp_judged,
            row.tlp_violations,
            3.0 / row.cases as f64,
        );
        // Without this the line sits in a pipe buffer, which is the same problem again when
        // the output is piped to `tail` or a log.
        let _ = std::io::Write::flush(&mut std::io::stdout());
        table.push(row);
    }

    let total_diverged: usize = table.iter().map(|r| r.diverged).sum();
    let total_violations: usize = table.iter().map(|r| r.tlp_violations).sum();
    let total_judged: usize = table.iter().map(|r| r.tlp_judged).sum();
    let total_cases: usize = table.iter().map(|r| r.cases).sum();

    println!("\nreading");
    println!(
        "  {total_cases} differential cases, {total_judged} metamorphic engine-checks.\n  \
         {total_diverged} divergences, {total_violations} self-inconsistencies."
    );
    if total_diverged == 0 && total_violations == 0 {
        println!(
            "  Combined bound across all axes: {:.1e} per case. Note this is NOT a bound on\n  \
             `V1_ALL`: axes were run alone, so nothing here says anything about their\n  \
             *interactions*, which is what the campaign tests.",
            3.0 / total_cases as f64
        );
    }

    // The corpus-shape check, promoted to a printed warning rather than left to the reader.
    let baseline_ordered = table
        .first()
        .map(|r| r.ordered as f64 / r.cases as f64)
        .unwrap_or(0.0);
    for row in &table {
        let share = row.ordered as f64 / row.cases as f64;
        if baseline_ordered > 0.0 && share < baseline_ordered * 0.6 {
            println!(
                "  WARNING  {} dropped the ordered share from {:.0}% to {:.0}% — an axis must\n  \
                 add cases, not remove them. Its zero covers a narrower corpus than it looks.",
                row.axis,
                100.0 * baseline_ordered,
                100.0 * share
            );
        }
    }
}

fn measure(axis: &'static str, bounds: Bounds, diff_cases: usize, tlp_cases: usize) -> Row {
    let generator = SqlGenerator::new(bounds);
    let oracle = SqlDifferentialOracle;
    let (mut diverged, mut skipped, mut ordered) = (0, 0, 0);

    let started = Instant::now();
    for seed in 0..diff_cases as u64 {
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
        match oracle.check(&case, &outputs) {
            Verdict::Diverged { .. } => diverged += 1,
            Verdict::Skipped { .. } => skipped += 1,
            Verdict::Agree => {}
        }
    }
    let per_second = diff_cases as f64 / started.elapsed().as_secs_f64();

    // The metamorphic half, on DuckDB only: this table is about relative yield per axis, and
    // running both engines would double the time for a second copy of the same signal. The
    // campaign runs both.
    let (mut tlp_judged, mut tlp_violations) = (0, 0);
    for seed in 0..tlp_cases as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let run = |c: &_| -> Option<SqlOutcome> { DuckDbImpl.run(c).ok() };

        let relation = if let Some(parts) = partition_having(&case) {
            four(
                &run,
                &parts.whole,
                &parts.is_true,
                &parts.is_false,
                &parts.is_unknown,
            )
            .map(|(w, t, f, u)| check(&w, &t, &f, &u))
        } else if let Some(parts) = partition(&case) {
            four(
                &run,
                &parts.whole,
                &parts.is_true,
                &parts.is_false,
                &parts.is_unknown,
            )
            .map(|(w, t, f, u)| {
                if parts.distinct {
                    check_distinct(&w, &t, &f, &u)
                } else {
                    check(&w, &t, &f, &u)
                }
            })
        } else if let Some(parts) = partition_aggregate(&case) {
            four(
                &run,
                &parts.whole,
                &parts.is_true,
                &parts.is_false,
                &parts.is_unknown,
            )
            .map(|(w, t, f, u)| check_aggregate(parts.func, &w, &t, &f, &u))
        } else if let Some(parts) = partition_grouped(&case) {
            four(
                &run,
                &parts.whole,
                &parts.is_true,
                &parts.is_false,
                &parts.is_unknown,
            )
            .map(|(w, t, f, u)| check_grouped(parts.keys, &parts.funcs, &w, &t, &f, &u))
        } else {
            None
        };

        match relation {
            Some(Relation::Holds) => tlp_judged += 1,
            Some(Relation::Violated { .. }) => {
                tlp_judged += 1;
                tlp_violations += 1;
            }
            _ => {}
        }
    }

    Row {
        axis,
        diverged,
        skipped,
        ordered,
        cases: diff_cases,
        per_second,
        tlp_judged,
        tlp_violations,
    }
}

/// Run four variants, or `None` if any could not be run.
fn four<F, C>(
    run: &F,
    a: &C,
    b: &C,
    c: &C,
    d: &C,
) -> Option<(SqlOutcome, SqlOutcome, SqlOutcome, SqlOutcome)>
where
    F: Fn(&C) -> Option<SqlOutcome>,
{
    Some((run(a)?, run(b)?, run(c)?, run(d)?))
}
