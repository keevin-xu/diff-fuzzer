//! Run a batch of generated cases, and turn anything that diverges into a finding.
//!
//! The full pipeline: generate, run on both engines, normalize, judge — and on a
//! divergence, **minimize, sign, and save**. It reports counts by verdict and groups the
//! findings by signature, so one problem found a thousand times is one line.
//!
//! **The whole `SqlCase` is written, not the seed.** A seed reproduces a case only for the
//! exact generator that produced it, and generators change — the tensor domain recorded 814
//! findings by seed and later found 810 could no longer be reproduced.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example hunt -- [cases] [label]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::minimize::minimize;
use diff_fuzzer_core::traits::{
    Generator, Implementation, Normalizer, Oracle, SkipReason, Verdict,
};
use sql_adapter::DIFFERENTIAL_ROOT;
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::generator::SqlGenerator;
use sql_adapter::known::known_comma_join_defect;
use sql_adapter::normalize::{CanonicalResult, SqlNormalizer};
use sql_adapter::oracle::{SortMode, SqlDifferentialOracle};
use sql_adapter::render::Dialect;
use sql_adapter::report::{Environment, Minimisation, SqlDivergence};
use sql_adapter::shrink::complexity;
use sql_adapter::signature::{DisagreementKind, signature};
use std::collections::BTreeMap;
use std::time::Instant;

/// Run one case on both engines and canonicalize what came back.
fn outcomes(case: &SqlCase) -> Option<Vec<(String, CanonicalResult)>> {
    let sqlite = SqliteImpl.run(case).ok()?;
    let duckdb = DuckDbImpl.run(case).ok()?;
    Some(vec![
        ("sqlite".to_string(), SqlNormalizer.normalize(sqlite)),
        ("duckdb".to_string(), SqlNormalizer.normalize(duckdb)),
    ])
}

/// Does this case still diverge? The predicate minimization is driven by.
fn diverges(case: &SqlCase) -> bool {
    let Some(results) = outcomes(case) else {
        return false;
    };
    let named: Vec<_> = results
        .into_iter()
        .map(|(name, output)| diff_fuzzer_core::traits::NamedOutput {
            implementation: name,
            output,
        })
        .collect();
    matches!(
        SqlDifferentialOracle.check(case, &named),
        Verdict::Diverged(_)
    )
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let total: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let label = arguments.next().unwrap_or_else(|| "hunt".to_string());
    // A third argument switches on the overflow-reachable bounds, so the two settings can
    // be measured against each other rather than one being chosen by argument.
    let bounds = match arguments.next().as_deref() {
        Some("wide") => sql_adapter::gen_schema::Bounds::V1_WIDE_ARITHMETIC,
        Some("aggregates") => sql_adapter::gen_schema::Bounds::V1_AGGREGATES,
        Some("setops") => sql_adapter::gen_schema::Bounds::V1_SET_OPS,
        Some("chained") => sql_adapter::gen_schema::Bounds::V1_CHAINED_SET_OPS,
        Some("joins") => sql_adapter::gen_schema::Bounds::V1_JOINS,
        Some("not-in") => sql_adapter::gen_schema::Bounds::V1_NOT_IN,
        Some("not-in-list") => sql_adapter::gen_schema::Bounds::V1_NOT_IN_LIST,
        Some("distinct") => sql_adapter::gen_schema::Bounds::V1_DISTINCT,
        Some("having") => sql_adapter::gen_schema::Bounds::V1_HAVING,
        Some("not-in-correlated") => sql_adapter::gen_schema::Bounds::V1_NOT_IN_CORRELATED,
        Some("multi-group-by") => sql_adapter::gen_schema::Bounds::V1_MULTI_GROUP_BY,
        Some("case") => sql_adapter::gen_schema::Bounds::V1_CASE,
        Some("window") => sql_adapter::gen_schema::Bounds::V1_WINDOW,
        Some("indexes") => sql_adapter::gen_schema::Bounds::V1_INDEXES,
        Some("large") => sql_adapter::gen_schema::Bounds::V1_LARGE,
        Some("subqueries") => sql_adapter::gen_schema::Bounds::V1_SUBQUERIES,
        Some("all") => sql_adapter::gen_schema::Bounds::V1_ALL,
        Some("all-large") => sql_adapter::gen_schema::Bounds::V1_ALL_LARGE,
        Some("comma-joins") => sql_adapter::gen_schema::Bounds::V1_COMMA_JOINS,
        None => sql_adapter::gen_schema::Bounds::V1,
        // **An unrecognised name is a hard error, not a silent fallback to the default.**
        //
        // It read `_ => <default>`, so a typo or a preset this runner had never been taught
        // about produced a full clean run *under a different configuration than the one named*.
        // That is not hypothetical: a 30,000-case sweep labelled `comma-joins` ran the baseline
        // and reported "0 divergences" while the real comma-joins axis diverges at **12%**. The
        // label goes into the findings directory and into every summary quoting the run, so the
        // wrong number outlives the command that produced it.
        //
        // Listing the valid names in the message matters too — the missing name here *was*
        // `comma-joins`, and a bare "unknown axis" would not have shown that it was absent
        // rather than misspelled.
        Some(unknown) => {
            eprintln!(
                "unknown axis {unknown:?}. valid: wide, aggregates, setops, chained, joins, \
                 comma-joins, not-in, not-in-list, not-in-correlated, distinct, having, \
                 multi-group-by, case, window, indexes, large, subqueries, all, all-large"
            );
            std::process::exit(2);
        }
    };

    // Counted and reported separately from `agreed`: a case suppressed by the catalog is not a
    // case where the engines agreed, and folding the two together would overstate agreement by
    // exactly the size of the catalog's reach.
    let mut known_legal = 0usize;

    let generator = SqlGenerator::new(bounds);
    let directory = format!("{DIFFERENTIAL_ROOT}/{label}");
    let environment = Environment::detect();

    println!("generator: {}", generator.description());
    println!(
        "engines:   sqlite {} · duckdb {}",
        environment.sqlite, environment.duckdb
    );
    println!("cases:     {total}");
    println!("findings:  {directory}/\n");

    let (mut agreed, mut diverged, mut skipped) = (0usize, 0usize, 0usize);
    let mut by_signature: BTreeMap<String, usize> = BTreeMap::new();
    let mut ordered_cases = 0usize;

    let started = Instant::now();
    // **Progress prints every 100,000 cases, flushed.** A campaign that reports only at the end
    // loses everything when it is interrupted — which happened: a 50-minute run over ~500,000
    // cases left four header lines and no verdict counts. The same defect had already been
    // found and fixed in `axis_table` and was not carried across. A long measurement should be
    // resumable by inspection: whatever it got through is on screen.
    let progress_every = 100_000usize;

    for seed in 0..total as u64 {
        if seed > 0 && (seed as usize).is_multiple_of(progress_every) {
            let done = seed as usize;
            let rate = done as f64 / started.elapsed().as_secs_f64();
            println!(
                "  … {done:>9} cases | agreed {agreed} diverged {diverged} skipped {skipped} \
                 | {rate:.0}/sec | ordered {:.0}%",
                100.0 * ordered_cases as f64 / done as f64
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if SortMode::for_case(&case) == SortMode::Ordered {
            ordered_cases += 1;
        }

        let Some(results) = outcomes(&case) else {
            skipped += 1;
            continue;
        };
        let named: Vec<_> = results
            .iter()
            .map(|(name, output)| diff_fuzzer_core::traits::NamedOutput {
                implementation: name.clone(),
                output: output.clone(),
            })
            .collect();

        match SqlDifferentialOracle.check(&case, &named) {
            Verdict::Agree => agreed += 1,
            Verdict::Skipped(reason) => {
                skipped += 1;
                if matches!(reason, SkipReason::CouldNotRun { .. }) && skipped <= 3 {
                    println!("skipped seed {seed}: {reason}");
                }
            }
            Verdict::Diverged(divergence) => {
                // **The catalogued comma-join defect, filtered here rather than in the oracle.**
                //
                // `known_comma_join_defect` was written at S10.4, unit-tested, and **never
                // called by any runner** — dead code guarding nothing. `PENDING` 2.21 had
                // predicted the consequence exactly: *"required before any campaign enables
                // comma-joins, or this one mechanism swamps every run"*. Measured on the fixed
                // generator, it swamps at **12.0%** — 3,602 of 30,000 cases, two signatures,
                // both tracing to SQLite's documented parser defect (`SPECS.md` §2.11).
                //
                // It cannot live in the `Oracle` seam, and that is structural rather than an
                // oversight: `legal_difference` judges *outputs*, while this defect is a
                // property of the *query* — a comma-join and an explicit join binding against
                // each other. The same mismatch recorded at G-S8. The runner is the first place
                // that holds both the case and the verdict, so it is where the filter belongs.
                if let Some(entry) = known_comma_join_defect(&case) {
                    known_legal += 1;
                    if known_legal <= 3 {
                        println!(
                            "seed {seed}: suppressed as a known legal difference ({})",
                            entry.name
                        );
                    }
                    continue;
                }

                diverged += 1;

                // Minimize before recording. A raw generated divergence is a whole database
                // program; nobody can tell from it which part matters.
                let before = complexity(&case);
                let minimized = minimize(case.clone(), diverges);
                let after = complexity(&minimized.input);

                // What kind of disagreement, computed from the *minimized* case — and
                // checked, because a minimized case that no longer diverges would mean the
                // shrinker walked off the finding.
                let small = minimized.input.clone();
                let Some(small_results) = outcomes(&small) else {
                    println!("seed {seed}: minimized case no longer runs — not recording");
                    continue;
                };
                let Some(kind) =
                    DisagreementKind::between(&small_results[0].1, &small_results[1].1)
                else {
                    println!("seed {seed}: minimized case no longer diverges — not recording");
                    continue;
                };

                let key = signature(&small, kind);
                *by_signature.entry(key.clone()).or_default() += 1;

                let finding = SqlDivergence {
                    signature: key.clone(),
                    kind,
                    disagreeing: vec!["sqlite".to_string(), "duckdb".to_string()],
                    seed,
                    generator: generator.description(),
                    sql: small.statements(Dialect::Sqlite),
                    case: small,
                    minimisation: Minimisation {
                        steps: minimized.steps,
                        candidates_tried: minimized.candidates_tried,
                        complete: minimized.is_minimal(),
                        complexity_before: before,
                        complexity_after: after,
                    },
                    outputs: small_results
                        .iter()
                        .map(|(name, output)| (name.clone(), format!("{output:?}")))
                        .collect(),
                    environment: environment.clone(),
                    summary: divergence.summary.clone(),
                };

                match finding.save(&directory) {
                    Ok(path) => {
                        if diverged <= 5 {
                            println!("DIVERGED seed {seed}  [{key}]");
                            println!("{}", finding.script());
                            for (name, output) in &finding.outputs {
                                println!("  {name}: {output}");
                            }
                            println!(
                                "  minimized {before:?} -> {after:?} in {} steps ({} candidates){}",
                                minimized.steps,
                                minimized.candidates_tried,
                                if minimized.is_minimal() {
                                    ""
                                } else {
                                    ", BUDGET EXHAUSTED"
                                }
                            );
                            println!("  saved {path}\n");
                        }
                    }
                    Err(error) => println!("could not save seed {seed}: {error}"),
                }
            }
        }
    }
    let elapsed = started.elapsed();

    let percent = |count: usize| 100.0 * count as f64 / total as f64;
    println!("verdicts over {total} cases");
    println!("  agreed    {agreed:>7} ({:>5.1}%)", percent(agreed));
    println!(
        "  known-legal {known_legal:>5} ({:>5.1}%) — suppressed by the catalog, NOT agreement",
        percent(known_legal)
    );
    println!("  diverged  {diverged:>7} ({:>5.1}%)", percent(diverged));
    println!("  skipped   {skipped:>7} ({:>5.1}%)", percent(skipped));

    if !by_signature.is_empty() {
        println!("\ndistinct problems: {}", by_signature.len());
        for (key, count) in &by_signature {
            println!("  {count:>5}x  {key}");
        }
    }

    println!(
        "\ncorpus: {ordered_cases} totally ordered ({:.0}%), {} unordered",
        percent(ordered_cases),
        total - ordered_cases
    );
    println!(
        "{:.1}s, {:.0} cases/sec",
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64()
    );

    if diverged == 0 {
        println!(
            "\nno divergences — a result, not a failure. It means something only because \
             the fault-injection tests prove detection works."
        );
    }
}
