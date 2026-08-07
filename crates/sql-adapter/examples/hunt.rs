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
use sql_adapter::FINDINGS_ROOT;
use sql_adapter::ast::SqlCase;
use sql_adapter::backends::{DuckDbImpl, SqliteImpl};
use sql_adapter::generator::SqlGenerator;
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
        Some("subqueries") => sql_adapter::gen_schema::Bounds::V1_SUBQUERIES,
        Some("all") => sql_adapter::gen_schema::Bounds::V1_ALL,
        _ => sql_adapter::gen_schema::Bounds::V1,
    };

    let generator = SqlGenerator::new(bounds);
    let directory = format!("{FINDINGS_ROOT}/runs/{label}");
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
    for seed in 0..total as u64 {
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
