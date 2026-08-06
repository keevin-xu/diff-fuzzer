//! Run a batch of generated cases through the whole pipeline and report what came back.
//!
//! This is the first point where the tool does the thing it exists to do: generate, run on
//! both engines, normalize, judge. It reports counts by verdict and writes every divergence
//! to disk as a complete, self-contained case.
//!
//! **The whole `SqlCase` is written, not the seed.** A seed reproduces a case only for the
//! exact generator that produced it, and generators change — the tensor domain recorded 814
//! findings by seed and later found 810 of them could no longer be reproduced.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example hunt -- [cases] [label]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::{Generator, SkipReason, Verdict};
use sql_adapter::FINDINGS_ROOT;
use sql_adapter::driver::check_case;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::oracle::SortMode;
use sql_adapter::render::Dialect;
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let total: usize = arguments
        .next()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5_000);
    let label = arguments.next().unwrap_or_else(|| "hunt".to_string());

    let generator = SqlGenerator::default();
    let directory = format!("{FINDINGS_ROOT}/runs/{label}");

    println!("generator: {}", generator.description());
    println!("cases:     {total}");
    println!("findings:  {directory}/\n");

    let (mut agreed, mut diverged, mut skipped) = (0usize, 0usize, 0usize);
    let mut skip_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut ordered_cases = 0usize;

    let started = Instant::now();
    for seed in 0..total as u64 {
        // Regenerate the case alongside the verdict: the driver owns the run, but a finding
        // has to carry the case itself.
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if SortMode::for_case(&case) == SortMode::Ordered {
            ordered_cases += 1;
        }

        match check_case(seed).verdict {
            Verdict::Agree => agreed += 1,
            Verdict::Skipped(reason) => {
                skipped += 1;
                let key = match reason {
                    SkipReason::TooFewResults { .. } => "too few results",
                    SkipReason::CouldNotRun { .. } => "could not run",
                    SkipReason::NothingComparable { .. } => "nothing comparable",
                    SkipReason::KnownLegal { .. } => "known legal",
                };
                *skip_reasons.entry(key.to_string()).or_default() += 1;
            }
            Verdict::Diverged(divergence) => {
                diverged += 1;

                let record = serde_json::json!({
                    "seed": seed,
                    "generator": generator.description(),
                    "summary": divergence.summary,
                    "sort_mode": format!("{:?}", SortMode::for_case(&case)),
                    "sql": case.statements(Dialect::Sqlite),
                    "outputs": divergence.outputs,
                    "case": case,
                });

                std::fs::create_dir_all(&directory).expect("create findings directory");
                let path = format!("{directory}/diverged-{seed}.json");
                std::fs::write(
                    &path,
                    serde_json::to_string_pretty(&record).expect("serialize"),
                )
                .expect("write finding");

                if diverged <= 5 {
                    println!("DIVERGED seed {seed}: {}", divergence.summary);
                    println!("  {}", case.statements(Dialect::Sqlite).last().unwrap());
                    println!("  saved to {path}\n");
                }
            }
        }
    }
    let elapsed = started.elapsed();

    println!("verdicts over {total} cases");
    let percent = |count: usize| 100.0 * count as f64 / total as f64;
    println!("  agreed    {agreed:>7} ({:>5.1}%)", percent(agreed));
    println!("  diverged  {diverged:>7} ({:>5.1}%)", percent(diverged));
    println!("  skipped   {skipped:>7} ({:>5.1}%)", percent(skipped));
    for (reason, count) in &skip_reasons {
        println!("      {reason}: {count}");
    }

    println!("\ncorpus shape");
    println!(
        "  totally ordered  {ordered_cases:>7} ({:>5.1}%) — compared with row order intact",
        percent(ordered_cases)
    );
    println!(
        "  unordered        {:>7} ({:>5.1}%) — sorted before comparing",
        total - ordered_cases,
        percent(total - ordered_cases)
    );

    println!(
        "\n{:.1}s, {:.0} cases/sec",
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64()
    );

    if diverged == 0 {
        println!(
            "\nno divergences. that is a result, not a failure — but it only means \
                  something because the fault-injection tests prove detection works."
        );
    }
}
