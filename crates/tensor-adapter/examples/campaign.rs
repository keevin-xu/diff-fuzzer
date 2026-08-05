//! Run the oracle over a large batch of generated cases and report everything it flags.
//!
//! This is the tool doing the job it was built for, at volume. Every divergence is
//! printed with the seed that produced it, the operation, and the size of the
//! disagreement — the minimum needed to go back and investigate one.
//!
//! Two things are worth knowing about how to read the output.
//!
//! **Zero divergences is a real result, not a failure.** It means the two backends
//! agree everywhere the tolerance policy says they should, which is what correct
//! implementations do. The number that would be alarming is a *large* one, because that
//! usually means the tolerance is wrong rather than that a library is broken.
//!
//! **A clean run only means something if the tool can still detect faults.** That is
//! what the injected-fault tests in `testing.rs` are for, and why they run on every
//! `cargo test`. Without them, "no divergences found" would be indistinguishable from a
//! comparison that had quietly stopped working.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example campaign            # 100k cases
//! cargo run --release -p tensor-adapter --example campaign 500000     # more
//! cargo run --release -p tensor-adapter --example campaign 50000 wide # wider shapes/values
//! ```

use diff_fuzzer_core::{
    DifferentialOracle, DivergenceReport, Finding, FindingsLog, Generator, MinimisationRecord,
    NamedOutput, NormalizedRunner, Oracle, Runner, SeededRng, Seen, TolerancePolicy, Verdict,
    driver::run_once, minimize,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    Bounds, CanonicalTensor, FaultyBackend, TensorNormalizer, TensorOp, TensorOpGenerator,
    TensorTolerancePolicy, environment, flex, libtorch, signature_across, wgpu,
};

/// Divergences printed in full before switching to a count. A campaign that flags
/// thousands of cases has a policy problem, not a discovery — and scrolling past all of
/// them helps nobody.
const MAX_PRINTED: usize = 20;

/// How much of each recorded field to keep. Enough to see the shape of the data and the
/// first values; far short of a full tensor.
const FINDING_FIELD_CHARS: usize = 1_000;

#[derive(Default)]
struct Tally {
    agreed: usize,
    diverged: usize,
    skipped: usize,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cases: u64 = args.first().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let wide = args.iter().any(|a| a == "wide");
    // `open` lifts the generator's domain restrictions, so `sqrt` receives negatives and
    // divisors may be zero — undefined and infinite results start occurring.
    let open = args.iter().any(|a| a == "open");
    // `fault` swaps the libtorch backend for one deliberately wrong by a known amount.
    // The real backends agree on everything the policy permits, so this is how the
    // reporting path — shrink, describe, save — can be exercised on demand rather than
    // only when a genuine divergence happens to turn up.
    let fault = args.iter().any(|a| a == "fault");
    // `gpu` adds `burn-wgpu` as a third implementation. Opt-in rather than default: GPU
    // reductions are not deterministic on this device (see `examples/wgpu_check.rs`), so
    // a run including it answers a different question from a CPU-only one and the two
    // should not be silently mixed.
    let gpu = args.iter().any(|a| a == "gpu");

    // The wider setting exists to stress the accumulating operations. Since the
    // tolerance for those is derived per case from the actual shapes and values, a
    // larger generator should *not* produce more false positives — the allowance grows
    // with the arithmetic. That is a claim worth testing rather than assuming.
    let bounds = if wide {
        Bounds {
            // Rank is held at 3 while dimensions grow. Accumulation depth is what
            // stresses the tolerance — a reduction sums the length of one axis, and a
            // matmul sums its inner dimension — whereas extra ranks mostly multiply the
            // element count and slow the run down without testing anything new.
            max_rank: 3,
            max_dim: 64,
            magnitude: 1000.0,
            ..Bounds::default()
        }
    } else {
        Bounds::default()
    };
    let bounds = Bounds {
        restrict_domains: !open,
        ..bounds
    };

    let generator = TensorOpGenerator::new(bounds);
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let faulty = NormalizedRunner::new(FaultyBackend::new(flex(), 0.5), TensorNormalizer);

    let gpu_runner = NormalizedRunner::new(wgpu(), TensorNormalizer);

    let second: &dyn Runner<In = TensorOp, Canon = CanonicalTensor> =
        if fault { &faulty } else { &torch };

    // A `Vec` rather than an array: the driver has always taken a slice and never assumed
    // two, so a third implementation needs no change beyond appending one.
    let mut runners: Vec<&dyn Runner<In = TensorOp, Canon = CanonicalTensor>> = vec![&cpu, second];
    if gpu {
        runners.push(&gpu_runner);
    }
    let runners = runners.as_slice();

    let oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy> =
        DifferentialOracle::new(TensorTolerancePolicy);

    // Does this case diverge? Used both for the campaign's own verdicts and as the
    // predicate that drives shrinking.
    //
    // Note it re-asks the *policy* for each candidate rather than reusing the original
    // case's tolerance. That is deliberate: a shrunk case is only a legitimate finding if
    // it diverges under the tolerance that would apply to it. Since the policy tightens
    // as a case gets smaller, this is the stricter reading of the two.
    let diverges = |case: &TensorOp| -> bool {
        let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
            .iter()
            .filter_map(|runner| {
                runner
                    .run_and_normalize(case)
                    .ok()
                    .map(|output| NamedOutput {
                        implementation: runner.name().to_string(),
                        output,
                    })
            })
            .collect();

        matches!(oracle.check(case, &outputs), Verdict::Diverged(_))
    };

    println!(
        "campaign: {cases} cases, {} bounds{}{} (rank <= {}, dim <= {}, |value| <= {})",
        if wide { "wide" } else { "default" },
        if open { ", domains unrestricted" } else { "" },
        if fault { ", FAULT INJECTED" } else { "" },
        bounds.max_rank,
        bounds.max_dim,
        bounds.magnitude
    );
    // Naming the actual pair rather than the usual one: a header that says `burn-tch`
    // while a fault is injected would misdescribe every finding below it.
    println!(
        "  {}, tolerance derived per operation\n",
        runners
            .iter()
            .map(|r| r.name())
            .collect::<Vec<_>>()
            .join(" vs ")
    );

    // Which run directory this campaign's output belongs to.
    //
    // The fuzz target files findings under `findings/runs/<run>/<operation>/`; this runner
    // used to write to `findings/` directly, so the two paths scattered their output
    // across two layouts. **Findings split across two places is how one set of them gets
    // forgotten** — the same failure that was already fixed once for the fuzz target, and
    // then reintroduced here by not applying it to both.
    //
    // `DIFF_FUZZER_RUN` names the run when set, matching the fuzz target. Unset, the label
    // says what the campaign was rather than inventing a plausible one.
    let run = std::env::var("DIFF_FUZZER_RUN")
        .unwrap_or_else(|_| format!("seeded-{}", if wide { "wide" } else { "default" }));
    let run_dir = format!("findings/runs/{run}");

    // Opened up front rather than on the first divergence, so a permissions or path
    // problem surfaces immediately instead of after an hour of work is already lost.
    let log_path = format!("{run_dir}/campaign.jsonl");
    let mut log = FindingsLog::open(&log_path).expect("findings log is writable");

    // Tracks which problems have already been reported. A single defect is reachable
    // from a huge number of inputs, so without this a campaign's output is one finding
    // repeated — and a genuinely *second* problem would be invisible in the noise.
    let mut seen = Seen::new();

    let mut totals = Tally::default();
    let mut per_operation: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut printed = 0usize;

    let started = std::time::Instant::now();

    for seed in 0..cases {
        // Regenerated only to label the row; the driver builds the identical case from
        // the same seed, which is the determinism guarantee being relied on.
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let outcome = run_once(seed, &generator, runners, &oracle);

        let entry = per_operation.entry(case.name()).or_default();

        match &outcome.verdict {
            Verdict::Agree => {
                totals.agreed += 1;
                entry.agreed += 1;
            }
            Verdict::Skipped(reason) => {
                totals.skipped += 1;
                entry.skipped += 1;
                if totals.skipped <= 5 {
                    println!("  SKIPPED seed {seed} ({}): {reason}", case.name());
                }
            }
            Verdict::Diverged(divergence) => {
                totals.diverged += 1;
                entry.diverged += 1;

                // Shrink before recording. A divergence arrives at whatever size the
                // generator produced — often hundreds of values, nearly all irrelevant —
                // and nobody can act on that. Doing it here, at the moment of discovery,
                // means every saved report is already the smallest form we could find.
                // Group before doing any work. Shrinking and writing a report for the
                // thousandth instance of a problem already recorded costs real time and
                // adds nothing.
                let outputs = normalized(&case, runners);
                // The pair that will name this finding, so the tolerance cited matches the
                // comparison described. With per-pair bounds there is no single number for
                // a case.
                let pair = worst_pair(&case, &outputs);
                let tolerance =
                    TensorTolerancePolicy.tolerance_for(&case, (pair.0.as_str(), pair.1.as_str()));
                // Across *all* implementations, not the first two. With three backends the
                // old form computed the label from two CPUs that agreed, so a GPU
                // divergence came out labelled `.../agree`.
                let (fingerprint, disagreeing) = signature_across(&case, &outputs, tolerance);
                let _ = &disagreeing;

                if !seen.is_new(&fingerprint) {
                    // Counted, not recorded. How often a problem is reachable is worth
                    // knowing; a thousand copies of its report is not.
                    continue;
                }

                let minimized = minimize(case.clone(), diverges);

                // Re-judge the *minimised* case to describe it. The original's outputs
                // and summary belong to a case the report no longer contains — pairing a
                // one-element input with a description of thirty-five values would leave
                // a reader unable to tell which they were looking at.
                let shrunk_divergence = match describe(&minimized.input, runners, &oracle) {
                    Some(divergence) => divergence,
                    // Cannot happen — the predicate only accepted candidates that
                    // diverge — but falling back to the original description is better
                    // than losing the finding to an assertion.
                    None => divergence.clone(),
                };

                let report = DivergenceReport {
                    seed,
                    label: case.name().to_string(),
                    generator: format!("{bounds:?}"),
                    input: minimized.input.clone(),
                    minimisation: MinimisationRecord::from(&minimized),
                    outputs: shrunk_divergence.outputs,
                    tolerance: TensorTolerancePolicy
                        .tolerance_for(&minimized.input, (pair.0.as_str(), pair.1.as_str())),
                    environment: environment(),
                    summary: shrunk_divergence.summary,
                };

                // Grouped by operation inside the run, so `triage_findings` sees the same
                // shape of tree whichever runner produced it.
                let report_path = format!("{run_dir}/{}/{}", case.name(), report.filename());
                report.save(&report_path).expect("report is writable");

                if printed < MAX_PRINTED {
                    println!(
                        "  shrank {} values -> {} values in {} reductions ({}), saved to {report_path}",
                        element_count(&case),
                        element_count(&minimized.input),
                        minimized.steps,
                        minimized.stopped
                    );
                }

                // Written before anything is printed. The terminal is where a finding
                // is noticed; the file is where it survives.
                //
                // Truncated, because a wide-bounds case holds tens of thousands of
                // values and its full text runs to about a megabyte — storing those
                // verbatim produced a 224 MB log for 235 findings. The seed regenerates
                // the case exactly, and the summary carries the error magnitudes, which
                // is what triage actually reads.
                log.append(&Finding::new(
                    seed,
                    case.name(),
                    // Without the configuration, the seed above identifies nothing — it
                    // names a case only in combination with the bounds that produced it.
                    format!("{bounds:?}"),
                    &fingerprint,
                    divergence.truncated(FINDING_FIELD_CHARS),
                ))
                .expect("findings log is writable");

                if printed < MAX_PRINTED {
                    printed += 1;
                    println!("  DIVERGED seed {seed}");
                    println!("    operation: {}", case.name());
                    println!("    shapes:    {}", shapes_of(&case));
                    println!("    detail:    {}", divergence.summary);
                    println!();
                } else if printed == MAX_PRINTED {
                    printed += 1;
                    println!("  ... further divergences counted but not printed\n");
                }
            }
        }
    }

    let elapsed = started.elapsed();

    println!("---");
    println!("  agreed    {:>8}", totals.agreed);
    println!("  diverged  {:>8}", totals.diverged);
    println!("  skipped   {:>8}", totals.skipped);
    println!(
        "  {:.0} cases/sec ({:.2?} total)\n",
        cases as f64 / elapsed.as_secs_f64(),
        elapsed
    );

    println!(
        "  {:<8} {:>9} {:>9} {:>9}",
        "op", "agreed", "diverged", "skipped"
    );
    for (name, tally) in &per_operation {
        println!(
            "  {:<8} {:>9} {:>9} {:>9}",
            name, tally.agreed, tally.diverged, tally.skipped
        );
    }

    if seen.distinct() > 0 {
        println!();
        println!(
            "  distinct problems: {} (from {} diverging cases)",
            seen.distinct(),
            seen.total()
        );
        for (fingerprint, count) in seen.counts() {
            println!("    {count:>6}x  {fingerprint}");
        }
    }

    println!();
    if log.written() > 0 {
        println!("  {} findings written to {log_path}\n", log.written());
    }

    if totals.diverged == 0 {
        println!("  no divergences. The backends agree everywhere the policy allows.");
        println!("  (The injected-fault tests in `cargo test` are what make this claim");
        println!("   mean something — they prove the detector still detects.)");
    } else {
        println!(
            "  {} divergences to triage: reproducible? float noise? legal? real?",
            totals.diverged
        );
    }
}

/// Run a case on every implementation and return the canonical results.
fn normalized(
    case: &TensorOp,
    runners: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
) -> Vec<(String, CanonicalTensor)> {
    runners
        .iter()
        .filter_map(|runner| {
            runner
                .run_and_normalize(case)
                .ok()
                .map(|output| (runner.name().to_string(), output))
        })
        .collect()
}

/// Run a case and return the divergence it produces, if any.
///
/// Used to describe a *shrunk* case, so that every field of a report refers to the same
/// thing.
fn describe(
    case: &TensorOp,
    runners: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
    oracle: &DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
) -> Option<diff_fuzzer_core::Divergence> {
    let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
        .iter()
        .filter_map(|runner| {
            runner
                .run_and_normalize(case)
                .ok()
                .map(|output| NamedOutput {
                    implementation: runner.name().to_string(),
                    output,
                })
        })
        .collect();

    match oracle.check(case, &outputs) {
        Verdict::Diverged(divergence) => Some(divergence),
        _ => None,
    }
}

/// How many values a case holds, for reporting what shrinking achieved.
fn element_count(case: &TensorOp) -> usize {
    match case {
        TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => arg.len(),
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => lhs.len() + rhs.len(),
    }
}

/// A compact description of a case's argument shapes, for the log line.
fn shapes_of(case: &TensorOp) -> String {
    match case {
        TensorOp::Unary { arg, .. } => format!("{:?}", arg.shape()),
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => {
            format!("{:?} x {:?}", lhs.shape(), rhs.shape())
        }
        TensorOp::Reduce { arg, axis, .. } => format!("{:?} axis {axis}", arg.shape()),
    }
}

/// Which two implementations a finding should be attributed to.
///
/// The worst disagreeing pair when one exists, so the tolerance recorded in a report is the
/// one that governed the comparison the report describes. Falls back to the first two, which
/// only happens on a case that did not diverge.
fn worst_pair(case: &TensorOp, outputs: &[(String, CanonicalTensor)]) -> (String, String) {
    let probe = TensorTolerancePolicy.tolerance_for(case, ("", ""));
    match signature_across(case, outputs, probe).1 {
        Some(pair) => (pair.left, pair.right),
        None => match outputs {
            [(a, _), (b, _), ..] => (a.clone(), b.clone()),
            _ => (String::new(), String::new()),
        },
    }
}
