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
    DifferentialOracle, Finding, FindingsLog, Generator, NormalizedRunner, Runner, SeededRng,
    Verdict, driver::run_once,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    Bounds, CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, TensorTolerancePolicy,
    libtorch, ndarray,
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
    let mut args = std::env::args().skip(1);
    let cases: u64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(100_000);
    let wide = args.next().is_some_and(|a| a == "wide");

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
        }
    } else {
        Bounds::default()
    };

    let generator = TensorOpGenerator::new(bounds);
    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &torch];
    let oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy> =
        DifferentialOracle::new(TensorTolerancePolicy);

    println!(
        "campaign: {cases} cases, {} bounds (rank <= {}, dim <= {}, |value| <= {})",
        if wide { "wide" } else { "default" },
        bounds.max_rank,
        bounds.max_dim,
        bounds.magnitude
    );
    println!("  burn-ndarray vs burn-tch, tolerance derived per operation\n");

    // Opened up front rather than on the first divergence, so a permissions or path
    // problem surfaces immediately instead of after an hour of work is already lost.
    let log_path = format!(
        "findings/campaign-{}.jsonl",
        if wide { "wide" } else { "default" }
    );
    let mut log = FindingsLog::open(&log_path).expect("findings log is writable");

    let mut totals = Tally::default();
    let mut per_operation: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut printed = 0usize;

    let started = std::time::Instant::now();

    for seed in 0..cases {
        // Regenerated only to label the row; the driver builds the identical case from
        // the same seed, which is the determinism guarantee being relied on.
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let outcome = run_once(seed, &generator, &runners, &oracle);

        let entry = per_operation.entry(case.name()).or_default();

        match &outcome.verdict {
            Verdict::Agree => {
                totals.agreed += 1;
                entry.agreed += 1;
            }
            Verdict::Skipped(reason) => {
                totals.skipped += 1;
                entry.skipped += 1;
                println!("  SKIPPED seed {seed} ({}): {reason}", case.name());
            }
            Verdict::Diverged(divergence) => {
                totals.diverged += 1;
                entry.diverged += 1;

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
