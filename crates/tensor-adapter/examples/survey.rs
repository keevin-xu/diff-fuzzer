//! Run many generated cases through both backends and summarise what happened.
//!
//! This is the first honest look at what the tool reports on varied input. The
//! comparison is still **exact equality**, which is deliberately the wrong tool for
//! floating-point results — two correct implementations routinely differ in the last
//! bits, because addition is not associative and different kernels accumulate in
//! different orders. So the disagreements counted here are expected to be mostly
//! noise, and the point of running it is to find out *which operations* produce that
//! noise and how much.
//!
//! Run with: `cargo run --release -p tensor-adapter --example survey`

use diff_fuzzer_core::{
    DifferentialOracle, FixedTolerance, Generator, NormalizedRunner, Runner, SeededRng, Tolerance,
    Verdict, driver::run_once,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, libtorch, ndarray,
};

const CASES: u64 = 5_000;

#[derive(Default)]
struct Tally {
    agreed: usize,
    diverged: usize,
    skipped: usize,
}

fn main() {
    let generator = TensorOpGenerator::default();
    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &torch];
    // Exact comparison for now: the tolerance policy that varies by operation
    // arrives next.
    let oracle: DifferentialOracle<TensorOp, CanonicalTensor, FixedTolerance> =
        DifferentialOracle::new(FixedTolerance(Tolerance::EXACT));

    let mut totals = Tally::default();
    let mut per_operation: BTreeMap<&str, Tally> = BTreeMap::new();
    let mut first_disagreement: Option<(u64, String)> = None;

    let started = std::time::Instant::now();

    for seed in 0..CASES {
        // Regenerated purely to label the row; the driver builds the identical case
        // from the same seed, which is the determinism guarantee being relied on.
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let outcome = run_once(seed, &generator, &runners, &oracle);

        let entry = per_operation.entry(case.name()).or_default();
        match &outcome.verdict {
            Verdict::Agree => {
                totals.agreed += 1;
                entry.agreed += 1;
            }
            Verdict::Skipped(_) => {
                totals.skipped += 1;
                entry.skipped += 1;
            }
            Verdict::Diverged(divergence) => {
                totals.diverged += 1;
                entry.diverged += 1;
                if first_disagreement.is_none() {
                    first_disagreement = Some((seed, divergence.to_string()));
                }
            }
        }
    }

    let elapsed = started.elapsed();

    println!("{CASES} cases, exact comparison, burn-ndarray vs burn-tch\n");
    println!(
        "  agreed   {:>6}  ({:.1}%)",
        totals.agreed,
        100.0 * totals.agreed as f64 / CASES as f64
    );
    println!(
        "  disagreed{:>6}  ({:.1}%)",
        totals.diverged,
        100.0 * totals.diverged as f64 / CASES as f64
    );
    println!("  skipped  {:>6}", totals.skipped);
    println!(
        "\n  {:.0} cases/sec ({:.2?} total)",
        CASES as f64 / elapsed.as_secs_f64(),
        elapsed
    );

    println!("\nby operation:");
    println!(
        "  {:<8} {:>7} {:>10} {:>8}",
        "op", "agreed", "disagreed", "rate"
    );
    for (name, tally) in &per_operation {
        let total = tally.agreed + tally.diverged + tally.skipped;
        println!(
            "  {:<8} {:>7} {:>10} {:>7.1}%",
            name,
            tally.agreed,
            tally.diverged,
            100.0 * tally.diverged as f64 / total as f64
        );
    }

    if let Some((seed, text)) = first_disagreement {
        println!("\nfirst disagreement (seed {seed}):");
        for line in text.lines().take(4) {
            // Long tensors make this unreadable; the full case is reproducible from the
            // seed, which is the point of recording it.
            let trimmed: String = line.chars().take(160).collect();
            println!("  {trimmed}");
        }
    }
}
