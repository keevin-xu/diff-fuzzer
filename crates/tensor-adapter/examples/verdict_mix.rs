//! Are the operations judgeable now, and what do they actually say?
//!
//! The tolerance audit at PHASE-7F found that 81% of `exp` cases and 65% of `softmax` cases
//! carried a bound nothing could fail, and were reported as **agreement**. Capping the
//! condition-number term where the function saturates fixed the bound. This asks the question
//! that actually matters: with the fix in, what verdicts do those operations now produce?
//!
//! **Agreement, divergence and skip are reported separately per operation**, because a
//! campaign that "found nothing" can mean two entirely different things and the whole point
//! of PHASE-7F was that the difference had been invisible.
use diff_fuzzer_core::{
    DifferentialOracle, Generator, NamedOutput, NormalizedRunner, Oracle, Runner, SeededRng,
    Verdict,
};
use std::collections::BTreeMap;
use tensor_adapter::ops::Bounds;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, TensorTolerancePolicy, flex,
    libtorch, wgpu,
};

#[derive(Default)]
struct Tally {
    agreed: usize,
    diverged: usize,
    unjudgeable: usize,
    other_skip: usize,
}

fn main() {
    let cases: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(4_000);

    // The operations whose backends run genuinely different algorithms, given the whole
    // budget instead of competing with a class already understood.
    // `control` adds back `max`/`min`, whose disagreement is known. **A run reporting zero
    // divergences is uninformative unless the instrument is shown to still detect one** — the
    // same reasoning as the fault-injected backend, applied to a configuration rather than to
    // the harness.
    let control = std::env::args().any(|a| a == "control");
    let bounds = Bounds {
        restrict_domains: false,
        magnitude: 1000.0,
        selecting_reductions: control,
        ..Bounds::NUMERICALLY_INTERESTING
    };
    let generator = TensorOpGenerator::new(bounds);

    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let gpu = NormalizedRunner::new(wgpu(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 3] = [&cpu, &torch, &gpu];

    let mut tally: BTreeMap<&str, Tally> = BTreeMap::new();

    for seed in 0..cases {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
            .iter()
            .filter_map(|r| {
                r.run_and_normalize(&case).ok().map(|output| NamedOutput {
                    implementation: r.name().to_string(),
                    output,
                })
            })
            .collect();

        let entry = tally.entry(case.name()).or_default();
        match oracle.check(&case, &outputs) {
            Verdict::Agree => entry.agreed += 1,
            Verdict::Diverged(_) => entry.diverged += 1,
            Verdict::Skipped(reason) => {
                if format!("{reason:?}").starts_with("Unjudgeable") {
                    entry.unjudgeable += 1;
                } else {
                    entry.other_skip += 1;
                }
            }
        }
    }

    println!("{cases} cases, three backends, unrestricted domains, magnitude 1000\n");
    println!(
        "{:<10} {:>8} {:>9} {:>12} {:>11}",
        "op", "agreed", "diverged", "unjudgeable", "other skip"
    );
    for (name, t) in &tally {
        println!(
            "{name:<10} {:>8} {:>9} {:>12} {:>11}",
            t.agreed, t.diverged, t.unjudgeable, t.other_skip
        );
    }

    let total_unjudgeable: usize = tally.values().map(|t| t.unjudgeable).sum();
    println!(
        "\n{} case(s) carried a bound nothing could fail — before PHASE-7F these were counted \
         as agreement.",
        total_unjudgeable
    );
}
