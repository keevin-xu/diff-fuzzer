//! Which axis of `Bounds` actually produces divergences?
//!
//! The wide generator diverged on 7 of 4,000 cases and the default on 0 of 4,000, but they
//! differ on three axes at once. Changing the fuzzer's decode bounds on the strength of
//! that would be guessing at which one mattered. This varies one axis at a time.
use diff_fuzzer_core::{
    DifferentialOracle, NamedOutput, NormalizedRunner, Oracle, Runner, Verdict,
};
use tensor_adapter::validation;
use tensor_adapter::{
    Bounds, CanonicalTensor, Predicate, TensorNormalizer, TensorOp, TensorOpGenerator,
    TensorTolerancePolicy, flex, libtorch,
};

const EVERYTHING: Predicate = Predicate {
    required: 0,
    forbidden: 0,
};

fn main() {
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);

    let diverges = |case: &TensorOp| {
        let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &torch];
        let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
            .iter()
            .filter_map(|r| {
                r.run_and_normalize(case).ok().map(|output| NamedOutput {
                    implementation: r.name().to_string(),
                    output,
                })
            })
            .collect();
        matches!(oracle.check(case, &outputs), Verdict::Diverged(_))
    };

    let base = Bounds::default();

    // **The decision metric is divergences per second**, not hit rate alone. A wider bound
    // raises the hit rate and lowers throughput at the same time, and only their product
    // says which setting finds more disagreements in a campaign of fixed length.
    let configs: Vec<(String, Bounds)> = vec![
        ("max_dim 8 (historical)".into(), base),
        (
            "max_dim 64, budget 4k".into(),
            Bounds {
                max_dim: 64,
                ..base
            },
        ),
        (
            "max_dim 64, budget 64k".into(),
            Bounds {
                max_dim: 64,
                max_elements: 65_536,
                ..base
            },
        ),
        (
            "max_dim 64, budget 1M".into(),
            Bounds {
                max_dim: 64,
                max_elements: 1_048_576,
                ..base
            },
        ),
        (
            "magnitude 1000".into(),
            Bounds {
                magnitude: 1000.0,
                ..base
            },
        ),
    ];

    println!(
        "{:<26} {:>8} {:>7} {:>8} {:>12}",
        "bounds", "diverged", "of", "sec", "diverg/sec"
    );
    for (name, bounds) in configs {
        let started = std::time::Instant::now();
        let result = validation::validate(
            EVERYTHING,
            &TensorOpGenerator::new(bounds),
            20_260_805,
            2_000,
            &diverges,
        );
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "{name:<26} {:>8} {:>7} {:>8.1} {:>12.3}",
            result.diverged,
            result.matched,
            elapsed,
            result.diverged as f64 / elapsed
        );
    }
}
