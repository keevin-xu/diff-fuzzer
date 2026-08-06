//! How many cases per second, and where the time goes.
//!
//! Throughput is not a vanity metric for a fuzzer — bugs found scale with cases
//! executed, so a harness that wastes time directly costs findings. The number worth
//! watching is the *split*: time spent in the backends is time spent testing them,
//! while time spent generating or comparing is overhead. Overhead that grows is a
//! regression even if the total looks respectable.
//!
//! Build in release. Debug builds are several times slower and say nothing useful
//! about a real campaign.
//!
//! Run with: `cargo run --release -p tensor-adapter --example throughput`

use diff_fuzzer_core::{
    DifferentialOracle, FixedTolerance, Generator, Implementation, NormalizedRunner, Normalizer,
    Oracle, Runner, SeededRng, Tolerance, driver::run_once, traits::NamedOutput,
};
use std::time::Instant;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, flex, libtorch,
};

const CASES: u64 = 50_000;

fn rate(count: u64, seconds: f64) -> String {
    let per_second = count as f64 / seconds;
    if per_second >= 1_000_000.0 {
        format!("{:.1}M/sec", per_second / 1_000_000.0)
    } else {
        format!("{:.0}k/sec", per_second / 1_000.0)
    }
}

fn main() {
    // `wide` measures the cost of the shapes the fuzzer's decoder now reaches, so the
    // throughput consequence of widening `max_dim` is a measurement rather than a guess.
    let generator = match std::env::args().nth(1).as_deref() {
        Some("wide") => TensorOpGenerator::new(tensor_adapter::Bounds {
            max_dim: 64,
            ..Default::default()
        }),
        _ => TensorOpGenerator::default(),
    };

    // Generation alone.
    let start = Instant::now();
    let cases: Vec<TensorOp> = (0..CASES)
        .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
        .collect();
    let generation = start.elapsed().as_secs_f64();

    // Execution on each backend separately, so a slow backend is visible rather than
    // averaged away.
    let cpu = flex();
    let start = Instant::now();
    for case in &cases {
        let _ = cpu.run(case).expect("valid case");
    }
    let cpu_execution = start.elapsed().as_secs_f64();

    let torch = libtorch();
    let start = Instant::now();
    for case in &cases {
        let _ = torch.run(case).expect("valid case");
    }
    let torch_execution = start.elapsed().as_secs_f64();

    // Comparison alone. Everything the oracle needs is built *before* the clock
    // starts, so this measures the comparison itself rather than the cost of setting
    // it up — otherwise the allocation would be attributed to the wrong stage.
    // Exact comparison for now: the tolerance policy that varies by operation
    // arrives next.
    let oracle: DifferentialOracle<TensorOp, CanonicalTensor, FixedTolerance> =
        DifferentialOracle::new(FixedTolerance(Tolerance::EXACT));
    let prepared: Vec<(TensorOp, [NamedOutput<CanonicalTensor>; 2])> = cases
        .iter()
        .map(|case| {
            let outputs = [
                NamedOutput {
                    implementation: "flex".to_string(),
                    output: TensorNormalizer.normalize(cpu.run(case).unwrap()),
                },
                NamedOutput {
                    implementation: "libtorch".to_string(),
                    output: TensorNormalizer.normalize(torch.run(case).unwrap()),
                },
            ];
            (case.clone(), outputs)
        })
        .collect();

    let start = Instant::now();
    for (case, outputs) in &prepared {
        let _ = oracle.check(case, outputs);
    }
    let comparison = start.elapsed().as_secs_f64();

    // The whole pipeline, which is what a campaign actually runs.
    let cpu_runner = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch_runner = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] =
        [&cpu_runner, &torch_runner];

    let start = Instant::now();
    for seed in 0..CASES {
        let _ = run_once(seed, &generator, &runners, &oracle);
    }
    let pipeline = start.elapsed().as_secs_f64();

    println!("{CASES} cases\n");
    println!("  {:<22} {:>10}", "generation", rate(CASES, generation));
    println!(
        "  {:<22} {:>10}",
        "execute on flex",
        rate(CASES, cpu_execution)
    );
    println!(
        "  {:<22} {:>10}",
        "execute on libtorch",
        rate(CASES, torch_execution)
    );
    println!(
        "  {:<22} {:>10}",
        "normalise + compare",
        rate(CASES, comparison)
    );
    println!("  {:<22} {:>10}", "whole pipeline", rate(CASES, pipeline));

    let overhead = generation + comparison;
    let backends = cpu_execution + torch_execution;
    println!(
        "\n  time in the backends: {:.0}%   overhead (generate + compare): {:.0}%",
        100.0 * backends / (backends + overhead),
        100.0 * overhead / (backends + overhead)
    );
}
