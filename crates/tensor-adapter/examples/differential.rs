//! The pipeline end to end: a seed goes in, a verdict comes out.
//!
//! Everything the finished tool does is present here in miniature — generate a case,
//! run it on both backends, put the results in a comparable form, judge them, and say
//! what happened, with the seed attached so any outcome can be replayed. What is
//! missing is depth, not structure: one hardcoded case instead of generated ones, and
//! exact comparison instead of a tolerance.
//!
//! Run with: `cargo run -p tensor-adapter --example differential`

use diff_fuzzer_core::{DifferentialOracle, NormalizedRunner, Runner, Verdict, driver::run_once};
use tensor_adapter::{
    CanonicalTensor, FaultyBackend, FixedAddGenerator, TensorNormalizer, TensorOp, libtorch,
    ndarray,
};

fn main() {
    // The engine only emits log events; a program decides whether to listen. Turning
    // this on shows the per-case detail, including the seed on every line.
    tracing_subscriber::fmt()
        .with_max_level(tracing_subscriber::filter::LevelFilter::DEBUG)
        .with_target(false)
        .init();

    // Each backend is paired with the normaliser for its output. After pairing, the
    // two have the same type from the driver's point of view, despite producing
    // different kinds of tensor internally — which is why the driver takes a list and
    // not exactly two arguments.
    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &torch];

    let oracle: DifferentialOracle<TensorOp, CanonicalTensor> = DifferentialOracle::new();

    for seed in 0..3 {
        let outcome = run_once(seed, &FixedAddGenerator, &runners, &oracle);

        let verdict = match &outcome.verdict {
            Verdict::Agree => "agree".to_string(),
            Verdict::Skipped(reason) => format!("skipped ({reason})"),
            Verdict::Diverged(divergence) => format!("DIVERGED: {}", divergence.summary),
        };
        println!("seed {:<3} -> {verdict}", outcome.seed);
    }

    // The same seed must always give the same verdict. It is stated here as well as in
    // the tests because it is the property everything else rests on: a finding that
    // cannot be replayed from its seed is a defect in this tool rather than a
    // discovery about anything else.
    let first = run_once(7, &FixedAddGenerator, &runners, &oracle);
    let again = run_once(7, &FixedAddGenerator, &runners, &oracle);
    println!("\nseed 7 replayed identically: {}", first == again);

    // Two correct backends agreeing proves very little on its own — a comparison that
    // had quietly stopped working would report exactly the same thing. So: introduce a
    // backend known to be wrong by a fixed amount, and confirm it gets caught.
    let faulty = NormalizedRunner::new(FaultyBackend::new(ndarray(), 0.5), TensorNormalizer);
    let with_fault: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &faulty];

    println!("\nwith a deliberately faulty backend:");
    match run_once(0, &FixedAddGenerator, &with_fault, &oracle).verdict {
        Verdict::Diverged(divergence) => print!("{divergence}"),
        other => println!("  fault NOT caught — the detector is broken: {other:?}"),
    }
}
