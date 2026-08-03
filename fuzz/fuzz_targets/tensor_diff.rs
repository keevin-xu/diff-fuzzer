//! The fuzzing entry point: bytes in, a verdict out.
//!
//! libFuzzer calls this function repeatedly with byte strings it generates, and — the
//! part that matters — it **watches which branches the program takes**. Inputs that
//! reach new code get kept and mutated further, so over time the corpus evolves toward
//! parts of the target nothing has exercised yet. That is the difference between this
//! and the seed-driven campaign: the campaign generates blindly, while this one is
//! steered by what the code under test actually does.
//!
//! Because `burn` compiles from source as an ordinary dependency, the instrumentation
//! reaches **into the library being tested**, not just into our harness. That is the
//! capability this project chose Rust for, and it is not available to a fuzzer driving
//! a compiled library from another language.
//!
//! **A panic is how a bug is signalled.** libFuzzer treats it as a crash and saves the
//! offending bytes to `fuzz/artifacts/`, so the case can be replayed. Hence the shape
//! below: on a confirmed divergence, write a report and then panic.
//!
//! Run with:
//! ```text
//! cargo +nightly fuzz run tensor_diff
//! ```

#![no_main]

use diff_fuzzer_core::{
    DifferentialOracle, DivergenceReport, Generator, MinimisationRecord, NamedOutput,
    NormalizedRunner, Oracle, Runner, SeededRng, TolerancePolicy, Verdict, minimize,
};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, TensorTolerancePolicy,
    environment, libtorch, ndarray,
};

/// Everything that is expensive to build, constructed once and reused.
///
/// Backend setup — and libtorch initialisation in particular — costs far more than
/// running a small tensor operation. Doing it per execution would make the harness
/// itself the bottleneck, and **throughput is bugs found**: a harness that halves the
/// execution rate halves the ground covered in a campaign.
struct Harness {
    generator: TensorOpGenerator,
    cpu: NormalizedRunner<tensor_adapter::NdArrayBackend, TensorNormalizer>,
    torch: NormalizedRunner<tensor_adapter::LibTorchBackend, TensorNormalizer>,
    oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
}

static HARNESS: OnceLock<Harness> = OnceLock::new();

fn harness() -> &'static Harness {
    HARNESS.get_or_init(|| Harness {
        generator: TensorOpGenerator::default(),
        cpu: NormalizedRunner::new(ndarray(), TensorNormalizer),
        torch: NormalizedRunner::new(libtorch(), TensorNormalizer),
        oracle: DifferentialOracle::new(TensorTolerancePolicy),
    })
}

fuzz_target!(|data: &[u8]| {
    let harness = harness();
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] =
        [&harness.cpu, &harness.torch];

    // Bytes to a case.
    //
    // **This is the weak part of the target, and it is deliberate for now.** Turning the
    // bytes into a seed and generating from that means a one-bit mutation produces a
    // completely unrelated case, so libFuzzer's coverage feedback has nothing to act on
    // — it cannot learn that "this input was close, try something like it". Decoding the
    // bytes *into the case structure*, so that small mutations make small changes, is
    // the next step, and is what makes the coverage guidance worth having.
    let Some(seed) = seed_from(data) else { return };
    let case = harness.generator.generate(&mut SeededRng::from_seed(seed));

    let outputs = run_all(&case, &runners);
    let Verdict::Diverged(divergence) = harness.oracle.check(&case, &outputs) else {
        return;
    };

    // A divergence: shrink it, record it, then panic so libFuzzer preserves the input.
    let diverges = |candidate: &TensorOp| {
        let outputs = run_all(candidate, &runners);
        matches!(
            harness.oracle.check(candidate, &outputs),
            Verdict::Diverged(_)
        )
    };
    let minimized = minimize(case.clone(), diverges);

    let report = DivergenceReport {
        seed,
        label: case.name().to_string(),
        generator: format!("{:?}", harness.generator.bounds),
        input: minimized.input.clone(),
        minimisation: MinimisationRecord::from(&minimized),
        outputs: divergence.outputs.clone(),
        tolerance: TensorTolerancePolicy.tolerance_for(&minimized.input),
        environment: environment(),
        summary: divergence.summary.clone(),
    };

    // Written before panicking. The panic is what makes libFuzzer keep the input; the
    // report is what makes the finding readable by a person.
    let path = format!("findings/{}", report.filename());
    if let Err(error) = report.save(&path) {
        eprintln!("could not save report to {path}: {error}");
    }

    panic!("divergence in {} (seed {}): {}", case.name(), seed, report.summary);
});

/// Run a case on every implementation, keeping whatever succeeded.
///
/// An implementation that cannot run a case is left out rather than counted against it;
/// the oracle then decides whether enough remains to compare.
fn run_all(
    case: &TensorOp,
    runners: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
) -> Vec<NamedOutput<CanonicalTensor>> {
    runners
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
        .collect()
}

/// The first eight bytes as a seed.
///
/// Returns `None` for anything shorter, which tells libFuzzer nothing interesting
/// happened. Padding short inputs instead would map many distinct byte strings onto the
/// same case, wasting the fuzzer's budget on duplicates.
fn seed_from(data: &[u8]) -> Option<u64> {
    let bytes: [u8; 8] = data.get(..8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}
