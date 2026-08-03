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
    DifferentialOracle, DivergenceReport, MinimisationRecord, NamedOutput, NormalizedRunner,
    Oracle, Runner, TolerancePolicy, Verdict, minimize,
};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, environment, libtorch,
    ndarray,
};

/// Everything that is expensive to build, constructed once and reused.
///
/// Backend setup — and libtorch initialisation in particular — costs far more than
/// running a small tensor operation. Doing it per execution would make the harness
/// itself the bottleneck, and **throughput is bugs found**: a harness that halves the
/// execution rate halves the ground covered in a campaign.
struct Harness {
    cpu: NormalizedRunner<tensor_adapter::NdArrayBackend, TensorNormalizer>,
    torch: NormalizedRunner<tensor_adapter::LibTorchBackend, TensorNormalizer>,
    oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
}

static HARNESS: OnceLock<Harness> = OnceLock::new();

fn harness() -> &'static Harness {
    HARNESS.get_or_init(|| Harness {
        cpu: NormalizedRunner::new(ndarray(), TensorNormalizer),
        torch: NormalizedRunner::new(libtorch(), TensorNormalizer),
        oracle: DifferentialOracle::new(TensorTolerancePolicy),
    })
}

// The case arrives already decoded. `TensorOp` implements `Arbitrary` by mapping the
// fuzzer's bytes onto a fixed layout — operation, rank, dimensions, then one byte per
// value — so a mutation late in the input perturbs a value while leaving the shape
// intact. That locality is what lets libFuzzer's coverage feedback mean anything: it can
// explore *around* an interesting input instead of being thrown to an unrelated one.
fuzz_target!(|case: TensorOp| {
    let harness = harness();
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] =
        [&harness.cpu, &harness.torch];

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
        // No seed: this case came from the fuzzer's bytes, not from a seeded generator.
        // Zero records that honestly rather than inventing a number that would reproduce
        // something else entirely — the `input` field is what reproduces this finding,
        // and libFuzzer keeps the bytes in `fuzz/artifacts/` besides.
        seed: 0,
        label: case.name().to_string(),
        generator: "decoded from fuzzer bytes".to_string(),
        input: minimized.input.clone(),
        minimisation: MinimisationRecord::from(&minimized),
        outputs: divergence.outputs.clone(),
        tolerance: TensorTolerancePolicy.tolerance_for(&minimized.input),
        environment: environment(),
        summary: divergence.summary.clone(),
    };

    // Written before panicking. The panic is what makes libFuzzer keep the input; the
    // report is what makes the finding readable by a person.
    // Named by the operation and a hash of the case, since there is no seed to key on.
    let path = format!("findings/fuzz-{}-{:x}.json", case.name(), case_digest(&case));
    if let Err(error) = report.save(&path) {
        eprintln!("could not save report to {path}: {error}");
    }

    panic!("divergence in {}: {}", case.name(), report.summary);
});

/// A short stable digest of a case, for naming its report file.
///
/// Derived from the case itself so that re-finding the same divergence overwrites one
/// file rather than accumulating copies.
fn case_digest(case: &TensorOp) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{case:?}").hash(&mut hasher);
    hasher.finish()
}

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
