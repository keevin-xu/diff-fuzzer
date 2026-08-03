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
    Budget, DifferentialOracle, Divergence, DivergenceReport, MinimisationRecord, NamedOutput,
    NormalizedRunner, Oracle, Runner, TolerancePolicy, Verdict, minimize_within,
};
use libfuzzer_sys::fuzz_target;
use std::sync::OnceLock;
use tensor_adapter::{
    CanonicalTensor, FaultyNdArray, TensorNormalizer, TensorOp, TensorTolerancePolicy, environment,
    faulty as faulty_backend, libtorch, ndarray,
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
    /// A backend wrong by a known amount, used only when the fault switch is set.
    faulty: NormalizedRunner<FaultyNdArray, TensorNormalizer>,
    oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
    /// Whether to compare against the faulty backend instead of libtorch.
    inject_fault: bool,
}

static HARNESS: OnceLock<Harness> = OnceLock::new();

fn harness() -> &'static Harness {
    HARNESS.get_or_init(|| Harness {
        cpu: NormalizedRunner::new(ndarray(), TensorNormalizer),
        torch: NormalizedRunner::new(libtorch(), TensorNormalizer),
        faulty: NormalizedRunner::new(faulty_backend(0.5), TensorNormalizer),
        oracle: DifferentialOracle::new(TensorTolerancePolicy),
        // **The switch that makes a clean campaign mean something.**
        //
        // A fuzzing run that finds nothing is indistinguishable from one whose detection
        // has quietly broken — and unlike the library code, this target's divergence path
        // is not covered by `cargo test`, because a fuzz target cannot be unit-tested.
        // Setting `DIFF_FUZZER_INJECT_FAULT=1` compares against a backend wrong by a known
        // amount, so the whole path — detect, shrink, report, panic, preserve the input —
        // can be demonstrated on demand rather than trusted.
        inject_fault: std::env::var("DIFF_FUZZER_INJECT_FAULT").is_ok(),
    })
}

/// How hard to shrink inside a fuzz iteration.
///
/// Tighter than the default. Every candidate is a full run on both backends, and libFuzzer
/// measures how long an input takes — a minimisation that ran for minutes would look like a
/// hang and could get the input reported as a timeout rather than a divergence. A slightly
/// larger reproduction that arrives promptly is the better trade here; the campaign runner
/// has no such constraint and uses the full budget.
const FUZZ_MINIMISATION: Budget = Budget {
    max_steps: 50,
    max_candidates: 500,
    max_duration: None,
};

// The case arrives already decoded. `TensorOp` implements `Arbitrary` by mapping the
// fuzzer's bytes onto a fixed layout — operation, rank, dimensions, then one byte per
// value — so a mutation late in the input perturbs a value while leaving the shape
// intact. That locality is what lets libFuzzer's coverage feedback mean anything: it can
// explore *around* an interesting input instead of being thrown to an unrelated one.
fuzz_target!(|case: TensorOp| {
    let harness = harness();
    let second: &dyn Runner<In = TensorOp, Canon = CanonicalTensor> = if harness.inject_fault {
        &harness.faulty
    } else {
        &harness.torch
    };
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&harness.cpu, second];

    let outputs = run_all(&case, &runners);
    if !matches!(harness.oracle.check(&case, &outputs), Verdict::Diverged(_)) {
        return;
    }

    // A divergence: shrink it, record it, then panic so libFuzzer preserves the input.
    let diverges = |candidate: &TensorOp| {
        let outputs = run_all(candidate, &runners);
        matches!(
            harness.oracle.check(candidate, &outputs),
            Verdict::Diverged(_)
        )
    };
    let minimized = minimize_within(case.clone(), FUZZ_MINIMISATION, diverges);

    // Describe the *minimised* case, not the original. Pairing a one-element input with
    // a summary of a hundred values leaves a reader unable to tell which they are
    // looking at — the same incoherence the campaign runner had, and worth fixing in
    // both places rather than only where it was noticed.
    let described = describe(&minimized.input, &runners, &harness.oracle);
    let Some(divergence) = described else {
        // Unreachable: the predicate only accepted candidates that diverge.
        return;
    };

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
        outputs: divergence.outputs,
        tolerance: TensorTolerancePolicy.tolerance_for(&minimized.input),
        environment: environment(),
        summary: divergence.summary,
    };

    // Written before panicking. The panic is what makes libFuzzer keep the input; the
    // report is what makes the finding readable by a person.
    // Resolved against this crate's location rather than the working directory, which
    // for a fuzz target depends on how it was invoked. Reports otherwise land in
    // `fuzz/findings/` while the campaign runner writes to `findings/`, and findings
    // split across two directories is how one set of them gets forgotten.
    //
    // Named by the operation and a hash of the case, since there is no seed to key on.
    //
    // Filed under `runs/<run>/<operation>/`. A sustained campaign produces hundreds of
    // reports, and a few hundred files in one directory is a pile rather than a result:
    // two campaigns weeks apart become indistinguishable, and the operation — the first
    // thing anyone wants to sort by — is buried in the filename.
    //
    // The run name comes from the environment because **each crash happens in a fresh
    // process** under `-fork=1`, so the children cannot agree on anything held in memory.
    // An environment variable is inherited by every child, which makes it the one channel
    // that survives. (Unlike `DYLD_*` on macOS, which SIP strips — the reason fork mode
    // was silently spawning children that died before executing anything.)
    let path = format!(
        "{}/../findings/runs/{}/{}/fuzz-{}-{:x}.json",
        env!("CARGO_MANIFEST_DIR"),
        run_label(),
        case.name(),
        case.name(),
        case_digest(&case)
    );
    if let Err(error) = report.save(&path) {
        eprintln!("could not save report to {path}: {error}");
    }

    panic!("divergence in {}: {}", case.name(), report.summary);
});

/// Which run's directory this process should file its findings under.
///
/// Read from `DIFF_FUZZER_RUN`, set once by whoever launches the campaign. When it is
/// unset the label is `unlabelled` rather than something plausible-looking: an ad-hoc
/// replay must not be able to quietly deposit its output into a real campaign's results,
/// and a directory named `unlabelled` says what happened instead of hiding it.
///
/// Sanitised to a conservative character set, because the value ends up in a path and an
/// environment variable is caller-supplied input like any other.
fn run_label() -> String {
    let raw = std::env::var("DIFF_FUZZER_RUN").unwrap_or_default();
    let cleaned: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        .take(96)
        .collect();

    // `.` survives the filter because dates and versions want it, which leaves `..` — a
    // label of `..` would escape the run directory and scatter findings a level up. Any
    // all-dots label is rejected rather than trusted.
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        "unlabelled".to_string()
    } else {
        cleaned
    }
}

/// Run a case and return the divergence it produces, if any.
fn describe(
    case: &TensorOp,
    runners: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
    oracle: &DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
) -> Option<Divergence> {
    match oracle.check(case, &run_all(case, runners)) {
        Verdict::Diverged(divergence) => Some(divergence),
        _ => None,
    }
}

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
