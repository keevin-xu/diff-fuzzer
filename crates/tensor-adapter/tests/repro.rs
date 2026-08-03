//! The whole reporting path, end to end: diverge, shrink, save, load, reproduce.
//!
//! Each stage is tested in isolation elsewhere. What this file checks is that they
//! compose — that a divergence found at generated size survives being shrunk, written to
//! disk, read back in a fresh process's worth of state, and re-run to the same
//! conclusion. A pipeline whose stages all pass individually can still lose the finding
//! at a seam.
//!
//! The divergence is produced by a deliberately faulty backend rather than waited for,
//! since the real backends agree on everything the policy permits. That is the right way
//! round: the mechanism has to be provable on demand, not only when a genuine bug happens
//! to turn up.

use diff_fuzzer_core::{
    Agreement, ApproxEq, DivergenceReport, MinimisationRecord, NormalizedRunner, Runner, Tolerance,
    load_report, minimize,
};
use tensor_adapter::{
    BinaryOp, CanonicalTensor, FaultyBackend, TensorNormalizer, TensorOp, TensorValue, environment,
    libtorch, ndarray, repro::reproduce,
};

type AnyRunner<'a> = &'a dyn Runner<In = TensorOp, Canon = CanonicalTensor>;

/// The tolerance findings are judged against here. Exact, because the injected fault is
/// far larger than any rounding and nothing should absorb it.
const TOLERANCE: Tolerance = Tolerance::EXACT;

fn temp_path(name: &str) -> std::path::PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("diff-fuzzer-{name}-{unique}.json"))
}

/// A large case: 4 x 6 x 5, or 120 values per operand.
fn large_case() -> TensorOp {
    let count = 4 * 6 * 5;
    let values = |offset: f32| {
        TensorValue::new(
            vec![4, 6, 5],
            (0..count).map(|i| i as f32 * 0.25 + offset).collect(),
        )
    };
    TensorOp::binary(BinaryOp::Add, values(1.0), values(2.0))
}

fn element_count(case: &TensorOp) -> usize {
    match case {
        TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => arg.len(),
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => lhs.len() + rhs.len(),
    }
}

/// Does this case still diverge between a correct backend and a faulty one?
fn still_diverges(case: &TensorOp) -> bool {
    let correct = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let faulty = NormalizedRunner::new(FaultyBackend::new(ndarray(), 0.5), TensorNormalizer);

    let (Ok(a), Ok(b)) = (
        correct.run_and_normalize(case),
        faulty.run_and_normalize(case),
    ) else {
        return false;
    };

    !matches!(a.approx_compare(&b, TOLERANCE), Agreement::Agree(_))
}

/// **The end-to-end property this file exists for.** A divergence found at generated size
/// must survive shrinking, serialisation, and reloading, and still reproduce.
#[test]
fn a_divergence_survives_shrinking_saving_and_reloading() {
    let original = large_case();
    assert!(
        still_diverges(&original),
        "the injected fault was not caught"
    );

    // Shrink.
    let minimized = minimize(original.clone(), still_diverges);
    assert!(
        still_diverges(&minimized.input),
        "shrinking produced a case that no longer diverges"
    );

    // Report.
    let report = DivergenceReport {
        seed: 4242,
        label: minimized.input.name().to_string(),
        generator: "hand-built large case".to_string(),
        input: minimized.input.clone(),
        minimisation: MinimisationRecord::from(&minimized),
        outputs: vec![
            ("burn-ndarray".to_string(), "recorded".to_string()),
            (
                "burn-ndarray+fault(0.5)".to_string(),
                "recorded".to_string(),
            ),
        ],
        tolerance: TOLERANCE,
        environment: environment(),
        summary: "injected fault".to_string(),
    };

    // Save and reload.
    let path = temp_path("round-trip");
    report.save(&path).unwrap();
    let loaded: DivergenceReport<TensorOp> = load_report(&path).unwrap();
    assert_eq!(loaded.input, minimized.input, "the case changed on disk");

    // Reproduce from what was loaded, not from what is still in memory.
    let correct = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let faulty = NormalizedRunner::new(FaultyBackend::new(ndarray(), 0.5), TensorNormalizer);
    let implementations: [AnyRunner; 2] = [&correct, &faulty];

    let outcome = reproduce(&loaded, &implementations);
    assert!(
        outcome.reproduced,
        "a saved report did not reproduce: {}",
        outcome.detail
    );

    std::fs::remove_file(&path).ok();
}

/// Shrinking has to actually shrink. A minimiser that returns its input unchanged would
/// pass every correctness test in the suite while being useless.
#[test]
fn minimisation_substantially_reduces_the_case() {
    let original = large_case();
    let before = element_count(&original);

    let minimized = minimize(original, still_diverges);
    let after = element_count(&minimized.input);

    assert!(
        after * 10 <= before,
        "only shrank from {before} to {after} values"
    );
    assert!(
        minimized.is_minimal(),
        "stopped early: {}",
        minimized.stopped
    );
    assert!(minimized.steps > 0);
}

/// The shrunk case must remain **valid** — runnable on a real backend, not merely
/// smaller. An invalid "minimal" case is worse than none, since it cannot be sent to
/// anyone.
#[test]
fn the_minimised_case_still_runs_on_a_real_backend() {
    let minimized = minimize(large_case(), still_diverges);

    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);

    assert!(cpu.run_and_normalize(&minimized.input).is_ok());
    assert!(torch.run_and_normalize(&minimized.input).is_ok());
}

/// Minimising the same divergence twice must give the same case, or a "minimal
/// reproduction" would differ between runs and be no more actionable than the original.
#[test]
fn minimisation_of_the_same_case_is_reproducible() {
    let first = minimize(large_case(), still_diverges);
    let second = minimize(large_case(), still_diverges);

    assert_eq!(first.input, second.input);
    assert_eq!(first.steps, second.steps);
}

/// A report whose divergence has gone away must say so plainly rather than quietly
/// passing. Two genuinely correct backends stand in for "the bug is gone".
#[test]
fn a_report_that_no_longer_diverges_is_reported_as_such() {
    let minimized = minimize(large_case(), still_diverges);

    let record = MinimisationRecord::from(&minimized);
    let report = DivergenceReport {
        seed: 1,
        label: minimized.input.name().to_string(),
        generator: "hand-built".to_string(),
        input: minimized.input,
        minimisation: record,
        outputs: vec![],
        tolerance: TOLERANCE,
        environment: environment(),
        summary: "injected fault".to_string(),
    };

    // Replayed against two *correct* backends, so the fault is absent.
    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let implementations: [AnyRunner; 2] = [&cpu, &torch];

    let outcome = reproduce(&report, &implementations);
    assert!(!outcome.reproduced);
    // And it must not leap to "fixed" — several things could explain a disappearance,
    // and only one of them is good news.
    assert!(
        outcome.detail.contains("Possible causes"),
        "{}",
        outcome.detail
    );
}

/// A report carries everything a stranger needs: the case itself, the tolerance it was
/// judged against, and the versions it applies to. Checked on the *loaded* copy, since
/// that is what a recipient actually has.
#[test]
fn a_saved_report_is_self_contained() {
    let minimized = minimize(large_case(), still_diverges);
    let record = MinimisationRecord::from(&minimized);
    let report = DivergenceReport {
        seed: 99,
        label: minimized.input.name().to_string(),
        generator: "Bounds { max_dim: 8 }".to_string(),
        input: minimized.input,
        minimisation: record,
        outputs: vec![("a".to_string(), "[1.0]".to_string())],
        tolerance: TOLERANCE,
        environment: environment(),
        summary: "injected fault".to_string(),
    };

    let path = temp_path("self-contained");
    report.save(&path).unwrap();
    let loaded: DivergenceReport<TensorOp> = load_report(&path).unwrap();

    // The case, in full — not a seed to be regenerated.
    assert_eq!(loaded.input, report.input);
    // The threshold the claim was made against.
    assert_eq!(loaded.tolerance, TOLERANCE);
    // The versions it applies to.
    assert!(
        loaded
            .environment
            .components
            .iter()
            .any(|(name, _)| name == "burn")
    );
    assert!(!loaded.environment.platform.is_empty());

    std::fs::remove_file(&path).ok();
}
