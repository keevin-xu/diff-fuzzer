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
    BinaryOp, CanonicalTensor, FaultyBackend, ReduceOp, TensorNormalizer, TensorOp, TensorValue,
    UnaryOp, environment, libtorch, ndarray, repro::reproduce,
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

/// A tensor of `shape` where every value is `fill`.
fn filled(shape: &[usize], fill: f32) -> TensorValue {
    let count = shape.iter().product();
    TensorValue::new(shape.to_vec(), vec![fill; count])
}

/// The largest absolute disagreement this case produces, or `None` if it agrees.
///
/// Used as a **signature**: the injected fault offsets one value by a known amount, so
/// this number identifies *which* bug a case exhibits, and it must survive shrinking.
fn worst_absolute_error(case: &TensorOp) -> Option<f64> {
    let correct = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let faulty = NormalizedRunner::new(FaultyBackend::new(ndarray(), 0.5), TensorNormalizer);

    let a = correct.run_and_normalize(case).ok()?;
    let b = faulty.run_and_normalize(case).ok()?;

    match a.approx_compare(&b, TOLERANCE) {
        Agreement::Disagree(comparison) => Some(comparison.max_absolute_error),
        _ => None,
    }
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

/// Shrinking must work for **every** operation class, not just the elementwise one that
/// happens to be easiest. Each class has its own constraints — a reduction's axis, a
/// matrix multiplication's shared dimension — and a shrinker can quite easily work for
/// one shape of case while producing nothing usable for another.
#[test]
fn every_operation_class_shrinks() {
    let cases = [
        TensorOp::unary(UnaryOp::Neg, filled(&[4, 6], 1.5)),
        TensorOp::binary(BinaryOp::Mul, filled(&[3, 5], 2.0), filled(&[3, 5], 3.0)),
        TensorOp::reduce(ReduceOp::Sum, filled(&[4, 5, 6], 1.0), 2),
        TensorOp::matmul(filled(&[4, 5], 1.0), filled(&[5, 6], 2.0)),
    ];

    for case in cases {
        let before = element_count(&case);
        let name = case.name();

        assert!(still_diverges(&case), "{name}: the fault was not caught");
        let minimized = minimize(case, still_diverges);
        let after = element_count(&minimized.input);

        assert!(
            after < before,
            "{name} did not shrink at all: {before} values"
        );
        assert!(
            minimized.is_minimal(),
            "{name} stopped early: {}",
            minimized.stopped
        );
        assert!(
            still_diverges(&minimized.input),
            "{name} shrank into a case that no longer diverges"
        );
    }
}

/// **Shrinking must not change what the bug is.**
///
/// The risk is subtle and specific: a smaller case can still diverge while diverging for
/// a *different reason* than the original, and the resulting "minimal reproduction" then
/// sends a maintainer after the wrong thing. Here the injected fault offsets one value by
/// a known amount, so the signature is the absolute error — it must survive shrinking
/// unchanged.
#[test]
fn shrinking_preserves_the_nature_of_the_divergence() {
    const BIAS: f32 = 0.5;

    let original = TensorOp::binary(BinaryOp::Add, filled(&[3, 4], 2.0), filled(&[3, 4], 5.0));
    let before = worst_absolute_error(&original).expect("the original diverges");

    let minimized = minimize(original, still_diverges);
    let after = worst_absolute_error(&minimized.input).expect("the minimised case diverges");

    assert!(
        (before - BIAS as f64).abs() < 1e-6,
        "unexpected original error {before}"
    );
    assert!(
        (after - before).abs() < 1e-6,
        "the error changed from {before} to {after}: shrinking found a different bug"
    );
}

/// A fault far smaller than the injected one must still be caught and shrunk. If only
/// obvious faults survived minimisation, the mechanism would be useless for the subtle
/// divergences that actually need shrinking to be understood.
#[test]
fn a_small_fault_is_still_caught_and_shrunk() {
    let tiny = |case: &TensorOp| -> bool {
        let correct = NormalizedRunner::new(ndarray(), TensorNormalizer);
        let faulty = NormalizedRunner::new(FaultyBackend::new(ndarray(), 1e-4), TensorNormalizer);

        let (Ok(a), Ok(b)) = (
            correct.run_and_normalize(case),
            faulty.run_and_normalize(case),
        ) else {
            return false;
        };
        !matches!(a.approx_compare(&b, TOLERANCE), Agreement::Agree(_))
    };

    let original = TensorOp::unary(UnaryOp::Abs, filled(&[5, 5], 3.0));
    assert!(tiny(&original), "a small fault went undetected");

    let minimized = minimize(original, tiny);
    assert!(element_count(&minimized.input) <= 2);
    assert!(tiny(&minimized.input));
}

/// A fault on the *other* side of the comparison shrinks identically. Nothing about
/// minimisation may depend on which implementation happens to be wrong — in a real
/// finding, we do not know.
#[test]
fn a_fault_on_either_implementation_shrinks_the_same_way() {
    let against_libtorch = |case: &TensorOp| -> bool {
        let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
        let faulty = NormalizedRunner::new(FaultyBackend::new(libtorch(), 0.5), TensorNormalizer);

        let (Ok(a), Ok(b)) = (cpu.run_and_normalize(case), faulty.run_and_normalize(case)) else {
            return false;
        };
        !matches!(a.approx_compare(&b, TOLERANCE), Agreement::Agree(_))
    };

    let minimized = minimize(
        TensorOp::binary(BinaryOp::Sub, filled(&[4, 4], 7.0), filled(&[4, 4], 2.0)),
        against_libtorch,
    );

    assert!(element_count(&minimized.input) <= 4);
    assert!(against_libtorch(&minimized.input));
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
