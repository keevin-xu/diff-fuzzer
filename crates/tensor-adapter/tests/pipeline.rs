//! Generated cases running through the whole pipeline on both backends.
//!
//! An integration test rather than a unit test, so it goes through the crate's public
//! interface exactly as a real user of it would. What it checks is **validity**: every
//! generated case must actually execute. Whether the two backends then *agree* is a
//! separate question, and not this test's business — the comparison is still exact
//! equality, which is the wrong tool for floating-point results and is replaced later.

use diff_fuzzer_core::{
    DifferentialOracle, FixedTolerance, Generator, Implementation, NormalizedRunner, Runner,
    SeededRng, Tolerance, Verdict, driver::run_once,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    Bounds, CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, libtorch, ndarray,
};

type AnyRunner<'a> = &'a dyn Runner<In = TensorOp, Canon = CanonicalTensor>;

/// Every generated case must run on both backends without being rejected.
///
/// This is the claim correct-by-construction generation makes, and the one that decides
/// whether the fuzzer is worth running at all. A case rejected as malformed exercises
/// nothing but the validation code — so a generator whose output is only occasionally
/// valid spends the campaign proving that shape checks work, while the kernels it
/// exists to test go untouched.
#[test]
fn every_generated_case_executes_on_both_backends() {
    let generator = TensorOpGenerator::default();
    let (cpu, torch) = (ndarray(), libtorch());

    let mut rejected = Vec::new();

    for seed in 0..2_000 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        if let Err(error) = cpu.run(&case) {
            rejected.push(format!("seed {seed}: {} on cpu: {error}", case.name()));
        }
        if let Err(error) = torch.run(&case) {
            rejected.push(format!("seed {seed}: {} on libtorch: {error}", case.name()));
        }
    }

    assert!(
        rejected.is_empty(),
        "{} of 2000 cases were rejected as invalid:\n  {}",
        rejected.len(),
        rejected
            .iter()
            .take(10)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// Both backends must agree on the *shape* of every result, whatever they think of the
/// numbers.
///
/// Shape is settled by the operation's definition rather than by arithmetic, so a
/// disagreement here would not be floating-point noise — it would mean the two
/// backends disagree about what the operation *means*. Worth separating from numeric
/// comparison, because it stays a hard error even once numbers are compared loosely.
#[test]
fn both_backends_agree_on_result_shapes() {
    let generator = TensorOpGenerator::default();
    let (cpu, torch) = (ndarray(), libtorch());

    for seed in 0..2_000 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let from_cpu = cpu.run(&case).expect("valid case");
        let from_torch = torch.run(&case).expect("valid case");

        assert_eq!(
            from_cpu.shape.to_vec(),
            from_torch.shape.to_vec(),
            "seed {seed}: {} produced different shapes",
            case.name()
        );
    }
}

/// The pipeline must never skip a generated case. A skip means something could not be
/// compared, and with a generator that only produces valid cases there should be
/// nothing that cannot be.
#[test]
fn no_generated_case_is_skipped() {
    let generator = TensorOpGenerator::default();
    let cpu = NormalizedRunner::new(ndarray(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let runners: [AnyRunner; 2] = [&cpu, &torch];
    // Exact comparison for now: the tolerance policy that varies by operation
    // arrives next.
    let oracle: DifferentialOracle<TensorOp, CanonicalTensor, FixedTolerance> =
        DifferentialOracle::new(FixedTolerance(Tolerance::EXACT));

    for seed in 0..1_000 {
        let outcome = run_once(seed, &generator, &runners, &oracle);
        if let Verdict::Skipped(reason) = outcome.verdict {
            panic!("seed {seed} was skipped: {reason}");
        }
    }
}

/// The real validity check: tens of thousands of cases, from seeds spread across the
/// whole 64-bit range rather than a tidy run starting at zero.
///
/// Seed choice matters more than it looks. Seeds `0..n` are adjacent numbers, and a
/// generator with a subtle dependence on seed *magnitude* — or one accidentally
/// producing similar cases for nearby seeds — would sail through a small sequential
/// run and fail in a real campaign, where seeds are arbitrary. So this samples three
/// widely separated regions.
///
/// Any failure is reported grouped by operation, because the fix is virtually always a
/// constraint in one operation's module rather than something general.
#[test]
fn tens_of_thousands_of_cases_all_execute() {
    const SEQUENTIAL: u64 = 30_000;
    const SCATTERED: u64 = 10_000;
    const HIGH: u64 = 10_000;

    let generator = TensorOpGenerator::default();
    let (cpu, torch) = (ndarray(), libtorch());

    // Three regions: low sequential, scattered across the range by a large odd stride,
    // and the very top of the range.
    let seeds = (0..SEQUENTIAL)
        .chain((0..SCATTERED).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)))
        .chain((0..HIGH).map(|i| u64::MAX - i));

    let mut failures: BTreeMap<&'static str, Vec<String>> = BTreeMap::new();
    let mut executed = 0usize;

    for seed in seeds {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        executed += 1;

        for (backend_name, result) in [("cpu", cpu.run(&case)), ("libtorch", torch.run(&case))] {
            if let Err(error) = result {
                failures
                    .entry(case.name())
                    .or_default()
                    .push(format!("seed {seed} on {backend_name}: {error}"));
            }
        }
    }

    let total_failures: usize = failures.values().map(Vec::len).sum();
    assert_eq!(
        total_failures,
        0,
        "{total_failures} of {executed} cases failed to execute.\n{}",
        failures
            .iter()
            .map(|(op, errors)| format!("  {op}: {} failures, e.g. {}", errors.len(), errors[0]))
            .collect::<Vec<_>>()
            .join("\n")
    );

    assert_eq!(executed as u64, SEQUENTIAL + SCATTERED + HIGH);
}

/// Narrow bounds must stay valid too. Degenerate shapes — every dimension of length
/// one — are a classic source of bugs, so it matters that they are reachable *and*
/// that they execute.
#[test]
fn degenerate_shapes_still_execute() {
    let generator = TensorOpGenerator::new(Bounds {
        max_rank: 4,
        max_dim: 1,
        magnitude: 1.0,
        ..Bounds::default()
    });
    let (cpu, torch) = (ndarray(), libtorch());

    for seed in 0..500 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        cpu.run(&case)
            .unwrap_or_else(|e| panic!("seed {seed}: {} on cpu: {e}", case.name()));
        torch
            .run(&case)
            .unwrap_or_else(|e| panic!("seed {seed}: {} on libtorch: {e}", case.name()));
    }
}
