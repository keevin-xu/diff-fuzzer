//! Generated cases running through the whole pipeline on both backends.
//!
//! An integration test rather than a unit test, so it goes through the crate's public
//! interface exactly as a real user of it would. What it checks is **validity**: every
//! generated case must actually execute. Whether the two backends then *agree* is a
//! separate question, and not this test's business — the comparison is still exact
//! equality, which is the wrong tool for floating-point results and is replaced later.

use diff_fuzzer_core::{
    DifferentialOracle, Generator, Implementation, NormalizedRunner, Runner, SeededRng, Verdict,
    driver::run_once,
};
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
    let oracle: DifferentialOracle<TensorOp, CanonicalTensor> = DifferentialOracle::new();

    for seed in 0..1_000 {
        let outcome = run_once(seed, &generator, &runners, &oracle);
        if let Verdict::Skipped(reason) = outcome.verdict {
            panic!("seed {seed} was skipped: {reason}");
        }
    }
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
