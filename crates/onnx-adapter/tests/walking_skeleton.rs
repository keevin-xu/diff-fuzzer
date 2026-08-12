//! The N1 walking skeleton, exercised end to end through the **engine's own driver**.
//!
//! Every other test in this crate calls the pieces directly. These go through
//! `diff_fuzzer_core::driver::run_once`, which is the path a campaign actually takes:
//! seed → generator → case → every runtime → normalizer → oracle → verdict.
//!
//! That distinction matters. A unit test can prove the oracle is correct while the engine
//! never reaches it — and a check that is perfectly correct and unreachable is the failure
//! this project has hit ten times. These tests are what prove the wiring.

use diff_fuzzer_core::driver::run_once;
use diff_fuzzer_core::runner::{NormalizedRunner, Runner};
use diff_fuzzer_core::traits::{Implementation, Verdict};

use onnx_adapter::case::{OnnxCase, OpKind};
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::testing::{FaultClass, Panicking, WrongValues, classify_fault};

type BoxedRunner = Box<dyn Runner<In = OnnxCase, Canon = Canonical>>;

fn real_runners() -> Vec<BoxedRunner> {
    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut runners: Vec<BoxedRunner> = vec![
        Box::new(NormalizedRunner::new(OrtRuntime, OnnxNormalizer)),
        Box::new(NormalizedRunner::new(TractRuntime, OnnxNormalizer)),
    ];
    #[cfg(feature = "candle")]
    runners.push(Box::new(NormalizedRunner::new(
        onnx_adapter::runtimes::CandleRuntime,
        OnnxNormalizer,
    )));
    runners
}

fn verdict_for(seed: u64, runners: &[BoxedRunner]) -> Verdict {
    let borrowed: Vec<&dyn Runner<In = OnnxCase, Canon = Canonical>> =
        runners.iter().map(std::convert::AsRef::as_ref).collect();
    run_once(seed, &OnnxGenerator::default(), &borrowed, &OnnxOracle).verdict
}

/// **N1.9.** Same seed, same case, same verdict — through the whole loop.
///
/// The property every finding depends on: a divergence that cannot be replayed from its
/// seed is a defect in this tool, not a discovery about anything else.
#[test]
fn the_same_seed_gives_the_same_verdict() {
    let runners = real_runners();
    for seed in [0, 1, 7, 1234, 99_999] {
        let first = verdict_for(seed, &runners);
        let second = verdict_for(seed, &runners);
        assert_eq!(first, second, "seed {seed} was not reproducible");
    }
}

/// The real runtimes agree across a broad sweep of seeds.
///
/// **This is not yet a result about ONNX runtimes.** It is a statement about a skeleton
/// generator producing four simple operators at tiny shapes, and it is reported here only
/// as evidence that the loop runs clean. The claim that means something — a measured
/// surface with fault injection behind it — belongs to N7 and N8.
#[test]
fn the_real_runtimes_agree_across_many_seeds() {
    let runners = real_runners();
    let mut agreed = 0;
    let mut skipped = 0;
    let mut diverged = Vec::new();

    for seed in 0..500 {
        match verdict_for(seed, &runners) {
            Verdict::Agree => agreed += 1,
            Verdict::Skipped(_) => skipped += 1,
            Verdict::Diverged(d) => diverged.push((seed, d.summary)),
        }
    }

    assert!(
        diverged.is_empty(),
        "unexpected divergences on the skeleton corpus: {diverged:?}"
    );
    // A run that was *entirely* skipped would satisfy the assertion above while proving
    // nothing at all. The engine must actually have compared something.
    assert!(
        agreed > 100,
        "only {agreed} of 500 cases were judged ({skipped} skipped) — the corpus is not \
         exercising the oracle"
    );
}

/// **The pairing that makes the test above mean anything.**
///
/// A clean sweep is worth nothing on its own: it is equally consistent with "the runtimes
/// agree" and "the detector could never fire". Running the identical loop with one
/// corrupted participant is what separates those, and it must be the *same* corpus and the
/// *same* driver, not a proxy.
#[test]
fn the_same_corpus_with_a_corrupted_runtime_is_caught_every_time() {
    let corrupted: Vec<BoxedRunner> = vec![
        Box::new(NormalizedRunner::new(OrtRuntime, OnnxNormalizer)),
        Box::new(NormalizedRunner::new(
            WrongValues::new(TractRuntime, 1.0),
            OnnxNormalizer,
        )),
    ];
    let clean = real_runners();

    let faulty_implementation = WrongValues::new(TractRuntime, 1.0);
    let mut exercised = 0;
    let mut inert = 0;
    let mut caught = 0;
    let mut missed = Vec::new();

    for seed in 0..500 {
        let clean_verdict = verdict_for(seed, &clean);
        if !matches!(clean_verdict, Verdict::Agree) {
            continue;
        }
        let case = fresh_case(seed);

        // **Classify the fault; do not guess at it.**
        //
        // The first version of this test used `total_elements() != 0` as a proxy for "the
        // fault did something", and it was wrong on 50 of 349 seeds — because adding 1.0
        // changes nothing when the element is `NaN`, `±inf`, or `f32::MAX` (which rounds
        // straight back to itself). Those faults are **inert**, and the oracle agreeing on
        // them is correct behaviour, not a miss.
        //
        // This is exactly the trap `05-MEASUREMENT-AND-CAMPAIGNS.md` describes, and the
        // function that avoids it already existed in `testing.rs`. A proxy for "did the
        // fault do anything" is not an answer to it.
        let before = TractRuntime.run(&case).expect("never Err");
        let after = faulty_implementation.run(&case).expect("never Err");
        match classify_fault(&before, &after) {
            FaultClass::Exercised => exercised += 1,
            FaultClass::Inert => {
                inert += 1;
                continue;
            }
            FaultClass::Unrunnable => continue,
        }

        match verdict_for(seed, &corrupted) {
            Verdict::Diverged(_) => caught += 1,
            // A `Skipped` verdict counts as a miss. Otherwise a normalizer that declined
            // every case would score perfect.
            other => missed.push((seed, format!("{other:?}"))),
        }
    }

    assert!(
        exercised > 100,
        "only {exercised} seeds exercised the fault"
    );
    assert!(
        inert > 0,
        "no inert faults at all — the special-value pool is not reaching element 0, which \
         means this test is not covering the case that broke its first version"
    );
    assert_eq!(
        caught,
        exercised,
        "the oracle missed {} of {exercised} exercised faults ({inert} inert): {:?}",
        exercised - caught,
        &missed[..missed.len().min(5)]
    );
}

/// The crash path, through the driver rather than through a unit test.
///
/// The engine's default for a participant that cannot produce a result is
/// `SkipReason::CouldNotRun`. This domain routes crashes around that as *values*, and this
/// test is what proves the routing survives the real driver — the thesis is only true if it
/// holds on the path a campaign takes.
#[test]
fn a_crashing_runtime_is_reported_by_the_driver() {
    let runners: Vec<BoxedRunner> = vec![
        Box::new(NormalizedRunner::new(OrtRuntime, OnnxNormalizer)),
        Box::new(NormalizedRunner::new(TractRuntime, OnnxNormalizer)),
        Box::new(NormalizedRunner::new(Panicking::new(), OnnxNormalizer)),
    ];

    let mut reported = 0;
    for seed in 0..50 {
        if let Verdict::Diverged(divergence) = verdict_for(seed, &runners) {
            assert!(
                divergence.summary.contains("panicking"),
                "the crasher must be named: {}",
                divergence.summary
            );
            reported += 1;
        }
    }
    assert_eq!(reported, 50, "every crash must be reported, none skipped");
}

/// Every case the driver runs is one our own validator accepts.
///
/// The first of the two gates in front of a crash report: if our model were invalid, a
/// runtime falling over on it would be our bug, and reporting it would be manufacturing a
/// finding.
#[test]
fn every_case_the_driver_runs_is_valid() {
    for seed in 0..2_000 {
        let case = fresh_case(seed);
        let problems = onnx_adapter::validation::validate(&case);
        assert!(problems.is_empty(), "seed {seed}: {problems:?}");
    }
}

/// Regenerate the case a seed produces, so a test can inspect it.
///
/// Uses the same generator and the same seeded rng the driver does, which is exactly why
/// this is sound — and exactly why it would stop being sound if the generator changed.
/// That is the reason findings store the case rather than the seed.
fn fresh_case(seed: u64) -> OnnxCase {
    use diff_fuzzer_core::rng::SeededRng;
    use diff_fuzzer_core::traits::Generator;
    OnnxGenerator::default().generate(&mut SeededRng::from_seed(seed))
}

/// Every operator reaches the driver. An operator the loop never runs is an operator this
/// phase has not actually tested, however green the suite looks.
#[test]
fn every_operator_is_exercised_through_the_driver() {
    let mut seen: Vec<OpKind> = Vec::new();
    for seed in 0..500 {
        let op = fresh_case(seed).op;
        if !seen.contains(&op) {
            seen.push(op);
        }
    }
    for op in OpKind::ALL {
        assert!(seen.contains(&op), "{op:?} never reached the driver");
    }
}
