//! **N4.8** — does the detector still fire at *this* configuration?
//!
//! The fault-injection test in `tests/walking_skeleton.rs` runs on `Bounds::one_axis()`: four
//! elementwise operators, ordinary values, tiny shapes. It has passed since N1. That says
//! nothing about the corpus this phase actually produces — 33 operators across five element
//! types, adversarial values, degenerate shapes, and a capability layer that reclassifies
//! outcomes *before* the oracle sees them.
//!
//! Every one of those additions is a chance to break the detector, and the capability layer is
//! the sharpest: its whole job is to turn some disagreements into skips. A layer that excused
//! too much would make a campaign look clean while catching nothing, which is precisely the
//! failure mode `05-MEASUREMENT-AND-CAMPAIGNS.md` puts first. So the check is re-run **on the
//! configuration that will actually be used**, not on the one that was convenient to write.
//!
//! All three wrappers, because they catch different broken oracles: `WrongValues` (an oracle
//! that ignores values), `WrongShape` (one that compares element-by-element without checking
//! shape), and `Panicking` (one that routes crashes into the skip path — the domain's thesis).
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n4_detector --features candle
use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation, NamedOutput, Oracle, Verdict};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::case::OnnxCase;
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::testing::{FaultClass, Panicking, WrongShape, WrongValues, classify_fault};

const SEEDS: u64 = 3000;

fn canon(outcome: OnnxOutcome) -> Canonical {
    OnnxNormalizer.normalize(outcome)
}

fn named(name: &str, outcome: OnnxOutcome) -> NamedOutput<Canonical> {
    NamedOutput {
        implementation: name.to_string(),
        output: canon(outcome),
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // The N4 configuration: everything the generator can produce, adversarial values on.
    let bounds = Bounds {
        special_values: true,
        ..Bounds::default()
    };
    let generator = OnnxGenerator::new(bounds);

    let path = format!("{}/census.json", onnx_adapter::FINDINGS_ROOT);
    let caps = Capabilities::load(&path).expect("run the n2_census example first");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    assert!(drift.is_empty(), "the census is stale: {drift:?}");

    let tract = WithCapabilities::new(TractRuntime, &caps);
    let ort = WithCapabilities::new(OrtRuntime, &caps);

    // Each fault is applied *outside* the capability layer, so reclassification happens first
    // and the corruption sits where a real defect would: in the answer we are willing to judge.
    let mut results: Vec<(&str, usize, usize, usize, Vec<u64>)> = Vec::new();

    for (label, faulty) in [
        (
            "WrongValues(+1.0)",
            Box::new(|c: &OnnxCase| WrongValues::new(TractRuntime, 1.0).run(c).unwrap())
                as Box<dyn Fn(&OnnxCase) -> OnnxOutcome>,
        ),
        (
            "WrongShape",
            Box::new(|c: &OnnxCase| WrongShape::new(TractRuntime).run(c).unwrap()),
        ),
        (
            "Panicking",
            Box::new(|c: &OnnxCase| Panicking::new().run(c).unwrap()),
        ),
    ] {
        let (mut exercised, mut inert, mut caught) = (0usize, 0usize, 0usize);
        let mut missed = Vec::new();

        for seed in 0..SEEDS {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !onnx_adapter::validation::is_valid(&case) {
                continue;
            }
            let clean_tract = tract.run(&case).unwrap();
            let clean_ort = ort.run(&case).unwrap();

            // Only seeds the clean pair agrees on can demonstrate anything: on a seed that
            // already diverges, a divergence with the fault applied proves nothing.
            let clean = OnnxOracle.check(
                &case,
                &[
                    named("tract", clean_tract.clone()),
                    named("onnxruntime", clean_ort.clone()),
                ],
            );
            if !matches!(clean, Verdict::Agree) {
                continue;
            }

            // Classify the fault rather than assuming it did something. A fault that changed
            // nothing is **inert**, and the oracle agreeing on it is correct — counting that
            // as a miss is the error that broke this check's first version elsewhere.
            let corrupted = faulty(&case);
            match classify_fault(&clean_tract, &corrupted) {
                FaultClass::Exercised => exercised += 1,
                FaultClass::Inert => {
                    inert += 1;
                    continue;
                }
                FaultClass::Unrunnable => continue,
            }

            match OnnxOracle.check(
                &case,
                &[named("tract", corrupted), named("onnxruntime", clean_ort)],
            ) {
                Verdict::Diverged(_) => caught += 1,
                // A skip counts as a miss. An oracle that declined every case would otherwise
                // score perfectly, which is the whole failure this check exists to rule out.
                _ => missed.push(seed),
            }
        }
        results.push((label, exercised, inert, caught, missed));
    }
    std::panic::set_hook(previous);

    println!(
        "\nfault injection at the N4 configuration ({SEEDS} seeds, 33 operators, specials on)\n"
    );
    println!(
        "{:<20} {:>10} {:>8} {:>8} {:>8}",
        "fault", "exercised", "inert", "caught", "missed"
    );
    let mut all_caught = true;
    for (label, exercised, inert, caught, missed) in &results {
        println!(
            "{label:<20} {exercised:>10} {inert:>8} {caught:>8} {:>8}",
            missed.len()
        );
        if !missed.is_empty() {
            all_caught = false;
            println!("    MISSED seeds: {:?}", &missed[..missed.len().min(10)]);
        }
        if *exercised == 0 {
            all_caught = false;
            println!("    NOTHING EXERCISED — this row proves nothing");
        }
    }
    println!(
        "\n{}",
        if all_caught {
            "every exercised fault was caught: the detector fires at this configuration"
        } else {
            "THE DETECTOR HAS A HOLE — see the missed seeds above"
        }
    );
}
