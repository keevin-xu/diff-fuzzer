//! **N5.7, N5.8** — the domain's thesis, measured.
//!
//! # The claim being tested
//!
//! Every earlier domain in this project routed a failure-to-produce-a-result into
//! `SkipReason::CouldNotRun` — a **skip**, something uninteresting that happened to the tool
//! rather than something wrong with the implementation. This domain's contribution is that some
//! of those are the most valuable findings available: a runtime that **claims** an operator and
//! then panics on a **valid** model is defective, and roughly 76% of published bugs in this
//! space are that class rather than wrong answers.
//!
//! So this measures three things:
//!
//! 1. **How many outcomes the old error model would have discarded.** Every `Crashed` and
//!    `TimedOut` on a valid model is one, because under `Out = Result<_, RunError>` there is no
//!    channel that carries them to the oracle as anything but a skip.
//! 2. **How many of those are genuinely attributable** — the runtime claims the operator (per the
//!    N2 census) and the specification's own implementation accepts the model. Anything failing
//!    either test is ours, not theirs.
//! 3. **The class split**: crash-class findings versus wrong-answer findings, so the literature's
//!    ~76% can be *compared against* rather than assumed (`PENDING` 1.8).
//!
//! # The honesty constraint
//!
//! A crash on a model **we** malformed is our bug. Two gates stand in front of every crash
//! counted here: our own `validate()`, and `onnx.reference` accepting the model. A number
//! produced without both would be measuring the generator, not the runtimes.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n5_thesis --features candle
use std::collections::BTreeMap;

use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation, NamedOutput, Oracle, Verdict};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::sentinel::CrashSentinel;
use onnx_adapter::timeout::WithTimeout;

const SEEDS: u64 = 4000;

/// What the outcome was, for the census of outcome kinds.
fn kind(outcome: &OnnxOutcome) -> &'static str {
    match outcome {
        OnnxOutcome::Ok(_) => "ok",
        OnnxOutcome::Rejected { .. } => "rejected",
        OnnxOutcome::Unsupported { .. } => "unsupported",
        OnnxOutcome::Crashed { .. } => "crashed",
        OnnxOutcome::TimedOut { .. } => "timed-out",
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    assert!(drift.is_empty(), "the census is stale: {drift:?}");

    // Timeout inside, capability classification outside. The capability layer borrows the census
    // and is not `'static`; it also must never rewrite a `TimedOut`, exactly as it never rewrites
    // a `Crashed`.
    let tract = WithCapabilities::new(WithTimeout::new(TractRuntime), &caps);
    let ort = WithCapabilities::new(WithTimeout::new(OrtRuntime), &caps);
    #[cfg(feature = "candle")]
    let candle = WithCapabilities::new(
        WithTimeout::new(onnx_adapter::runtimes::CandleRuntime),
        &caps,
    );

    let reference = ReferenceRuntime::start().expect("reference");
    let (mut sentinel, recovered) =
        CrashSentinel::open(format!("{}/in-flight.json", onnx_adapter::FINDINGS_ROOT))
            .expect("sentinel");
    if let Some(in_flight) = &recovered {
        println!(
            "\n*** a previous run died on {} / {} (seed {}) ***\n",
            in_flight.runtime,
            in_flight.case.op.onnx_name(),
            in_flight.seed
        );
    }

    for (label, bounds) in [
        ("ordinary values (the N3 corpus)", Bounds::default()),
        (
            "adversarial values (the N4 corpus)",
            Bounds::default().with_special_values(),
        ),
    ] {
        let generator = OnnxGenerator::new(bounds);

        // Outcome kinds, per participant.
        let mut kinds: BTreeMap<(&str, &'static str), usize> = BTreeMap::new();
        // Crash-class outcomes, split by whether they are attributable.
        let (mut defects_total, mut defects_attributable, mut defects_ours) = (0usize, 0, 0);
        let mut defect_signatures: BTreeMap<String, (usize, u64)> = BTreeMap::new();
        // Verdicts, split into the two finding classes.
        let (mut crash_divergences, mut answer_divergences) = (0usize, 0usize);
        let (mut agreed, mut skipped) = (0usize, 0usize);

        for seed in 0..SEEDS {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !onnx_adapter::validation::is_valid(&case) {
                continue;
            }

            // Armed before each execution and disarmed after, so a process death names both
            // the case and the runtime that was holding it.
            let mut outcomes: Vec<(&str, OnnxOutcome)> = Vec::new();
            sentinel.arm("tract", seed, &case).expect("arm");
            outcomes.push(("tract", tract.run(&case).expect("never Err")));
            sentinel.arm("onnxruntime", seed, &case).expect("arm");
            outcomes.push(("onnxruntime", ort.run(&case).expect("never Err")));
            #[cfg(feature = "candle")]
            {
                sentinel.arm("candle", seed, &case).expect("arm");
                outcomes.push(("candle", candle.run(&case).expect("never Err")));
            }
            sentinel.disarm().expect("disarm");

            for (name, outcome) in &outcomes {
                *kinds.entry((name, kind(outcome))).or_default() += 1;

                // ── The thesis number ──────────────────────────────────────────────
                // Crash-class outcomes: what the engine's error model would have turned
                // into `SkipReason::CouldNotRun` and shown to nobody.
                if matches!(
                    outcome,
                    OnnxOutcome::Crashed { .. } | OnnxOutcome::TimedOut { .. }
                ) {
                    defects_total += 1;

                    // Attributable only if the specification's own implementation accepts the
                    // model. A crash on a model we malformed is our bug, not the runtime's.
                    let spec_accepts =
                        matches!(reference.run(&case).expect("never Err"), OnnxOutcome::Ok(_));
                    if spec_accepts {
                        defects_attributable += 1;
                        let detail = match outcome {
                            OnnxOutcome::Crashed { detail } => {
                                detail.lines().next().unwrap_or("").to_string()
                            }
                            _ => "timed out".to_string(),
                        };
                        let key = format!(
                            "{name} | {} | {:?} | {detail}",
                            case.op.onnx_name(),
                            onnx_adapter::ops::data_elem_type(&case)
                        );
                        let entry = defect_signatures.entry(key).or_insert((0, seed));
                        entry.0 += 1;
                    } else {
                        defects_ours += 1;
                    }
                }
            }

            let named: Vec<NamedOutput<Canonical>> = outcomes
                .iter()
                .map(|(n, o)| NamedOutput {
                    implementation: (*n).to_string(),
                    output: OnnxNormalizer.normalize(o.clone()),
                })
                .collect();

            match OnnxOracle.check(&case, &named) {
                Verdict::Agree => agreed += 1,
                Verdict::Skipped(_) => skipped += 1,
                Verdict::Diverged(_) => {
                    // The split N5.8 asks for: was this divergence caused by something
                    // falling over, or by two runtimes answering differently?
                    if named.iter().any(|o| o.output.is_self_evident_defect()) {
                        crash_divergences += 1;
                    } else {
                        answer_divergences += 1;
                    }
                }
            }
        }

        println!("\n═══ {label} ═══");
        println!("\n  outcome kinds per participant:");
        let mut participants_seen: Vec<&str> = kinds.keys().map(|(n, _)| *n).collect();
        participants_seen.sort_unstable();
        participants_seen.dedup();
        println!(
            "  {:<14} {:>8} {:>10} {:>13} {:>9} {:>10}",
            "", "ok", "rejected", "unsupported", "crashed", "timed-out"
        );
        for name in participants_seen {
            let get = |k| kinds.get(&(name, k)).copied().unwrap_or(0);
            println!(
                "  {name:<14} {:>8} {:>10} {:>13} {:>9} {:>10}",
                get("ok"),
                get("rejected"),
                get("unsupported"),
                get("crashed"),
                get("timed-out")
            );
        }

        println!("\n  ── the thesis ──");
        println!("  crash-class outcomes (would have been silent skips):  {defects_total}");
        println!(
            "    of which the specification accepts the model:       {defects_attributable}  <- findings"
        );
        println!("    of which the model is ours to fix:                  {defects_ours}");
        if !defect_signatures.is_empty() {
            println!("\n  distinct crash signatures on valid models:");
            for (key, (count, seed)) in &defect_signatures {
                println!("    {count:>4}x  seed {seed:<6} {key}");
            }
        }

        let findings = crash_divergences + answer_divergences;
        println!("\n  ── the class split (N5.8) ──");
        println!("  agreed {agreed}, skipped {skipped}, diverged {findings}");
        if findings > 0 {
            println!(
                "  crash-class divergences   {crash_divergences:>5}  ({:.1}%)",
                100.0 * crash_divergences as f64 / findings as f64
            );
            println!(
                "  wrong-answer divergences  {answer_divergences:>5}  ({:.1}%)",
                100.0 * answer_divergences as f64 / findings as f64
            );
            println!("  the literature's split    ~76% crash / ~24% wrong answer");
        }
    }

    // ── The control ────────────────────────────────────────────────────────────────────
    //
    // **A zero is only worth reporting if a non-zero was reachable.** Every number above is
    // zero, and that is indistinguishable from a broken counter unless something proves the
    // counting path fires. So the identical loop is run again with a participant that crashes
    // on purpose, and the same counters must light up.
    //
    // This is the same discipline as the fault-injection check on the oracle, applied one level
    // out: there, a clean sweep needed proof the *oracle* could fire; here, a zero needs proof
    // the *measurement* could.
    {
        use onnx_adapter::testing::Panicking;
        let generator = OnnxGenerator::new(Bounds::default());
        let (mut control_defects, mut control_attributable) = (0usize, 0usize);
        let mut control_crash_divergences = 0usize;

        for seed in 0..200u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !onnx_adapter::validation::is_valid(&case) {
                continue;
            }
            let outcomes: Vec<(&str, OnnxOutcome)> = vec![
                ("onnxruntime", ort.run(&case).expect("never Err")),
                ("saboteur", Panicking::new().run(&case).expect("never Err")),
            ];
            for (_, outcome) in &outcomes {
                if matches!(
                    outcome,
                    OnnxOutcome::Crashed { .. } | OnnxOutcome::TimedOut { .. }
                ) {
                    control_defects += 1;
                    if matches!(reference.run(&case).expect("never Err"), OnnxOutcome::Ok(_)) {
                        control_attributable += 1;
                    }
                }
            }
            let named: Vec<NamedOutput<Canonical>> = outcomes
                .iter()
                .map(|(n, o)| NamedOutput {
                    implementation: (*n).to_string(),
                    output: OnnxNormalizer.normalize(o.clone()),
                })
                .collect();
            if let Verdict::Diverged(_) = OnnxOracle.check(&case, &named)
                && named.iter().any(|o| o.output.is_self_evident_defect())
            {
                control_crash_divergences += 1;
            }
        }

        println!("\n═══ control: the same counters, against a runtime that crashes on purpose ═══");
        println!("  crash-class outcomes counted:     {control_defects}");
        println!("  of which attributable:            {control_attributable}");
        println!("  crash-class divergences counted:  {control_crash_divergences}");
        println!(
            "\n  {}",
            if control_defects > 0 && control_crash_divergences > 0 {
                "the counting path fires, so the zeros above are a result and not a broken counter"
            } else {
                "*** THE COUNTER IS BROKEN — every zero above is meaningless ***"
            }
        );
    }

    std::panic::set_hook(previous);
}
