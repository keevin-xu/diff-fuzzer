//! Triage the generator's corpus: what crashes, what diverges, and whether it is **ours**.
//!
//! `08-RISKS.md` §2 is blunt that most early findings in any differential project are the
//! tool's own invalid models — one SQL sweep produced 825 from its own invalid queries. So this
//! reports three things per signature, not one:
//!
//! - what the runtime did (crashed, rejected, answered),
//! - **whether the specification's own implementation accepts the model**, which is the
//!   practical definition of validity, and
//! - how many cases share the signature, so a flood of one mechanism is visible immediately.
//!
//! A crash on a model the reference accepts is a candidate finding. A crash on a model it
//! rejects is our bug. Nothing here decides which; it lays them out so a human can.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n3_triage --features candle
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use std::collections::BTreeMap;

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let generator = OnnxGenerator::default();
    let reference = ReferenceRuntime::start().expect("reference");

    // The capability model, pulled forward from N5 (`PENDING` 1.12). Without it, a runtime that
    // does not implement an operator reports a typed error string, the adapter records it as
    // `Rejected`, and the oracle correctly reports "one rejected while others answered" — a real
    // disagreement in form and no finding at all in substance.
    let path = format!("{}/census.json", onnx_adapter::FINDINGS_ROOT);
    let caps = onnx_adapter::capability::Capabilities::load(&path)
        .expect("run the n2_census example first — the corpus is not judgeable without it");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    if !drift.is_empty() {
        eprintln!("REFUSING: the census was taken under different versions: {drift:?}");
        std::process::exit(1);
    }
    let tract = onnx_adapter::capability::WithCapabilities::new(TractRuntime, &caps);
    let ort = onnx_adapter::capability::WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let candle = onnx_adapter::capability::WithCapabilities::new(
        onnx_adapter::runtimes::CandleRuntime,
        &caps,
    );
    let mut crashes: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut diverged: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut invalid = 0;

    for seed in 0..2000u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if !onnx_adapter::validation::is_valid(&case) {
            invalid += 1;
            continue;
        }

        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut participants: Vec<(&str, OnnxOutcome)> = vec![
            ("tract", tract.run(&case).unwrap()),
            ("onnxruntime", ort.run(&case).unwrap()),
        ];
        #[cfg(feature = "candle")]
        participants.push(("candle", candle.run(&case).unwrap()));

        // Group participants by what they produced; more than one group is a disagreement.
        use diff_fuzzer_core::Normalizer;
        use diff_fuzzer_core::traits::{NamedOutput, Oracle, Verdict};
        let outputs: Vec<NamedOutput<_>> = participants
            .iter()
            .map(|(n, o)| NamedOutput {
                implementation: (*n).to_string(),
                output: onnx_adapter::normalize::OnnxNormalizer.normalize(o.clone()),
            })
            .collect();
        if let Verdict::Diverged(d) = onnx_adapter::oracle::OnnxOracle.check(&case, &outputs) {
            let kinds: Vec<String> = participants
                .iter()
                .map(|(n, o)| format!("{n}={}", o.kind()))
                .collect();
            let key = format!(
                "{} | {:?} | {} | {}",
                case.op.onnx_name(),
                case.inputs[0].elem_type(),
                kinds.join(" "),
                d.summary
            );
            let entry = diverged.entry(key).or_insert((0, seed));
            entry.0 += 1;
        }

        for (name, outcome) in &participants {
            if let OnnxOutcome::Crashed { detail } = outcome {
                let spec_ok = matches!(reference.run(&case).unwrap(), OnnxOutcome::Ok(_));
                let key = format!(
                    "{name} | {} | {:?} | spec-accepts={spec_ok} | {}",
                    case.op.onnx_name(),
                    case.inputs[0].elem_type(),
                    detail.lines().next().unwrap_or("")
                );
                let entry = crashes.entry(key).or_insert((0, seed));
                entry.0 += 1;
            }
        }
    }
    std::panic::set_hook(previous);

    println!("invalid cases generated: {invalid} (must be 0)");
    println!("\ndistinct DIVERGENCE signatures:");
    for (key, (count, seed)) in &diverged {
        println!("  {count:>4}x  seed {seed:<6} {key}");
    }
    println!("\ndistinct crash signatures over 2000 seeds:");
    for (key, (count, seed)) in &crashes {
        println!("  {count:>4}x  seed {seed:<6} {key}");
    }
}
