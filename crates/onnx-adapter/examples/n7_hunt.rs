//! **N7.9** — hunt the Tier A + Tier B surface, minimise every distinct problem, store it, and
//! replay it back.
//!
//! # This is the line
//!
//! `04-ADOPTING-A-THIRD-DOMAIN.md`: *do not build breadth before you can tell a real finding from
//! your own bug.* Everything before this makes a finding possible; everything after makes the
//! next one cheaper. So this example is the whole pipeline end to end, and each stage has to hold
//! its own weight:
//!
//! 1. **generate** over the full surface with adversarial values;
//! 2. **judge** with the capability layer and the legal-difference catalog in force;
//! 3. **group** by signature, so a problem hit a hundred times is one problem;
//! 4. **minimise** each distinct problem, requiring the *same signature* throughout — a case that
//!    shrinks into a different bug is a report describing something it never demonstrated;
//! 5. **store** the whole case with its policy, versions and generator description;
//! 6. **replay** it from the record, which is the only check that the stored artifact is
//!    self-contained.
//!
//! Step 6 is the one that is easy to skip and the only one that proves the rest. A finding that
//! cannot be replayed from its own record is a note, not a reproduction.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n7_hunt --features candle
use std::collections::BTreeMap;

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation, NamedOutput, Oracle, Verdict};
use diff_fuzzer_core::{Budget, Normalizer, axes::GenerationAxes, minimize_within};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::case::OnnxCase;
use onnx_adapter::findings::{CampaignLog, Minimisation, Run, StoredFinding};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::repro::{Replay, replay};
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::shrink::complexity;

const SEEDS: u64 = 4000;

/// The run name. Findings land in `runs/differential/{RUN_NAME}`, the log in `logs/{RUN_NAME}.log`.
const RUN_NAME: &str = "n7-first-findings";

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let tract = WithCapabilities::new(TractRuntime, &caps);
    let ort = WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let candle = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);

    // One place that runs a case on everybody, used by the sweep *and* by the minimiser's
    // predicate. Two code paths would let the predicate judge a candidate differently from the
    // way the finding was found, which is how a minimiser walks off into a different bug.
    let judge = |case: &OnnxCase| -> Vec<NamedOutput<Canonical>> {
        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut outs = vec![
            ("tract", tract.run(case).expect("never Err")),
            ("onnxruntime", ort.run(case).expect("never Err")),
        ];
        #[cfg(feature = "candle")]
        outs.push(("candle", candle.run(case).expect("never Err")));
        outs.into_iter()
            .map(|(n, o)| NamedOutput {
                implementation: n.to_string(),
                output: OnnxNormalizer.normalize(o),
            })
            .collect()
    };

    let signature_of = |case: &OnnxCase| -> Option<String> {
        let outputs = judge(case);
        match OnnxOracle.check(case, &outputs) {
            Verdict::Diverged(_) => onnx_adapter::signature::of(case, &outputs).map(|s| s.key()),
            _ => None,
        }
    };

    // ── 1-3. Sweep and group ────────────────────────────────────────────────────────
    let bounds = Bounds::default().with_special_values();
    let generator = OnnxGenerator::new(bounds.clone());
    let mut log = CampaignLog::open(RUN_NAME).expect("opening the campaign log");
    log.header(RUN_NAME, &bounds.description())
        .expect("writing the log header");
    let mut first_seen: BTreeMap<String, (u64, OnnxCase)> = BTreeMap::new();
    let mut hits: BTreeMap<String, usize> = BTreeMap::new();

    for seed in 0..SEEDS {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if !onnx_adapter::validation::is_valid(&case) {
            continue;
        }
        if let Some(key) = signature_of(&case) {
            *hits.entry(key.clone()).or_default() += 1;
            first_seen.entry(key).or_insert((seed, case));
        }
    }

    log.say(format!(
        "{} distinct signatures across {SEEDS} seeds ({} occurrences)",
        first_seen.len(),
        hits.values().sum::<usize>()
    ))
    .expect("writing the log");
    log.say(String::new()).expect("writing the log");

    // ── 4-6. Minimise, store, replay ────────────────────────────────────────────────
    let mut run = Run::open(onnx_adapter::OracleKind::Differential, RUN_NAME)
        .expect("opening the run directory");
    let budget = Budget {
        max_steps: 200,
        max_candidates: 4000,
        max_duration: None,
    };

    for (key, (seed, case)) in &first_seen {
        let before = complexity(case);

        // **The predicate requires the same signature**, not merely a divergence.
        let minimized = minimize_within(case.clone(), budget, |candidate: &OnnxCase| {
            signature_of(candidate).as_deref() == Some(key.as_str())
        });

        let after = complexity(&minimized.input);
        let outputs = judge(&minimized.input);
        let signature = onnx_adapter::signature::of(&minimized.input, &outputs)
            .expect("the minimised case must still have a signature");

        let rendered: Vec<(String, String)> = outputs
            .iter()
            .map(|o| (o.implementation.clone(), render(&o.output)))
            .collect();

        let finding = StoredFinding::new(
            signature.key(),
            OnnxOracle
                .check(&minimized.input, &outputs)
                .diverged_summary()
                .unwrap_or_default(),
            *seed,
            bounds.description(),
            minimized.input.clone(),
            rendered.clone(),
        )
        .with_signature(signature.clone())
        .with_minimisation(Minimisation {
            steps: minimized.steps,
            candidates_tried: minimized.candidates_tried,
            complete: minimized.is_minimal(),
            elements_before: before.elements,
            elements_after: after.elements,
        });

        let stored = run.record(&finding).expect("writing a finding");

        log.say(format!("── {key}")).expect("log");
        log.say(format!(
            "   seen {}x · minimised {} -> {} elements, rank sum {} -> {} ({} steps, {} candidates, {})",
            hits[key],
            before.elements,
            after.elements,
            before.rank_sum,
            after.rank_sum,
            minimized.steps,
            minimized.candidates_tried,
            minimized.stopped
        ))
        .expect("log");
        log.say(format!("   {}", describe_case(&minimized.input)))
            .expect("log");
        for (name, text) in &rendered {
            log.say(format!("     {name:<12} {}", truncate(text, 96)))
                .expect("log");
        }

        // ── 6. Replay from the record ───────────────────────────────────────────────
        let participants: Vec<(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)> = vec![
            ("tract", &tract),
            ("onnxruntime", &ort),
            #[cfg(feature = "candle")]
            ("candle", &candle),
        ];
        let result = replay(&finding, &participants);
        log.say(format!(
            "   replay: {}",
            match &result {
                Replay::Reproduced { .. } => "REPRODUCED from its own record".to_string(),
                other => format!("*** {other:?} ***"),
            }
        ))
        .expect("log");
        log.say(format!(
            "   {} in {}\n",
            if stored {
                "recorded"
            } else {
                "already present"
            },
            run.directory().display()
        ))
        .expect("log");
    }
    std::panic::set_hook(previous);

    log.say(format!(
        "\n{} distinct findings written to {}",
        run.distinct(),
        run.directory().display()
    ))
    .expect("writing the log");
}

/// A one-line description of what the case actually is.
fn describe_case(case: &OnnxCase) -> String {
    let inputs: Vec<String> = case
        .inputs
        .iter()
        .map(|i| {
            format!(
                "{}{:?}:{:?}={}",
                if i.is_initializer() { "init " } else { "" },
                i.dims,
                i.elem_type(),
                truncate(&format!("{:?}", i.data), 44)
            )
        })
        .collect();
    format!(
        "{} opset {} · {}{}",
        case.op.onnx_name(),
        case.opset,
        inputs.join(" , "),
        if case.attrs.is_empty() {
            String::new()
        } else {
            format!(" · attrs {}", case.attrs.describe())
        }
    )
}

fn render(canonical: &Canonical) -> String {
    if !canonical.is_ok() {
        return format!("{}: {}", canonical.kind, canonical.detail);
    }
    canonical
        .tensors
        .iter()
        .map(|t| {
            let bits: Vec<String> = t.bits.iter().take(8).map(|b| format!("{b:#x}")).collect();
            format!("{:?}[{}]", t.dims, bits.join(" "))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn truncate(text: &str, at: usize) -> String {
    if text.chars().count() <= at {
        return text.to_string();
    }
    let head: String = text.chars().take(at).collect();
    format!("{head}…")
}

/// The oracle's summary, when it diverged.
trait DivergedSummary {
    fn diverged_summary(&self) -> Option<String>;
}

impl DivergedSummary for Verdict {
    fn diverged_summary(&self) -> Option<String> {
        match self {
            Verdict::Diverged(d) => Some(d.summary.clone()),
            _ => None,
        }
    }
}
