//! **N6.5, N6.6** — the false-positive funnel: what gets filtered, why, and what survives.
//!
//! # Why a funnel rather than a number
//!
//! "The oracle reports 66 divergences" is not a defensible statement on its own. Every filter
//! between a generated case and a reported divergence is a place where a **real bug can be
//! silently eaten**, and an over-broad filter looks exactly like success: the count goes down,
//! the survivors look clean, and nothing indicates that something real went with them.
//!
//! `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §6 and `02-METHODOLOGY.md` both insist the filtered set
//! be logged with reasons rather than summarised away. So this reports every stage, in order,
//! with what it removed — and the survivors are checked against the finding drafts, so a
//! signature nobody has explained is reported as **unexplained** rather than counted as clean.
//!
//! # The two sides of the funnel
//!
//! Filters live in two places, and they are not equivalent:
//!
//! - **Generator-side**: cases whose answer the specification does not determine are never
//!   produced. Sound either way — being wrong costs coverage, which is visible.
//! - **Oracle-side**: differences the comparison forgives or declines to judge. Being wrong here
//!   hides defects silently.
//!
//! This measures the oracle side directly, stage by stage. The generator side cannot be measured
//! retrospectively — the cases do not exist — so it is reported from `known.rs`, where each rule
//! carries the count it was measured at.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n6_funnel --features candle
use std::collections::BTreeMap;

use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Oracle, SkipReason, Verdict,
};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::known::{CATALOG, Handling};
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

const SEEDS: u64 = 4000;

/// The surviving signatures we can account for, and what accounts for them.
///
/// **This list is the point of N6.6.** A signature not in it is not "probably fine" — it is
/// unexplained, and the phase's acceptance criterion is that no such signature remains.
fn explanation(signature: &str) -> Option<&'static str> {
    // **Keyed on the operator *and* who dissented.** The first version of this matched on the
    // operator alone, and credited "Sign | candle disagrees" to F-001 — which is a *tract*
    // defect. That is the over-broad filter this whole example exists to detect, built into the
    // detector. A signature must be explained by something that actually explains it.
    match signature {
        // F-001 appears in two signature forms depending on whether candle took part: as a
        // two-faction split when candle abstained, and as a named lone dissenter when candle
        // ran and agreed with ONNX Runtime. Both are tract dissenting on `Sign`, which is the
        // property that identifies the finding — not the shape of the summary string.
        s if s.starts_with("Sign |")
            && (s.contains("onnxruntime | tract") || s.contains("tract disagrees with")) =>
        {
            Some("F-001 — tract returns Sign(0)=1 for integers; the spec states 0. CONFIRMED")
        }
        s if s.starts_with("Where |") && s.contains("onnxruntime | tract") => {
            Some("F-004 — ONNX Runtime returns +0.0 for a -0.0 selected from X. CONFIRMED")
        }
        s if s.starts_with("Reshape |") => Some(
            "F-002 — tract and candle reject a rank-changing Reshape of a zero-size tensor that \
             the reference and ONNX Runtime both accept. CANDIDATE",
        ),
        _ => None,
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let classified_tract = WithCapabilities::new(TractRuntime, &caps);
    let classified_ort = WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let classified_candle = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);

    for (label, bounds) in [
        ("ordinary values", Bounds::default()),
        (
            "adversarial values",
            Bounds::default().with_special_values(),
        ),
    ] {
        let generator = OnnxGenerator::new(bounds);

        let mut generated = 0usize;
        let mut invalid = 0usize;
        let mut raw_divergences = 0usize;
        let mut after_capability = 0usize;
        let mut skipped_too_few = 0usize;
        let mut skipped_degenerate_or_licensed = 0usize;
        let mut agreed = 0usize;
        let mut survivors: BTreeMap<String, usize> = BTreeMap::new();
        let mut removed_by_capability: BTreeMap<String, usize> = BTreeMap::new();

        for seed in 0..SEEDS {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            generated += 1;
            if !onnx_adapter::validation::is_valid(&case) {
                invalid += 1;
                continue;
            }

            // ── Stage 1: raw, with no capability classification at all ──────────────
            // What the oracle would report if a runtime declining an operator it never
            // implemented counted as a disagreement. This is the number the domain started
            // with before the capability model was pulled forward from N5.
            let raw: Vec<NamedOutput<Canonical>> = {
                #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
                let mut outs = vec![
                    ("tract", TractRuntime.run(&case).expect("never Err")),
                    ("onnxruntime", OrtRuntime.run(&case).expect("never Err")),
                ];
                #[cfg(feature = "candle")]
                outs.push((
                    "candle",
                    onnx_adapter::runtimes::CandleRuntime
                        .run(&case)
                        .expect("never Err"),
                ));
                outs.into_iter()
                    .map(|(n, o)| NamedOutput {
                        implementation: n.to_string(),
                        output: OnnxNormalizer.normalize(o),
                    })
                    .collect()
            };
            let raw_verdict = OnnxOracle.check(&case, &raw);
            if matches!(raw_verdict, Verdict::Diverged(_)) {
                raw_divergences += 1;
            }

            // ── Stage 2: with capability classification ─────────────────────────────
            #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
            let mut outs = vec![
                ("tract", classified_tract.run(&case).expect("never Err")),
                ("onnxruntime", classified_ort.run(&case).expect("never Err")),
            ];
            #[cfg(feature = "candle")]
            outs.push(("candle", classified_candle.run(&case).expect("never Err")));
            let classified: Vec<NamedOutput<Canonical>> = outs
                .into_iter()
                .map(|(n, o)| NamedOutput {
                    implementation: n.to_string(),
                    output: OnnxNormalizer.normalize(o),
                })
                .collect();

            match OnnxOracle.check(&case, &classified) {
                Verdict::Agree => agreed += 1,
                Verdict::Skipped(SkipReason::TooFewResults { .. }) => skipped_too_few += 1,
                Verdict::Skipped(_) => skipped_degenerate_or_licensed += 1,
                Verdict::Diverged(d) => {
                    after_capability += 1;
                    *survivors
                        .entry(format!("{} | {}", case.op.onnx_name(), d.summary))
                        .or_default() += 1;
                }
            }

            // What capability classification specifically removed, and on what grounds.
            if matches!(raw_verdict, Verdict::Diverged(_))
                && !matches!(OnnxOracle.check(&case, &classified), Verdict::Diverged(_))
            {
                let who: Vec<&str> = classified
                    .iter()
                    .filter(|o| o.output.is_unsupported())
                    .map(|o| o.implementation.as_str())
                    .collect();
                *removed_by_capability
                    .entry(format!(
                        "{} | {:?} rank {} | abstained: {}",
                        case.op.onnx_name(),
                        onnx_adapter::ops::data_elem_type(&case),
                        onnx_adapter::ops::data_rank(&case),
                        who.join(", ")
                    ))
                    .or_default() += 1;
            }
        }

        println!("\n═══════ the funnel — {label} ═══════\n");
        println!("  {generated:>6}  cases generated");
        println!("  {invalid:>6}  dropped: invalid by our own validator (must be 0)");
        println!("  {raw_divergences:>6}  disagreements with NO capability classification");
        println!(
            "  {:>6}  removed by capability classification  ({} distinct grounds)",
            raw_divergences.saturating_sub(after_capability),
            removed_by_capability.len()
        );
        println!("  {skipped_too_few:>6}  skipped: fewer than two runtimes left to compare");
        println!(
            "  {skipped_degenerate_or_licensed:>6}  skipped: degenerate result, or agreement resting on a licensed difference"
        );
        println!("  {agreed:>6}  agreed");
        println!("  {after_capability:>6}  SURVIVE as reported divergences");

        println!("\n  what capability classification removed (top 10, with grounds):");
        let mut grounds: Vec<_> = removed_by_capability.iter().collect();
        grounds.sort_by(|a, b| b.1.cmp(a.1));
        for (reason, count) in grounds.iter().take(10) {
            println!("    {count:>4}x  {reason}");
        }

        println!("\n  survivors — each must be defensible as plausibly real:");
        let mut unexplained = 0;
        for (signature, count) in &survivors {
            match explanation(signature) {
                Some(why) => println!("    {count:>4}x  {signature}\n            └─ {why}"),
                None => {
                    unexplained += 1;
                    println!("    {count:>4}x  {signature}\n            └─ *** UNEXPLAINED ***");
                }
            }
        }
        println!(
            "\n  {}",
            if unexplained == 0 {
                "every surviving signature is accounted for by a finding draft"
            } else {
                "*** SOME SIGNATURES ARE UNEXPLAINED — N6.6 is not met ***"
            }
        );
    }
    std::panic::set_hook(previous);

    // ── The generator side, which cannot be measured retrospectively ────────────────
    println!("\n═══════ filtered before the oracle ever saw them ═══════\n");
    println!("  These cases are never generated, so they cannot appear in the funnel above.");
    println!("  Each carries the measurement it was established at.\n");
    for entry in CATALOG {
        let (marker, detail) = match entry.handling {
            Handling::DeclinedByGenerator => ("declined", ""),
            Handling::ForgivenByComparison => ("forgiven", " <- the only oracle-side loosening"),
            Handling::ExcludedByConfiguration { becomes_live_if } => ("excluded", becomes_live_if),
        };
        println!(
            "  [{marker}] {:<26} SPECS §{}{}",
            entry.id, entry.citation.specs_section, detail
        );
    }
}
