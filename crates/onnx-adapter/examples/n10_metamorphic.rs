//! **N10.5, N10.6** — run every metamorphic relation over the differential oracle's own corpus,
//! and report what each one contributed.
//!
//! # Why composition, not a total
//!
//! `PHASE-N10` marks N10.6 in red, and the reason is a mistake this project has already made:
//! *"six oracles agree" and "one relation covers 96%" are indistinguishable in a total.* A headline
//! number of checks says nothing about whether the coverage is broad or a monoculture, and the
//! flattering reading is the one nobody questions.
//!
//! So every relation reports its own held / violated / not-applicable, and the not-applicable
//! column is printed rather than folded away — a relation that almost never applies is a relation
//! whose zero means almost nothing.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n10_metamorphic --features candle -- [cases]
use std::collections::BTreeMap;

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use onnx_adapter::case::{OnnxCase, TensorValue};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::metamorphic::{self, Tally, Verdict};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

fn cases() -> u64 {
    std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000)
}

fn produced(outcome: &OnnxOutcome) -> Option<&Vec<TensorValue>> {
    match outcome {
        OnnxOutcome::Ok(t) => Some(t),
        _ => None,
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    // **The same corpus the differential oracle uses** (N10.5), so the two are comparable. A
    // relation run over a different corpus produces a number that cannot be set beside anything.
    let bounds = Bounds::default().with_special_values().with_quantized();
    let generator = OnnxGenerator::new(bounds.clone());

    let participants: Vec<(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)> =
        vec![("tract", &TractRuntime), ("onnxruntime", &OrtRuntime)];

    // relation -> runtime -> tally
    let mut results: BTreeMap<&'static str, BTreeMap<&'static str, Tally>> = BTreeMap::new();
    let mut violations: Vec<String> = Vec::new();

    for seed in 0..cases() {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if !onnx_adapter::validation::is_valid(&case) {
            continue;
        }

        for (name, runtime) in &participants {
            let outcome = runtime.run(&case).expect("never Err");

            // ── N10.1: inferred shape versus produced shape ─────────────────────────
            let verdict = match produced(&outcome) {
                Some(tensors) => metamorphic::shape_matches_inference(&case, tensors),
                None => Verdict::NotApplicable,
            };
            results
                .entry("shape-inference")
                .or_default()
                .entry(name)
                .or_default()
                .record(verdict);
            if verdict == Verdict::Violated && violations.len() < 10 {
                let (elem, dims) = onnx_adapter::ops::output_spec(&case);
                violations.push(format!(
                    "shape-inference | {name} | seed {seed} | {} | inferred {:?}{:?} produced {:?}",
                    case.op.onnx_name(),
                    elem,
                    dims,
                    produced(&outcome).map(|t| (t[0].elem_type(), t[0].dims.clone()))
                ));
            }

            // ── N10.2: the same model at an opset where the operator is unchanged ───
            let verdict = 'opset: {
                let Some(tensors) = produced(&outcome) else {
                    break 'opset Verdict::NotApplicable;
                };
                let Some(earlier) = metamorphic::opset_invariant(&case) else {
                    break 'opset Verdict::NotApplicable;
                };
                match runtime.run(&earlier).expect("never Err") {
                    OnnxOutcome::Ok(other) => {
                        if other.len() == tensors.len()
                            && other
                                .iter()
                                .zip(tensors.iter())
                                .all(|(a, b)| metamorphic::identical(a, b))
                        {
                            Verdict::Held
                        } else {
                            Verdict::Violated
                        }
                    }
                    _ => Verdict::NotApplicable,
                }
            };
            results
                .entry("opset-invariance")
                .or_default()
                .entry(name)
                .or_default()
                .record(verdict);
            if verdict == Verdict::Violated && violations.len() < 10 {
                violations.push(format!(
                    "opset-invariance | {name} | seed {seed} | {} | opset {} vs {}",
                    case.op.onnx_name(),
                    onnx_adapter::ops::spec(case.op).since,
                    case.opset
                ));
            }

            // ── N10.3: Transpose composed with its inverse is the identity ──────────
            let verdict = 'transpose: {
                let Some(tensors) = produced(&outcome) else {
                    break 'transpose Verdict::NotApplicable;
                };
                let Some(second) = metamorphic::transpose_inverse(&case, &tensors[0]) else {
                    break 'transpose Verdict::NotApplicable;
                };
                match runtime.run(&second).expect("never Err") {
                    OnnxOutcome::Ok(back) => {
                        if metamorphic::identical(&case.inputs[0], &back[0]) {
                            Verdict::Held
                        } else {
                            Verdict::Violated
                        }
                    }
                    _ => Verdict::NotApplicable,
                }
            };
            results
                .entry("transpose-inverse")
                .or_default()
                .entry(name)
                .or_default()
                .record(verdict);
            if verdict == Verdict::Violated && violations.len() < 10 {
                violations.push(format!(
                    "transpose-inverse | {name} | seed {seed} | dims {:?}",
                    case.inputs[0].dims
                ));
            }

            // ── N10.3: a widening Cast round-trips exactly ──────────────────────────
            let verdict = 'cast: {
                let Some((widen, back_to)) = metamorphic::cast_round_trip(&case) else {
                    break 'cast Verdict::NotApplicable;
                };
                let OnnxOutcome::Ok(wide) = runtime.run(&widen).expect("never Err") else {
                    break 'cast Verdict::NotApplicable;
                };
                let narrow = metamorphic::cast_back(&wide[0], back_to, case.opset);
                match runtime.run(&narrow).expect("never Err") {
                    OnnxOutcome::Ok(back) => {
                        if metamorphic::identical(&case.inputs[0], &back[0]) {
                            Verdict::Held
                        } else {
                            Verdict::Violated
                        }
                    }
                    _ => Verdict::NotApplicable,
                }
            };
            results
                .entry("cast-round-trip")
                .or_default()
                .entry(name)
                .or_default()
                .record(verdict);
            if verdict == Verdict::Violated && violations.len() < 10 {
                violations.push(format!(
                    "cast-round-trip | {name} | seed {seed} | {:?}",
                    case.inputs[0].elem_type()
                ));
            }
        }
    }
    std::panic::set_hook(previous);

    println!("\nmetamorphic relations over {} seeds", cases());
    println!("corpus: {}\n", bounds.description());
    println!(
        "{:<20} {:<14} {:>10} {:>10} {:>18} {:>10}",
        "relation", "runtime", "held", "VIOLATED", "not applicable", "judged"
    );

    let mut total_judged = 0usize;
    let mut total_violated = 0usize;
    for (relation, per_runtime) in &results {
        for (runtime, tally) in per_runtime {
            println!(
                "{relation:<20} {runtime:<14} {:>10} {:>10} {:>18} {:>10}",
                tally.held,
                tally.violated,
                tally.not_applicable,
                tally.judged()
            );
            total_judged += tally.judged();
            total_violated += tally.violated;
        }
    }

    println!("\n── composition (N10.6) ──");
    println!("  {total_judged} checks judged in total, {total_violated} violated");
    for (relation, per_runtime) in &results {
        let judged: usize = per_runtime.values().map(Tally::judged).sum();
        let share = 100.0 * judged as f64 / total_judged.max(1) as f64;
        println!("  {relation:<20} {judged:>9} checks  ({share:>5.1}% of all judging)");
    }
    println!(
        "\n  **A relation's zero means as much as its share of the judging.** A relation that\n  \
         almost never applies contributes a zero that nothing rests on."
    );

    if violations.is_empty() {
        println!("\nno relation was violated");
    } else {
        println!("\nVIOLATIONS:");
        for v in &violations {
            println!("  {v}");
        }
    }
}
