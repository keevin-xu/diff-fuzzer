//! **N10.5, N10.6** — the metamorphic campaign: every relation over the differential oracle's own
//! corpus, with findings filed under the metamorphic tree.
//!
//! # Why composition, not a total
//!
//! `PHASE-N10` marks N10.6 in red, and the reason is a mistake this project has already made:
//! *"six oracles agree" and "one relation covers 96%" are indistinguishable in a total.* A headline
//! number of checks says nothing about whether the coverage is broad or a monoculture, and the
//! flattering reading is the one nobody questions. It happened here too — with three relations,
//! one accounted for 94.7% of all judging.
//!
//! So every relation reports its own held / violated / not-applicable, and not-applicable is
//! printed rather than folded away: a relation that almost never applies contributes a zero that
//! nothing rests on.
//!
//! # Findings go in a separate tree
//!
//! `runs/metamorphic/` rather than `runs/differential/`, because the two are **different claims**.
//! A differential finding names implementations that disagreed and asks which is wrong. A
//! metamorphic violation names one implementation contradicting a relation that must hold, where
//! no second implementation is involved and no legal-difference argument is available at all.
//!
//! # The control
//!
//! `--control` substitutes a runtime whose outputs are deliberately reshaped. A relation suite that
//! stays clean against *that* could not have detected anything, and its zero would mean nothing.
//! `WrongShape` is the right injection here: the dominant relation is about shapes.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n10_metamorphic --features candle -- \
//!       --name <run> [--seeds A..B] [--control]
use std::collections::BTreeMap;

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use onnx_adapter::case::{OnnxCase, TensorValue};
use onnx_adapter::findings::{CampaignLog, Run, StoredFinding};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::metamorphic::{self, Tally, Verdict};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::testing::WrongShape;

/// Log every this many cases; print to stdout far less often. A background process here is
/// reclaimed after ~1,029 printed lines, which cost two campaigns before it was diagnosed.
const PROGRESS_EVERY: u64 = 2_000;
const PRINT_EVERY: u64 = 50_000;

struct Args {
    name: String,
    seeds: std::ops::Range<u64>,
    control: bool,
}

fn parse_args() -> Args {
    let mut name = "metamorphic".to_string();
    let mut seeds = 0..20_000u64;
    let mut control = false;
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--name" => {
                i += 1;
                name = raw.get(i).cloned().unwrap_or(name);
            }
            "--seeds" => {
                i += 1;
                if let Some((a, b)) = raw.get(i).and_then(|r| r.split_once("..")) {
                    seeds = a.parse().unwrap_or(0)..b.parse().unwrap_or(20_000);
                }
            }
            "--control" => control = true,
            other => eprintln!("ignoring unrecognised argument {other}"),
        }
        i += 1;
    }
    Args {
        name,
        seeds,
        control,
    }
}

fn produced(outcome: &OnnxOutcome) -> Option<&Vec<TensorValue>> {
    match outcome {
        OnnxOutcome::Ok(t) => Some(t),
        _ => None,
    }
}

/// Record a violation as a finding in the **metamorphic** tree.
///
/// # Why the signature is keyed on the relation
///
/// A metamorphic violation is not "these two disagreed"; it is "this one contradicted a rule".
/// Keying on the relation as well as the operator keeps two different broken rules on the same
/// operator apart — the same reason the differential signature keys on the disagreement kind.
///
/// The runtime is deliberately **not** in the key, matching the differential scheme: the same
/// broken relation on the same operator is one problem however many runtimes exhibit it. Which
/// runtimes did is recorded beside it.
#[allow(clippy::too_many_arguments)]
fn record_violation(
    run: &mut Run,
    seen: &mut Vec<String>,
    log: &mut CampaignLog,
    relation: &'static str,
    runtime: &str,
    case: &OnnxCase,
    seed: u64,
    detail: String,
    generator: &str,
) {
    let elem = onnx_adapter::ops::data_elem_type(case);
    let rank = onnx_adapter::ops::data_rank(case);
    let key = format!(
        "{relation}/{}/{}/{elem:?}/rank{rank}",
        case.op.onnx_name(),
        case.opset
    );
    if seen.contains(&key) {
        return;
    }
    seen.push(key.clone());

    let mut finding = StoredFinding::new(
        &key,
        format!("{relation} violated by {runtime}: {detail}"),
        seed,
        generator,
        case.clone(),
        vec![(runtime.to_string(), detail)],
    );
    finding.kind = relation.to_string();
    finding.disagreeing = vec![runtime.to_string()];
    run.record(&finding).expect("writing a metamorphic finding");
    let _ = log.line(format!("  VIOLATION recorded: {key}"));
}

fn main() {
    let args = parse_args();
    let run_name = if args.control {
        format!("{}-control", args.name)
    } else {
        args.name.clone()
    };
    let cases = args.seeds.end.saturating_sub(args.seeds.start);

    // **The same corpus the differential oracle uses** (N10.5), so the two are comparable. A
    // relation run over a different corpus produces a number that cannot be set beside anything.
    let bounds = Bounds::default().with_special_values().with_quantized();

    println!("\n════════ metamorphic campaign: {run_name} ════════");
    println!(
        "  seeds        {}..{} ({cases} cases)",
        args.seeds.start, args.seeds.end
    );
    println!(
        "  relations    shape-inference, opset-invariance, transpose-inverse, cast-round-trip"
    );
    if args.control {
        println!(
            "  MODE         CONTROL — every output is deliberately reshaped.\n\
             \x20              Violations are EXPECTED; a clean control means the relations\n\
             \x20              could not have detected anything and their zero means nothing."
        );
    }
    println!("  writes       findings/onnx/runs/metamorphic/{run_name}/*.json");
    println!("               findings/onnx/logs/{run_name}.log");
    println!("  NOTE         findings go under runs/metamorphic/, a different tree from the");
    println!("               differential campaign, because they are a different kind of claim.\n");

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let generator = OnnxGenerator::new(bounds.clone());

    let saboteur_tract = WrongShape::new(TractRuntime);
    let saboteur_ort = WrongShape::new(OrtRuntime);
    let participants: Vec<(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)> =
        if args.control {
            vec![("tract", &saboteur_tract), ("onnxruntime", &saboteur_ort)]
        } else {
            vec![("tract", &TractRuntime), ("onnxruntime", &OrtRuntime)]
        };

    let mut log = CampaignLog::open(&run_name).expect("opening the campaign log");
    log.header(&run_name, &bounds.description())
        .expect("log header");
    log.line(format!(
        "seeds {}..{}{}",
        args.seeds.start,
        args.seeds.end,
        if args.control { "  (CONTROL RUN)" } else { "" }
    ))
    .expect("log");

    // **A separate tree from the differential findings**, chosen by the oracle kind rather than by
    // a path string, so a metamorphic violation cannot be filed as a differential one.
    let mut run =
        Run::open(onnx_adapter::OracleKind::Metamorphic, &run_name).expect("run directory");
    let mut seen: Vec<String> = Vec::new();

    // relation -> runtime -> tally
    let mut results: BTreeMap<&'static str, BTreeMap<&'static str, Tally>> = BTreeMap::new();
    let mut violations: Vec<String> = Vec::new();

    let started = std::time::Instant::now();
    let mut generated = 0u64;
    for seed in args.seeds.clone() {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        generated += 1;
        if generated.is_multiple_of(PROGRESS_EVERY) {
            let judged: usize = results
                .values()
                .flat_map(|m| m.values())
                .map(Tally::judged)
                .sum();
            let violated: usize = results
                .values()
                .flat_map(|m| m.values())
                .map(|t| t.violated)
                .sum();
            let rate = generated as f64 / started.elapsed().as_secs_f64();
            let line = format!(
                "  … {generated}/{cases} cases · {judged} checks judged · {violated} violated · {rate:.0}/s"
            );
            if generated.is_multiple_of(PRINT_EVERY) {
                log.say(line).expect("log");
            } else {
                log.line(line).expect("log");
            }
        }
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
            if verdict == Verdict::Violated {
                let (elem, dims) = onnx_adapter::ops::output_spec(&case);
                let detail = format!(
                    "inferred {elem:?}{dims:?}, produced {:?}",
                    produced(&outcome).map(|t| (t[0].elem_type(), t[0].dims.clone()))
                );
                if violations.len() < 10 {
                    violations.push(format!(
                        "shape-inference | {name} | seed {seed} | {} | {detail}",
                        case.op.onnx_name()
                    ));
                }
                record_violation(
                    &mut run,
                    &mut seen,
                    &mut log,
                    "shape-inference",
                    name,
                    &case,
                    seed,
                    detail,
                    &bounds.description(),
                );
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
            if verdict == Verdict::Violated {
                let detail = format!(
                    "opset {} and {} disagree though the operator is unchanged between them",
                    onnx_adapter::ops::spec(case.op).since,
                    case.opset
                );
                if violations.len() < 10 {
                    violations.push(format!(
                        "opset-invariance | {name} | seed {seed} | {} | {detail}",
                        case.op.onnx_name()
                    ));
                }
                record_violation(
                    &mut run,
                    &mut seen,
                    &mut log,
                    "opset-invariance",
                    name,
                    &case,
                    seed,
                    detail,
                    &bounds.description(),
                );
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
            if verdict == Verdict::Violated {
                let detail = format!(
                    "transposing by the permutation then its inverse did not return the input \
                     (dims {:?})",
                    case.inputs[0].dims
                );
                if violations.len() < 10 {
                    violations.push(format!(
                        "transpose-inverse | {name} | seed {seed} | {detail}"
                    ));
                }
                record_violation(
                    &mut run,
                    &mut seen,
                    &mut log,
                    "transpose-inverse",
                    name,
                    &case,
                    seed,
                    detail,
                    &bounds.description(),
                );
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
            if verdict == Verdict::Violated {
                let detail = format!(
                    "a widening cast from {:?} did not round-trip losslessly",
                    case.inputs[0].elem_type()
                );
                if violations.len() < 10 {
                    violations.push(format!("cast-round-trip | {name} | seed {seed} | {detail}"));
                }
                record_violation(
                    &mut run,
                    &mut seen,
                    &mut log,
                    "cast-round-trip",
                    name,
                    &case,
                    seed,
                    detail,
                    &bounds.description(),
                );
            }
        }
    }
    std::panic::set_hook(previous);

    println!("\nmetamorphic relations over {generated} cases");
    let findings_path = run.directory().display().to_string();
    let log_path = log.path().display().to_string();
    let mut say = |line: String| log.say(line).expect("log");
    say(format!(
        "\n════════ {run_name} — result ({generated} cases in {:.1}s) ════════\n",
        started.elapsed().as_secs_f64()
    ));
    say(format!(
        "{:<20} {:<14} {:>10} {:>10} {:>18} {:>10}",
        "relation", "runtime", "held", "VIOLATED", "not applicable", "judged"
    ));

    let mut total_judged = 0usize;
    let mut total_violated = 0usize;
    for (relation, per_runtime) in &results {
        for (runtime, tally) in per_runtime {
            say(format!(
                "{relation:<20} {runtime:<14} {:>10} {:>10} {:>18} {:>10}",
                tally.held,
                tally.violated,
                tally.not_applicable,
                tally.judged()
            ));
            total_judged += tally.judged();
            total_violated += tally.violated;
        }
    }

    say("\n── composition (N10.6) ──".to_string());
    say(format!(
        "  {total_judged} checks judged in total, {total_violated} violated"
    ));
    for (relation, per_runtime) in &results {
        let judged: usize = per_runtime.values().map(Tally::judged).sum();
        let share = 100.0 * judged as f64 / total_judged.max(1) as f64;
        println!("  {relation:<20} {judged:>9} checks  ({share:>5.1}% of all judging)");
    }
    say(
        "\n  A relation's zero means as much as its share of the judging. A relation that almost\n  \
         never applies contributes a zero that nothing rests on."
            .to_string(),
    );

    if violations.is_empty() {
        say(if args.control {
            "\n  *** CONTROL FAILED — the relations could not detect a deliberately reshaped\n  \
             output, so a clean real run proves NOTHING ***"
                .to_string()
        } else {
            "\nno relation was violated".to_string()
        });
    } else {
        say(format!(
            "\n{} VIOLATIONS (first 10 shown):",
            violations.len()
        ));
        for v in &violations {
            say(format!("  {v}"));
        }
        if args.control {
            say("\n  CONTROL PASSED — the relations detect a broken implementation.".to_string());
        }
    }
    say(format!("\n  findings: {}", findings_path));
    say(format!("  log:      {}", log_path));
}
