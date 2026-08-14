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
//! **One fault is not enough, and the numbers say so.** `WrongShape` was chosen because the
//! dominant relation is about shapes — and a 3,000,000-case control with it moved
//! `shape-inference` (49.1% of judging) and `cast-round-trip` (0.8%) while leaving
//! `opset-invariance` (48.2%) and `transpose-inverse` (1.9%) at **exactly zero**. Half the
//! judging behind an 11,750,917-check zero had never been shown able to fire.
//!
//! `opset-invariance` compares two runs of the **same** runtime, so any fault applied equally to
//! both is invisible to it; `transpose-inverse` *declined to apply* under a broken shape, its
//! judged count falling from 113,947 to 28,577 — the fault removing the check rather than
//! tripping it. So the control takes a `--fault`, and the gate asks that **every** relation have
//! one that fires.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n10_metamorphic --features candle -- \
//!       --name <run> [--seeds A..B] [--control]
use std::collections::BTreeMap;

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::case::{OnnxCase, TensorValue};
use onnx_adapter::findings::{CampaignLog, Run, StoredFinding};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::metamorphic::{self, Tally, Verdict};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::testing::{WrongAtOpset, WrongShape, WrongValues};

/// Log every this many cases; print to stdout far less often. A background process here is
/// reclaimed after ~1,029 printed lines, which cost two campaigns before it was diagnosed.
const PROGRESS_EVERY: u64 = 2_000;
const PRINT_EVERY: u64 = 50_000;

struct Args {
    name: String,
    seeds: std::ops::Range<u64>,
    control: bool,
    fault: Fault,
}

/// Which deliberate fault a control run injects.
///
/// **A relation is only controlled by a fault it can see.** Listed with what each one moves,
/// measured rather than assumed — see the module note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// Flatten the output shape. Moves `shape-inference` and `cast-round-trip`.
    Shape,
    /// Perturb one element of every output. Moves `transpose-inverse`, which composes two runs
    /// and therefore accumulates the perturbation, and `cast-round-trip`.
    Values,
    /// Perturb one element **only at opset 22 and above**. The one fault `opset-invariance` can
    /// see, because it compares two runs of the same runtime at different opsets.
    AtOpset,
}

impl Fault {
    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "shape" => Some(Fault::Shape),
            "values" => Some(Fault::Values),
            "opset" => Some(Fault::AtOpset),
            _ => None,
        }
    }

    /// The relations this fault is expected to move. Named so the run can *check* rather than
    /// leave it to a reader to notice a zero in the wrong column.
    fn expected_to_move(self) -> &'static [&'static str] {
        match self {
            Fault::Shape => &["shape-inference", "cast-round-trip"],
            Fault::Values => &["transpose-inverse", "cast-round-trip"],
            Fault::AtOpset => &["opset-invariance"],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Fault::Shape => "shape",
            Fault::Values => "values",
            Fault::AtOpset => "opset",
        }
    }
}

fn parse_args() -> Args {
    let mut name = "metamorphic".to_string();
    let mut seeds = 0..20_000u64;
    let mut control = false;
    let mut fault = Fault::Shape;
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
            "--fault" => {
                i += 1;
                match raw.get(i).and_then(|r| Fault::parse(r)) {
                    Some(chosen) => fault = chosen,
                    None => eprintln!("unrecognised --fault; keeping {}", fault.label()),
                }
            }
            other => eprintln!("ignoring unrecognised argument {other}"),
        }
        i += 1;
    }
    Args {
        name,
        seeds,
        control,
        fault,
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
        // The fault is in the name, because three controls must not overwrite each other's
        // findings — and because a control's result is a claim about *one* fault.
        format!("{}-control-{}", args.name, args.fault.label())
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
            "  MODE         CONTROL, fault = {}. Violations are EXPECTED.\n\
             \x20              Expected to move: {}\n\
             \x20              A relation this fault cannot see keeps an UNCONTROLLED zero;\n\
             \x20              only the relations listed above are validated by this run.",
            args.fault.label(),
            args.fault.expected_to_move().join(", ")
        );
    }
    println!("  writes       findings/onnx/runs/metamorphic/{run_name}/*.json");
    println!("               findings/onnx/logs/{run_name}.log");
    println!("  NOTE         findings go under runs/metamorphic/, a different tree from the");
    println!("               differential campaign, because they are a different kind of claim.\n");

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let generator = OnnxGenerator::new(bounds.clone());

    // **Every participant is capability-gated, exactly as the differential campaign gates them.**
    //
    // This example originally used the bare runtimes. For `tract` and ONNX Runtime that is
    // provably inert — the census records both as representing all seven element types, so the
    // representability gate never fires, and the wrapper's only other effect is to reclassify
    // `Rejected` as `Unsupported`, which the relations map to `NotApplicable` either way.
    //
    // For candle it is decisive. The census measures its representable types as **`F32`, `F64`
    // and `I64` only** — it has no `Bool`. Run ungated, candle answers `Less` with a `uint8`
    // tensor, and `shape_matches_inference` correctly observes that the inferred type was
    // `Bool`: **1,501 violations in 20,000 cases, every one of them ours.** `capability.rs` had
    // already written this hazard down, about `Cast` to `int32`.
    //
    // The gate must wrap the runtime *inside* the saboteur, not outside it: `WithCapabilities`
    // keys on `inner.name()`, and a saboteur renames its inner runtime, so an outer gate would
    // look up "candle-wrong-shape", find nothing measured, and conclude nothing.
    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    assert!(drift.is_empty(), "the census is stale: {drift:?}");

    let tract = WithCapabilities::new(TractRuntime, &caps);
    let ort = WithCapabilities::new(OrtRuntime, &caps);

    // All three saboteurs are built regardless of which is used, so each binding outlives the
    // trait-object references taken below. A `match` returning them directly would drop the
    // unselected ones at the end of the arm.
    let shape_tract = WrongShape::new(WithCapabilities::new(TractRuntime, &caps));
    let shape_ort = WrongShape::new(WithCapabilities::new(OrtRuntime, &caps));
    // A delta far larger than any rounding difference, for the same reason `WrongValues` documents:
    // a fault the oracle might mistake for noise proves nothing.
    let values_tract = WrongValues::new(WithCapabilities::new(TractRuntime, &caps), 7.0);
    let values_ort = WrongValues::new(WithCapabilities::new(OrtRuntime, &caps), 7.0);
    // Corrupt at opset 22 and above. Cases are generated at 22 and `opset-invariance` derives the
    // *older* twin, so exactly one side of that comparison is corrupted.
    let opset_tract = WrongAtOpset::new(WithCapabilities::new(TractRuntime, &caps), 22, 7.0);
    let opset_ort = WrongAtOpset::new(WithCapabilities::new(OrtRuntime, &caps), 22, 7.0);

    // **candle, the runtime the relations had never been asked about.**
    //
    // The first metamorphic campaign ran on `tract` and ONNX Runtime — the two most mature
    // participants — and returned 0 violations over 11,750,917 checks. A zero drawn only from the
    // strongest implementations is the weakest form of that zero, and candle is the one already
    // implicated in two findings (`Reshape` of a zero-size tensor, and `allowzero`).
    //
    // It is also where the relations' structural advantage lies: a relation judges **one**
    // implementation against a rule, so it can reach a case only candle supports — 159,711 cases
    // (5.3%) of the differential campaign were skipped for having fewer than two runtimes to
    // compare. `PENDING` 2.12.
    #[cfg(feature = "candle")]
    let candle = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);
    #[cfg(feature = "candle")]
    let shape_candle = WrongShape::new(WithCapabilities::new(
        onnx_adapter::runtimes::CandleRuntime,
        &caps,
    ));
    #[cfg(feature = "candle")]
    let values_candle = WrongValues::new(
        WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps),
        7.0,
    );
    #[cfg(feature = "candle")]
    let opset_candle = WrongAtOpset::new(
        WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps),
        22,
        7.0,
    );

    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut participants: Vec<(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)> =
        if args.control {
            match args.fault {
                Fault::Shape => vec![("tract", &shape_tract), ("onnxruntime", &shape_ort)],
                Fault::Values => vec![("tract", &values_tract), ("onnxruntime", &values_ort)],
                Fault::AtOpset => vec![("tract", &opset_tract), ("onnxruntime", &opset_ort)],
            }
        } else {
            vec![("tract", &tract), ("onnxruntime", &ort)]
        };

    // Appended rather than written into each arm, so the sabotaged and real paths cannot drift
    // apart on which runtimes take part — the failure the differential campaign's participant
    // list has already had once.
    #[cfg(feature = "candle")]
    if args.control {
        participants.push(match args.fault {
            Fault::Shape => ("candle", &shape_candle),
            Fault::Values => ("candle", &values_candle),
            Fault::AtOpset => ("candle", &opset_candle),
        });
    } else {
        participants.push(("candle", &candle));
    }

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

    // **A control passes when the relations it targets moved, not when *something* moved.**
    //
    // The first version asked only whether the violation list was non-empty. Against a shape
    // fault that is satisfied by `shape-inference` alone — and `shape-inference` is 49% of the
    // judging, so "CONTROL PASSED" was printed over `opset-invariance` and `transpose-inverse`
    // sitting at exactly zero, uncontrolled, for a full 3,000,000-case run.
    if args.control {
        let moved: Vec<&str> = args
            .fault
            .expected_to_move()
            .iter()
            .copied()
            .filter(|relation| {
                results
                    .get(*relation)
                    .is_some_and(|per| per.values().any(|tally| tally.violated > 0))
            })
            .collect();
        let missed: Vec<&str> = args
            .fault
            .expected_to_move()
            .iter()
            .copied()
            .filter(|relation| !moved.contains(relation))
            .collect();
        say(format!(
            "\n  fault `{}` moved: [{}]   did NOT move: [{}]",
            args.fault.label(),
            moved.join(", "),
            missed.join(", ")
        ));
        if missed.is_empty() {
            say("  CONTROL PASSED — every relation this fault targets was violated.".to_string());
        } else {
            say(
                "  *** CONTROL FAILED — a relation this fault should have tripped did not.\n  \
                 Its zero in the real run proves NOTHING ***"
                    .to_string(),
            );
        }
        // Which relations remain unvalidated *by any fault* is not knowable from one run, so the
        // reminder is unconditional rather than a computed claim this run cannot support.
        say(
            "  Relations outside that list keep an uncontrolled zero until another fault moves \n               them. See the module note for the measured fault-to-relation map."
                .to_string(),
        );
    }

    if violations.is_empty() {
        if !args.control {
            say("\nno relation was violated".to_string());
        }
    } else {
        say(format!(
            "\n{} VIOLATIONS (first 10 shown):",
            violations.len()
        ));
        for v in &violations {
            say(format!("  {v}"));
        }
    }
    say(format!("\n  findings: {}", findings_path));
    say(format!("  log:      {}", log_path));
}
