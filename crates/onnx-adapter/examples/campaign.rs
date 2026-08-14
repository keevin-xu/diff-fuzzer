//! A sustained campaign whose numbers can be defended.
//!
//! # What this reports, and why each number is here
//!
//! `05-MEASUREMENT-AND-CAMPAIGNS.md` catalogues the ways a campaign lies, and every one of them
//! has already cost this project time. Each is answered by a specific line of output:
//!
//! | the lie | the answer |
//! |---|---|
//! | a rate with no baseline | a **control run** where divergence is expected, beside the real one |
//! | totals instead of composition | per-operator, per-kind and per-implementation breakdowns |
//! | the nominal bound | the **effective** bound over *substantive* cases only |
//! | one bug counted a thousand times | de-duplication by signature |
//! | many signatures read as many bugs | **problems** reported beside signatures (`PENDING` 2.7) |
//! | a count with the cases discarded | every finding written to disk, case included |
//!
//! # Survivability
//!
//! Three separate properties, learned one interruption at a time:
//!
//! - **inspectable** — progress is printed and flushed as it goes, so a running campaign can be
//!   watched and a killed one still shows how far it got;
//! - **resumable** — takes a seed range, so a restart continues from where it stopped rather than
//!   repeating work. Re-running a range that was already covered is **idempotent** rather than
//!   skipped: a finding's filename is derived from its signature, so the same problem rewrites the
//!   same file. Deliberately not "skip anything already found" — re-confirming a known problem on
//!   a later run is exactly how a fixed one gets noticed;
//! - **survivable** — every distinct problem is written to disk **the moment it is first seen**,
//!   unminimised, then rewritten in place once minimisation has run. A campaign killed at 2.9
//!   million of 3 million cases still leaves behind every problem it found.
//!
//! That last property was *claimed before it was true*. The first version accumulated findings in
//! a map and wrote them only after the sweep finished, so an interrupted campaign lost **all** of
//! them and left only a progress log — while the comment above it read exactly as it does now.
//! Caught because a monitor reported `findings-on-disk=0` six minutes into a run that had already
//! found forty signatures.
//!
//! # Usage
//!
//! ```text
//! campaign --name <run> [--seeds A..B] [--control] [--announce]
//!
//!   --announce   print what the run would do, and exit without running it
//!   --control    inject a deliberately wrong participant; divergence is EXPECTED
//!   --seeds A..B seed range, default 0..20000
//!   --opsets     draw each case's opset from the operator's own span instead of pinning 22.
//!                Off by default, for the same reason as --quantized: the pinned corpus is the
//!                baseline this axis's yield is measured against.
//!   --quantized  include the Tier Q surface (PHASE-N9); off by default, so the default IS
//!                the baseline the quantized yield is measured against
//!   --quantized-only  ONLY the Tier Q surface. Quantized operators are ~4.7% of a mixed
//!                corpus, so a mixed campaign spends most of its budget on a surface already
//!                known to be saturated.
//! ```
use std::collections::BTreeMap;
use std::time::Instant;

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{
    Generator, Implementation, NamedOutput, Oracle, SkipReason, Verdict,
};
use diff_fuzzer_core::{Budget, Normalizer, axes::GenerationAxes, minimize_within};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::case::OnnxCase;
use onnx_adapter::findings::{CampaignLog, Minimisation, Run, StoredFinding, write_rough_draft};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::problems::{Status, group};
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::sentinel::CrashSentinel;
use onnx_adapter::shrink::complexity;
use onnx_adapter::signature::Signature;
use onnx_adapter::testing::WrongValues;

/// How often to record progress **in the log file**. Frequent, because the log is read after the
/// fact and detail there is free.
const PROGRESS_EVERY: u64 = 2000;

/// How often to also print progress **to stdout**.
///
/// # Why these are different numbers
///
/// A background process in this environment is reclaimed after a bounded number of stdout lines.
/// Measured, not guessed: **four separate campaign runs across two independent pairs all stopped
/// at exactly 1,029 progress lines**, which at one line per 2,000 cases is 2,058,000 cases every
/// time — the same count at different throughputs, so it is a volume limit rather than a time
/// limit. Two campaigns were lost to it before the cause was identified, both mistaken for
/// external interruption.
///
/// Printing every 50,000 cases instead puts the ceiling above 50 million, while the log keeps full
/// resolution. A campaign is watched through the log anyway.
const PRINT_EVERY: u64 = 50_000;

struct Args {
    name: String,
    seeds: std::ops::Range<u64>,
    control: bool,
    announce: bool,
    quantized: bool,
    quantized_only: bool,
    opsets: bool,
}

fn parse_args() -> Args {
    let mut name = "campaign".to_string();
    let mut seeds = 0..20_000u64;
    let (mut control, mut announce, mut quantized) = (false, false, false);
    let mut quantized_only = false;
    let mut opsets = false;

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
            "--opsets" => opsets = true,
            "--quantized" => quantized = true,
            "--quantized-only" => {
                quantized = true;
                quantized_only = true;
            }
            "--announce" => announce = true,
            other => eprintln!("ignoring unrecognised argument {other}"),
        }
        i += 1;
    }
    Args {
        name,
        seeds,
        control,
        announce,
        quantized,
        quantized_only,
        opsets,
    }
}

fn main() {
    let args = parse_args();
    // The quantized surface is a separate axis so its yield can be measured *against* the
    // Tier A/B baseline. N9.7 asks for exactly that comparison, and a rate without a baseline
    // is not a measurement.
    // **`--quantized-only` exists because of a composition problem, not a preference.**
    //
    // Quantized operators are 4 of 37, so they are only ~4.7% of a mixed corpus. A three-million
    // case campaign with `--quantized` would spend roughly 95% of its compute re-testing the
    // Tier A/B surface that N8 already measured to saturate at ~50,000 seeds — buying a bound
    // that already exists — to collect ~69,000 judged quantized cases.
    //
    // Turning the other axes off spends the whole budget on the surface being measured.
    let bounds = if args.quantized_only {
        Bounds {
            float_elementwise: false,
            comparisons: false,
            logical: false,
            structural: false,
            shape_input_operators: false,
            ..Bounds::default().with_special_values().with_quantized()
        }
    } else if args.quantized {
        Bounds::default().with_special_values().with_quantized()
    } else {
        Bounds::default().with_special_values()
    };
    let bounds = if args.opsets {
        bounds.with_opsets()
    } else {
        bounds
    };
    let cases = args.seeds.end.saturating_sub(args.seeds.start);

    // ── N8.2: announce before running ───────────────────────────────────────────────
    // Printed whether or not `--announce` was passed, so a campaign never starts without
    // having said what it is about to do.
    let run_name = if args.control {
        format!("{}-control", args.name)
    } else {
        args.name.clone()
    };
    println!("\n════════ campaign: {run_name} ════════");
    println!(
        "  seeds        {}..{}  ({cases} cases)",
        args.seeds.start, args.seeds.end
    );
    println!(
        "  participants tract, onnxruntime{}",
        if cfg!(feature = "candle") {
            ", candle"
        } else {
            ""
        }
    );
    if args.control {
        println!(
            "  MODE         CONTROL — tract is replaced by a deliberately wrong participant.\n\
             \x20              Divergence is EXPECTED here; a clean control means the campaign\n\
             \x20              could not have found anything and its zero means nothing."
        );
    }
    println!("  generator    {}", bounds.description());
    println!(
        "  writes       findings/onnx/runs/differential/{run_name}/*.json\n\
         \x20              findings/onnx/logs/{run_name}.log"
    );
    // A rough figure from the N5 measurement: ~0.2 ms per case per participant, plus
    // minimisation on whatever is found. Deliberately stated as an estimate.
    println!(
        "  estimate     ~{:.0} min of judging, plus minimisation of whatever is found",
        cases as f64 * 0.0006 / 60.0
    );
    if args.announce {
        println!("\n  --announce given; not running.\n");
        return;
    }
    println!();

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    assert!(drift.is_empty(), "the census is stale: {drift:?}");

    let tract = WithCapabilities::new(TractRuntime, &caps);
    let ort = WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let candle = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);
    // The control's wrong participant. Wraps the *bare* runtime, so the corruption sits where a
    // real defect would rather than being reclassified away by the capability layer.
    let saboteur = WithCapabilities::new(WrongValues::new(TractRuntime, 1.0), &caps);

    let judge = |case: &OnnxCase| -> Vec<NamedOutput<Canonical>> {
        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut outs: Vec<(&str, OnnxOutcome)> = if args.control {
            vec![
                ("tract", saboteur.run(case).expect("never Err")),
                ("onnxruntime", ort.run(case).expect("never Err")),
            ]
        } else {
            vec![
                ("tract", tract.run(case).expect("never Err")),
                ("onnxruntime", ort.run(case).expect("never Err")),
            ]
        };
        #[cfg(feature = "candle")]
        outs.push(("candle", candle.run(case).expect("never Err")));
        outs.into_iter()
            .map(|(n, o)| NamedOutput {
                implementation: n.to_string(),
                output: OnnxNormalizer.normalize(o),
            })
            .collect()
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

    let mut run = Run::open(onnx_adapter::OracleKind::Differential, &run_name).expect("run dir");
    // **Per-run, not shared.** Two campaigns running concurrently — which is exactly how a real
    // run and its control are meant to be run — would otherwise overwrite each other's in-flight
    // record, and the evidence of which case killed a process would name the wrong process.
    let (mut sentinel, recovered) = CrashSentinel::open(format!(
        "{}/in-flight-{run_name}.json",
        onnx_adapter::FINDINGS_ROOT
    ))
    .expect("sentinel");
    if let Some(in_flight) = &recovered {
        log.say(format!(
            "*** a previous run died on {} / {} (seed {}) — that case is itself a finding ***",
            in_flight.runtime,
            in_flight.case.op.onnx_name(),
            in_flight.seed
        ))
        .expect("log");
    }

    let generator = OnnxGenerator::new(bounds.clone());
    let started = Instant::now();

    // Composition, not totals.
    let (mut generated, mut invalid, mut agreed, mut diverged) = (0u64, 0u64, 0u64, 0u64);
    let (mut skipped_too_few, mut skipped_nothing_comparable) = (0u64, 0u64);
    let mut by_operator: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut by_kind: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut by_implementation: BTreeMap<String, u64> = BTreeMap::new();
    let mut signatures: BTreeMap<String, (Signature, usize, u64, OnnxCase)> = BTreeMap::new();

    for seed in args.seeds.clone() {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        generated += 1;
        if !onnx_adapter::validation::is_valid(&case) {
            invalid += 1;
            continue;
        }

        sentinel.arm("campaign", seed, &case).expect("arm");
        let outputs = judge(&case);
        sentinel.disarm().expect("disarm");

        match OnnxOracle.check(&case, &outputs) {
            Verdict::Agree => agreed += 1,
            Verdict::Skipped(SkipReason::TooFewResults { .. }) => skipped_too_few += 1,
            Verdict::Skipped(_) => skipped_nothing_comparable += 1,
            Verdict::Diverged(_) => {
                diverged += 1;
                if let Some(signature) = onnx_adapter::signature::of(&case, &outputs) {
                    *by_operator.entry(case.op.onnx_name()).or_default() += 1;
                    *by_kind.entry(signature.kind.token()).or_default() += 1;
                    for (name, outcome) in &signature.participants {
                        if outcome != "unsupported" {
                            *by_implementation.entry(name.clone()).or_default() += 1;
                        }
                    }
                    let key = signature.key();
                    let first_time = !signatures.contains_key(&key);

                    // **Written on first sight, not at the end.** Provisional: unminimised, and
                    // with an occurrence count the final pass corrects — but *on disk*, where an
                    // interrupted campaign leaves it behind. The filename derives from the
                    // signature, so the minimised version overwrites this one rather than
                    // accumulating beside it.
                    if first_time {
                        let provisional = StoredFinding::new(
                            &key,
                            "provisional — written when first seen; not yet minimised",
                            seed,
                            bounds.description(),
                            case.clone(),
                            outputs
                                .iter()
                                .map(|o| {
                                    (o.implementation.clone(), format!("{:?}", o.output.tensors))
                                })
                                .collect(),
                        )
                        .with_signature(signature.clone());
                        run.record(&provisional)
                            .expect("writing a provisional finding");
                    }

                    let entry = signatures
                        .entry(key)
                        .or_insert((signature, 0, seed, case.clone()));
                    entry.1 += 1;
                }
            }
        }

        if generated % PROGRESS_EVERY == 0 {
            let rate = generated as f64 / started.elapsed().as_secs_f64();
            let line = format!(
                "  … {generated}/{cases} cases · {} distinct signatures · {diverged} divergences · {rate:.0}/s",
                signatures.len()
            );
            // The log gets every line; stdout gets one in twenty-five. See `PRINT_EVERY`.
            if generated % PRINT_EVERY == 0 {
                log.say(line).expect("log");
            } else {
                log.line(line).expect("log");
            }
        }
    }
    let elapsed = started.elapsed();
    std::panic::set_hook(previous);

    // ── Minimise and store, de-duplicated by signature ──────────────────────────────
    let budget = Budget {
        max_steps: 200,
        max_candidates: 4000,
        max_duration: None,
    };
    let mut drafted: Vec<String> = Vec::new();
    for (key, (signature, hits, seed, case)) in &signatures {
        let before = complexity(case);
        let minimized = minimize_within(case.clone(), budget, |candidate: &OnnxCase| {
            let outputs = judge(candidate);
            matches!(OnnxOracle.check(candidate, &outputs), Verdict::Diverged(_))
                && onnx_adapter::signature::of(candidate, &outputs)
                    .map(|s| s.key())
                    .as_deref()
                    == Some(key.as_str())
        });
        let outputs = judge(&minimized.input);
        let rendered: Vec<(String, String)> = outputs
            .iter()
            .map(|o| (o.implementation.clone(), format!("{:?}", o.output.tensors)))
            .collect();
        let finding = StoredFinding::new(
            key,
            format!("{hits} occurrences in this run"),
            *seed,
            bounds.description(),
            minimized.input.clone(),
            rendered,
        )
        .with_signature(signature.clone())
        .with_minimisation(Minimisation {
            steps: minimized.steps,
            candidates_tried: minimized.candidates_tried,
            complete: minimized.is_minimal(),
            elements_before: before.elements,
            elements_after: complexity(&minimized.input).elements,
        });
        run.record(&finding).expect("writing a finding");

        // **A signature nobody has explained gets a draft, written now.** Everything needed to
        // start investigating it is in memory at this moment and nowhere else afterwards; the
        // alternative is a line in a log at the end of a half-hour run. The draft carries the
        // evidence and the triage ladder, never the analysis.
        //
        // **Never from a control.** A control injects a wrong answer into every operator, so every
        // one of its signatures is unexplained — not because it might be novel, but because we
        // broke it deliberately. Drafting them produced 533 files describing our own injected
        // faults, filed beside eight real reports. An unexplained signature is evidence only when
        // nothing was arranged to make it so.
        if !args.control
            && !onnx_adapter::problems::PROBLEMS
                .iter()
                .any(|p| p.covers(signature))
            && write_rough_draft(&finding, &run_name, onnx_adapter::OracleKind::Differential)
                .expect("writing a rough draft")
        {
            drafted.push(key.clone());
        }
    }

    // ── The report ──────────────────────────────────────────────────────────────────
    // **`judged` is already the substantive set.** A case that was skipped — too few runtimes, or
    // a result nothing could disagree about — never entered `agreed + diverged`, so subtracting
    // the skips from `judged` removes them a second time.
    //
    // The first version did exactly that, and the real run's rate looked entirely plausible at
    // 2.638%. The **control run** is what exposed it, by reporting a divergence rate of
    // **119.875%** — a rate above 100% being the one form of the error nobody can read past.
    // That is what N8.4 means by a baseline: it is not only a check on the runtimes, it is a
    // check on the arithmetic reporting them.
    let judged = agreed + diverged;
    let substantive = judged;
    let counted: Vec<(Signature, usize)> = signatures
        .values()
        .map(|(s, n, _, _)| (s.clone(), *n))
        .collect();
    let grouping = group(&counted);

    // Captured before the closure borrows the log, so the closing lines can still name them.
    let findings_path = run.directory().display().to_string();
    let log_path = log.path().display().to_string();
    let mut say = |line: String| log.say(line).expect("log");

    say(format!("\n════════ {run_name} — result ════════"));
    say(format!(
        "  ran {generated} cases in {:.1}s ({:.0}/s)",
        elapsed.as_secs_f64(),
        generated as f64 / elapsed.as_secs_f64().max(0.001)
    ));
    say(String::new());

    say("  ── composition (N8.5) ──".to_string());
    say(format!("  {generated:>8}  generated"));
    say(format!(
        "  {invalid:>8}  invalid by our own validator (must be 0)"
    ));
    say(format!("  {agreed:>8}  agreed"));
    say(format!(
        "  {skipped_too_few:>8}  skipped: fewer than two runtimes left to compare"
    ));
    say(format!(
        "  {skipped_nothing_comparable:>8}  skipped: nothing comparable (degenerate, or a licensed difference)"
    ));
    say(format!("  {diverged:>8}  diverged"));

    say(String::new());
    say("  ── the bound (N8.6) ──".to_string());
    say(format!(
        "  substantive           {substantive} of {generated} generated ({:.1}%) — cases that were",
        100.0 * substantive as f64 / generated.max(1) as f64
    ));
    say(format!(
        "                        actually compared and could have disagreed. The other {} were",
        generated - substantive - invalid
    ));
    say(format!(
        "                        skipped: {skipped_too_few} with too few runtimes, {skipped_nothing_comparable} with nothing comparable."
    ));
    if diverged == 0 {
        say(format!(
            "  EFFECTIVE BOUND       no divergence in {substantive} substantive cases"
        ));
    } else {
        say(format!(
            "  divergence rate       {:.3}% of substantive cases (1 in {:.0})",
            100.0 * diverged as f64 / substantive.max(1) as f64,
            substantive as f64 / diverged as f64
        ));
    }

    say(String::new());
    say("  ── problems, not signatures (N8.3, PENDING 2.7) ──".to_string());
    say(format!("  {:>8}  occurrences", diverged));
    say(format!("  {:>8}  distinct signatures", signatures.len()));
    // **The coarse count, and why it is here.**
    //
    // The signature key includes the opset — deliberately, since the scheme errs finer rather
    // than coarser. With `--opsets` on, a defect present at every version of an operator produces
    // one signature per version: measured at 60,000 seeds, **44 signatures pinned at 22 became
    // 243, for the same four problems**.
    //
    // So the fine count stopped being comparable with earlier runs. Dropping the opset gives the
    // number a reader actually reaches for — 43 against the pinned corpus's 44 — while the fine
    // count still shows how much surface each behaviour was seen across.
    let behaviours: std::collections::BTreeSet<(String, String, usize, &str)> = signatures
        .values()
        .map(|(signature, _, _, _)| {
            (
                signature.operator.to_string(),
                format!("{:?}", signature.elem_type),
                signature.rank,
                signature.kind.token(),
            )
        })
        .collect();
    say(format!(
        "  {:>8}  distinct behaviours (operator/type/rank/kind — comparable across opsets)",
        behaviours.len()
    ));
    say(format!(
        "  {:>8}  distinct PROBLEMS  <- the number to quote",
        grouping.problems()
    ));
    for (problem, sigs, occurrences) in &grouping.matched {
        say(format!(
            "      {} [{}] {} — {sigs} signatures, {occurrences} occurrences — {}",
            problem.id,
            match problem.status {
                Status::Ready => "READY",
                Status::Candidate => "CANDIDATE",
                Status::NotFiling => "not filing",
            },
            problem.implementation,
            problem.what
        ));
    }
    for (key, count) in &grouping.unexplained {
        say(format!("      *** UNEXPLAINED *** {count}x {key}"));
    }
    if !drafted.is_empty() {
        say(String::new());
        say(format!(
            "  {} rough draft(s) written to issues/onnx-runtime/ for the unexplained signatures:",
            drafted.len()
        ));
        for key in &drafted {
            say(format!("      DRAFT-… for {key}"));
        }
    }

    if !by_operator.is_empty() {
        say(String::new());
        say("  ── where the divergences are ──".to_string());
        let mut ops: Vec<_> = by_operator.iter().collect();
        ops.sort_by(|a, b| b.1.cmp(a.1));
        for (operator, count) in ops {
            say(format!("      {count:>6}  {operator}"));
        }
        for (kind, count) in &by_kind {
            say(format!("      {count:>6}  kind={kind}"));
        }
        for (implementation, count) in &by_implementation {
            say(format!("      {count:>6}  involved {implementation}"));
        }
    }

    say(String::new());
    if args.control {
        say(if diverged > 0 {
            "  CONTROL PASSED — the campaign could have found something.".to_string()
        } else {
            "  *** CONTROL FAILED — a clean control means a clean real run proves NOTHING ***"
                .to_string()
        });
    } else if grouping.has_unexplained() {
        say(
            "  Unexplained signatures above need triage (N8.8) before any number is quoted."
                .to_string(),
        );
    } else {
        say("  Every signature is accounted for by a known problem.".to_string());
    }
    say(format!("  findings: {findings_path}"));
    say(format!("  log:      {log_path}"));
}
