//! N11.5–N11.7 — search for trigger rules, validate them by prediction, and report the gaps.
//!
//! # What this does, in order
//!
//! 1. **Load the findings** of a completed differential run. These are the positives: cases that
//!    really diverged, minimised, each with the generator description it was produced under.
//! 2. **Collect negatives** by re-running the generator and keeping cases every runtime agreed
//!    on — under the **same** bounds the findings came from, which `Pool::matched` then verifies
//!    rather than trusts.
//! 3. **Search** every conjunction of at most three features for rules that cover findings and
//!    fire on no negative.
//! 4. **Validate by prediction**: for each rule, draw fresh cases, keep the ones it matches, run
//!    them, and measure how often they diverge. A rule that describes the findings but predicts
//!    nothing is a coincidence and is reported as one.
//! 5. **Report the vocabulary gap** — the findings no rule could separate. This is the output
//!    most likely to be worth something, and it is printed last so it is the thing left on
//!    screen.
//!
//! # What a good result looks like, stated before running
//!
//! **Not "everything explained".** `PHASE-N11` names the hazard directly: *"a run in which
//! everything is neatly explained is a warning sign, not a success"*, because the vocabulary
//! could simply have been fitted to findings someone had already read. This domain's atoms come
//! from the phase file's pre-registered list, written before N7 found anything, which is what
//! makes either outcome informative.
//!
//! Prior art from the other two domains: in tensors **763 of 814** findings landed in
//! `unexplained`; in SQL the atoms were wrong *in kind*, describing one table when the trigger
//! was a relationship between two.
//!
//! # Usage
//!
//! ```text
//! n11_predicates [--run <name>] [--negatives N] [--validate N]
//!
//!   --run        the differential run to read findings from (default n10-diff)
//!   --negatives  cases to draw looking for agreeing ones (default 20000)
//!   --validate   cases to draw per rule when validating by prediction (default 20000)
//! ```

use std::collections::BTreeMap;

use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation, NamedOutput, Oracle, Verdict};
use onnx_adapter::capability::{Capabilities, WithCapabilities};
use onnx_adapter::case::OnnxCase;
use onnx_adapter::features::{FEATURES, features};
use onnx_adapter::findings::Run;
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::negatives::{Negative, Pool, SamplingContext, Source, is_interesting};
use onnx_adapter::normalize::{Canonical, OnnxNormalizer};
use onnx_adapter::oracle::OnnxOracle;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::rule_validation::{Outcome, validate};
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use onnx_adapter::search::search;

/// The seed every validation sampling runs from. **Fixed and reported**, because a rule's
/// evidence is only evidence if someone else can reproduce the number.
const VALIDATION_SEED: u64 = 20_250_814;

struct Args {
    run: String,
    negatives: usize,
    validate: usize,
    /// Feature names to validate as a hand-written rule, instead of searching.
    ///
    /// **The search ranks by fit, and fit cannot see prediction.** Two rules covering the same
    /// findings with the same number of terms tie, and enumeration order breaks it — so the
    /// committed rule may be the coincidence and its twin the real trigger. This flag exists to
    /// measure the twin rather than argue about it.
    probe: Vec<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        run: "n10-diff".to_string(),
        negatives: 20_000,
        validate: 20_000,
        probe: Vec::new(),
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--run" => {
                i += 1;
                args.run = raw.get(i).cloned().unwrap_or(args.run);
            }
            "--negatives" => {
                i += 1;
                args.negatives = raw
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.negatives);
            }
            "--validate" => {
                i += 1;
                args.validate = raw
                    .get(i)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.validate);
            }
            "--probe" => {
                i += 1;
                args.probe = raw
                    .get(i)
                    .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
                    .unwrap_or_default();
            }
            other => eprintln!("ignoring unrecognised argument {other}"),
        }
        i += 1;
    }
    args
}

fn main() {
    let args = parse_args();

    println!(
        "\n════════ N11 — predicate search over run `{}` ════════",
        args.run
    );

    // ── 1. The positives ──────────────────────────────────────────────────────────────
    let stored = Run::load(onnx_adapter::OracleKind::Differential, &args.run)
        .expect("run the campaign example first");
    if stored.is_empty() {
        println!("  no findings in that run — nothing to group.");
        return;
    }

    // **The findings carry the generator they were produced under**, and the negatives must be
    // drawn under the same one. Reading it off the findings rather than off whatever is compiled
    // today is the only way to be sure of the configuration that actually ran.
    let descriptions: Vec<&str> = {
        let mut d: Vec<&str> = stored.iter().map(|f| f.generator.as_str()).collect();
        d.sort_unstable();
        d.dedup();
        d
    };
    if descriptions.len() != 1 {
        println!(
            "  *** the findings come from {} different generator configurations. Scoring them \
             against one pool of negatives would compare distributions, not triggers. ***",
            descriptions.len()
        );
        for d in &descriptions {
            println!("      {d}");
        }
        return;
    }
    let description = descriptions[0].to_string();
    let findings: Vec<OnnxCase> = stored.iter().map(|f| f.case.clone()).collect();

    println!("  findings     {} cases", findings.len());
    println!("  generator    {description}");

    // Rebuild the bounds the campaign ran under. The quantized axis is the one that changes what
    // is reachable, and it is visible in the description the findings recorded.
    // **Rebuilt axis by axis from the description the findings recorded**, then checked against
    // it. Every axis must be listed here; the assertion below is what makes forgetting one a loud
    // failure rather than a silent distribution mismatch — which is exactly what it caught when
    // `vary-opset` was added and this was not updated.
    let mut bounds = Bounds::default().with_special_values();
    if description.contains("quantized=on") {
        bounds = bounds.with_quantized();
    }
    if description.contains("vary-opset=on") {
        bounds = bounds.with_opsets();
    }
    assert_eq!(
        bounds.description(),
        description,
        "the reconstructed bounds are not the ones the findings recorded, so the negatives would \
         be drawn from a different distribution than the positives. A new generation axis was \
         probably added without a matching branch above."
    );

    // ── the runtimes, gated exactly as the campaign gates them ────────────────────────
    let caps = Capabilities::load(&format!("{}/census.json", onnx_adapter::FINDINGS_ROOT))
        .expect("run the n2_census example first");
    let drift = caps.is_stale_for(&onnx_adapter::environment::environment().components);
    assert!(drift.is_empty(), "the census is stale: {drift:?}");

    let tract = WithCapabilities::new(TractRuntime, &caps);
    let ort = WithCapabilities::new(OrtRuntime, &caps);
    #[cfg(feature = "candle")]
    let candle = WithCapabilities::new(onnx_adapter::runtimes::CandleRuntime, &caps);

    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let judge = |case: &OnnxCase| -> Vec<NamedOutput<Canonical>> {
        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut outs: Vec<(&str, OnnxOutcome)> = vec![
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
    let diverges =
        |case: &OnnxCase| matches!(OnnxOracle.check(case, &judge(case)), Verdict::Diverged(_));

    let runtime_names = onnx_adapter::runtimes::compiled_runtime_names();
    let context = SamplingContext::new(description.clone(), &runtime_names);

    // ── 2. The negatives ──────────────────────────────────────────────────────────────
    //
    // Agreeing cases only. Stratified by `is_interesting` so the pool is not all easy cases: a
    // rule that only has to beat trivial negatives has not been tested against anything.
    println!("\n  ── collecting negatives ({} draws) ──", args.negatives);
    let generator = OnnxGenerator::new(bounds.clone());
    let mut collected: Vec<Negative> = Vec::new();
    let (mut agreeing, mut interesting) = (0usize, 0usize);
    for seed in 0..args.negatives as u64 {
        // Offset the seed space so negatives are not the very cases the findings came from.
        let case = generator.generate(&mut SeededRng::from_seed(seed + 100_000_000));
        if !onnx_adapter::validation::is_valid(&case) {
            continue;
        }
        if !matches!(OnnxOracle.check(&case, &judge(&case)), Verdict::Agree) {
            continue;
        }
        agreeing += 1;
        let source = if is_interesting(&case) {
            interesting += 1;
            Source::Interesting
        } else {
            Source::Ordinary
        };
        collected.push(Negative {
            case,
            source,
            provenance: context.provenance(),
            generator: context.generator.clone(),
            runtimes: context.runtimes.clone(),
        });
    }
    println!(
        "     {agreeing} agreed, of which {interesting} carry an awkward value ({}%)",
        (interesting * 100).checked_div(agreeing).unwrap_or(0)
    );

    let pool = match Pool::matched(collected, &context) {
        Ok(pool) => pool,
        Err(e) => {
            println!("\n  *** the negative pool was refused: {e} ***");
            println!(
                "      A rule that survives an empty or mismatched pool has survived nothing."
            );
            std::panic::set_hook(previous);
            return;
        }
    };
    println!("     pool of {} negatives, accepted", pool.len());

    // How often each feature holds, on each side. **Printed before the search**, because a
    // feature that never holds among the findings cannot appear in any rule, and a feature that
    // holds everywhere on both sides cannot separate anything — both facts explain a poor result
    // far better than the result itself does.
    println!("\n  ── feature incidence: findings vs negatives ──");
    println!(
        "     {:<28} {:>10} {:>12}",
        "feature", "findings", "negatives"
    );
    let negative_features: Vec<_> = pool.negatives().iter().map(|n| features(&n.case)).collect();
    for name in FEATURES {
        let in_findings = findings.iter().filter(|c| features(c).has(name)).count();
        let in_negatives = negative_features.iter().filter(|f| f.has(name)).count();
        let flag = if in_findings == 0 {
            "  <- never in a finding"
        } else if in_findings == findings.len() && in_negatives == pool.len() {
            "  <- holds everywhere"
        } else {
            ""
        };
        println!(
            "     {name:<28} {:>4}/{:<5} {:>5}/{:<6}{flag}",
            in_findings,
            findings.len(),
            in_negatives,
            pool.len()
        );
    }

    // ── 2b. A hand-written rule, if one was asked for ─────────────────────────────────
    if !args.probe.is_empty() {
        let names: Vec<&str> = args.probe.iter().map(String::as_str).collect();
        let probe = onnx_adapter::predicate::Predicate::new(&names, &[]);
        let covered = findings
            .iter()
            .filter(|c| probe.matches(features(c)))
            .count();
        let matched_negatives = negative_features
            .iter()
            .filter(|f| probe.matches(**f))
            .count();
        println!("\n  ── probe: {} ──", probe.describe());
        println!(
            "     FOR      covers {covered} of {} findings",
            findings.len()
        );
        println!(
            "     AGAINST  matches {matched_negatives} of {} negatives",
            pool.len()
        );
        let validation = validate(probe, &generator, VALIDATION_SEED, args.validate, diverges);
        println!("     PREDICTS {}", validation.describe());
        std::panic::set_hook(previous);
        println!();
        return;
    }

    // ── 3. The search ─────────────────────────────────────────────────────────────────
    println!("\n  ── search ──");
    let result = search(&findings, &pool);
    println!(
        "     {} predicates considered, {} rule(s) committed, {} finding(s) unexplained",
        result.considered,
        result.classes.len(),
        result.unexplained.len()
    );

    // ── 4. Validation by prediction ───────────────────────────────────────────────────
    //
    // The rules above are guaranteed to fit — the search found them by fitting. The only honest
    // test is whether they predict divergence in cases nobody has run.
    if !result.classes.is_empty() {
        println!("\n  ── candidates, with evidence for and against (N11.6) ──");
    }
    for (i, class) in result.classes.iter().enumerate() {
        println!("\n     [{}] {}", i + 1, class.predicate.describe());
        println!(
            "         FOR      covers {} of {} findings",
            class.covered.len(),
            findings.len()
        );
        // Signatures of the covered findings, so a reader can see what the rule grouped.
        let mut covered_signatures: Vec<&str> = class
            .covered
            .iter()
            .map(|&i| stored[i].signature.as_str())
            .collect();
        covered_signatures.sort_unstable();
        covered_signatures.dedup();
        for s in covered_signatures.iter().take(6) {
            println!("                  {s}");
        }
        if covered_signatures.len() > 6 {
            println!(
                "                  … and {} more",
                covered_signatures.len() - 6
            );
        }

        println!("         AGAINST negatives matched, by source — never summed:");
        for (source, matched, total) in &class.negatives_by_source {
            println!("                  {:<14} {matched}/{total}", source.label());
        }

        let validation = validate(
            class.predicate,
            &generator,
            // A fixed seed, so every claim in this report replays. Distinct from the negative
            // collection offset so validation never re-draws the pool.
            VALIDATION_SEED,
            args.validate,
            diverges,
        );
        println!("         PREDICTS {}", validation.describe());
        if validation.outcome == Outcome::Coincidence {
            println!(
                "                  ^ the rule described the findings, not the trigger. Discard."
            );
        }

        // **Every rule that tied is validated too.** The scoring cannot tell a coincidence from a
        // trigger when both fit identically, so the choice is deferred to the reader with the
        // prediction beside each one — measured, not argued.
        for alternate in &class.tied_with {
            let alt = validate(
                *alternate,
                &generator,
                VALIDATION_SEED,
                args.validate,
                diverges,
            );
            let marker =
                if alt.outcome == Outcome::Trigger && validation.outcome != Outcome::Trigger {
                    "  <- TIED ON FIT, AND THIS ONE PREDICTS"
                } else {
                    ""
                };
            println!("         TIED     {}{marker}", alternate.describe());
            println!("                  {}", alt.describe());
        }
    }

    // ── 5. The vocabulary gap ─────────────────────────────────────────────────────────
    //
    // Last, so it is what remains on screen.
    println!("\n  ── vocabulary gaps (N11.7) ──");
    if result.unexplained.is_empty() {
        println!("     Every finding was separated from the negatives by some rule.");
        println!(
            "     **Treat this with suspicion rather than satisfaction.** The phase file names \
             a neat\n     result as a warning sign: it is what fitting the vocabulary to \
             already-read findings\n     also looks like. The defence here is that the atoms come \
             from the pre-registered list."
        );
    } else {
        println!(
            "     {} finding(s) that NO conjunction of {} features could separate from the \
             negatives.\n     **This is the most useful output of the tool**: the vocabulary does \
             not contain the\n     property that distinguishes these cases.",
            result.unexplained.len(),
            FEATURES.len()
        );
        let mut by_signature: BTreeMap<&str, usize> = BTreeMap::new();
        for &i in &result.unexplained {
            *by_signature
                .entry(stored[i].signature.as_str())
                .or_default() += 1;
        }
        for (signature, n) in &by_signature {
            let example = result
                .unexplained
                .iter()
                .find(|&&i| stored[i].signature == *signature)
                .expect("present");
            println!(
                "\n       {signature}  ({n} finding(s))\n         features held: [{}]",
                features(&findings[*example]).names().join(", ")
            );
        }
    }

    std::panic::set_hook(previous);
    println!();
}
