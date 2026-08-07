//! Propose falsifiable trigger claims for a directory of findings, and write the evidence.
//!
//! # What this does
//!
//! 1. Load findings (cases that diverged) and negatives (cases that did not).
//! 2. Search every rule of at most three input properties for ones that cover findings
//!    without ever firing on a negative.
//! 3. Test each surviving rule against **freshly generated cases it was never fitted to**.
//! 4. Write `CANDIDATES.md`.
//!
//! # What this deliberately does NOT do
//!
//! **It never writes `known.rs`.** An entry there carries a mechanism story and a status,
//! and both are human judgments. This tool produces candidates and the evidence for and
//! against them; ratifying one is a person's decision, made by reading the file.
//!
//! Every entry therefore **leads with the validation evidence, not with the rule.** The
//! failure mode this format exists to resist is a reviewer scrolling a list of confident-
//! looking rules and approving all of them. A number that says `12/247 matched cases
//! diverged` stops that; a heading that says `overflow_product ∧ cancellation` does not.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example propose_predicates -- <findings-dir>
//! ```

use diff_fuzzer_core::report::{DivergenceReport, load_report};
use diff_fuzzer_core::{
    DifferentialOracle, NamedOutput, NormalizedRunner, Oracle, Runner, Verdict,
};
use std::fmt::Write as _;
use std::path::Path;
use tensor_adapter::negatives::{self, Pool, SamplingContext};
use tensor_adapter::validation::{self, Outcome};
use tensor_adapter::{
    Bounds, CanonicalTensor, TensorNormalizer, TensorOp, TensorOpGenerator, TensorTolerancePolicy,
    flex, libtorch, search,
};

/// The seed every validation run uses.
///
/// Fixed and written into the report, because a rate nobody can reproduce is not evidence
/// (`CLAUDE.md` §3 — determinism is sacred).
const VALIDATION_SEED: u64 = 20_260_805;

/// Cases drawn per candidate. Most are rejected without ever touching a backend, so the
/// number of *runs* is far smaller than this.
const VALIDATION_BUDGET: usize = 4_000;

/// A rule that matches everything, used to measure the **baseline divergence rate**.
///
/// `Predicate::default()` is vacuous — `is_vacuous()` is true and the search rejects it for
/// exactly that reason. Here that is the point: it selects nothing, so the rate it produces
/// is the rate for cases drawn with no rule at all.
const EVERYTHING: tensor_adapter::Predicate = tensor_adapter::Predicate {
    required: 0,
    forbidden: 0,
};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| format!("{}/runs", tensor_adapter::FINDINGS_ROOT));

    let (findings, observed_on, produced_by) = load_findings(Path::new(&dir));
    println!("{} findings loaded from {dir}", findings.len());
    if findings.is_empty() {
        println!("nothing to explain — no report written");
        return;
    }

    // **The backend set is read from the findings, not assumed.**
    //
    // This hardcoded `[flex, libtorch]`, written when the harness ran two backends. Pointed
    // at a three-backend campaign it asked for a pool that could not exist, and the guard
    // refused — correctly, and for a reason that looked like a data problem rather than a
    // stale constant. Deriving it means the question can only be asked about the run that
    // actually happened.
    let backends: Vec<&str> = observed_on.iter().map(String::as_str).collect();
    println!("findings were observed on {backends:?}");
    println!("produced by: {produced_by}");
    // **Both halves of the context read from the findings.** The backend set was fixed at
    // PHASE-7E; the generator description followed at PHASE-7F for the same reason. A
    // campaign records how it was configured, and that record — not whatever this binary was
    // compiled against — is what its negatives must be matched on.
    let context = SamplingContext::new(&produced_by, &backends);
    let pool = match Pool::matched(negatives::load(tensor_adapter::NEGATIVES_ROOT), &context) {
        Ok(pool) => pool,
        Err(error) => {
            // Declining is the correct outcome, not a crash: scoring against a mismatched
            // or empty pool produces a confident number that means nothing.
            println!("cannot score: {error}");
            return;
        }
    };
    println!("{} negatives in the matched pool", pool.len());

    let result = search::search(&findings, &pool);
    println!(
        "{} predicates considered, {} classes, {} unexplained",
        result.considered,
        result.classes.len(),
        result.unexplained.len()
    );

    // Validation is the expensive part: it runs backends. Build them once.
    let differential = Differential::new();
    println!("\nvalidating {} candidates...", result.classes.len());

    // **The control, and the report is uninterpretable without it.** A candidate's rate
    // only means something relative to the rate for cases drawn with no rule at all. If the
    // backend pair diverges on 0% of unfiltered cases, then 0% for a matched sample is not
    // evidence against the rule — it is evidence that nothing diverges.
    //
    // **The diverging cases are kept, not just counted.** A count is a claim nobody can
    // check; the cases are the evidence, and they are findings in their own right.
    let mut discovered: Vec<TensorOp> = Vec::new();
    let baseline = validate_collecting(
        EVERYTHING,
        &differential,
        Bounds::default(),
        &mut discovered,
    );
    let baseline_wide =
        validate_collecting(EVERYTHING, &differential, wide_bounds(), &mut discovered);
    if !discovered.is_empty() {
        let path = Path::new(&dir).join("baseline-divergences.json");
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&discovered).expect("serialise"),
        )
        .expect("write baseline divergences");
        println!(
            "  saved {} baseline divergence(s) to {}",
            discovered.len(),
            path.display()
        );
    }
    println!(
        "  baseline (no rule): default {}/{}, wide {}/{}",
        baseline.diverged, baseline.matched, baseline_wide.diverged, baseline_wide.matched
    );

    let mut report = String::new();
    write_header(&mut report, &dir, &findings, &pool, &result);
    write_baseline(&mut report, &baseline, &baseline_wide);

    for (index, class) in result.classes.iter().enumerate() {
        // **Both generators.** Under default bounds three features never occur at all, so a
        // rule mentioning one comes back unreachable; the wide generator reaches two of
        // them. Reporting both keeps "not reachable here" separate from "not a trigger".
        let default = validate(class.predicate, &differential, Bounds::default());
        let wide = validate(class.predicate, &differential, wide_bounds());
        println!(
            "  {}/{}  {}",
            index + 1,
            result.classes.len(),
            default.describe()
        );
        write_candidate(
            &mut report,
            index + 1,
            class,
            &findings,
            &default,
            &wide,
            (&baseline, &baseline_wide),
        );
    }

    write_gaps(&mut report, &result, &findings);

    let path = Path::new(&dir).join("CANDIDATES.md");
    std::fs::write(&path, report).expect("write CANDIDATES.md");
    println!("\nwrote {}", path.display());
}

/// The wide generator used for validation.
///
/// **`restrict_domains: false` is not optional here.** Without it the generator cannot
/// produce a non-finite input, so any candidate mentioning `NaN` comes back "not reachable"
/// — which reads as a statement about the rule and is really a statement about this
/// function. That happened on the first `max`-versus-`NaN` finding, and the same omission
/// was found in `examples/reach.rs` the same day.
///
/// The rule of thumb it cost: **a validation generator must be configured like the campaign
/// it is validating**, not like a tidy default.
fn wide_bounds() -> Bounds {
    Bounds {
        max_rank: 3,
        max_dim: 64,
        magnitude: 1000.0,
        special_value_rate: 0.125,
        restrict_domains: false,
        ..Bounds::default()
    }
}

/// Like [`validate`], but keeps every case that diverged rather than only counting them.
fn validate_collecting(
    predicate: tensor_adapter::Predicate,
    differential: &Differential,
    bounds: Bounds,
    found: &mut Vec<TensorOp>,
) -> validation::Validation {
    validation::validate(
        predicate,
        &TensorOpGenerator::new(bounds),
        VALIDATION_SEED,
        VALIDATION_BUDGET,
        |case| {
            let diverged = differential.diverges(case);
            if diverged {
                found.push(case.clone());
            }
            diverged
        },
    )
}

fn validate(
    predicate: tensor_adapter::Predicate,
    differential: &Differential,
    bounds: Bounds,
) -> validation::Validation {
    validation::validate(
        predicate,
        &TensorOpGenerator::new(bounds),
        VALIDATION_SEED,
        VALIDATION_BUDGET,
        |case| differential.diverges(case),
    )
}

// --- the differential ------------------------------------------------------------------

/// The same comparison the fuzz target makes: two backends, one oracle, one policy.
struct Differential {
    cpu: NormalizedRunner<tensor_adapter::FlexBackend, TensorNormalizer>,
    torch: NormalizedRunner<tensor_adapter::LibTorchBackend, TensorNormalizer>,
    oracle: DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
}

impl Differential {
    fn new() -> Self {
        Self {
            cpu: NormalizedRunner::new(flex(), TensorNormalizer),
            torch: NormalizedRunner::new(libtorch(), TensorNormalizer),
            oracle: DifferentialOracle::new(TensorTolerancePolicy),
        }
    }

    /// Whether a case diverges.
    ///
    /// `Skipped` counts as **not** diverging: a case the policy declines to judge is not
    /// evidence for a trigger claim. Counting it either way would be wrong, and counting it
    /// as divergence would inflate every rate in the report.
    fn diverges(&self, case: &TensorOp) -> bool {
        let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] =
            [&self.cpu, &self.torch];
        let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
            .iter()
            .filter_map(|runner| {
                runner
                    .run_and_normalize(case)
                    .ok()
                    .map(|output| NamedOutput {
                        implementation: runner.name().to_string(),
                        output,
                    })
            })
            .collect();

        matches!(self.oracle.check(case, &outputs), Verdict::Diverged(_))
    }
}

// --- loading ---------------------------------------------------------------------------

/// The findings, and **the implementations they were observed on**.
///
/// The backend set comes from the reports themselves — every one records which
/// implementations ran — so a stale constant cannot silently ask about the wrong campaign.
fn load_findings(dir: &Path) -> (Vec<TensorOp>, Vec<String>, String) {
    let mut out = Vec::new();
    let mut observed: Vec<String> = Vec::new();
    let mut generators: Vec<String> = Vec::new();
    let mut unreadable = 0usize;
    collect(
        dir,
        &mut out,
        &mut observed,
        &mut generators,
        &mut unreadable,
    );
    generators.sort();
    generators.dedup();
    if generators.len() > 1 {
        eprintln!(
            "⚠ these findings came from {} different generator configurations:\n  {}\n               Scoring them together would compare runs that explored different spaces.",
            generators.len(),
            generators.join("\n  ")
        );
    }
    let produced_by = generators.first().cloned().unwrap_or_default();
    observed.sort();
    observed.dedup();

    // **A finding that cannot be parsed is not a finding that does not exist.** Reporting
    // "nothing to explain" for reports that failed to load is the same data-loss-as-success
    // failure that `triage_findings` had; both are fixed, and both are loud.
    if unreadable > 0 {
        eprintln!(
            "⚠ {unreadable} report(s) could not be read and are excluded. Every number below \
             is computed from an incomplete set."
        );
    }
    (out, observed, produced_by)
}

fn collect(
    dir: &Path,
    out: &mut Vec<TensorOp>,
    observed: &mut Vec<String>,
    generators: &mut Vec<String>,
    unreadable: &mut usize,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out, observed, generators, unreadable);
        } else if path.extension().is_some_and(|e| e == "json") {
            match load_report::<TensorOp>(&path) {
                Ok(report) => {
                    let report: DivergenceReport<TensorOp> = report;
                    observed.extend(report.outputs.iter().map(|(name, _)| name.clone()));
                    generators.push(report.generator.clone());
                    out.push(report.input);
                }
                Err(error) => {
                    eprintln!("could not read {}: {error}", path.display());
                    *unreadable += 1;
                }
            }
        }
    }
}

// --- the report ------------------------------------------------------------------------

fn write_header(
    out: &mut String,
    dir: &str,
    findings: &[TensorOp],
    pool: &Pool,
    result: &search::SearchResult,
) {
    let _ = writeln!(out, "# Candidate trigger predicates\n");
    let _ = writeln!(
        out,
        "Generated by `examples/propose_predicates.rs` from `{dir}`. \
         **Nothing here is ratified.** Each entry is a falsifiable claim plus the evidence \
         for and against it; promoting one to `known.rs` is a human judgment and this tool \
         will never do it.\n"
    );
    let _ = writeln!(
        out,
        "| | |\n|---|---|\n\
         | findings | {} |\n\
         | negatives (matched pool) | {} |\n\
         | predicates considered | {} |\n\
         | classes proposed | {} |\n\
         | findings left unexplained | {} |\n",
        findings.len(),
        pool.len(),
        result.considered,
        result.classes.len(),
        result.unexplained.len()
    );
    let _ = writeln!(
        out,
        "Validation samples {VALIDATION_BUDGET} cases per candidate from seed \
         `{VALIDATION_SEED}`, keeps the ones the rule matches, and runs those. Every rate \
         below is reproducible from that seed.\n"
    );
}

/// The baseline, written directly under the header because every rate below depends on it.
fn write_baseline(
    out: &mut String,
    default: &validation::Validation,
    wide: &validation::Validation,
) {
    let _ = writeln!(out, "## Baseline — cases drawn with no rule at all\n");
    let _ = writeln!(
        out,
        "| generator | sampled | diverged | rate |\n|---|---|---|---|"
    );
    for (name, v) in [("default", default), ("wide", wide)] {
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {} |",
            v.matched,
            v.diverged,
            v.rate()
                .map(|r| format!("{:.2}%", r * 100.0))
                .unwrap_or_else(|| "—".to_string())
        );
    }
    let baseline_is_zero = default.diverged == 0 && wide.diverged == 0;
    let _ = writeln!(out);
    if baseline_is_zero {
        let _ = writeln!(
            out,
            "> ⚠️ **The backend pair did not diverge on a single unfiltered case.** Every \
             candidate rate below is therefore uninterpretable as a judgment of the rule: a \
             0% rate is exactly what a correct rule would also score against a pair that \
             never disagrees. **No candidate below has been fairly tested**, and none should \
             be discarded on this evidence.\n"
        );
    } else {
        let _ = writeln!(
            out,
            "A candidate is only informative if its rate is well above this baseline.\n"
        );
    }
}

fn write_candidate(
    out: &mut String,
    number: usize,
    class: &search::Candidate,
    findings: &[TensorOp],
    default: &validation::Validation,
    wide: &validation::Validation,
    baseline: (&validation::Validation, &validation::Validation),
) {
    // **Evidence first, rule second.** The heading is the verdict a reviewer must weigh,
    // not the rule they might wave through.
    let _ = writeln!(
        out,
        "---\n\n## Candidate {number} — {}\n",
        verdict(default, wide)
    );
    let _ = writeln!(out, "**Prediction on unseen cases**\n");
    let _ = writeln!(
        out,
        "| generator | matched | diverged | rate | baseline rate | outcome |\n\
         |---|---|---|---|---|---|"
    );
    for (name, v, base) in [("default", default, baseline.0), ("wide", wide, baseline.1)] {
        // The baseline column is what makes the rate mean anything: a rule is only
        // informative if it lifts the rate above what drawing at random already gives.
        let _ = writeln!(
            out,
            "| {name} | {} | {} | {} | {} | {:?} |",
            v.matched,
            v.diverged,
            v.rate()
                .map(|r| format!("{:.1}%", r * 100.0))
                .unwrap_or_else(|| "—".to_string()),
            base.rate()
                .map(|r| format!("{:.2}%", r * 100.0))
                .unwrap_or_else(|| "—".to_string()),
            v.outcome
        );
    }
    // **Lift is a different question from the verdict, and both are reported.** The
    // verdict answers "does this rule predict divergence at least half the time". Lift
    // answers "does it concentrate divergence at all". A rule can fail the first and still
    // be the most useful thing in the file — and the threshold is NOT retuned to make that
    // come out tidily, because tuning it after seeing these numbers is the fitting-to-data
    // error the whole module exists to catch.
    for (name, v, base) in [("default", default, baseline.0), ("wide", wide, baseline.1)] {
        if let (Some(rate), Some(base_rate)) = (v.rate(), base.rate())
            && base_rate > 0.0
            && rate > base_rate * 2.0
            && v.outcome != Outcome::Trigger
        {
            let _ = writeln!(
                out,
                "\n> **Concentrates divergence {:.1}x above baseline on the {name} \
                 generator** ({} of {} matched cases, against {:.2}% for cases drawn with \
                 no rule). It does not meet the {:.0}% trigger threshold and is therefore \
                 not ratified — but a rule that multiplies the divergence rate is evidence \
                 about *something*, and discarding it silently would throw that away.\n",
                rate / base_rate,
                v.diverged,
                v.matched,
                base_rate * 100.0,
                validation::TRIGGER_RATE * 100.0
            );
        }
    }

    let _ = writeln!(out, "\n**The rule:** `{}`\n", class.predicate.describe());
    let _ = writeln!(
        out,
        "Covers **{} of {} findings**; {} are not covered by this rule.\n",
        class.covered.len(),
        findings.len(),
        findings.len() - class.covered.len()
    );
    let _ = writeln!(
        out,
        "**Negatives it fires on** — by source, never summed:\n"
    );
    let _ = writeln!(out, "| source | matched | total |\n|---|---|---|");
    for (source, matched, total) in &class.negatives_by_source {
        let _ = writeln!(out, "| {} | {matched} | {total} |", source.label());
    }
    let _ = writeln!(out);
    if class
        .negatives_by_source
        .iter()
        .all(|(source, _, _)| *source != negatives::Source::NearMiss)
    {
        let _ = writeln!(
            out,
            "> ⚠️ **No `NearMiss` negatives in the pool.** Near-misses are cases one edit \
             away from diverging, and they are the only negatives strong enough to make \
             surviving one mean much. This rule survived easier cases only.\n"
        );
    }
}

/// The one-line verdict a reviewer reads first.
fn verdict(default: &validation::Validation, wide: &validation::Validation) -> String {
    // The stronger of the two generators decides, because `NeverSampled` under one and
    // `Trigger` under the other means the rule was reachable and held.
    // Rank by outcome, then by how many cases actually matched: between two rows with the
    // same verdict, the one that tested more cases is the stronger evidence, and quoting
    // the weaker one understates what is known.
    let best = [default, wide]
        .into_iter()
        .min_by_key(|v| {
            let rank = match v.outcome {
                Outcome::Trigger => 0,
                Outcome::Coincidence => 1,
                Outcome::Inconclusive => 2,
                Outcome::NeverSampled => 3,
            };
            (rank, usize::MAX - v.matched)
        })
        .expect("two elements");
    match best.outcome {
        Outcome::Trigger => format!(
            "PREDICTS DIVERGENCE ({}/{} unseen cases)",
            best.diverged, best.matched
        ),
        Outcome::Coincidence => format!(
            "FAILED PREDICTION ({}/{} unseen cases) — discard",
            best.diverged, best.matched
        ),
        Outcome::Inconclusive => format!(
            "UNPROVEN — only {} unseen cases matched, too few to judge",
            best.matched
        ),
        Outcome::NeverSampled => {
            "UNTESTABLE — no generator reaches this rule, so it is untested, not wrong".into()
        }
    }
}

/// The vocabulary gap. **This section is the most useful output of the whole tool**, so it
/// is written even when it is empty — an absent section reads as an absent problem.
fn write_gaps(out: &mut String, result: &search::SearchResult, findings: &[TensorOp]) {
    let _ = writeln!(out, "---\n\n## Vocabulary gaps\n");
    if result.unexplained.is_empty() {
        let _ = writeln!(
            out,
            "Every finding is covered by a candidate above.\n\n\
             > ⚠️ **Treat this as a warning, not a success.** A run in which everything is \
             neatly explained usually means the negatives were too easy or the feature \
             vocabulary was fitted to these findings.\n"
        );
        return;
    }
    let _ = writeln!(
        out,
        "**{} of {} findings are unexplained.** No rule over the current 17 features \
         separates them from cases that did not diverge — so the vocabulary is missing the \
         property that distinguishes them. This is a statement about the features, not a \
         failure of the search.\n",
        result.unexplained.len(),
        findings.len()
    );
    let _ = writeln!(
        out,
        "Adding a feature is a deliberate act: it widens every future search and it can be \
         fitted to exactly these findings. The gap is recorded here so that choice is made \
         knowingly rather than by accident.\n"
    );
}
