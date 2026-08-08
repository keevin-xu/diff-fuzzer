//! Group a directory of saved reports by what problem they represent.
//!
//! The existing `triage` example reads a campaign's JSONL log. This one reads the
//! **directory of `DivergenceReport` files** that a fuzzing run leaves behind — which is a
//! different shape of problem, because the fuzz target writes one file per crashing input
//! and cannot de-duplicate as it goes.
//!
//! Why it cannot: with `-fork=1`, each crash happens in a fresh child process, so nothing
//! held in memory survives to the next one. De-duplication for fuzzing therefore has to
//! happen **after the fact**, here. A long campaign that trips repeatedly over one
//! problem otherwise leaves a folder of thousands of files that all say the same thing,
//! and reading it by hand is exactly the work this project exists to avoid.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example triage_findings           # findings/
//! cargo run --release -p tensor-adapter --example triage_findings some/dir
//! ```

use diff_fuzzer_core::{
    Agreement, ApproxEq, DivergenceReport, Implementation, Normalizer, Seen, load_report,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    CanonicalTensor, DisagreeingPair, Known, Relation, TensorNormalizer, TensorOp, extract, flex,
    known_by_predicate, known_issue, libtorch, signature_across, wgpu,
};

/// One representative of a problem, plus how often it was hit.
struct Group {
    /// Which two implementations the signature was computed from.
    ///
    /// Kept **beside** the signature rather than inside it: a key containing backend names
    /// would change when a backend is added, orphaning `known.rs`. See
    /// `signature_across`'s documentation.
    disagreeing: Option<DisagreeingPair>,
    occurrences: usize,
    /// The smallest case seen for this problem — the one worth reading.
    smallest: DivergenceReport<TensorOp>,
    smallest_size: usize,
    /// Where that smallest case lives, so the index can point at a file to open
    /// rather than leaving the reader to grep a few hundred of them.
    smallest_path: std::path::PathBuf,
    reproduced: usize,
    checked: usize,
    /// Which recorded class, if any, **explains the trigger** of the smallest case.
    ///
    /// Distinct from matching a signature. A signature match says "this looked like
    /// something we have seen"; a predicate match says "something we understand explains
    /// why an input like this fails". The two can disagree, and where they do is the most
    /// informative thing this tool produces — see [`Relation`].
    explained_by: Option<&'static Known>,
}

fn main() {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "findings".to_string());

    let (reports, unreadable) = load_all(&directory);

    // **An unreadable report is a lost finding, and must never be reported as an absent
    // one.** This printed "a campaign that found nothing" for three findings it had failed to
    // parse — the most reassuring possible message for a data-loss bug. Saying so first, and
    // loudly, is the fix that matters; the parse bug behind it was only the occasion.
    if unreadable > 0 {
        eprintln!(
            "\n⚠ {unreadable} report(s) in {directory}/ could not be read — see the errors \
             above.\n  These are findings that exist and are being ignored. Do not read what \
             follows as a complete picture."
        );
    }

    if reports.is_empty() {
        if unreadable > 0 {
            println!(
                "\nno readable reports in {directory}/ — but {unreadable} exist and failed to parse"
            );
            println!("  THIS IS NOT A CLEAN RESULT. Fix the reader before drawing any conclusion.");
        } else {
            println!("no reports in {directory}/");
            println!(
                "  (a campaign that found nothing leaves none — that is a result, not a failure)"
            );
        }
        return;
    }

    let total = reports.len();
    println!("triage: {total} reports from {directory}/\n");

    let mut seen = Seen::new();
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();

    for (path, report) in reports {
        // Re-derive the signature from the case rather than trusting one recorded in the
        // file. A report written by an older build may carry a signature computed by an
        // older rule, and grouping by two different rules at once would split a problem
        // in half without saying so.
        let (fingerprint, disagreeing) = fingerprint_of(&report);
        seen.is_new(&fingerprint);

        // Rung 1 of the ladder, on every report: does it still diverge? A finding that
        // does not reproduce is a defect in *this tool*, and nothing has been learned
        // about the target.
        let reproduced = still_diverges(&report);
        let size = element_count(&report.input);
        // Computed from the **case**, so it is available for an input nobody has run.
        let explained_by = known_by_predicate(extract(&report.input));

        groups
            .entry(fingerprint)
            .and_modify(|group| {
                group.occurrences += 1;
                group.checked += 1;
                group.reproduced += reproduced as usize;
                if size < group.smallest_size {
                    group.smallest_size = size;
                    group.smallest = report.clone();
                    group.smallest_path = path.clone();
                    group.explained_by = explained_by;
                }
            })
            .or_insert_with(|| Group {
                disagreeing,
                occurrences: 1,
                smallest_size: size,
                smallest: report,
                smallest_path: path,
                reproduced: reproduced as usize,
                checked: 1,
                explained_by,
            });
    }

    report(&groups, &seen);

    // The printed output scrolls away; the file is what gets read later, quoted in a
    // write-up, and diffed against the next campaign's.
    let index = format!("{directory}/TRIAGE.md");
    match write_index(&index, &directory, &groups, total) {
        Ok(()) => println!("\nwritten: {index}"),
        Err(error) => eprintln!("\ncould not write {index}: {error}"),
    }
}

/// Write the triage index: what was found, what is new, and what to do next.
///
/// **New signatures come first.** A campaign's output is mostly things already known — the
/// `matmul` overflow is reachable within a few hundred thousand executions, so every long
/// run rediscovers it. Listed with equal prominence, a genuinely new problem is what gets
/// missed. The scarce resource in triage is attention, not disk.
///
/// Known findings are still counted, still checked for reproduction, still listed. Nothing
/// is filtered: suppressing them would mean a *change* in a known problem's behaviour went
/// unnoticed, which is exactly the sort of thing worth noticing.
fn write_index(
    path: &str,
    directory: &str,
    groups: &BTreeMap<String, Group>,
    total: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Partition before sorting, so the ordering rule is stated once and visibly.
    let (fresh, familiar): (Vec<_>, Vec<_>) = {
        let mut ordered: Vec<(&String, &Group)> = groups.iter().collect();
        ordered.sort_by_key(|(_, group)| std::cmp::Reverse(group.occurrences));
        // **Partitioned by whether a human is needed, not by whether the signature is
        // recognised.** A known signature whose trigger nothing explains belongs at the
        // top — that is the merge signal, and sorting it below with the familiar problems
        // is exactly how a class holding two bugs stays invisible.
        ordered.into_iter().partition(|(signature, group)| {
            Relation::of(signature, group.explained_by).needs_attention()
        })
    };

    out.push_str("# Triage\n\n");
    out.push_str(
        "Regenerated by `cargo run --release -p tensor-adapter --example triage_findings`.\n\
         **Do not edit** — rerun it instead. Signatures are recomputed from each case under\n\
         the *current* policy, so this file is only ever as current as its last run.\n\n",
    );
    out.push_str(&format!(
        "`{directory}` — **{total} reports**, **{} distinct {}** \
         ({} new, {} already investigated).\n\n",
        groups.len(),
        if groups.len() == 1 {
            "problem"
        } else {
            "problems"
        },
        fresh.len(),
        familiar.len()
    ));

    let unreproduced: usize = groups.values().map(|g| g.checked - g.reproduced).sum();
    if unreproduced > 0 {
        out.push_str(&format!(
            "> ⚠ **{unreproduced} report(s) no longer diverge.** Start there. A finding that\n\
             > cannot be replayed is evidence about *this tool*, not about the target.\n\n",
        ));
    }

    out.push_str("## Needs triage\n\n");
    if fresh.is_empty() {
        out.push_str(
            "None — every signature here has been investigated before. That is a real\n\
             result, not a failed campaign.\n\n",
        );
    } else {
        out.push_str(
            "Each of these needs a person. Work it up the ladder: does it reproduce → is it\n\
             our tool → is it float noise → is it legal → report it.\n\n\
             **Three different reasons appear here**, and they are not equally urgent. A\n\
             *novel* group is simply new. A group whose **trigger is known but signature is\n\
             not** is likely one bug wearing two symptoms. A group whose **signature is known\n\
             but trigger is not** is the dangerous one: the class may hold a second problem\n\
             that symptom grouping cannot see.\n\n",
        );
        for (signature, group) in &fresh {
            section(&mut out, signature, group, None);
        }
    }

    out.push_str("## Already investigated\n\n");
    if familiar.is_empty() {
        out.push_str("None.\n\n");
    } else {
        out.push_str(
            "Listed, not hidden — a change in a known problem's behaviour is worth\n\
             noticing. **`reported` is not `settled`:** filing an issue is not the same as\n\
             having an answer.\n\n",
        );
        for (signature, group) in &familiar {
            section(&mut out, signature, group, known_issue(signature));
        }
    }

    std::fs::write(path, out)
}

/// One signature's entry.
fn section(out: &mut String, signature: &str, group: &Group, known: Option<&'static Known>) {
    out.push_str(&format!("### `{signature}`\n\n"));

    let relation = Relation::of(signature, group.explained_by);
    out.push_str(&format!("**{}**\n\n", relation.label()));
    if let Some(explained) = group.explained_by {
        out.push_str(&format!(
            "Trigger explained by [{}]({}).\n\n",
            explained.signature, explained.reference
        ));
    }

    if let Some(known) = known {
        out.push_str(&format!(
            "**{}** — [{}]({})\n\n{}\n\n",
            known.status.label(),
            known.reference.trim_start_matches("https://"),
            known.reference,
            known.note
        ));
        if !known.status.is_settled() {
            out.push_str("*The question is still open; this is not a closed case.*\n\n");
        }
    }

    if let Some(pair) = &group.disagreeing {
        out.push_str(&format!(
            "- disagreeing pair: **{}** vs **{}**\n",
            pair.left, pair.right
        ));
    }
    out.push_str(&format!(
        "- **{}** case(s) matched this signature — *matching is not evidence of a shared cause*\n\
         - reproduces: **{} of {}** checked{}\n\
         - smallest case: {} values, `{}`\n\n",
        group.occurrences,
        group.reproduced,
        group.checked,
        if group.reproduced < group.checked {
            "  ⚠ **investigate this tool first**"
        } else {
            ""
        },
        group.smallest_size,
        group.smallest_path.display()
    ));

    out.push_str("```\n");
    out.push_str(&truncate(&format!("{:?}", group.smallest.input), 400));
    out.push_str("\n\n");
    out.push_str(&truncate(&group.smallest.summary, 400));
    out.push_str("\n```\n\n");
}

/// Collect every report at or below `directory`.
///
/// **Recursive, because the layout is nested**: the fuzz target files findings under
/// `runs/<run>/<operation>/`, so a campaign's output is several directories deep. Pointing
/// this at `findings/` reads everything; pointing it at one run reads that run alone.
///
/// The recursion is written with an explicit stack rather than a recursive function — a
/// findings tree is shallow, but a directory loop through a symlink is not something a
/// triage tool should die on.
/// Every readable report, **and how many were not**.
///
/// The count is returned rather than only logged, because a caller that receives just a list
/// cannot tell an empty campaign from a broken reader — and will say the reassuring thing.
fn load_all(directory: &str) -> (Vec<(std::path::PathBuf, DivergenceReport<TensorOp>)>, usize) {
    let mut reports = Vec::new();
    let mut unreadable = 0usize;
    let mut pending = vec![std::path::PathBuf::from(directory)];

    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };

        for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "json") {
                match load_report(&path) {
                    Ok(report) => reports.push((path, report)),
                    // Named rather than silently skipped: an unreadable report is a lost
                    // finding, and losing one quietly is worse than failing loudly.
                    // **Counted as well as named** — printing an error and then returning a
                    // list the caller reads as "nothing found" is how the loud failure became
                    // a quiet one anyway.
                    Err(error) => {
                        eprintln!("could not read {}: {error}", path.display());
                        unreadable += 1;
                    }
                }
            }
        }
    }

    (reports, unreadable)
}

/// Recompute a report's signature from its case, across every implementation.
///
/// Re-derived rather than read from the file: a report written by an older build carries a
/// label computed by an older rule, and grouping under two rules at once splits a problem
/// in half without saying so.
///
/// **Every backend is run, not just the two CPUs.** Hardcoding a pair here had the same
/// defect the campaign runner did — a finding whose only dissenter was the GPU would be
/// re-labelled from two backends that agree.
fn fingerprint_of(report: &DivergenceReport<TensorOp>) -> (String, Option<DisagreeingPair>) {
    let mut outputs: Vec<(String, CanonicalTensor)> = Vec::new();
    for runner in backends() {
        if let Ok(raw) = runner.run(&report.input) {
            outputs.push((runner.name().to_string(), TensorNormalizer.normalize(raw)));
        }
    }
    if outputs.len() < 2 {
        return (format!("{}/could-not-run", report.label), None);
    }

    signature_across(&report.input, &outputs, report.tolerance)
}

/// Every backend a finding is re-checked against.
///
/// The GPU is included so a divergence that only it exhibits is labelled from the pair that
/// actually disagreed. Cheap: triage runs once over a directory, not per fuzzing execution.
fn backends() -> Vec<Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>> {
    vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())]
}

/// Does this report's case still diverge, under the tolerance it was judged against?
///
/// **Compares every pair, not one hardcoded pair.**
///
/// This checked `flex` against `libtorch` only, which was written when the harness ran two
/// backends. A finding where the *GPU* disagrees with two CPU backends that agree with each
/// other then replays as "does not reproduce" — and triage says so in its most alarming
/// language: *"a finding that cannot be replayed is evidence about this tool, not the
/// target."* Which was true, but about the replay rather than the finding.
///
/// That is the worst possible failure for this function: it does not lose a finding quietly,
/// it actively argues the finding is spurious. Caught at PHASE-7E on the first `max`-versus-
/// `NaN` result, which is the fifth place two-implementation blindness has been found in this
/// project — assume there is a sixth.
fn still_diverges(report: &DivergenceReport<TensorOp>) -> bool {
    let outputs: Vec<(String, CanonicalTensor)> = backends()
        .iter()
        .filter_map(|backend| {
            backend
                .run(&report.input)
                .ok()
                .map(|out| (backend.name().to_string(), TensorNormalizer.normalize(out)))
        })
        .collect();

    if outputs.len() < 2 {
        return false;
    }

    // Any pair disagreeing means the case still diverges. The recorded tolerance is used
    // rather than the current one, so tightening a bound later cannot rewrite what an old
    // finding meant.
    for (i, (_, left)) in outputs.iter().enumerate() {
        for (_, right) in outputs.iter().skip(i + 1) {
            if !matches!(
                left.approx_compare(right, report.tolerance),
                Agreement::Agree(_)
            ) {
                return true;
            }
        }
    }
    false
}

fn element_count(case: &TensorOp) -> usize {
    match case {
        TensorOp::Unary { arg, .. }
        | TensorOp::Reduce { arg, .. }
        | TensorOp::Activation { arg, .. }
        | TensorOp::Scan { arg, .. } => arg.len(),
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => lhs.len() + rhs.len(),
    }
}

fn report(groups: &BTreeMap<String, Group>, seen: &Seen) {
    println!(
        "{} distinct problems from {} reports\n",
        seen.distinct(),
        seen.total()
    );

    // Most frequently hit first: how reachable a problem is is a reasonable proxy for how
    // much it matters, and it is the ordering a reader wants.
    let mut ordered: Vec<(&String, &Group)> = groups.iter().collect();
    ordered.sort_by_key(|(_, group)| std::cmp::Reverse(group.occurrences));

    for (fingerprint, group) in &ordered {
        println!("── {fingerprint}");
        // "matched this signature" rather than "hit N times". The map counted signature
        // matches; it did not establish that N cases share a mechanism. A count that reads
        // as N confirmations of one bug overstates what was computed.
        println!("   {} case(s) matched this signature", group.occurrences);
        println!(
            "   {}",
            Relation::of(fingerprint, group.explained_by).label()
        );
        if let Some(pair) = &group.disagreeing {
            println!("   disagreeing pair: {} vs {}", pair.left, pair.right);
        }
        println!(
            "   reproduces: {} of {} checked{}",
            group.reproduced,
            group.checked,
            if group.reproduced < group.checked {
                "   ⚠ SOME DO NOT REPRODUCE — investigate this tool first"
            } else {
                ""
            }
        );
        println!("   smallest case ({} values):", group.smallest_size);
        println!(
            "     {:?}",
            truncate(&format!("{:?}", group.smallest.input), 160)
        );
        println!("     {}", truncate(&group.smallest.summary, 160));
        println!();
    }

    let unreproduced: usize = ordered.iter().map(|(_, g)| g.checked - g.reproduced).sum();

    println!("next, per the triage ladder:");
    if unreproduced > 0 {
        println!("  1. ⚠ {unreproduced} report(s) no longer diverge. Start there — a finding");
        println!("     that cannot be replayed is evidence about this tool, not the target.");
    } else {
        println!("  1. ✓ every report still reproduces");
    }
    println!("  2. is each within what floating-point arithmetic predicts? (`triage` example)");
    println!("  3. is each a difference the specification permits?");
    println!("  4. what remains is worth reporting — one issue per distinct problem, not per case");
}

fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{kept}…")
}
