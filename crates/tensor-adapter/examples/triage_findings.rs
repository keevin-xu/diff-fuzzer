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
    CanonicalTensor, DisagreeingPair, Known, TensorNormalizer, TensorOp, known_issue, libtorch,
    ndarray, signature_across, wgpu,
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
}

fn main() {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "findings".to_string());

    let reports = load_all(&directory);
    if reports.is_empty() {
        println!("no reports in {directory}/");
        println!("  (a campaign that found nothing leaves none — that is a result, not a failure)");
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
        ordered
            .into_iter()
            .partition(|(signature, _)| known_issue(signature).is_none())
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
            "Not seen before. Work each up the ladder: does it reproduce → is it our\n\
             tool → is it float noise → is it legal → report it.\n\n",
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
fn load_all(directory: &str) -> Vec<(std::path::PathBuf, DivergenceReport<TensorOp>)> {
    let mut reports = Vec::new();
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
                    Err(error) => eprintln!("could not read {}: {error}", path.display()),
                }
            }
        }
    }

    reports
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
    vec![Box::new(ndarray()), Box::new(libtorch()), Box::new(wgpu())]
}

/// Does this report's case still diverge, under the tolerance it was judged against?
fn still_diverges(report: &DivergenceReport<TensorOp>) -> bool {
    let (cpu, torch) = (ndarray(), libtorch());
    let (Ok(left), Ok(right)) = (cpu.run(&report.input), torch.run(&report.input)) else {
        return false;
    };

    let left = TensorNormalizer.normalize(left);
    let right = TensorNormalizer.normalize(right);

    !matches!(
        left.approx_compare(&right, report.tolerance),
        Agreement::Agree(_)
    )
}

fn element_count(case: &TensorOp) -> usize {
    match case {
        TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => arg.len(),
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
