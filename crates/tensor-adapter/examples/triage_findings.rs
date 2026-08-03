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
use tensor_adapter::{TensorNormalizer, TensorOp, libtorch, ndarray, signature};

/// One representative of a problem, plus how often it was hit.
struct Group {
    occurrences: usize,
    /// The smallest case seen for this problem — the one worth reading.
    smallest: DivergenceReport<TensorOp>,
    smallest_size: usize,
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

    println!("triage: {} reports from {directory}/\n", reports.len());

    let mut seen = Seen::new();
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();

    for report in reports {
        // Re-derive the signature from the case rather than trusting one recorded in the
        // file. A report written by an older build may carry a signature computed by an
        // older rule, and grouping by two different rules at once would split a problem
        // in half without saying so.
        let fingerprint = fingerprint_of(&report);
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
                }
            })
            .or_insert_with(|| Group {
                occurrences: 1,
                smallest_size: size,
                smallest: report,
                reproduced: reproduced as usize,
                checked: 1,
            });
    }

    report(&groups, &seen);
}

fn load_all(directory: &str) -> Vec<DivergenceReport<TensorOp>> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };

    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .filter_map(|path| match load_report(&path) {
            Ok(report) => Some(report),
            Err(error) => {
                // Named rather than silently skipped: an unreadable report is a lost
                // finding, and losing one quietly is worse than failing loudly.
                eprintln!("could not read {}: {error}", path.display());
                None
            }
        })
        .collect()
}

/// Recompute a report's signature from its case.
fn fingerprint_of(report: &DivergenceReport<TensorOp>) -> String {
    let (cpu, torch) = (ndarray(), libtorch());
    let (Ok(left), Ok(right)) = (cpu.run(&report.input), torch.run(&report.input)) else {
        return format!("{}/could-not-run", report.label);
    };

    signature(
        &report.input,
        &TensorNormalizer.normalize(left),
        &TensorNormalizer.normalize(right),
        report.tolerance,
    )
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
        println!("   hit {} time(s)", group.occurrences);
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
