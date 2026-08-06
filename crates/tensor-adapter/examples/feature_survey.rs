//! What features do the real findings and negatives actually carry?
//!
//! A sanity check on the vocabulary before any search is built: a feature that fires on
//! everything, or on nothing, discriminates nothing and is dead weight in the search space.
use diff_fuzzer_core::{DivergenceReport, load_report};
use tensor_adapter::{FEATURES, TensorOp, extract, negatives};

/// Findings are `DivergenceReport`s, not `Negative`s — a different on-disk shape.
fn load_findings(root: &str) -> Vec<TensorOp> {
    let mut cases = Vec::new();
    let mut pending = vec![std::path::PathBuf::from(root)];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for path in entries.filter_map(Result::ok).map(|e| e.path()) {
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "json")
                && let Ok(r) = load_report::<TensorOp>(&path)
            {
                let r: DivergenceReport<TensorOp> = r;
                cases.push(r.input);
            }
        }
    }
    cases
}

fn main() {
    let findings = load_findings(&format!(
        "{}/runs/archive/pre-flex-swap",
        tensor_adapter::FINDINGS_ROOT
    ));
    let negs = negatives::load(tensor_adapter::NEGATIVES_ROOT);
    println!(
        "{} archived findings, {} negatives\n",
        findings.len(),
        negs.len()
    );

    println!("{:<28} {:>10} {:>10}", "feature", "findings", "negatives");
    for name in FEATURES {
        let f = findings.iter().filter(|c| extract(c).has(name)).count();
        let g = negs.iter().filter(|n| extract(&n.case).has(name)).count();
        let flag = if f == 0 && g == 0 {
            "  <- never fires"
        } else if f == findings.len() && g == negs.len() && !findings.is_empty() {
            "  <- always fires"
        } else {
            ""
        };
        println!("{name:<28} {f:>10} {g:>10}{flag}");
    }
}
