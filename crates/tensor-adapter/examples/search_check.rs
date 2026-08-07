//! A mechanism check for the predicate search, run over whatever findings are on disk.
//!
//! NOT a result about burn. The findings available today were produced before the flex
//! backend replaced ndarray, so the pair they describe no longer exists. What this shows is
//! whether the *search* behaves on real data rather than on hand-built test cases.
use diff_fuzzer_core::report::{DivergenceReport, load_report};
use std::path::Path;
use tensor_adapter::input::TensorOp;
use tensor_adapter::negatives::{self, Pool, SamplingContext};
use tensor_adapter::search;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: search_check <dir>");
    let mut findings: Vec<TensorOp> = Vec::new();
    let mut observed: Vec<String> = Vec::new();
    collect(Path::new(&dir), &mut findings, &mut observed);
    observed.sort();
    observed.dedup();
    println!("{} findings loaded from {dir}", findings.len());

    let all = negatives::load(tensor_adapter::NEGATIVES_ROOT);
    // **Read from the findings, not assumed** — the same fix `propose_predicates` needed.
    // A hardcoded pair asks for a pool that a three-backend campaign never produced, and the
    // refusal reads as missing data rather than as a stale constant.
    let backends: Vec<&str> = observed.iter().map(String::as_str).collect();
    println!("findings were observed on {backends:?}");
    let context = SamplingContext::new(negatives::FUZZER_GENERATOR, &backends);
    let pool = match Pool::matched(all, &context) {
        Ok(p) => p,
        Err(e) => {
            println!("no usable negatives: {e}");
            return;
        }
    };
    println!("{} negatives in the matched pool\n", pool.len());

    let result = search::search(&findings, &pool);
    println!("{} predicates considered\n", result.considered);

    for (i, class) in result.classes.iter().enumerate() {
        println!("class {}: {}", i + 1, class.predicate.describe());
        println!("  covers {} findings", class.covered.len());
        for (source, matched, total) in &class.negatives_by_source {
            println!("  {:<12} {matched} of {total} matched", source.label());
        }
    }

    println!(
        "\nunexplained: {} findings — no rule separates them from the negatives",
        result.unexplained.len()
    );
}

fn collect(dir: &Path, out: &mut Vec<TensorOp>, observed: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out, observed);
        } else if path.extension().is_some_and(|e| e == "json")
            && let Ok(report) = load_report::<TensorOp>(&path)
        {
            let report: DivergenceReport<TensorOp> = report;
            observed.extend(report.outputs.iter().map(|(name, _)| name.clone()));
            out.push(report.input);
        }
    }
}
