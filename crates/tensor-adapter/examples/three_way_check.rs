//! Does a known two-backend divergence still report as one when wgpu is added?
//!
//! 0 findings in a three-way campaign, where the two-way campaign averaged one per ~500k
//! executions, is either bad luck or the third backend changing the verdict. This decides
//! which, by replaying findings that are *known* to diverge on flex-vs-tch.
use diff_fuzzer_core::report::{DivergenceReport, load_report};
use diff_fuzzer_core::{
    DifferentialOracle, NamedOutput, NormalizedRunner, Oracle, Runner, Verdict,
};
use std::path::Path;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, flex, libtorch, wgpu,
};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: three_way_check <findings-dir>");

    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let gpu = NormalizedRunner::new(wgpu(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);

    let mut cases: Vec<(String, TensorOp)> = Vec::new();
    collect(Path::new(&dir), &mut cases);
    println!("{} recorded findings\n", cases.len());

    for (name, case) in cases {
        let two: Vec<&dyn Runner<In = TensorOp, Canon = CanonicalTensor>> = vec![&cpu, &torch];
        let three: Vec<&dyn Runner<In = TensorOp, Canon = CanonicalTensor>> =
            vec![&cpu, &torch, &gpu];

        println!("{name}");
        println!("  two-way:   {}", verdict(&oracle, &case, &two));
        println!("  three-way: {}", verdict(&oracle, &case, &three));
    }
}

fn verdict(
    oracle: &DifferentialOracle<TensorOp, CanonicalTensor, TensorTolerancePolicy>,
    case: &TensorOp,
    runners: &[&dyn Runner<In = TensorOp, Canon = CanonicalTensor>],
) -> String {
    let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
        .iter()
        .filter_map(|r| {
            r.run_and_normalize(case).ok().map(|output| NamedOutput {
                implementation: r.name().to_string(),
                output,
            })
        })
        .collect();
    let ran: Vec<&str> = outputs.iter().map(|o| o.implementation.as_str()).collect();
    match oracle.check(case, &outputs) {
        Verdict::Diverged(_) => format!("DIVERGED   (ran: {})", ran.join(", ")),
        Verdict::Agree => format!("agree      (ran: {})", ran.join(", ")),
        Verdict::Skipped(reason) => format!("SKIPPED {reason:?} (ran: {})", ran.join(", ")),
    }
}

fn collect(dir: &Path, out: &mut Vec<(String, TensorOp)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "json")
            && let Ok(report) = load_report::<TensorOp>(&path)
        {
            let report: DivergenceReport<TensorOp> = report;
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            out.push((name, report.input));
        }
    }
}
