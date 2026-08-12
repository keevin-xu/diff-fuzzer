//! **N3.7–N3.9** — what the corpus actually contains, what the budget costs, and how fast it runs.
//!
//! # Three measurements, and why each is not the obvious one
//!
//! **Corpus shape (N3.7).** Not "how many cases" but *what is in them*. A construct at 0% is
//! untested no matter what the verdicts say, and `05-MEASUREMENT-AND-CAMPAIGNS.md` insists on
//! tracking **interactions**, not just single constructs — one axis once read healthy on both of
//! its constituent constructs while their *combination* was 0%.
//!
//! **The element budget, measured on both sides (N3.8).** `02-METHODOLOGY.md` records a cap
//! justified as "the old worst case, so it costs nothing" that took a divergence rate from
//! 9-in-2,000 to **0-in-2,000** — it had clamped away exactly the shapes that diverge, while
//! costing the full runtime anyway. So the budget is measured *against a larger one*, not
//! asserted to be free.
//!
//! **Throughput with the tail (N3.9).** A sample small enough to be convenient is systematically
//! optimistic, because it is fast precisely for the reason that it rarely contains the expensive
//! cases. A convenient SQL sample read 84 cases/sec against a true 18. So this reports the
//! **distribution** and the build-versus-run split, never a bare mean.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n3_corpus --features candle

use std::collections::BTreeMap;
use std::time::Instant;

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};

use onnx_adapter::case::{ElemType, OnnxCase};
use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::model::build_bytes;
use onnx_adapter::ops;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

/// Seeds spread across the whole `u64` range, not `0..n`.
fn wide_seeds(count: u64) -> impl Iterator<Item = u64> {
    (0..count).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn corpus(bounds: &Bounds, count: u64) -> Vec<OnnxCase> {
    let generator = OnnxGenerator::new(bounds.clone());
    wide_seeds(count)
        .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
        .collect()
}

fn main() {
    let cases: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20_000);

    let bounds = Bounds::default();
    println!("{}", bounds.description());
    println!("{cases} cases from widely separated seeds\n");

    let sample = corpus(&bounds, cases);

    // ── N3.7: what is in the corpus ───────────────────────────────────────────────
    println!("── operator distribution ──");
    let mut per_op: BTreeMap<&str, usize> = BTreeMap::new();
    for case in &sample {
        *per_op.entry(case.op.onnx_name()).or_default() += 1;
    }
    let admitted = bounds.operators();
    for op in &admitted {
        let n = per_op.get(op.onnx_name()).copied().unwrap_or(0);
        let flag = if n == 0 { "   <-- 0%, UNTESTED" } else { "" };
        println!(
            "  {:<12} {n:>6}  {:>5.1}%{flag}",
            op.onnx_name(),
            100.0 * n as f64 / sample.len() as f64
        );
    }

    println!("\n── element-type distribution (of the type each case is about) ──");
    let mut per_type: BTreeMap<String, usize> = BTreeMap::new();
    for case in &sample {
        *per_type
            .entry(format!("{:?}", ops::data_elem_type(case)))
            .or_default() += 1;
    }
    for elem in bounds.element_types() {
        let key = format!("{elem:?}");
        let n = per_type.get(&key).copied().unwrap_or(0);
        let flag = if n == 0 { "   <-- 0%, UNTESTED" } else { "" };
        println!(
            "  {key:<8} {n:>6}  {:>5.1}%{flag}",
            100.0 * n as f64 / sample.len() as f64
        );
    }

    println!("\n── shape distribution ──");
    let mut per_rank: BTreeMap<usize, usize> = BTreeMap::new();
    let (mut empty, mut scalar, mut largest) = (0usize, 0usize, 0usize);
    for case in &sample {
        let input = &case.inputs[0];
        *per_rank.entry(input.dims.len()).or_default() += 1;
        if input.dims.is_empty() {
            scalar += 1;
        }
        if input.element_count() == 0 {
            empty += 1;
        }
        largest = largest.max(input.element_count());
    }
    for (rank, n) in &per_rank {
        println!(
            "  rank {rank}     {n:>6}  {:>5.1}%",
            100.0 * *n as f64 / sample.len() as f64
        );
    }
    println!(
        "  rank-0 scalars {scalar} ({:.1}%), empty tensors {empty} ({:.1}%), largest {largest} elements",
        100.0 * scalar as f64 / sample.len() as f64,
        100.0 * empty as f64 / sample.len() as f64
    );

    // ── interactions, not just constructs ─────────────────────────────────────────
    println!("\n── interactions (operator x element type), the ones at 0% ──");
    let mut pairs: BTreeMap<(&str, String), usize> = BTreeMap::new();
    for case in &sample {
        *pairs
            .entry((
                case.op.onnx_name(),
                format!("{:?}", ops::data_elem_type(case)),
            ))
            .or_default() += 1;
    }
    let mut missing = 0;
    for op in &admitted {
        for elem in bounds.element_types() {
            if ops::probe(*op, elem, bounds.opset).is_none() {
                continue; // the specification forbids it; not a gap in the corpus
            }
            let key = (op.onnx_name(), format!("{elem:?}"));
            if !pairs.contains_key(&key) {
                println!(
                    "  {:<12} {:<6}  0%   <-- reachable but never generated",
                    key.0, key.1
                );
                missing += 1;
            }
        }
    }
    println!(
        "  {} of {} spec-permitted pairs never generated",
        missing,
        admitted
            .iter()
            .flat_map(|op| bounds.element_types().into_iter().map(move |e| (*op, e)))
            .filter(|(op, e)| ops::probe(*op, *e, bounds.opset).is_some())
            .count()
    );

    // ── N3.8: the budget, measured on both sides ──────────────────────────────────
    println!("\n── the element budget, measured against a 16x larger one ──");
    let generous = Bounds {
        element_budget: bounds.element_budget * 16,
        ..bounds.clone()
    };
    let wide = corpus(&generous, cases.min(4_000));
    let narrow = corpus(&bounds, cases.min(4_000));
    let describe = |name: &str, sample: &[OnnxCase]| {
        let total: usize = sample.iter().map(OnnxCase::total_elements).sum();
        let largest = sample
            .iter()
            .map(OnnxCase::total_elements)
            .max()
            .unwrap_or(0);
        let empties = sample.iter().filter(|c| c.total_elements() == 0).count();
        println!(
            "  {name:<22} mean {:>8.1} elements   largest {largest:>7}   empty {:>4.1}%",
            total as f64 / sample.len() as f64,
            100.0 * empties as f64 / sample.len() as f64
        );
    };
    describe(&format!("budget {}", bounds.element_budget), &narrow);
    describe(&format!("budget {}", generous.element_budget), &wide);

    // **Size is not the measurement that matters.** `02-METHODOLOGY.md`'s warning is about a cap
    // that took a *divergence rate* from 9-in-2,000 to 0-in-2,000 — it had clamped away exactly
    // the shapes that diverge. A cap that halves the mean case size and costs no findings is
    // free; one that halves the findings is not, and only counting them can tell the difference.
    println!("\n  disagreement rate on each side (the measurement that actually matters):");
    let caps = onnx_adapter::capability::Capabilities::load(&format!(
        "{}/census.json",
        onnx_adapter::FINDINGS_ROOT
    ))
    .expect("run n2_census first");
    let ort = onnx_adapter::capability::WithCapabilities::new(OrtRuntime, &caps);
    let tract = onnx_adapter::capability::WithCapabilities::new(TractRuntime, &caps);

    for (label, sample) in [("budget 256", &narrow), ("budget 4096", &wide)] {
        use diff_fuzzer_core::Normalizer;
        use diff_fuzzer_core::traits::{NamedOutput, Oracle, Verdict};
        let (mut diverged, mut judged, mut skipped) = (0, 0, 0);
        for case in sample.iter() {
            let outputs: Vec<NamedOutput<_>> = [
                ("onnxruntime", ort.run(case).expect("never Err")),
                ("tract", tract.run(case).expect("never Err")),
            ]
            .into_iter()
            .map(|(n, o)| NamedOutput {
                implementation: n.to_string(),
                output: onnx_adapter::normalize::OnnxNormalizer.normalize(o),
            })
            .collect();
            match onnx_adapter::oracle::OnnxOracle.check(case, &outputs) {
                Verdict::Diverged(_) => {
                    diverged += 1;
                    judged += 1;
                }
                Verdict::Agree => judged += 1,
                Verdict::Skipped(_) => skipped += 1,
            }
        }
        println!(
            "    {label:<12} {diverged:>4} disagreements in {judged} judged ({:.2}%), {skipped} skipped",
            100.0 * diverged as f64 / judged.max(1) as f64
        );
    }

    // ── N3.9: throughput, tail included, build vs run ─────────────────────────────
    println!("\n── throughput on a tail-inclusive sample ──");
    let mut build = Vec::with_capacity(narrow.len());
    for case in &narrow {
        let start = Instant::now();
        std::hint::black_box(build_bytes(case));
        build.push(start.elapsed().as_secs_f64());
    }
    report("model build (protobuf)", &build);

    /// A participant reduced to "give me the outcome for this case", so the timing loop is the
    /// same code for each and cannot accidentally measure them differently.
    type Runner<'a> = (&'a str, Box<dyn Fn(&OnnxCase) -> OnnxOutcome>);

    let participants: Vec<Runner<'_>> = vec![
        (
            "onnxruntime",
            Box::new(|c: &OnnxCase| OrtRuntime.run(c).expect("never Err")),
        ),
        (
            "tract",
            Box::new(|c: &OnnxCase| TractRuntime.run(c).expect("never Err")),
        ),
    ];
    for (name, run) in &participants {
        // Warm up before timing. A one-time cost inside the first loop is what produced a
        // number 12x cheaper than a step it contained, earlier in this project.
        for case in narrow.iter().take(50) {
            std::hint::black_box(run(case));
        }
        let mut times = Vec::with_capacity(narrow.len());
        for case in &narrow {
            let start = Instant::now();
            std::hint::black_box(run(case));
            times.push(start.elapsed().as_secs_f64());
        }
        report(name, &times);
    }

    println!("\nThe max column is the point: a mean alone cannot show a tail.");
    let _ = ElemType::F32;
}

fn report(name: &str, times: &[f64]) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in a duration"));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let at = |q: f64| sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)];
    println!(
        "  {name:<22} mean {:>8.4} ms  median {:>8.4}  p99 {:>8.4}  max {:>8.4}  ({:.0}/sec)",
        mean * 1e3,
        at(0.50) * 1e3,
        at(0.99) * 1e3,
        sorted[sorted.len() - 1] * 1e3,
        1.0 / mean
    );
}
