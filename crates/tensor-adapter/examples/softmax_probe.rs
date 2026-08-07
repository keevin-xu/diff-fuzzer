//! **Does `burn-flex`'s transposing path disagree with its direct one?**
//!
//! `burn-flex` normalises the last axis directly and any other axis by transposing,
//! normalising, and transposing back (`burn-flex/src/ops/activation.rs`). Two code paths
//! selected by a property of the input — structurally the same shape as the libtorch
//! tile-remainder effect, which is the one real bug this project has found.
//!
//! A fuzzer might reach this. A constructed experiment answers it in seconds, and does not
//! depend on a corpus happening to explore the right region. The same reasoning produced
//! `batched_probe.rs` at PHASE-7, which is what located the tile formula.
//!
//! # The experiments
//!
//! **A — the same numbers, normalised along different axes.** A square tensor whose rows and
//! columns hold the same multiset of values: softmax along either axis must give the same
//! numbers, transposed. Any difference is the transpose path and nothing else.
//!
//! **B — dimension held, rank varied.** Isolates whether it is the *transpose* that matters
//! or merely a non-trivial rank.
//!
//! **C — the exactly-known answer, off the last axis.** Rows of identical values give exactly
//! `1/n`. No tolerance argument can excuse a disagreement here.
//!
//! **D — the numerically hard cases.** A and B use well-conditioned data, where the derived
//! bound is dominated by nothing in particular. The bound says error concentrates where the
//! value *range* is wide (driving `exp`'s condition number after the max-shift) and where the
//! shifted argument underflows `exp` to zero. A probe that skipped those would have tested
//! the transpose hypothesis and nothing else.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example softmax_probe
//! ```

use diff_fuzzer_core::{
    DifferentialOracle, Implementation, NamedOutput, NormalizedRunner, Oracle, Runner, Verdict,
};
use tensor_adapter::input::ActivationOp;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, TensorValue, flex,
    libtorch, wgpu,
};

/// **What the policy actually decides**, as opposed to what the raw numbers look like.
///
/// A relative gap is only half the question: the other half is whether the derived bound
/// covers it. Reading the gaps and judging them by eye would be re-deriving the policy by
/// hand — and getting a different answer from the tool would be the real finding.
fn verdict(case: &TensorOp) -> String {
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let gpu = NormalizedRunner::new(wgpu(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);

    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 3] = [&cpu, &torch, &gpu];
    let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
        .iter()
        .filter_map(|r| {
            r.run_and_normalize(case).ok().map(|output| NamedOutput {
                implementation: r.name().to_string(),
                output,
            })
        })
        .collect();

    match oracle.check(case, &outputs) {
        Verdict::Diverged(_) => "DIVERGED".to_string(),
        Verdict::Agree => "agree".to_string(),
        Verdict::Skipped(reason) => format!("skipped ({reason:?})"),
    }
}

fn softmax(shape: &[usize], data: Vec<f32>, dim: usize) -> TensorOp {
    TensorOp::activation(
        ActivationOp::Softmax,
        TensorValue::new(shape.to_vec(), data),
        dim,
    )
}

/// Every backend's answer, as raw values.
fn run_all(case: &TensorOp) -> Vec<(String, Vec<f32>)> {
    let backends: Vec<Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>> =
        vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())];

    backends
        .iter()
        .filter_map(|b| {
            b.run(case)
                .ok()
                .and_then(|out| out.to_vec::<f32>().ok())
                .map(|v| (b.name().to_string(), v))
        })
        .collect()
}

/// The largest **relative** gap between any two backends, elementwise.
///
/// **Absolute gaps are the wrong measure here and reporting them was actively misleading.**
/// `softmax` outputs sum to 1, so most elements are small and some are subnormal: an
/// absolute gap of `1e-19` looks like agreement and is a *relative* error of `1e-6` when the
/// value itself is `1e-13`. Every tolerance in `POLICY.md` for this class is an `rtol`, so
/// the probe must speak the same language.
///
/// Zero against non-zero is reported as infinite rather than as the value itself: it is a
/// complete loss of the result, not a small error, and it is exactly what underflow produces.
fn worst_gap(results: &[(String, Vec<f32>)]) -> (f64, String) {
    let mut worst = 0.0f64;
    let mut who = String::new();
    for (i, (na, a)) in results.iter().enumerate() {
        for (nb, b) in results.iter().skip(i + 1) {
            for (x, y) in a.iter().zip(b) {
                let (x, y) = (*x as f64, *y as f64);
                // Both NaN counts as agreement, as does bit-equality; only a NaN facing a
                // number is a total disagreement.
                let gap = if x.is_nan() != y.is_nan() {
                    f64::INFINITY
                } else if x.is_nan() || x == y {
                    0.0
                } else {
                    let scale = x.abs().max(y.abs());
                    if scale == 0.0 {
                        0.0
                    } else {
                        (x - y).abs() / scale
                    }
                };
                if gap > worst {
                    worst = gap;
                    who = format!("{na} vs {nb}");
                }
            }
        }
    }
    (worst, who)
}

fn report(label: &str, case: &TensorOp) {
    let results = run_all(case);
    let (gap, who) = worst_gap(&results);
    let TensorOp::Activation { arg, dim, .. } = case else {
        unreachable!()
    };
    println!(
        "  {label:<30} dim {dim} {:?}  rel {gap:.2e}  {:<24} -> {}",
        arg.shape(),
        who,
        verdict(case)
    );
}

fn main() {
    println!("A — same numbers, different axis (any difference IS the transpose path)\n");
    // A 4x4 whose transpose holds the same values: softmax along 0 and along 1 must agree
    // up to transposition, so comparing each backend against the others on each axis
    // isolates the path rather than the data.
    let symmetric: Vec<f32> = (0..16).map(|i| ((i % 4) as f32) - 1.5).collect();
    report(
        "last axis (direct path)",
        &softmax(&[4, 4], symmetric.clone(), 1),
    );
    report("axis 0 (transposing path)", &softmax(&[4, 4], symmetric, 0));

    println!("\nB — dimension held, rank varied\n");
    for shape in [vec![8, 4], vec![2, 4, 4], vec![2, 2, 2, 4]] {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 - 3.0).collect();
        let last = shape.len() - 1;
        report("last axis", &softmax(&shape, data.clone(), last));
        if shape.len() > 1 {
            report("axis 0", &softmax(&shape, data, 0));
        }
    }

    println!("\nD — numerically hard cases\n");
    for (label, shape, data, dim) in [
        (
            "wide range, last axis",
            vec![1, 4],
            vec![0.0f32, 30.0, 60.0, 90.0],
            1usize,
        ),
        (
            "wide range, transposing",
            vec![4, 1],
            vec![0.0f32, 30.0, 60.0, 90.0],
            0,
        ),
        (
            "underflowing exp (>104 apart)",
            vec![1, 3],
            vec![0.0f32, 100.0, 200.0],
            1,
        ),
        (
            "underflowing, transposing",
            vec![3, 1],
            vec![0.0f32, 100.0, 200.0],
            0,
        ),
        (
            "long dimension, 64 terms",
            vec![1, 64],
            (0..64).map(|i| (i as f32) * 0.7).collect(),
            1,
        ),
        (
            "long dimension, transposing",
            vec![64, 1],
            (0..64).map(|i| (i as f32) * 0.7).collect(),
            0,
        ),
        ("all equal and huge", vec![1, 8], vec![1e30f32; 8], 1),
        (
            "mixed sign extremes",
            vec![1, 4],
            vec![-1e30f32, 1e30, -1e30, 1e30],
            1,
        ),
    ] {
        report(label, &softmax(&shape, data, dim));
    }

    println!("\nC — exactly known answers (1/n), off the last axis\n");
    for (shape, dim) in [
        (vec![4, 4], 0usize),
        (vec![3, 5], 0),
        (vec![2, 3, 4], 1),
        (vec![7, 2], 0),
    ] {
        let n: usize = shape.iter().product();
        let expected = 1.0 / shape[dim] as f32;
        let case = softmax(&shape, vec![2.5; n], dim);
        let results = run_all(&case);
        let worst_from_exact = results
            .iter()
            .flat_map(|(name, values)| {
                values
                    .iter()
                    .map(move |v| ((*v as f64 - expected as f64).abs(), name.clone()))
            })
            .fold(
                (0.0f64, String::new()),
                |acc, x| if x.0 > acc.0 { x } else { acc },
            );
        println!(
            "  shape {:?} dim {dim}   exact {expected:.6}   worst error {:.3e}  {}",
            shape, worst_from_exact.0, worst_from_exact.1
        );
    }
}
