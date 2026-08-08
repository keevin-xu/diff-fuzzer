//! Do three implementations of `cumsum` associate their additions the same way?
//!
//! A sequential scan keeps a running total. A parallel scan — Hillis–Steele or Blelloch —
//! computes partial sums in a tree and combines them, which **associates the additions
//! differently by construction**. Floating-point addition is not associative, so both can be
//! correct and differ.
//!
//! That makes this the strongest available candidate for a **numeric** disagreement, which is
//! the class 3.9 million cases of ordinary arithmetic failed to produce. This asks directly
//! rather than waiting for a fuzzer to stumble on it.
use diff_fuzzer_core::{
    DifferentialOracle, Implementation, NamedOutput, NormalizedRunner, Oracle, Runner,
    TolerancePolicy, Verdict,
};
use tensor_adapter::input::ScanOp;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, TensorValue, flex,
    libtorch, wgpu,
};

/// **What the policy decides, and what it allowed.**
///
/// A measured gap means nothing without the bound it is being measured against. Reading the
/// numbers and judging by eye would be re-deriving the policy by hand — and disagreeing with
/// the tool would itself be the finding.
fn verdict(data: Vec<f32>) -> String {
    let n = data.len();
    let case = TensorOp::scan(ScanOp::CumSum, TensorValue::new(vec![1, n], data), 1);
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let gpu = NormalizedRunner::new(wgpu(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 3] = [&cpu, &torch, &gpu];

    let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
        .iter()
        .filter_map(|r| {
            r.run_and_normalize(&case).ok().map(|output| NamedOutput {
                implementation: r.name().to_string(),
                output,
            })
        })
        .collect();

    let bound = TensorTolerancePolicy.tolerance_for(&case, ("burn-flex", "burn-tch"));
    let outcome = match oracle.check(&case, &outputs) {
        Verdict::Diverged(_) => "DIVERGED",
        Verdict::Agree => "agree",
        Verdict::Skipped(_) => "skipped",
    };
    format!("{outcome} (rtol {:.2e})", bound.rtol)
}

fn run(data: Vec<f32>) -> Vec<(String, Vec<f32>)> {
    let n = data.len();
    let case = TensorOp::scan(ScanOp::CumSum, TensorValue::new(vec![1, n], data), 1);
    let backends: Vec<Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>> =
        vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())];
    backends
        .iter()
        .filter_map(|b| {
            b.run(&case)
                .ok()
                .and_then(|o| o.to_vec::<f32>().ok())
                .map(|v| (b.name().to_string(), v))
        })
        .collect()
}

/// Worst relative gap between any two backends, and where.
fn worst(results: &[(String, Vec<f32>)]) -> (f64, String) {
    let mut worst = 0.0f64;
    let mut who = String::new();
    for (i, (na, a)) in results.iter().enumerate() {
        for (nb, b) in results.iter().skip(i + 1) {
            for (index, (x, y)) in a.iter().zip(b).enumerate() {
                let (x, y) = (*x as f64, *y as f64);
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
                    who = format!("{na}={x} vs {nb}={y} at element {index}");
                }
            }
        }
    }
    (worst, who)
}

fn show(label: &str, data: Vec<f32>) {
    let n = data.len();
    let (gap, who) = worst(&run(data.clone()));
    println!(
        "  {label:<28} n={n:<5} rel {gap:.2e}  {:<28} {who}",
        verdict(data)
    );
}

fn main() {
    println!("E — cumprod: where a running product overflows depends on grouping\n");
    {
        use tensor_adapter::input::ScanOp as S;
        let run_prod = |data: Vec<f32>| -> String {
            let n = data.len();
            let case = TensorOp::scan(S::CumProd, TensorValue::new(vec![1, n], data), 1);
            let backends: Vec<
                Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>,
            > = vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())];
            let outs: Vec<(String, Vec<f32>)> = backends
                .iter()
                .filter_map(|b| {
                    b.run(&case)
                        .ok()
                        .and_then(|o| o.to_vec::<f32>().ok())
                        .map(|v| (b.name().to_string(), v))
                })
                .collect();
            let (gap, who) = worst(&outs);
            format!("rel {gap:.2e}  {who}")
        };

        // Overflow then recovery: 1e20 * 1e20 is inf, but * 1e-20 afterwards would recover
        // it only if the product had not already saturated. Grouping decides.
        println!(
            "  overflow then divide back   {}",
            run_prod(vec![1e20, 1e20, 1e-20, 1e-20])
        );
        // A zero mid-sequence annihilates everything after it.
        println!(
            "  zero mid-sequence           {}",
            run_prod(vec![2.0, 0.0, 1e30, 1e30])
        );
        // Alternating magnitudes: the running value swings across the representable range.
        println!(
            "  alternating 1e18 / 1e-18    {}",
            run_prod(
                (0..32)
                    .map(|i| if i % 2 == 0 { 1e18 } else { 1e-18 })
                    .collect()
            )
        );
        // Long run of values just above 1: overflow arrives gradually.
        println!(
            "  1.5 repeated (overflows)    {}",
            run_prod(vec![1.5f32; 256])
        );
    }

    println!("\nF — underflow then recovery: does a flushed intermediate come back?\n");
    {
        use tensor_adapter::input::ScanOp as S;
        let show_prod = |label: &str, data: Vec<f32>| {
            let n = data.len();
            let case = TensorOp::scan(S::CumProd, TensorValue::new(vec![1, n], data), 1);
            let backends: Vec<
                Box<dyn Implementation<In = TensorOp, Out = burn::tensor::TensorData>>,
            > = vec![Box::new(flex()), Box::new(libtorch()), Box::new(wgpu())];
            let outs: Vec<String> = backends
                .iter()
                .filter_map(|b| {
                    b.run(&case)
                        .ok()
                        .and_then(|o| o.to_vec::<f32>().ok())
                        .map(|v| {
                            format!(
                                "{}=[{}]",
                                b.name(),
                                v.iter()
                                    .map(|x| format!("{x:e}"))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )
                        })
                })
                .collect();
            println!("  {label}");
            for o in outs {
                println!("      {o}");
            }
        };

        // Decay well into the subnormal range, then multiply back up. If an intermediate is
        // flushed to zero, zero is absorbing and the recovery never happens.
        show_prod(
            "1e-20 x4 then 1e20 x2",
            vec![1e-20, 1e-20, 1e-20, 1e-20, 1e20, 1e20],
        );
        show_prod("1e-30 x2 then 1e30", vec![1e-30, 1e-30, 1e30]);
        // Just below the smallest normal, then straight back up.
        show_prod("1e-38 x2 then 1e30", vec![1e-38, 1e-38, 1e30]);
    }

    println!("\nA — cancellation, where association order becomes visible\n");
    // (a + -a) + b is b; a + (-a + b) can lose b entirely to rounding first.
    for n in [4usize, 16, 64, 256] {
        let data: Vec<f32> = (0..n)
            .map(|i| match i % 3 {
                0 => 1e30,
                1 => -1e30,
                _ => 1.0,
            })
            .collect();
        show("alternating ±1e30 with ones", data);
    }

    println!("\nB — many small terms after one large one\n");
    // The classic association trap: a running total loses every small term, a tree does not.
    for n in [16usize, 256, 1024] {
        let mut data = vec![1e8f32];
        data.extend(std::iter::repeat_n(1.0f32, n - 1));
        show("1e8 then ones", data);
    }

    println!("\nC — long axes of ordinary values\n");
    for n in [64usize, 1024, 4096] {
        let data: Vec<f32> = (0..n).map(|i| ((i % 17) as f32) - 8.0).collect();
        show("mixed small values", data);
    }

    println!("\nD — exactly representable, so any difference is a defect\n");
    for n in [8usize, 64, 512] {
        let data: Vec<f32> = std::iter::repeat_n(1.0f32, n).collect();
        show("all ones (answer is 1..n)", data);
    }
}
