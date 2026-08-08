//! Do the backends' `erf` implementations switch approximations at the same point?
//!
//! `burn-flex` delegates to `libm::erff`, a piecewise-rational approximation whose formula
//! changes near `|x| = 0.84375` (`SPECS.md` §2b.5). `burn-tch` uses libtorch's own, which is
//! undocumented. **A switch point is a property of the input that selects a code path** — the
//! same shape as the tile remainder that explained the one bug filed upstream — so two
//! implementations choosing different boundaries should disagree most sharply either side of
//! one.
//!
//! The GPU is excluded on purpose: Metal's §8 gives no accuracy for `erf` in either table
//! (`SPECS.md` §4.1b), so no bound exists for that pair and the policy declines it.
use diff_fuzzer_core::{
    DifferentialOracle, Implementation, NamedOutput, NormalizedRunner, Oracle, Runner,
    TolerancePolicy, Verdict,
};
use tensor_adapter::input::UnaryOp;
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, TensorValue, flex, libtorch,
};

/// **Ask the oracle; do not re-derive the rule.**
///
/// The agreement test is `|a - b| <= atol + rtol * max(|a|, |b|)`, and a probe comparing a
/// relative gap against `rtol` alone ignores the floor that exists precisely for values too
/// small for a relative bound to mean anything. That mistake was made in this project's
/// `exp` probe and then repeated here, reporting "EXCEEDS BOUND" for subnormal results the
/// policy covers perfectly well. Asking the tool is the fix — and a disagreement between the
/// tool and a hand-rolled check would itself be the finding.
fn verdict(case: &TensorOp) -> &'static str {
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);
    let runners: [&dyn Runner<In = TensorOp, Canon = CanonicalTensor>; 2] = [&cpu, &torch];

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
        Verdict::Diverged(_) => "DIVERGED",
        Verdict::Agree => "agree",
        Verdict::Skipped(_) => "skipped",
    }
}

fn erf_at(values: Vec<f32>) -> (Vec<f32>, Vec<f32>, f64) {
    let n = values.len();
    let case = TensorOp::unary(UnaryOp::Erf, TensorValue::new(vec![n], values));
    let a = flex()
        .run(&case)
        .expect("flex")
        .to_vec::<f32>()
        .expect("f32");
    let b = libtorch()
        .run(&case)
        .expect("tch")
        .to_vec::<f32>()
        .expect("f32");
    let bound = TensorTolerancePolicy.tolerance_for(&case, ("burn-flex", "burn-tch"));
    (a, b, bound.rtol)
}

/// Worst relative gap, and how it compares with what the policy allows.
fn report(label: &str, values: Vec<f32>) {
    let (a, b, rtol) = erf_at(values.clone());
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let (x, y) = (*x as f64, *y as f64);
        let scale = x.abs().max(y.abs());
        let gap = if scale == 0.0 {
            0.0
        } else {
            (x - y).abs() / scale
        };
        if gap > worst {
            worst = gap;
            at = i;
        }
    }
    let n = values.len();
    let case = TensorOp::unary(UnaryOp::Erf, TensorValue::new(vec![n], values.clone()));
    println!(
        "  {label:<30} worst rel {worst:.2e}  rtol {rtol:.2e}  {:<9}  (x={:e})",
        verdict(&case),
        values[at]
    );
}

/// Steps `count` ULP from `start`, upward if `up`.
fn ulp_walk(start: f32, count: usize, up: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    let mut bits = start.to_bits();
    for _ in 0..count {
        out.push(f32::from_bits(bits));
        bits = if up { bits + 1 } else { bits - 1 };
    }
    out
}

fn main() {
    let switch = 0.84375f32;

    println!("A — straddling libm's switch point at |x| = 0.84375\n");
    report("32 ULP below the switch", ulp_walk(switch, 32, false));
    report("32 ULP above the switch", ulp_walk(switch, 32, true));
    report("negative side, below", ulp_walk(-switch, 32, false));
    report("negative side, above", ulp_walk(-switch, 32, true));

    println!("\nB — other plausible boundaries, for comparison\n");
    for point in [0.0f32, 0.5, 1.0, 2.0, 4.0, 6.0] {
        report(&format!("32 ULP around {point}"), ulp_walk(point, 32, true));
    }

    println!("\nC — the saturating tail, where erf approaches ±1\n");
    for point in [3.0f32, 4.0, 5.0, 10.0] {
        report(
            &format!("around {point} (erf ≈ 1)"),
            ulp_walk(point, 32, true),
        );
    }
}
