//! Do `burn-flex`'s five convolution paths agree with each other and with libtorch?
//!
//! **A hand-built sweep, not a campaign.** It answers the phase's actual question in minutes
//! by walking each guard in `burn-flex/src/ops/conv.rs` deliberately, instead of hoping a
//! fuzzer stumbles across the boundary. The same shape of experiment turned burn#5284 from a
//! symptom into the `(m mod 4) × (n mod 8)` mechanism.
//!
//! **The verdict comes from the oracle, never from arithmetic written here.** Reimplementing
//! the agreement rule in a probe has been done twice on this project and was wrong both
//! times — once comparing `rtol` while ignoring `atol`, once comparing only two of three
//! backends. `oracle.check` is the same code the campaign uses, so a probe cannot disagree
//! with a campaign about what counts as a divergence.
//!
//! Values are held fixed and small across the whole sweep, so the only thing varying is the
//! shape and the parameters — which is what makes a difference attributable to a code path
//! rather than to the numbers.

use diff_fuzzer_core::{
    Agreement, ApproxEq, DifferentialOracle, NamedOutput, NormalizedRunner, Oracle, Runner,
    SkipReason, TolerancePolicy, Verdict,
};
use tensor_adapter::input::{Conv2dParams, TensorValue};
use tensor_adapter::ops::conv::{Profile, classify};
use tensor_adapter::{
    CanonicalTensor, TensorNormalizer, TensorOp, TensorTolerancePolicy, flex, libtorch, wgpu,
};

/// Deterministic values that **require rounding**, which is the whole point.
///
/// # The mistake this replaced, because it produced a meaningless clean result
///
/// The first version of this probe used values like `±8 × 1.75` and `× 0.25`. Those are
/// exactly representable in binary, so every product and every partial sum was exact — and
/// all three backends agreed **bit-for-bit on all 42 configurations**. That reads as a strong
/// result and is nearly a vacuous one: with exact arithmetic no association order can
/// possibly matter, so the sweep could not have detected a rounding difference even if one
/// existed.
///
/// It was not entirely vacuous — an *indexing* bug like burn#4727's missing channel offset
/// sums the wrong elements and shows up regardless of representability. But that is a
/// narrower question than the one this probe claims to answer.
///
/// So the values now fill the mantissa: a multiplicative hash into a coprime divisor, which
/// is deterministic, reproducible, and essentially never exact.
///
/// A probe that injected infinities would be asking a different question — the special-value
/// policy already covers those, and here they would only obscure whether the *paths* agree.
fn tensor(shape: &[usize]) -> TensorValue {
    let count: usize = shape.iter().product();
    let data = (0..count)
        .map(|i| {
            let hashed = (i as u64).wrapping_mul(2_654_435_761) % 9_973;
            (hashed as f32 / 9_973.0) * 6.4 - 3.1
        })
        .collect();
    TensorValue::new(shape.to_vec(), data)
}

struct Case {
    label: String,
    op: TensorOp,
}

fn case(label: &str, image: &[usize], weight: &[usize], params: Conv2dParams, bias: bool) -> Case {
    Case {
        label: label.to_string(),
        op: TensorOp::conv2d(
            tensor(image),
            tensor(weight),
            bias.then(|| tensor(&[weight[0]])),
            params,
        ),
    }
}

fn main() {
    let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
    let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
    let gpu = NormalizedRunner::new(wgpu(), TensorNormalizer);
    let oracle = DifferentialOracle::new(TensorTolerancePolicy);
    let runners: Vec<&dyn Runner<In = TensorOp, Canon = CanonicalTensor>> =
        vec![&cpu, &torch, &gpu];

    let mut cases: Vec<Case> = Vec::new();

    // --- A. The grouped-and-padded region: burn#4727's exact trigger ----------------
    //
    // The fixed bug was a missing channel offset in the SIMD *remainder* path, reached only
    // when `groups > 1` AND `padding > 0`. Both conditions are swept, and both are swept
    // *separately* too — a difference that appears with either alone is a different story
    // from one that needs the conjunction.
    for groups in [1, 2, 4] {
        for padding in [0, 1, 2] {
            let in_channels = 4;
            cases.push(case(
                &format!("A groups={groups} padding={padding}"),
                &[1, in_channels, 6, 6],
                &[4, in_channels / groups, 3, 3],
                Conv2dParams {
                    groups,
                    padding: [padding, padding],
                    ..Default::default()
                },
                false,
            ));
        }
    }

    // --- B. Channel counts that are not a multiple of a vector width -----------------
    //
    // A remainder path exists because channels do not divide evenly into whatever the kernel
    // processes at once. Sweeping 1..=8 crosses every plausible width.
    for in_channels in 1..=8usize {
        cases.push(case(
            &format!("B in_channels={in_channels}"),
            &[1, in_channels, 5, 5],
            &[2, in_channels, 3, 3],
            Conv2dParams {
                padding: [1, 1],
                ..Default::default()
            },
            false,
        ));
    }

    // --- C. The pointwise guard: 1x1 skips im2col entirely --------------------------
    //
    // burn#4591 lived in `conv_im2col_1x1`. The neighbours matter as much as the case: 1x2
    // and 2x1 are one step off the guard and take the general path.
    for (kh, kw) in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1)] {
        cases.push(case(
            &format!("C kernel={kh}x{kw}"),
            &[1, 3, 5, 5],
            &[2, 3, kh, kw],
            Conv2dParams::default(),
            false,
        ));
    }

    // --- D. Depthwise: groups == in_channels == out_channels ------------------------
    //
    // And one step either side, because the guard is an equality: a case with one channel
    // too many is not depthwise and runs different code.
    for channels in [2, 3, 4, 8] {
        cases.push(case(
            &format!("D depthwise channels={channels}"),
            &[1, channels, 5, 5],
            &[channels, 1, 3, 3],
            Conv2dParams {
                groups: channels,
                ..Default::default()
            },
            false,
        ));
        cases.push(case(
            &format!("D near-depthwise channels={channels} (2 per group)"),
            &[1, channels * 2, 5, 5],
            &[channels, 2, 3, 3],
            Conv2dParams {
                groups: channels,
                ..Default::default()
            },
            false,
        ));
    }

    // --- E. A degenerate output, where the remainder path is all there is ------------
    //
    // With one output position the main tiled loop runs zero times, so a bug in the cleanup
    // path is unmasked rather than averaged away.
    for size in [3, 4, 5] {
        cases.push(case(
            &format!("E exact-fit kernel {size}x{size} (output 1x1)"),
            &[1, 4, size, size],
            &[4, 4, size, size],
            Conv2dParams::default(),
            false,
        ));
    }

    // --- F. Stride and dilation, which change how the window walks -------------------
    for stride in [1, 2, 3] {
        for dilation in [1, 2] {
            cases.push(case(
                &format!("F stride={stride} dilation={dilation}"),
                &[1, 3, 8, 8],
                &[3, 3, 3, 3],
                Conv2dParams {
                    stride: [stride, stride],
                    dilation: [dilation, dilation],
                    ..Default::default()
                },
                false,
            ));
        }
    }

    // --- G. Bias present or absent: fused into the accumulator, or a separate pass ---
    for bias in [false, true] {
        cases.push(case(
            &format!("G bias={bias}"),
            &[1, 4, 6, 6],
            &[3, 4, 3, 3],
            Conv2dParams {
                padding: [1, 1],
                ..Default::default()
            },
            bias,
        ));
    }

    println!(
        "{} configurations, values held fixed across all of them\n",
        cases.len()
    );

    let (mut agreed, mut diverged, mut skipped) = (0, 0, 0);
    let mut tightest_margin: f64 = f64::INFINITY;
    let mut divergences: Vec<String> = Vec::new();

    for Case { label, op } in &cases {
        let outputs: Vec<NamedOutput<CanonicalTensor>> = runners
            .iter()
            .filter_map(|r| {
                r.run_and_normalize(op).ok().map(|output| NamedOutput {
                    implementation: r.name().to_string(),
                    output,
                })
            })
            .collect();

        // **Every backend must have run.** Silently dropping one is the failure this project
        // has made five times: a `filter_map` over `.ok()` turns a backend that errored into
        // a backend that agreed, and a two-way comparison reported as three-way.
        assert_eq!(
            outputs.len(),
            runners.len(),
            "{label}: only {} of {} backends ran — a dropped backend would be reported as \
             agreement",
            outputs.len(),
            runners.len()
        );

        // **How much tighter could the bound be and still pass?**
        //
        // Reported, never decided on — the verdict above is the oracle's.
        //
        // This is the second version. The first divided the observed relative error by
        // `rtol`, which is **the exact mistake this file's header warns about**, made twice
        // before on this project and now a third time: the agreement rule is
        // `|a-b| <= atol + rtol*|b|`, so a ratio against `rtol` alone says nothing. It
        // printed values above 1 for cases that comfortably agreed, because `atol` was
        // carrying them.
        //
        // So instead of doing arithmetic here, the tolerance is *scaled down* and the real
        // comparison re-run. The largest factor that still agrees is a true margin, expressed
        // in the same rule the oracle uses.
        let mut margin = f64::INFINITY;
        for left in 0..outputs.len() {
            for right in (left + 1)..outputs.len() {
                let (a, b) = (&outputs[left], &outputs[right]);
                let full =
                    TensorTolerancePolicy.tolerance_for(op, (&a.implementation, &b.implementation));

                let mut tightest = 1.0f64;
                for exponent in 0..40 {
                    let factor = 2f64.powi(exponent);
                    let scaled =
                        diff_fuzzer_core::Tolerance::new(full.rtol / factor, full.atol / factor);
                    match a.output.approx_compare(&b.output, scaled) {
                        Agreement::Agree(_) => tightest = factor,
                        _ => break,
                    }
                }
                margin = margin.min(tightest);
            }
        }

        let profile = format!("{:?}", classify(op));
        let verdict = match oracle.check(op, &outputs) {
            Verdict::Agree => {
                agreed += 1;
                "agree".to_string()
            }
            Verdict::Diverged(divergence) => {
                diverged += 1;
                divergences.push(format!("{label}\n    {}", divergence.summary));
                "DIVERGED".to_string()
            }
            Verdict::Skipped(reason) => {
                skipped += 1;
                match reason {
                    SkipReason::Unjudgeable { rtol, atol } => {
                        format!("unjudged (rtol {rtol:.2e} atol {atol:.2e})")
                    }
                    other => format!("skipped {other}"),
                }
            }
        };
        let shown = if margin.is_finite() {
            format!("{margin:>10.0}x")
        } else {
            "         -".to_string()
        };
        println!("  {label:<44} {profile:<17} {verdict:<9} margin {shown}");
        if margin.is_finite() {
            tightest_margin = tightest_margin.min(margin);
        }
    }

    println!("\n{agreed} agree, {diverged} diverged, {skipped} unjudged");
    if tightest_margin.is_finite() {
        println!(
            "tightest margin anywhere: the bound could be {tightest_margin:.0}x narrower and \
             every case would still agree"
        );
    }

    if divergences.is_empty() {
        // **A clean sweep is a result, not a failure.** It says these five code paths agree
        // with each other and with libtorch on ordinary values — which is exactly the kind of
        // negative the project records rather than discards.
        println!("\nNo divergence across any path boundary swept here.");
    } else {
        println!("\nDivergences:");
        for entry in &divergences {
            println!("  {entry}");
        }
    }

    // Which profiles were actually exercised, so a silently-uncovered path is visible rather
    // than being mistaken for a path that agreed.
    let mut covered: Vec<String> = cases
        .iter()
        .map(|c| format!("{:?}", classify(&c.op)))
        .collect();
    covered.sort();
    covered.dedup();
    println!("\nProfiles exercised: {}", covered.join(", "));
    for profile in [
        Profile::GroupedAndPadded,
        Profile::Pointwise,
        Profile::Depthwise,
        Profile::FewChannels,
        Profile::General,
    ] {
        let name = format!("{profile:?}");
        if !covered.contains(&name) {
            println!("  NOT COVERED: {name}");
        }
    }
}
