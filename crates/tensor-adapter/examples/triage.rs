//! Work through a findings log and decide what each divergence actually is.
//!
//! The triage ladder, in order. Each rung must be cleared before the next is worth
//! asking:
//!
//! 1. **Is it our tool's fault?** Does it reproduce from its recorded seed? Is our
//!    comparison or normalisation wrong? If so, fix the tool — nothing has been learned
//!    about the target.
//! 2. **Is it floating-point noise?** Within a defensible tolerance for the operation?
//!    Then it is not a bug, and the tolerance policy is what needs attention.
//! 3. **Is it a legal difference?** Behaviour the specification leaves unspecified, so
//!    both implementations are within their rights.
//! 4. **Is it real?** Reproducible, beyond tolerance, and not legal. Only these are
//!    worth minimising and reporting upstream.
//!
//! This program automates rungs 1 and 2 — reproducibility, and comparing each error
//! against what floating-point arithmetic predicts for that operation. Rungs 3 and 4
//! need judgement, and the output is arranged to support it.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example triage findings/campaign-wide.jsonl wide
//! ```

use diff_fuzzer_core::{
    Generator, Implementation, Normalizer, SeededRng, Tolerance, TolerancePolicy, compare,
    read_findings,
};
use std::collections::BTreeMap;
use tensor_adapter::{
    Bounds, TensorNormalizer, TensorOp, TensorOpGenerator, TensorTolerancePolicy, libtorch, ndarray,
};

/// One rounding step for `f32`.
const EPSILON: f64 = f32::EPSILON as f64;

struct Triaged {
    label: &'static str,
    reproduced: bool,
    /// Largest magnitude among the case's arguments.
    largest_argument: f64,
    observed_relative_error: f64,
    /// What floating-point arithmetic predicts for this operation at this magnitude.
    predicted_relative_error: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: triage <findings.jsonl> [wide]");
        std::process::exit(1);
    });
    let wide = args.next().is_some_and(|a| a == "wide");

    // The bounds must match the campaign that produced the log, because a seed only
    // reproduces a case for the generator configuration that made it. **The log does
    // not record which bounds were used** — a gap worth fixing when the full report
    // artifact is built, since a finding that needs out-of-band knowledge to reproduce
    // is only half a finding.
    let bounds = if wide {
        Bounds {
            max_rank: 3,
            max_dim: 64,
            magnitude: 1000.0,
            ..Bounds::default()
        }
    } else {
        Bounds::default()
    };

    let findings = read_findings(&path).expect("findings log is readable");
    println!("triage: {} findings from {path}\n", findings.len());

    let generator = TensorOpGenerator::new(bounds);
    let (cpu, torch) = (ndarray(), libtorch());
    let policy = TensorTolerancePolicy;

    let mut triaged = Vec::new();

    for finding in &findings {
        let case = generator.generate(&mut SeededRng::from_seed(finding.seed));

        let left = TensorNormalizer.normalize(cpu.run(&case).expect("valid case"));
        let right = TensorNormalizer.normalize(torch.run(&case).expect("valid case"));

        // Rung 1: does it still diverge, from the seed alone?
        let tolerance = policy.tolerance_for(&case);
        let reproduced = !compare(&left.values, &right.values, tolerance).agrees();

        // Rung 2: how does the error compare with what the arithmetic predicts?
        let measured = compare(&left.values, &right.values, Tolerance::EXACT);
        let largest_argument = largest_argument(&case);

        triaged.push(Triaged {
            label: case.name(),
            reproduced,
            largest_argument,
            observed_relative_error: measured.max_relative_error,
            predicted_relative_error: predicted(case.name(), largest_argument),
        });
    }

    report(&triaged);
}

/// Largest absolute value among a case's arguments.
fn largest_argument(case: &TensorOp) -> f64 {
    let largest = |values: &[f32]| {
        values
            .iter()
            .map(|v| v.abs() as f64)
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max)
    };

    match case {
        TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => largest(arg.data()),
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => {
            largest(lhs.data()).max(largest(rhs.data()))
        }
    }
}

/// What relative error floating-point arithmetic predicts for this operation.
///
/// For `exp` the prediction is its **condition number** times epsilon. A function's
/// condition number says how much a relative perturbation of the input is magnified in
/// the output; for `exp(x)` it is `|x|`, because `exp(x + d) = exp(x) * e^d`. Since
/// implementations reduce the argument before approximating (`x = k*ln2 + r`), the error
/// in that reduction grows with `|x|` — so the two libraries drift further apart the
/// larger the argument.
///
/// The **factor of two** is the same correction the accumulating tolerance already
/// applies, and omitting it here was an inconsistency rather than a judgement: the
/// bound describes how far *one* implementation may sit from the true value, and two of
/// them may sit on opposite sides, so the gap between them can be twice it. The
/// justification is independent of the measurement — it is the same argument used in
/// `tolerance.rs` before any of this data existed.
fn predicted(label: &str, largest_argument: f64) -> f64 {
    match label {
        // A floor of two rounding steps, growing with the condition number, doubled for
        // the two-sided gap.
        "exp" => 2.0 * (2.0 * EPSILON).max(largest_argument * EPSILON),
        // Correctly rounded by IEEE-754: no difference is predicted at all.
        _ => 0.0,
    }
}

fn report(triaged: &[Triaged]) {
    // Rung 1 — reproducibility.
    let reproduced = triaged.iter().filter(|t| t.reproduced).count();
    println!("1. reproducible from the recorded seed");
    println!(
        "   {reproduced} of {} ({:.1}%)",
        triaged.len(),
        100.0 * reproduced as f64 / triaged.len() as f64
    );
    if reproduced < triaged.len() {
        println!("   WARNING: findings that do not reproduce are defects in this tool,");
        println!("   not discoveries about the target. Investigate before anything else.");
    }
    println!();

    // Which operations are involved at all.
    let mut by_label: BTreeMap<&str, usize> = BTreeMap::new();
    for t in triaged {
        *by_label.entry(t.label).or_default() += 1;
    }
    println!("2. by operation");
    for (label, count) in &by_label {
        println!("   {label:<8} {count}");
    }
    println!();

    // Rung 2 — is the error the size floating-point arithmetic predicts?
    println!("3. observed error against what the arithmetic predicts");
    println!(
        "   {:<10} {:>12} {:>12} {:>12} {:>8}",
        "op", "max |arg|", "observed", "predicted", "ratio"
    );

    let mut worst_ratio: f64 = 0.0;
    for label in by_label.keys() {
        let group: Vec<&Triaged> = triaged.iter().filter(|t| t.label == *label).collect();

        // The case with the largest observed error is the one that decides whether the
        // prediction holds.
        let worst = group
            .iter()
            .max_by(|a, b| {
                a.observed_relative_error
                    .total_cmp(&b.observed_relative_error)
            })
            .expect("group is non-empty");

        let ratio = if worst.predicted_relative_error > 0.0 {
            worst.observed_relative_error / worst.predicted_relative_error
        } else {
            f64::INFINITY
        };
        worst_ratio = worst_ratio.max(ratio);

        println!(
            "   {:<10} {:>12.3e} {:>12.3e} {:>12.3e} {:>8.2}",
            label,
            worst.largest_argument,
            worst.observed_relative_error,
            worst.predicted_relative_error,
            ratio
        );
    }
    println!();

    println!("4. provisional classification");
    if worst_ratio <= 1.0 {
        println!("   Every error is at or below what floating-point arithmetic predicts");
        println!("   for the operation at that magnitude. These are NOT bugs in either");
        println!("   library — they are the expected consequence of an approximation whose");
        println!("   accuracy degrades with argument size, and the tolerance policy is what");
        println!("   needs to account for it.");
        println!();
        println!("   Category: FLOAT NOISE / our tolerance is wrong, not the target.");
    } else {
        println!(
            "   At least one error EXCEEDS the predicted bound (worst ratio {worst_ratio:.2})."
        );
        println!("   That is not explained by rounding, and is worth investigating as a");
        println!("   candidate finding rather than tuned away.");
        println!();
        println!("   Category: NEEDS INVESTIGATION.");
    }
}
