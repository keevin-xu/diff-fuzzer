//! How *far* apart do the two backends actually get?
//!
//! The survey counts how often results are not bit-identical, which lumps a
//! one-bit rounding difference together with a catastrophically wrong answer. This
//! measures the size of the difference instead, per operation, so a tolerance can be
//! chosen by reading data rather than by turning a dial until the complaints stop.
//!
//! The distinction to watch for: a relative difference around `1e-7` is a single
//! rounding step in `f32` and means nothing. A difference around `1e-2` is not
//! rounding, and no tolerance should hide it — that would be a finding.
//!
//! Run with: `cargo run --release -p tensor-adapter --example error_distribution`

use diff_fuzzer_core::{Generator, Implementation, Normalizer, SeededRng, Tolerance, compare};
use std::collections::BTreeMap;
use tensor_adapter::{TensorNormalizer, TensorOpGenerator, libtorch, ndarray};

const CASES: u64 = 20_000;

/// Candidate thresholds to score against the measured data.
///
/// The first is what a single `f32` rounding step costs; the last two are the
/// established defaults from numpy and torch, included so our own choice can be
/// compared against what the field already does.
const CANDIDATES: [(&str, f64, f64); 5] = [
    ("exact", 0.0, 0.0),
    ("1 ulp (1.2e-7, 0)", 1.2e-7, 0.0),
    ("1e-6, 1e-9", 1e-6, 1e-9),
    ("numpy (1e-5, 1e-8)", 1e-5, 1e-8),
    ("torch (1.3e-6, 1e-5)", 1.3e-6, 1e-5),
];

#[derive(Default)]
struct Errors {
    /// Largest relative error seen in each case.
    relative: Vec<f64>,
    /// Largest absolute error seen in each case.
    absolute: Vec<f64>,
    /// Seed and relative error of the single worst case.
    worst: Option<(u64, f64)>,
    /// How many cases each candidate threshold would still flag.
    flagged: [usize; CANDIDATES.len()],
}

/// Value at a percentile of a sorted list.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

fn main() {
    let generator = TensorOpGenerator::default();
    let (cpu, torch) = (ndarray(), libtorch());
    let mut per_operation: BTreeMap<&str, Errors> = BTreeMap::new();

    for seed in 0..CASES {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        let left = TensorNormalizer.normalize(cpu.run(&case).expect("valid case"));
        let right = TensorNormalizer.normalize(torch.run(&case).expect("valid case"));
        assert_eq!(
            left.shape, right.shape,
            "shapes must match to compare values"
        );

        // Exact tolerance, because the maxima are reported regardless of whether
        // anything exceeded the threshold — what is wanted here is the magnitude, not a
        // verdict.
        let measured = compare(&left.values, &right.values, Tolerance::EXACT);

        let entry = per_operation.entry(case.name()).or_default();
        entry.relative.push(measured.max_relative_error);
        entry.absolute.push(measured.max_absolute_error);

        if entry
            .worst
            .is_none_or(|(_, error)| measured.max_relative_error > error)
        {
            entry.worst = Some((seed, measured.max_relative_error));
        }

        for (index, (_, rtol, atol)) in CANDIDATES.iter().enumerate() {
            let tolerance = Tolerance::new(*rtol, *atol);
            if !compare(&left.values, &right.values, tolerance).agrees() {
                entry.flagged[index] += 1;
            }
        }
    }

    println!("{CASES} cases, burn-ndarray vs burn-tch\n");
    println!("relative error per operation (worst element of each case)\n");
    println!(
        "  {:<8} {:>6} {:>11} {:>11} {:>11} {:>11}",
        "op", "cases", "median", "p99", "max", "max abs"
    );

    for (name, errors) in &mut per_operation {
        errors.relative.sort_by(|a, b| a.total_cmp(b));
        errors.absolute.sort_by(|a, b| a.total_cmp(b));

        println!(
            "  {:<8} {:>6} {:>11.2e} {:>11.2e} {:>11.2e} {:>11.2e}",
            name,
            errors.relative.len(),
            percentile(&errors.relative, 0.50),
            percentile(&errors.relative, 0.99),
            percentile(&errors.relative, 1.0),
            percentile(&errors.absolute, 1.0),
        );
    }

    println!("\ncases still flagged by each candidate threshold\n");
    print!("  {:<8}", "op");
    for (label, _, _) in CANDIDATES {
        print!(" {label:>22}");
    }
    println!();

    for (name, errors) in &per_operation {
        print!("  {name:<8}");
        for count in errors.flagged {
            let share = 100.0 * count as f64 / errors.relative.len() as f64;
            print!(" {:>15} ({:>4.1}%)", count, share);
        }
        println!();
    }

    println!("\nworst case per operation (for inspection)\n");
    for (name, errors) in &per_operation {
        if let Some((seed, error)) = errors.worst {
            println!("  {name:<8} seed {seed:<8} relative error {error:.3e}");
        }
    }

    // One rounding step in f32, for reference against the numbers above.
    println!(
        "\n  f32 machine epsilon: {:.3e}   (half-ulp: {:.3e})",
        f32::EPSILON,
        f32::EPSILON / 2.0
    );
}
