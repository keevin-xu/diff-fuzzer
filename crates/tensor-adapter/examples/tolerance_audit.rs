//! What tolerance does each operation actually get, on cases the fuzzer produces?
//!
//! A bound derived per case scales with that case's values. With `1e30` in the special-value
//! table, a bound proportional to the largest magnitude can become astronomically loose —
//! at which point the operation cannot diverge no matter what the backends do.
use diff_fuzzer_core::{Generator, SeededRng, TolerancePolicy};
use std::collections::BTreeMap;
use tensor_adapter::ops::Bounds;
use tensor_adapter::{TensorOpGenerator, TensorTolerancePolicy};

fn main() {
    let bounds = Bounds {
        max_rank: 3,
        max_dim: 64,
        magnitude: 10.0,
        special_value_rate: 0.125,
        restrict_domains: false,
        ..Bounds::default()
    };
    let generator = TensorOpGenerator::new(bounds);
    let policy = TensorTolerancePolicy;

    // Worst (largest) tolerance seen per operation, and how often it is hopeless.
    let mut worst: BTreeMap<&str, (f64, f64)> = BTreeMap::new();
    let mut hopeless: BTreeMap<&str, usize> = BTreeMap::new();
    let mut total: BTreeMap<&str, usize> = BTreeMap::new();

    for seed in 0..20_000u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let t = policy.tolerance_for(&case, ("burn-flex", "burn-wgpu"));
        let name = case.name();
        *total.entry(name).or_default() += 1;

        let entry = worst.entry(name).or_insert((0.0, 0.0));
        entry.0 = entry.0.max(t.rtol);
        entry.1 = entry.1.max(t.atol);

        // A relative bound above 1 permits a 100% error: nothing can fail it.
        if t.rtol >= 1.0 || t.atol >= 1e30 {
            *hopeless.entry(name).or_default() += 1;
        }
    }

    println!(
        "{:<10} {:>7} {:>12} {:>12} {:>10}",
        "op", "cases", "worst rtol", "worst atol", "hopeless"
    );
    for (name, count) in &total {
        let (r, a) = worst[name];
        let h = hopeless.get(name).copied().unwrap_or(0);
        println!(
            "{name:<10} {count:>7} {r:>12.2e} {a:>12.2e} {:>9.0}%",
            100.0 * h as f64 / *count as f64
        );
    }
}
