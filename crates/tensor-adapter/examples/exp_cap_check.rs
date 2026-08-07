//! Does the capped `exp` bound still cover what the backends actually do?
//!
//! Capping the condition-number term at `exp`'s saturation point tightened the bound from
//! 2.4e23 to 2.5e-5. A test encoding a *measured* worst case of 1.633e-4 then failed — so
//! either the cap is wrong, or that measurement came from cases the cap does not govern.
//! Guessing between those would be exactly the error the project exists to avoid.
use diff_fuzzer_core::{Generator, Implementation, SeededRng, TolerancePolicy};
use tensor_adapter::ops::Bounds;
use tensor_adapter::{TensorOpGenerator, TensorTolerancePolicy, flex, libtorch};

fn main() {
    let bounds = Bounds {
        max_rank: 3,
        max_dim: 64,
        magnitude: 1000.0,
        special_value_rate: 0.125,
        restrict_domains: false,
        ..Bounds::default()
    };
    let generator = TensorOpGenerator::new(bounds);
    let policy = TensorTolerancePolicy;
    let (cpu, torch) = (flex(), libtorch());

    let mut checked = 0usize;
    let mut violations = 0usize;
    let mut worst_ratio = 0.0f64;
    let mut worst_desc = String::new();

    for seed in 0..40_000u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if case.name() != "exp" {
            continue;
        }
        let (Ok(a), Ok(b)) = (cpu.run(&case), torch.run(&case)) else {
            continue;
        };
        let (Ok(a), Ok(b)) = (a.to_vec::<f32>(), b.to_vec::<f32>()) else {
            continue;
        };
        let t = policy.tolerance_for(&case, ("burn-flex", "burn-tch"));
        checked += 1;

        for (x, y) in a.iter().zip(&b) {
            // Only finite pairs: non-finite results are settled structurally, never by a
            // tolerance, so they say nothing about whether this bound is wide enough.
            if !x.is_finite() || !y.is_finite() || x == y {
                continue;
            }
            // **The whole rule, not just the relative half.** The agreement test is
            // `|a - b| <= atol + rtol * max(|a|, |b|)`; measuring against `rtol` alone
            // ignores the floor that exists precisely for values too small for a relative
            // bound to mean anything — which is the region these violations live in.
            let scale = (x.abs().max(y.abs())) as f64;
            let gap = (*x as f64 - *y as f64).abs();
            let allowed = t.atol + t.rtol * scale;
            let ratio = gap / allowed.max(f64::MIN_POSITIVE);
            if ratio > worst_ratio {
                worst_ratio = ratio;
                worst_desc =
                    format!("gap {gap:.3e} against allowance {allowed:.3e} (values ~{scale:.3e})");
            }
            if gap > allowed {
                violations += 1;
            }
        }
    }

    println!("{checked} exp cases compared, {violations} element(s) exceeded the bound");
    println!("worst observed / bound = {worst_ratio:.3}  ({worst_desc})");
    println!(
        "\n{}",
        if violations == 0 {
            "The capped bound covers everything measured. The old 1.633e-4 figure did not \
             come from a case this bound governs."
        } else {
            "THE CAP IS TOO TIGHT — it would produce false positives. Revert or raise it."
        }
    );
}
