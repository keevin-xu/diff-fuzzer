//! **Step 7.4, part 1** — what does the GPU actually do differently?
//!
//! # Why measure before deciding
//!
//! The first three-backend run produced 640 divergences from 2,000 cases, entirely in the
//! operations that demand **bit-exact** agreement. The obvious response — loosen the
//! tolerance until they pass — is exactly the error the whole tolerance policy exists to
//! prevent. A threshold fitted to observed differences encodes today's wgpu as the
//! standard, and the test then passes forever regardless of truth.
//!
//! The method that survived PHASE-4 is: **derive from what the specification permits →
//! measure → check the derivation covers the measurement with margin → never move the
//! bound to the data.** This program is the measuring half. It answers *what is the GPU
//! doing*, not *what number would make the failures stop*.
//!
//! # What it measures, and why each matters
//!
//! 1. **Subnormal handling.** GPUs commonly flush values below `f32::MIN_POSITIVE` to
//!    zero. That is not a rounding difference — a subnormal becoming `0.0` is a *relative*
//!    error of 1.0, which no sane `rtol` absorbs, and it would need `atol` instead.
//! 2. **Fused multiply-add.** If the GPU fuses `a*b + c` where the CPU does not (or the
//!    reverse), results differ by up to half an ULP of the product — the same mechanism
//!    behind burn#5284, on new hardware.
//! 3. **Error size for correctly-rounded operations**, in **ULPs**. IEEE-754 requires
//!    `+ - * / sqrt` to be correctly rounded, so a conforming implementation differs by
//!    **zero**. Any non-zero measurement here is the GPU declining to conform, and the
//!    size tells us whether it is one rounding step or something larger.
//! 4. **Non-determinism spread for reductions**, also in ULPs — the width of the band a
//!    repeated identical run can land in.
//!
//! Reported in ULPs rather than absolute error because an ULP is scale-free: "one ULP" is
//! the same statement about accuracy at `1e-30` and at `1e30`, where "0.001 apart" is not.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example gpu_numerics
//! ```

use burn::tensor::TensorData;
use diff_fuzzer_core::{Implementation, SeededRng};
use rand::RngExt;
use tensor_adapter::{BinaryOp, TensorOp, TensorValue, UnaryOp, libtorch, ndarray, wgpu};

/// Distance between two floats counted in representable values between them.
///
/// **The unit that makes accuracy claims comparable across scales.** Adjacent floats are
/// one ULP apart everywhere, so "3 ULPs" means the same degree of wrongness at `1e-30` as
/// at `1e30`, which an absolute difference does not.
///
/// Uses the standard trick: for same-signed finite floats, the bit patterns ordered as
/// integers are adjacent exactly when the floats are, so subtracting them counts the gap.
fn ulps_apart(a: f32, b: f32) -> Option<u32> {
    if a == b {
        return Some(0);
    }
    if !a.is_finite() || !b.is_finite() {
        return None; // inf/NaN are not a distance apart, they are a different kind
    }
    if (a < 0.0) != (b < 0.0) {
        // Opposite signs: count outward from zero on each side.
        return Some(a.to_bits() & 0x7fff_ffff)
            .and_then(|x| (b.to_bits() & 0x7fff_ffff).checked_add(x));
    }
    let (x, y) = (a.to_bits(), b.to_bits());
    Some(x.abs_diff(y))
}

fn run(op: &TensorOp, backend: &str) -> Option<Vec<f32>> {
    let out: TensorData = match backend {
        "cpu" => ndarray().run(op).ok()?,
        "tch" => libtorch().run(op).ok()?,
        _ => wgpu().run(op).ok()?,
    };
    out.to_vec::<f32>().ok()
}

/// Worst ULP gap between two backends over a batch of random cases.
struct Spread {
    worst_ulps: u32,
    cases: usize,
    exact: usize,
    incomparable: usize,
}

impl Spread {
    fn line(&self, label: &str) -> String {
        format!(
            "  {label:<28} worst {:>4} ULP   exact on {}/{}{}",
            self.worst_ulps,
            self.exact,
            self.cases,
            if self.incomparable > 0 {
                format!("   ({} inf/NaN, not counted)", self.incomparable)
            } else {
                String::new()
            }
        )
    }
}

/// Measure one operation across many random inputs, CPU against GPU.
fn measure(label: &str, build: impl Fn(&mut SeededRng) -> TensorOp, cases: usize) -> Spread {
    let mut rng = SeededRng::from_seed(7);
    let mut spread = Spread {
        worst_ulps: 0,
        cases,
        exact: 0,
        incomparable: 0,
    };

    for _ in 0..cases {
        let op = build(&mut rng);
        let (Some(cpu), Some(gpu)) = (run(&op, "cpu"), run(&op, "gpu")) else {
            continue;
        };

        let mut worst_here = 0u32;
        let mut comparable = true;
        for (a, b) in cpu.iter().zip(&gpu) {
            match ulps_apart(*a, *b) {
                Some(ulps) => worst_here = worst_here.max(ulps),
                None => comparable = false,
            }
        }
        if !comparable {
            spread.incomparable += 1;
            continue;
        }
        if worst_here == 0 {
            spread.exact += 1;
        }
        spread.worst_ulps = spread.worst_ulps.max(worst_here);
    }

    println!("{}", spread.line(label));
    spread
}

fn tensor(rng: &mut SeededRng, count: usize, magnitude: f32) -> TensorValue {
    let data: Vec<f32> = (0..count)
        .map(|_| rng.random_range(-magnitude..magnitude))
        .collect();
    TensorValue::new(vec![count], data)
}

fn main() {
    println!("step 7.4 — measuring what the GPU does differently\n");

    // ---- 1. Subnormals ------------------------------------------------------------
    //
    // Not a rounding question. A subnormal flushed to zero is a *relative* error of 1.0,
    // so no `rtol` can absorb it and only an `atol` at the subnormal scale would.
    println!("subnormals (does the GPU flush them to zero?):");
    let subnormals = vec![1e-45f32, -1e-45, f32::MIN_POSITIVE, f32::MIN_POSITIVE / 2.0];
    let op = TensorOp::unary(
        UnaryOp::Neg,
        TensorValue::new(vec![subnormals.len()], subnormals.clone()),
    );
    match (run(&op, "cpu"), run(&op, "gpu")) {
        (Some(cpu), Some(gpu)) => {
            println!("    input  {subnormals:?}");
            println!("    cpu    {cpu:?}");
            println!("    gpu    {gpu:?}");
            let flushed = gpu.iter().filter(|v| **v == 0.0).count()
                - cpu.iter().filter(|v| **v == 0.0).count();
            println!(
                "    → {}",
                if flushed > 0 {
                    format!("FLUSHES {flushed} of {} to zero", subnormals.len())
                } else {
                    "preserves subnormals".to_string()
                }
            );
        }
        _ => println!("    could not run"),
    }

    // ---- 2. Correctly-rounded operations -------------------------------------------
    //
    // IEEE-754 *requires* these to be correctly rounded, so a conforming implementation
    // differs by exactly zero ULPs. Anything above zero is the GPU declining to conform,
    // and the size says whether it is one rounding step or a different algorithm.
    println!("\ncorrectly-rounded operations (IEEE-754 requires 0 ULP):");
    measure(
        "add",
        |rng| {
            TensorOp::binary(
                BinaryOp::Add,
                tensor(rng, 64, 100.0),
                tensor(rng, 64, 100.0),
            )
        },
        200,
    );
    measure(
        "sub",
        |rng| {
            TensorOp::binary(
                BinaryOp::Sub,
                tensor(rng, 64, 100.0),
                tensor(rng, 64, 100.0),
            )
        },
        200,
    );
    measure(
        "mul",
        |rng| {
            TensorOp::binary(
                BinaryOp::Mul,
                tensor(rng, 64, 100.0),
                tensor(rng, 64, 100.0),
            )
        },
        200,
    );
    measure(
        "div",
        |rng| {
            let rhs = TensorValue::new(
                vec![64],
                (0..64).map(|_| rng.random_range(1.0..100.0)).collect(),
            );
            TensorOp::binary(BinaryOp::Div, tensor(rng, 64, 100.0), rhs)
        },
        200,
    );
    measure(
        "sqrt",
        |rng| {
            let arg = TensorValue::new(
                vec![64],
                (0..64).map(|_| rng.random_range(0.0..100.0)).collect(),
            );
            TensorOp::unary(UnaryOp::Sqrt, arg)
        },
        200,
    );
    measure(
        "neg",
        |rng| TensorOp::unary(UnaryOp::Neg, tensor(rng, 64, 100.0)),
        200,
    );
    measure(
        "abs",
        |rng| TensorOp::unary(UnaryOp::Abs, tensor(rng, 64, 100.0)),
        200,
    );

    // ---- 3. Approximated and accumulating ------------------------------------------
    println!("\napproximated / accumulating (differences expected, magnitude is the question):");
    measure(
        "exp",
        |rng| TensorOp::unary(UnaryOp::Exp, tensor(rng, 64, 5.0)),
        200,
    );

    // ---- 4. Non-determinism ---------------------------------------------------------
    //
    // The width of the band a repeated identical run lands in. This is not a CPU-vs-GPU
    // difference at all — it is the GPU against itself, and no tolerance between
    // *implementations* addresses it.
    // Which *kind* of reduction. `wgpu_check` found `sum()` — a full reduction to a
    // scalar — returning one of two values. The generator never emits that: `ReduceOp::Sum`
    // becomes an *axis* reduction, which may well be a different kernel. Worth separating,
    // because "GPU reductions are non-deterministic" and "one reduction kernel we never
    // call is non-deterministic" are very different problems.
    println!("\nnon-determinism by reduction kind (GPU against itself, 10 runs each):");
    {
        use burn::tensor::{Tensor, TensorData as TD};
        type G = burn::backend::Wgpu<f32, i32>;
        let device = Default::default();
        let data: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.1).collect();

        let full: std::collections::BTreeSet<u32> = (0..10)
            .map(|_| {
                let t = Tensor::<G, 1>::from_data(TD::new(data.clone(), [4096]), &device);
                t.sum().into_data().to_vec::<f32>().expect("read")[0].to_bits()
            })
            .collect();

        let axis: std::collections::BTreeSet<u32> = (0..10)
            .map(|_| {
                let t = Tensor::<G, 2>::from_data(TD::new(data.clone(), [1, 4096]), &device);
                t.sum_dim(1).into_data().to_vec::<f32>().expect("read")[0].to_bits()
            })
            .collect();

        println!(
            "  sum() full reduction to scalar   {} distinct value(s){}",
            full.len(),
            if full.len() > 1 {
                "   ◄ NON-DETERMINISTIC"
            } else {
                ""
            }
        );
        println!(
            "  sum_dim() axis reduction         {} distinct value(s){}",
            axis.len(),
            if axis.len() > 1 {
                "   ◄ NON-DETERMINISTIC"
            } else {
                ""
            }
        );
        println!("  (the generator only ever emits the axis form)");
    }

    println!("\nnon-determinism: the generator's reduction, repeated (GPU against itself):");
    for count in [256usize, 4096, 65536] {
        let mut rng = SeededRng::from_seed(11);
        let arg = tensor(&mut rng, count, 100.0);
        let op = TensorOp::reduce(tensor_adapter::ReduceOp::Sum, arg, 0);

        let runs: Vec<f32> = (0..10).filter_map(|_| Some(run(&op, "gpu")?[0])).collect();
        let (lo, hi) = runs
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), v| {
                (l.min(*v), h.max(*v))
            });
        let distinct: std::collections::BTreeSet<u32> = runs.iter().map(|v| v.to_bits()).collect();
        println!(
            "  sum of {count:<6}  {} distinct value(s) over 10 runs, spread {} ULP",
            distinct.len(),
            ulps_apart(lo, hi).map_or("?".to_string(), |u| u.to_string())
        );
    }
}
