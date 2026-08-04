//! **Step 7.1** — does the GPU work at all, before any of our code depends on it?
//!
//! # Why this exists as its own step
//!
//! A missing Metal device, a shader that fails to compile, an async readback that returns
//! before the GPU has finished, and a plain build error **all look identical** once they
//! are three layers down inside an adapter. This runs `burn-wgpu` directly, with nothing
//! of ours in the way, so a failure here points at exactly one thing.
//!
//! Deliberately does **not** implement `Implementation`, run a differential, or touch
//! `backends.rs`. That is step 7.2 — and keeping the two apart is what makes 7.2's real
//! question, *did `diff-fuzzer-core` have to change?*, mean anything.
//!
//! # What it checks, in order of how badly each would hurt later
//!
//! 1. **A device exists** and a tensor can be built on it.
//! 2. **Arithmetic is correct** on values with exact `f32` representations — if `2 + 2`
//!    is wrong, nothing downstream is worth debugging.
//! 3. **Readback is synchronised.** `into_data()` must not return before the GPU has
//!    finished. Getting this wrong yields *plausible stale numbers*, not an error, which
//!    is the worst failure mode available.
//! 4. **Every operation in the case vocabulary runs.** GPUs commonly support fewer ops
//!    than CPU kernels; each gap becomes a `SkipReason::CouldNotRun` at 7.2, and it is
//!    much cheaper to learn which ones now than to discover them mid-campaign.
//! 5. **Repeated runs of one input agree.** Not assumed — **this is the assumption the
//!    whole project rests on** (seeded replay, `repro.rs`, `still_diverges()`), and GPU
//!    reductions using atomics are the documented way it breaks.
//!
//! # WHAT IT FOUND (measured 2026-08-04)
//!
//! Everything works — device, arithmetic, readback synchronisation, and **every operation
//! in the case vocabulary**, including rank-4 and matmul. No `SkipReason::CouldNotRun`
//! wiring is needed for op gaps, which was the expected cost and is not there.
//!
//! **But GPU reductions are not deterministic.** Summing 4096 values returns one of two
//! results — `838656.0` or `838656.06`, exactly 1 ULP apart — in an unpredictable pattern,
//! and re-running the whole program gives a different pattern again. Two accumulation
//! orders, selected non-deterministically.
//!
//! **This was initially mistaken for warm-up** (autotuning settling on a kernel after the
//! first call), because in the first measurement only run 1 differed. Repeating the whole
//! program refuted it: the variation continues indefinitely and its position moves. The
//! lesson is the same one this phase keeps teaching — *five samples is not a measurement*.
//!
//! ## Why this matters more than a tolerance question
//!
//! **The whole project assumes that running the same input twice gives the same answer.**
//! Seeded replay, `repro.rs`, and `still_diverges()` in triage all rest on it, and a
//! finding that fails to reproduce is currently interpreted as *a defect in this tool*.
//!
//! The impact is real but bounded: a divergence comfortably past tolerance still
//! reproduces every time, and only findings sitting near the threshold will flake. That
//! argues for recording **"reproduced N of M attempts"** rather than a boolean — a change
//! to how reproduction is reported, not to what counts as a divergence. Decide at 7.4.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example wgpu_check
//! ```

use burn::backend::Wgpu;
use burn::tensor::{Tensor, TensorData};

type Gpu = Wgpu<f32, i32>;

fn main() {
    println!("step 7.1 — burn-wgpu standalone check\n");

    let device = Default::default();
    println!("device: {device:?}");

    // 1 & 2. Values with exact f32 representations, so any difference is a real fault
    // rather than a rounding question. Tolerance does not enter here on purpose.
    let a = Tensor::<Gpu, 1>::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [4]), &device);
    let doubled = (a.clone() + a.clone()).into_data().to_vec::<f32>();
    report("add (exact values)", &doubled, &[2.0, 4.0, 6.0, 8.0]);

    let squared = (a.clone() * a.clone()).into_data().to_vec::<f32>();
    report("mul (exact values)", &squared, &[1.0, 4.0, 9.0, 16.0]);

    // 3. Readback synchronisation. A chain long enough that an unsynchronised read would
    // land on an intermediate value rather than the final one.
    let chained = a
        .clone()
        .mul_scalar(2.0)
        .add_scalar(1.0)
        .mul_scalar(3.0)
        .into_data()
        .to_vec::<f32>();
    report(
        "chained ops (readback sync)",
        &chained,
        &[9.0, 15.0, 21.0, 27.0],
    );

    // 4. Which of the case vocabulary the GPU will actually run. Each failure here is a
    // `SkipReason::CouldNotRun` to wire up at 7.2, not a bug.
    println!("\noperation support:");
    supported("neg", || a.clone().neg().into_data().to_vec::<f32>());
    supported("abs", || a.clone().abs().into_data().to_vec::<f32>());
    supported("exp", || a.clone().exp().into_data().to_vec::<f32>());
    supported("sqrt", || a.clone().sqrt().into_data().to_vec::<f32>());
    supported("sum", || a.clone().sum().into_data().to_vec::<f32>());
    supported("matmul", || {
        let m = Tensor::<Gpu, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        m.clone().matmul(m).into_data().to_vec::<f32>()
    });
    supported("rank-4 tensor", || {
        Tensor::<Gpu, 4>::from_data(TensorData::new(vec![1.0f32; 16], [2, 2, 2, 2]), &device)
            .neg()
            .into_data()
            .to_vec::<f32>()
    });

    // 5. Determinism. A reduction over many terms is where atomics would show up, so the
    // sum is over enough values to make a differing accumulation order visible.
    println!("\ndeterminism across repeated runs:");
    let wide = Tensor::<Gpu, 1>::from_data(
        TensorData::new(
            (0..4096).map(|i| (i as f32) * 0.1).collect::<Vec<f32>>(),
            [4096],
        ),
        &device,
    );
    let runs: Vec<f32> = (0..20)
        .map(|_| {
            wide.clone()
                .sum()
                .into_data()
                .to_vec::<f32>()
                .expect("readable")[0]
        })
        .collect();

    let all_identical = runs.windows(2).all(|p| p[0].to_bits() == p[1].to_bits());
    // **The distinction that decides how much trouble this is.** If only the opening runs
    // differ and everything afterwards agrees, this is warm-up — `burn-cubecl` autotunes,
    // benchmarking candidate kernels and settling on one, so the first call can run a
    // different kernel from every later call. That is a *fixable* problem: warm the
    // backend once at startup, as the fuzz harness already does for setup cost.
    //
    // If results keep varying after warm-up, it is genuine non-determinism from atomic
    // accumulation, and no amount of warming helps.
    let settled_identical = runs[1..]
        .windows(2)
        .all(|p| p[0].to_bits() == p[1].to_bits());

    println!("  sum of 4096 values, first 5 of 20 runs: {:?}", &runs[..5]);
    let distinct: std::collections::BTreeSet<u32> = runs.iter().map(|v| v.to_bits()).collect();
    println!("  distinct values across all 20 runs: {}", distinct.len());

    if all_identical {
        println!("  ✓ bit-identical throughout — seeded replay holds for this operation");
    } else if settled_identical {
        println!("  ⚠ the FIRST run differs; every run after it agrees.");
        println!("    This is warm-up, not non-determinism — consistent with `burn-cubecl`");
        println!("    autotuning and settling on a kernel after the first call.");
        println!("    **Fixable**: warm the backend once before judging anything, or a");
        println!("    finding's first run would be compared against a different kernel");
        println!("    from the one that reproduces it. Decide explicitly at 7.4.");
    } else {
        println!("  ⚠ results keep varying after warm-up — genuine non-determinism.");
        println!("    Not a tolerance problem: a recorded finding may fail to reproduce for");
        println!("    reasons unrelated to the target, and `repro.rs` currently reads that");
        println!("    as a defect in this tool. See 7.4.");
    }
}

/// Compare against values chosen to be exactly representable, so equality is the right test.
fn report(label: &str, got: &Result<Vec<f32>, impl std::fmt::Debug>, want: &[f32]) {
    match got {
        Ok(values) if values == want => println!("  ✓ {label:<28} {values:?}"),
        Ok(values) => println!("  ✗ {label:<28} got {values:?}, want {want:?}"),
        Err(error) => println!("  ✗ {label:<28} failed to read back: {error:?}"),
    }
}

/// Whether an operation runs at all. A refusal is information, not a failure.
fn supported<E: std::fmt::Debug>(label: &str, run: impl FnOnce() -> Result<Vec<f32>, E>) {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
        Ok(Ok(_)) => println!("  ✓ {label}"),
        Ok(Err(error)) => println!("  ✗ {label} — readback failed: {error:?}"),
        Err(_) => println!("  ✗ {label} — panicked (unsupported on this backend?)"),
    }
}
