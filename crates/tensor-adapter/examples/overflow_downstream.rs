//! Does the `inf`-versus-`NaN` divergence stay confined, or spread?
//!
//! Written to check a claim before it went into a public issue, and it is kept because the
//! claim turned out to be wrong in a way that mattered. The draft asserted that
//! `clamp(0, 1)` maps `NaN` to `0.0`; it does not — `NaN` propagates. Rust's `f32::max`
//! swallows `NaN` while `f32::clamp` propagates it, and the two had been conflated.
//!
//! **Measure, do not reason, about floating-point edge cases.** A claim in an issue that a
//! maintainer can disprove in thirty seconds costs more credibility than the issue buys.
//!
//! What the measurement shows is stronger than the guess: after `recip`, `clamp`, or
//! `sigmoid`, the `ndarray` result is an ordinary finite number with nothing to flag,
//! while the `tch` result is still `NaN`. The divergence does not stay in the overflowing
//! element — one backend launders it into a plausible value.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example overflow_downstream
//! ```

use burn::backend::{Flex, LibTorch};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Run the overflowing matmul, then push its result through common downstream operations.
///
/// Generic over `B: Backend` — the same source runs on both backends, so any difference in
/// the output is the backend's, not the test's. This is the same trick the adapter uses.
fn probe<B: Backend>(name: &str) {
    let device = Default::default();

    // [1e30, -1e30] · [1e30, 1e30]ᵀ — both products overflow f32; the exact answer is 0.
    let a = Tensor::<B, 2>::from_data([[1e30, -1e30]], &device);
    let b = Tensor::<B, 2>::from_data([[1e30], [1e30]], &device);
    let out = a.matmul(b);

    // `clone()` because burn's tensor operations take `self` by value: each call consumes
    // the tensor, so reusing one means handing over a fresh handle each time.
    let raw = values(out.clone());
    let clamped = values(out.clone().clamp(0.0, 1.0));
    let relu = values(burn::tensor::activation::relu(out.clone()));
    let recip = values(out.clone().recip());
    let sigmoid = values(burn::tensor::activation::sigmoid(out.clone()));
    let positive = out
        .greater_elem(0.0)
        .into_data()
        .to_vec::<bool>()
        .expect("bool read");

    println!("{name}");
    println!("  raw           {raw:?}");
    println!("  clamp(0, 1)   {clamped:?}");
    println!("  relu          {relu:?}");
    println!("  recip         {recip:?}");
    println!("  sigmoid       {sigmoid:?}");
    println!("  > 0.0         {positive:?}");
}

fn values<B: Backend>(tensor: Tensor<B, 2>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 read")
}

fn main() {
    probe::<Flex<f32>>("flex");
    println!();
    probe::<LibTorch<f32>>("libtorch");

    println!();
    println!("the divergence does not stay put: after recip, clamp, or sigmoid one backend");
    println!("holds an ordinary finite value and the other still holds NaN.");
}
