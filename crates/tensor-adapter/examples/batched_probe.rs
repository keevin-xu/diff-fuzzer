//! Is the `matmul` overflow divergence confined to rank 2, or does batched matmul show it?
//!
//! Written because the seeded campaign reported 68 cases of `matmul/rank3/undefined` and
//! the obvious conclusion — "batched matmul has the same problem" — **failed its first
//! test**. A minimal batched case (two batches of `[1,2] × [2,1]`) produces `inf` on
//! *both* backends. Whatever happens at rank 3 is not simply the rank-2 story repeated
//! per batch.
//!
//! So this walks batch and matrix size upward to find where the backends start to
//! disagree. **The point is to locate the boundary, not to assume there is none.**
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example batched_probe
//! ```

use burn::backend::{LibTorch, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use diff_fuzzer_core::{Generator, Implementation, Normalizer, SeededRng};
use tensor_adapter::{
    Bounds, TensorNormalizer, TensorOpGenerator, libtorch as torch_impl, ndarray as nd_impl,
};

/// Run one batched matmul whose products overflow, on both backends.
///
/// `batch` batches of `[1, k] × [k, 1]`, every value `±1e30`, so every product overflows
/// and each dot product's exact answer is zero.
fn probe<B: Backend>(batch: usize, k: usize) -> Vec<f32> {
    let device = Default::default();

    let lhs: Vec<f32> = (0..batch * k)
        .map(|i| if i % 2 == 0 { 1e30 } else { -1e30 })
        .collect();
    let rhs: Vec<f32> = vec![1e30; batch * k];

    let a = Tensor::<B, 3>::from_data(TensorData::new(lhs, [batch, 1, k]), &device);
    let b = Tensor::<B, 3>::from_data(TensorData::new(rhs, [batch, k, 1]), &device);

    a.matmul(b).into_data().to_vec::<f32>().expect("f32 read")
}

/// How many values are infinite and how many are not-a-number.
fn describe(values: &[f32]) -> String {
    let infs = values.iter().filter(|v| v.is_infinite()).count();
    let nans = values.iter().filter(|v| v.is_nan()).count();
    format!("{infs} inf, {nans} NaN of {}", values.len())
}

/// Whether two result vectors disagree, treating `NaN` as distinct from any number.
fn disagree(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(a, b)| a.is_nan() != b.is_nan() || (!a.is_nan() && a != b))
        .count()
}

fn main() {
    println!("batched matmul, every product overflowing\n");
    println!(
        "{:>5} {:>4}  {:<24} {:<24}",
        "batch", "k", "ndarray", "libtorch"
    );

    for (batch, k) in [
        (1, 2),
        (2, 2),
        (2, 8),
        (8, 8),
        (8, 32),
        (36, 27),
        (36, 64),
        (64, 64),
    ] {
        let nd = probe::<NdArray<f32>>(batch, k);
        let tch = probe::<LibTorch<f32>>(batch, k);
        let differing = disagree(&nd, &tch);

        println!(
            "{batch:>5} {k:>4}  {:<24} {:<24} {}",
            describe(&nd),
            describe(&tch),
            if differing > 0 {
                format!("◄ {differing} differ")
            } else {
                String::new()
            }
        );
    }

    // The case the campaign actually found, rebuilt from its seed. Determinism is the
    // guarantee being leaned on here: the same seed and bounds must rebuild the same case.
    println!("\nthe campaign's own rank-3 case, rebuilt from seed 721:");
    let generator = TensorOpGenerator::new(Bounds {
        max_rank: 3,
        max_dim: 64,
        magnitude: 1000.0,
        ..Bounds::default()
    });
    let case = generator.generate(&mut SeededRng::from_seed(721));

    match (nd_impl().run(&case), torch_impl().run(&case)) {
        (Ok(left), Ok(right)) => {
            let left = TensorNormalizer.normalize(left);
            let right = TensorNormalizer.normalize(right);
            println!("  {} — shape {:?}", case.name(), left.shape);
            println!(
                "  {} of {} elements differ",
                disagree(&left.values, &right.values),
                left.values.len()
            );
        }
        _ => println!("  could not run"),
    }
}
