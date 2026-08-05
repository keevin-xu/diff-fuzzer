//! Does `burn-flex` reproduce the burn#5284 overflow divergence?
//!
//! The question that decides whether `flex` can replace `ndarray`. `ndarray`'s `inf` comes
//! from `matrixmultiply`'s NEON fused multiply-add — if `flex` fuses the same way, the
//! finding survives a swap. If it does not, the divergence is specific to a backend burn
//! has already dropped from its first-party CPU list, which is worth knowing before the
//! issue is leaned on further.

use burn::backend::{Flex, LibTorch, NdArray, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

/// The filed minimal case: `[1,2] x [2,1]`, both products overflowing, exact answer `0`.
fn overflow_case<B: Backend>() -> Vec<f32> {
    let device = Default::default();
    let a = Tensor::<B, 2>::from_data(TensorData::new(vec![1e30f32, -1e30], [1, 2]), &device);
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![1e30f32, 1e30], [2, 1]), &device);
    a.matmul(b).into_data().to_vec::<f32>().expect("f32 read")
}

/// The larger case, where only the trailing corner disagrees.
fn corner_case<B: Backend>() -> (usize, usize) {
    let device = Default::default();
    let (m, k, n) = (14, 4, 27);
    let lhs: Vec<f32> = (0..m * k)
        .map(|i| if (i % k) % 2 == 0 { 1e30 } else { -1e30 })
        .collect();
    let a = Tensor::<B, 2>::from_data(TensorData::new(lhs, [m, k]), &device);
    let b = Tensor::<B, 2>::from_data(TensorData::new(vec![1e30f32; k * n], [k, n]), &device);
    let out = a.matmul(b).into_data().to_vec::<f32>().expect("f32 read");
    (
        out.iter().filter(|v| v.is_infinite()).count(),
        out.iter().filter(|v| v.is_nan()).count(),
    )
}

fn main() {
    println!("burn#5284 minimal case — [1,2] x [2,1], exact answer 0\n");
    println!("  ndarray  {:?}", overflow_case::<NdArray<f32>>());
    println!("  tch      {:?}", overflow_case::<LibTorch<f32>>());
    println!(
        "  flex     {:?}   <-- the question",
        overflow_case::<Flex<f32>>()
    );

    println!("  wgpu     {:?}", overflow_case::<Wgpu<f32, i32>>());

    println!("\nlarger case — [14,4] x [4,27], 378 elements (inf, NaN):\n");
    let nd = corner_case::<NdArray<f32>>();
    let tc = corner_case::<LibTorch<f32>>();
    let fx = corner_case::<Flex<f32>>();
    let gp = corner_case::<Wgpu<f32, i32>>();
    println!("  ndarray  {nd:?}");
    println!("  tch      {tc:?}");
    println!("  flex     {fx:?}");
    println!("  wgpu     {gp:?}");

    // Which backends agree with which, on the case that would survive dropping ndarray.
    println!("\nagreement groups on the larger case:");
    let named = [("ndarray", nd), ("tch", tc), ("flex", fx), ("wgpu", gp)];
    for (a, x) in &named {
        let agrees: Vec<&str> = named
            .iter()
            .filter(|(b, y)| b != a && y == x)
            .map(|(b, _)| *b)
            .collect();
        println!(
            "  {a:<8} {x:?}  agrees with: {}",
            if agrees.is_empty() {
                "nobody — outlier".to_string()
            } else {
                agrees.join(", ")
            }
        );
    }
}
