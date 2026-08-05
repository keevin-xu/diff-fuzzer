//! Throwaway smoke test: run the *same* `matmul` on two different backends and print
//! both results.
//!
//! This is the whole project in miniature. One operation, two independent
//! implementations of the arithmetic — pure Rust on one side, PyTorch's C++ kernels
//! on the other — and a comparison of what they produced. Everything built later
//! (generating operations instead of hardcoding one, comparing within a tolerance
//! instead of exactly, shrinking failures) is an elaboration of these few lines.
//!
//! Run with: `cargo run -p tensor-adapter --example hello_tensor`

use burn::backend::{Flex, LibTorch};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Multiply a fixed 2x3 matrix by a fixed 3x2 matrix, returning the result as a flat
/// list of numbers.
///
/// The interesting part is the signature. `<B: Backend>` makes this function
/// *generic*: `B` is a type parameter, and `Backend` is a trait bound saying "`B` can
/// be anything, as long as it provides the operations the `Backend` trait requires."
/// The body never mentions a specific backend, so one piece of code runs on both.
///
/// That is the seam the entire project hangs on. Differential testing needs the same
/// input to reach two implementations *unchanged* — if each backend needed its own
/// hand-written version of the operation, any difference in results might just be a
/// difference in our two copies of the code. Here there is only one copy.
///
/// The compiler generates a separate specialised machine-code copy per backend used
/// (*monomorphisation*), so this abstraction costs nothing at runtime.
fn matmul_2x3_by_3x2<B: Backend>(device: &B::Device) -> Vec<f32> {
    // `Tensor::<B, 2>` — the 2 is the tensor's rank (a matrix), and it is part of the
    // type, checked at compile time. Rank mismatches are compile errors in burn,
    // where PyTorch would only complain at runtime.
    let a = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device);
    let b = Tensor::<B, 2>::from_data([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device);

    // matmul's constraint: the inner dimensions must agree. 2x3 by 3x2 shares the 3,
    // giving a 2x2 result. Generating arguments that satisfy constraints like this
    // one — rather than generating randomly and throwing away the failures — is the
    // job of the real generator.
    //
    // `a` and `b` are *moved* into `matmul`, not borrowed, so neither can be used
    // afterwards. Every value in Rust has exactly one owner, and passing it without
    // `&` transfers ownership.
    let c = a.matmul(b);

    // Pull the numbers out of the backend's internal representation into an ordinary
    // Rust `Vec`. This is normalisation in embryo: two backends store tensors in
    // completely different ways, so results have to be brought into a common,
    // comparable form before they can be compared at all.
    c.into_data()
        .to_vec::<f32>()
        .expect("result is a flat f32 tensor")
}

fn main() {
    // Each backend names its own device type, and both default to "the only device
    // available" here: the CPU.
    let cpu = matmul_2x3_by_3x2::<Flex<f32>>(&Default::default());
    let torch = matmul_2x3_by_3x2::<LibTorch<f32>>(&Default::default());

    println!("same matmul, two backends  (a 2x3 · b 3x2)");
    println!("  flex (pure Rust CPU) : {cpu:?}");
    println!("  libtorch (PyTorch C++)  : {torch:?}");
    println!("  expected                : [58.0, 64.0, 139.0, 154.0]");

    // Exact equality is deliberately the wrong tool for this job, and it is worth
    // seeing it work here so that it is obvious later why it stops working. These
    // inputs are small integers held exactly in f32, so both backends land on
    // bit-identical answers. Real generated inputs will not: two correct
    // implementations routinely differ in the last bits, because floating-point
    // addition is not associative and the two backends sum in different orders.
    // Comparing within a tolerance instead of exactly is what the oracle adds next.
    println!("  bit-identical           : {}", cpu == torch);
}
