//! Throwaway smoke test: run one `matmul` on the pure-Rust CPU backend and print it.
//!
//! The point is not the maths — it is proving that a tensor operation can be built
//! and executed at all, which is the foundation everything else sits on. The second
//! backend (libtorch) is added next, and then the same operation runs on both.
//!
//! Run with: `cargo run -p tensor-adapter --example hello_tensor`

use burn::backend::NdArray;
use burn::tensor::Tensor;

/// The backend under test: `burn`'s pure-Rust CPU implementation, using `f32`.
///
/// A `burn` "backend" is the code that actually performs the arithmetic. `burn`
/// itself is the API layer; each backend implements that API differently (here in
/// plain Rust, later by calling into libtorch's C++ kernels). Two backends running
/// the same operation and disagreeing is exactly the signal this project hunts for.
///
/// `type` here is a Rust *type alias* — a shorter name for an existing type, not a
/// new one. `Backend` and `NdArray<f32>` are interchangeable.
type Backend = NdArray<f32>;

fn main() {
    // Every tensor belongs to a device (which CPU/GPU it lives on). The ndarray
    // backend has exactly one, so the default is the only choice.
    let device = Default::default();

    // `Tensor::<Backend, 2>` — the 2 is the *rank* (number of dimensions), and it is
    // part of the type, checked at compile time. A rank-2 tensor is a matrix.
    // This is unusual and worth noting: shape errors that would be runtime crashes in
    // PyTorch are partly compile-time errors here.
    let a = Tensor::<Backend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let b = Tensor::<Backend, 2>::from_data([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], &device);

    // matmul's constraint: the inner dimensions must agree. `a` is 2x3 and `b` is
    // 3x2, so the shared 3 lines up and the result is 2x2. Generating argument shapes
    // that satisfy constraints like this one — rather than generating randomly and
    // discarding the failures — is what the real generator will do.
    //
    // `a` and `b` are *moved* into `matmul`, not borrowed: after this line neither can
    // be used again. Rust's ownership rules mean each value has exactly one owner, and
    // passing it without `&` hands that ownership over. `clone()` them if both the
    // inputs and the output are needed later.
    let c = a.matmul(b);

    println!("ndarray (CPU) backend");
    println!("  a (2x3) x b (3x2) = c {:?}", c.dims());
    println!("{c}");
    println!("  expected: [[58, 64], [139, 154]]");
}
