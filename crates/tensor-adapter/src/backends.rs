//! Executing a test case on a `burn` backend.
//!
//! There is **one** implementation of `run` here, generic over the backend, rather
//! than one per backend. That is not just brevity — it is a correctness requirement.
//! Differential testing concludes "these two systems disagree, so one is wrong", and
//! that conclusion only holds if the two really did receive the same input and perform
//! the same operation. Two hand-written copies of `run` could differ from each other,
//! and any disagreement would then be ambiguous: a bug in a backend, or a discrepancy
//! between our own two copies? With one copy the question cannot arise.
//!
//! Adding a third backend is consequently a type alias and a constructor.

use crate::input::{OpKind, TensorOp};
use burn::backend::{LibTorch, NdArray};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use diff_fuzzer_core::{Implementation, RunError};

/// Runs tensor cases on one `burn` backend.
///
/// `B` is the backend type. The struct holds its device — where tensors live and
/// where the arithmetic happens — created once and reused, rather than per case:
/// backend setup can be expensive, and a fuzzer's throughput is the number of cases
/// it gets through, so per-case setup would directly cost bugs found.
#[derive(Debug, Clone)]
pub struct BurnBackend<B: Backend> {
    name: &'static str,
    device: B::Device,
}

impl<B: Backend> BurnBackend<B> {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            device: B::Device::default(),
        }
    }

    /// Convert one of our backend-independent matrices into a tensor belonging to
    /// this backend.
    ///
    /// This is the boundary crossing: everything before it is plain Rust data that
    /// any backend could receive, everything after it belongs to one specific
    /// backend. Note the two sides of the differential cross this boundary through
    /// the same function.
    fn to_tensor(&self, matrix: &crate::input::Matrix) -> Tensor<B, 2> {
        let data = TensorData::new(matrix.data().to_vec(), matrix.shape());
        Tensor::<B, 2>::from_data(data, &self.device)
    }
}

impl<B: Backend> Implementation for BurnBackend<B> {
    type In = TensorOp;
    /// The backend's own tensor type. Two backends produce two *different* types
    /// here, which is precisely why they are not comparable yet — making them
    /// comparable is the normaliser's job.
    type Out = Tensor<B, 2>;

    fn name(&self) -> &str {
        self.name
    }

    fn run(&self, input: &TensorOp) -> Result<Self::Out, RunError> {
        let lhs = self.to_tensor(&input.lhs);
        let rhs = self.to_tensor(&input.rhs);

        // Matching on the operation rather than assuming one. With a single variant
        // the `match` is redundant today; it stops being redundant the moment a second
        // operation is added, and then the compiler will point at this spot and
        // require it to be handled.
        let out = match input.op {
            OpKind::Add => lhs.add(rhs),
        };

        Ok(out)
    }
}

/// The pure-Rust CPU backend.
pub type NdArrayBackend = BurnBackend<NdArray<f32>>;

/// The libtorch backend — the same arithmetic, performed by PyTorch's C++ kernels.
pub type LibTorchBackend = BurnBackend<LibTorch<f32>>;

/// Construct the CPU backend under test.
pub fn ndarray() -> NdArrayBackend {
    BurnBackend::new("burn-ndarray")
}

/// Construct the libtorch backend under test.
pub fn libtorch() -> LibTorchBackend {
    BurnBackend::new("burn-tch")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::FixedAddGenerator;
    use diff_fuzzer_core::{Generator, SeededRng};

    /// Pull the numbers out of a backend's tensor so a test can look at them. A
    /// throwaway version of what the normaliser will do properly.
    fn values<B: Backend>(t: Tensor<B, 2>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().expect("f32 tensor")
    }

    fn fixed_case() -> TensorOp {
        FixedAddGenerator.generate(&mut SeededRng::from_seed(0))
    }

    #[test]
    fn ndarray_backend_computes_the_addition() {
        let out = ndarray().run(&fixed_case()).expect("add is supported");
        assert_eq!(values(out), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn libtorch_backend_computes_the_addition() {
        let out = libtorch().run(&fixed_case()).expect("add is supported");
        assert_eq!(values(out), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn backends_identify_themselves_distinctly() {
        assert_ne!(ndarray().name(), libtorch().name());
    }

    /// Running the same case twice on one backend must give the same answer. If that
    /// failed, no comparison between two backends could mean anything, because a
    /// difference might just be noise from one of them.
    #[test]
    fn a_backend_is_self_consistent() {
        let backend = ndarray();
        let case = fixed_case();
        let first = values(backend.run(&case).unwrap());
        let second = values(backend.run(&case).unwrap());
        assert_eq!(first, second);
    }
}
