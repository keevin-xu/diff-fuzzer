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
//! This module is also the one place where a tensor's **rank stops being data and
//! becomes a type**. `burn` writes rank as a compile-time parameter (`Tensor<B, 2>`),
//! while a generated case carries its shape as an ordinary `Vec<usize>`. The `match`
//! in [`BurnBackend::run`] is where those meet, and confining it here is what lets
//! everything else treat rank as an ordinary value.

use crate::input::{BinaryOp, ReduceOp, TensorOp, TensorValue, UnaryOp};
use burn::backend::{Flex, LibTorch, Wgpu};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use diff_fuzzer_core::{Implementation, RunError};

/// The highest rank the generator may produce. Bounded because each supported rank is
/// a separate arm in the dispatch below — rank cannot be looped over, since each value
/// of it is a different type.
pub const MAX_RANK: usize = 4;

/// Runs tensor cases on one `burn` backend.
///
/// `B` is the backend type. The struct holds its device — where tensors live and where
/// the arithmetic happens — created once and reused, rather than per case: backend
/// setup can be expensive, and a fuzzer's throughput is the number of cases it gets
/// through, so per-case setup would directly cost bugs found.
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

    /// Convert one of our backend-independent values into a tensor belonging to this
    /// backend, at a rank known at compile time.
    ///
    /// This is the boundary crossing: everything before it is plain data any backend
    /// could receive, everything after belongs to one specific backend. Both sides of
    /// the differential cross it through this same function.
    fn tensor<const D: usize>(&self, value: &TensorValue) -> Tensor<B, D> {
        let data = TensorData::new(value.data().to_vec(), value.shape().to_vec());
        Tensor::<B, D>::from_data(data, &self.device)
    }

    fn unsupported(&self, reason: impl Into<String>) -> RunError {
        RunError::Unsupported {
            implementation: self.name.to_string(),
            reason: reason.into(),
        }
    }

    /// Execute a case whose arguments are all of rank `D`.
    ///
    /// Written once and instantiated by the compiler for each supported rank, so every
    /// rank runs identical logic rather than four hand-maintained copies.
    fn run_at_rank<const D: usize>(&self, op: &TensorOp) -> Result<TensorData, RunError> {
        let out: Tensor<B, D> = match op {
            TensorOp::Unary { kind, arg } => {
                let t = self.tensor::<D>(arg);
                match kind {
                    UnaryOp::Neg => t.neg(),
                    UnaryOp::Abs => t.abs(),
                    UnaryOp::Exp => t.exp(),
                    UnaryOp::Sqrt => t.sqrt(),
                }
            }
            TensorOp::Binary { kind, lhs, rhs } => {
                let (a, b) = (self.tensor::<D>(lhs), self.tensor::<D>(rhs));
                match kind {
                    BinaryOp::Add => a.add(b),
                    BinaryOp::Sub => a.sub(b),
                    BinaryOp::Mul => a.mul(b),
                    BinaryOp::Div => a.div(b),
                }
            }
            TensorOp::Reduce { kind, arg, axis } => {
                let t = self.tensor::<D>(arg);
                match kind {
                    // Collapses the axis to length one rather than removing it, so the
                    // result keeps rank `D` and this function has a single return type.
                    ReduceOp::Sum => t.sum_dim(*axis),
                }
            }
            TensorOp::Matmul { lhs, rhs } => {
                if D < 2 {
                    return Err(self.unsupported("matmul needs rank 2 or more"));
                }
                self.tensor::<D>(lhs).matmul(self.tensor::<D>(rhs))
            }
        };

        // Hand back `burn`'s own backend-independent data container. Returning the
        // backend's tensor type instead would make the return type depend on both the
        // backend *and* the rank, which is precisely what this function exists to
        // resolve.
        Ok(out.into_data())
    }
}

impl<B: Backend> Implementation for BurnBackend<B> {
    type In = TensorOp;
    /// Uniform across backends and ranks. Still not *comparable* — canonicalising it
    /// is the normaliser's job — but it is one type, which is what the driver needs.
    type Out = TensorData;

    fn name(&self) -> &str {
        self.name
    }

    fn run(&self, input: &TensorOp) -> Result<Self::Out, RunError> {
        // The single place a runtime rank becomes a compile-time one. Each arm is a
        // separate instantiation of the same generic function, so this is a dispatch
        // table, not four implementations.
        match input.rank() {
            1 => self.run_at_rank::<1>(input),
            2 => self.run_at_rank::<2>(input),
            3 => self.run_at_rank::<3>(input),
            4 => self.run_at_rank::<4>(input),
            other => Err(self.unsupported(format!(
                "rank {other} is beyond the supported maximum of {MAX_RANK}"
            ))),
        }
    }
}

/// The libtorch backend — the same arithmetic, performed by PyTorch's C++ kernels.
pub type LibTorchBackend = BurnBackend<LibTorch<f32>>;

/// The name every report, signature and sampling context uses for libtorch.
///
/// **A constant, not a literal repeated at each call site.** These names are matched
/// verbatim when deciding whether a negative may be scored against a finding, so a report
/// saying `burn-tch` and a scoring context saying `libtorch` silently never match — the
/// pool refuses everything and the reason looks like a data problem rather than a typo.
/// That happened once; hence the constants.
pub const LIBTORCH_NAME: &str = "burn-tch";

/// Construct the libtorch backend under test.
pub fn libtorch() -> LibTorchBackend {
    BurnBackend::new(LIBTORCH_NAME)
}

/// The wgpu backend — the same arithmetic again, this time on the GPU.
///
/// **Note the second type parameter.** `Flex<f32>` and `LibTorch<f32>` name only their
/// float type; `Wgpu` names its integer type as well, because a GPU kernel must know both
/// at compile time. That difference is absorbed here and reaches nothing else: the generic
/// `BurnBackend<B>` only requires `B: Backend`, so every operation, every rank, and the
/// whole dispatch `match` come across unchanged.
pub type WgpuBackend = BurnBackend<Wgpu<f32, i32>>;

/// The flex backend — burn's current first-party pure-Rust CPU option.
pub type FlexBackend = BurnBackend<Flex<f32>>;

/// The name every report, signature and sampling context uses for flex. See [`LIBTORCH_NAME`].
pub const FLEX_NAME: &str = "burn-flex";

/// Construct the flex CPU backend under test.
pub fn flex() -> FlexBackend {
    BurnBackend::new(FLEX_NAME)
}

/// Construct the GPU backend under test.
///
/// **This is the first backend on genuinely different hardware.** The two CPU backends
/// share an instruction set, which is what made exact agreement a defensible default.
/// A GPU shares none of it, and its reductions are not even deterministic between runs of
/// the same input (see `examples/wgpu_check.rs`) — so what counts as a legal difference
/// has to be re-derived rather than inherited. That work is step 7.4, not here.
/// The name every report, signature and sampling context uses for wgpu. See [`LIBTORCH_NAME`].
pub const WGPU_NAME: &str = "burn-wgpu";

pub fn wgpu() -> WgpuBackend {
    BurnBackend::new(WGPU_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The names are matched verbatim when scoring negatives**, so a constant that drifts
    /// from what the runner reports makes every pool refuse everything — and the failure
    /// reads as missing data rather than as a typo.
    #[test]
    fn the_name_constants_match_what_the_runners_report() {
        assert_eq!(flex().name(), FLEX_NAME);
        assert_eq!(libtorch().name(), LIBTORCH_NAME);
        assert_eq!(wgpu().name(), WGPU_NAME);
    }

    fn value(shape: &[usize], data: &[f32]) -> TensorValue {
        TensorValue::new(shape.to_vec(), data.to_vec())
    }

    fn values(out: TensorData) -> Vec<f32> {
        out.to_vec::<f32>().expect("f32 tensor")
    }

    /// Run a case on both backends and return their results, asserting the shape is
    /// what was expected. Most tests below care that the two agree, which is the
    /// property the whole tool depends on.
    fn on_both(op: &TensorOp, expected_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let cpu = flex().run(op).expect("cpu backend supports this");
        let torch = libtorch().run(op).expect("libtorch supports this");
        assert_eq!(cpu.shape.to_vec(), expected_shape.to_vec());
        assert_eq!(torch.shape.to_vec(), expected_shape.to_vec());
        (values(cpu), values(torch))
    }

    /// **The framework claim, as a test.** The GPU backend is reached through the same
    /// generic `run` as the CPU ones — no separate code path, no special-casing — so if
    /// this passes, adding an implementation really was a type alias and a constructor.
    ///
    /// Values are chosen to be exactly representable in `f32`, and every operation here
    /// is elementwise, so equality is the right test: no accumulation means no
    /// order-dependence, and none of the GPU's non-determinism can enter. Reductions are
    /// deliberately excluded — `sum` on this device returns one of two results (see
    /// `examples/wgpu_check.rs`), which is a tolerance question for step 7.4 rather than
    /// something a backend test should assert.
    #[test]
    fn the_gpu_backend_runs_through_the_same_dispatch_as_the_others() {
        let cases = [
            (
                TensorOp::binary(
                    BinaryOp::Add,
                    value(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                    value(&[2, 2], &[10.0, 20.0, 30.0, 40.0]),
                ),
                vec![11.0, 22.0, 33.0, 44.0],
            ),
            (
                TensorOp::unary(UnaryOp::Neg, value(&[3], &[-1.0, 0.0, 2.5])),
                vec![1.0, -0.0, -2.5],
            ),
            (
                TensorOp::unary(UnaryOp::Abs, value(&[2, 1, 2], &[-1.0, 2.0, -3.0, 4.0])),
                vec![1.0, 2.0, 3.0, 4.0],
            ),
        ];

        for (op, expected) in cases {
            let out = wgpu().run(&op).expect("the gpu backend supports this");
            assert_eq!(
                values(out),
                expected,
                "gpu disagreed on an exactly-representable case: {op:?}"
            );
        }
    }

    /// Rank is a *type* in burn, so each rank is a separate instantiation. A backend that
    /// works at rank 1 tells you nothing about rank 4 — worth asserting once per backend.
    #[test]
    fn the_gpu_backend_handles_every_rank() {
        for rank in 1..=MAX_RANK {
            let shape = vec![2; rank];
            let count = 1 << rank;
            let op = TensorOp::unary(UnaryOp::Neg, value(&shape, &vec![1.0; count]));

            let out = wgpu().run(&op).expect("the gpu backend supports this rank");
            assert_eq!(out.shape.to_vec(), shape, "wrong shape at rank {rank}");
            assert_eq!(
                values(out),
                vec![-1.0; count],
                "wrong values at rank {rank}"
            );
        }
    }

    #[test]
    fn elementwise_binary_runs_on_both_backends() {
        let op = TensorOp::binary(
            BinaryOp::Add,
            value(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            value(&[2, 2], &[10.0, 20.0, 30.0, 40.0]),
        );
        let (cpu, torch) = on_both(&op, &[2, 2]);
        assert_eq!(cpu, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(cpu, torch);
    }

    #[test]
    fn unary_runs_on_both_backends() {
        let op = TensorOp::unary(UnaryOp::Abs, value(&[3], &[-1.0, 0.0, 2.5]));
        let (cpu, torch) = on_both(&op, &[3]);
        assert_eq!(cpu, vec![1.0, 0.0, 2.5]);
        assert_eq!(cpu, torch);
    }

    #[test]
    fn matmul_runs_on_both_backends() {
        let op = TensorOp::matmul(
            value(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            value(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        );
        let (cpu, torch) = on_both(&op, &[2, 2]);
        assert_eq!(cpu, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(cpu, torch);
    }

    #[test]
    fn reduction_collapses_the_chosen_axis() {
        let op = TensorOp::reduce(ReduceOp::Sum, value(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), 1);
        let (cpu, torch) = on_both(&op, &[2, 1]);
        assert_eq!(cpu, vec![3.0, 7.0]);
        assert_eq!(cpu, torch);
    }

    /// Every supported rank must execute. Rank-specific code paths are exactly where
    /// shape-handling bugs live, so leaving any of them unexercised would be leaving
    /// the most likely place for a real finding untested.
    #[test]
    fn every_supported_rank_runs() {
        for shape in [vec![4], vec![2, 3], vec![2, 3, 2], vec![2, 2, 2, 2]] {
            let count = shape.iter().product();
            let op = TensorOp::unary(UnaryOp::Neg, value(&shape, &vec![1.0; count]));
            let (cpu, torch) = on_both(&op, &shape);
            assert_eq!(cpu, vec![-1.0; count]);
            assert_eq!(cpu, torch);
        }
    }

    /// Batched matmul, the reason ranks above two are worth supporting at all.
    #[test]
    fn batched_matmul_runs() {
        let op = TensorOp::matmul(
            value(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]),
            value(&[2, 2, 2], &[5.0, 6.0, 7.0, 8.0, 1.0, 1.0, 1.0, 1.0]),
        );
        let (cpu, torch) = on_both(&op, &[2, 2, 2]);
        // Identity times the first block, then twice the second.
        assert_eq!(cpu, vec![5.0, 6.0, 7.0, 8.0, 2.0, 2.0, 2.0, 2.0]);
        assert_eq!(cpu, torch);
    }

    #[test]
    fn backends_identify_themselves_distinctly() {
        assert_ne!(flex().name(), libtorch().name());
    }

    /// Running the same case twice on one backend must give the same answer. If that
    /// failed, no comparison between two backends could mean anything, because a
    /// difference might just be noise from one of them.
    #[test]
    fn a_backend_is_self_consistent() {
        let backend = flex();
        let op = TensorOp::unary(UnaryOp::Exp, value(&[3], &[0.5, 1.0, 2.0]));
        let first = values(backend.run(&op).unwrap());
        let second = values(backend.run(&op).unwrap());
        assert_eq!(first, second);
    }
}
