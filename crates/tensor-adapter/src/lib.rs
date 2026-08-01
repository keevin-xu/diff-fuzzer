//! # tensor-adapter
//!
//! The **DL/tensor adapter**: the per-software-type half of the project.
//!
//! Everything domain-specific about testing tensor libraries lives here — what a
//! valid tensor operation looks like, how to execute one on a given `burn` backend,
//! and how to canonicalize the resulting tensor for comparison. The engine in
//! `diff-fuzzer-core` drives all of it without knowing any of it.
//!
//! The differential is between two backends of the *same* framework (`burn`), which
//! means one generated op runs on both through an identical API — the design choice
//! that keeps false positives low. See `planning/05-TARGETS-AND-ORACLES.md`.
//!
//! ## Status
//!
//! A test case can be described and produced. Still to come: executing one on each
//! backend, canonicalising what comes back, and — replacing the placeholder generator
//! here — building cases that satisfy each operation's own rules.

pub mod backends;
pub mod generator;
pub mod input;
pub mod normalize;
pub mod ops;
pub mod testing;

pub use backends::{BurnBackend, LibTorchBackend, MAX_RANK, NdArrayBackend, libtorch, ndarray};
pub use generator::{FixedAddGenerator, TensorOpGenerator};
pub use input::{BinaryOp, ReduceOp, TensorOp, TensorValue, UnaryOp};
pub use normalize::{CanonicalTensor, TensorNormalizer};
pub use ops::Bounds;
pub use testing::FaultyBackend;
