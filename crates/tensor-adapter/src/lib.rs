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

/// Where this domain's outputs live, relative to the repository root.
///
/// **One constant, because a second domain is then a second constant** rather than a
/// search-and-replace across every example and harness. Findings, negatives and their
/// archives all hang off it, and nothing else in the crate names a path prefix.
///
/// Scoped by domain (`findings/tensor`) rather than flat, so a SQL adapter writing to
/// `findings/sql` cannot collide with this one — the two produce entirely different case
/// types under the same file names.
pub const FINDINGS_ROOT: &str = "findings/tensor";

/// Where this domain's non-diverging cases live.
pub const NEGATIVES_ROOT: &str = "findings/tensor/negatives";

pub mod backends;
/// Decoding fuzzer bytes into cases. Requires the `fuzzing` feature.
#[cfg(feature = "fuzzing")]
pub mod decode;
pub mod environment;
pub mod features;
pub mod generator;
pub mod input;
pub mod known;
pub mod negatives;
pub mod normalize;
pub mod ops;
pub mod predicate;
pub mod repro;
pub mod search;
/// A starting corpus for the fuzzer. Requires the `fuzzing` feature.
#[cfg(feature = "fuzzing")]
pub mod seeds;
pub mod shrink;
pub mod signature;
pub mod testing;
pub mod tolerance;
pub mod validation;

pub use backends::{
    BurnBackend, FLEX_NAME, FlexBackend, LIBTORCH_NAME, LibTorchBackend, MAX_RANK, WGPU_NAME,
    WgpuBackend, flex, libtorch, wgpu,
};
pub use environment::{BURN_VERSION, FLEX_VERSION, LIBTORCH_VERSION, TCH_VERSION, environment};
pub use features::{FEATURES, FeatureVec, extract};
pub use generator::{FixedAddGenerator, TensorOpGenerator};
pub use input::{BinaryOp, ReduceOp, TensorOp, TensorValue, UnaryOp};
pub use known::{KNOWN, Known, Relation, Status, known_by_predicate, known_issue};
pub use normalize::{CanonicalTensor, TensorNormalizer};
pub use ops::Bounds;
pub use predicate::Predicate;
pub use signature::{DisagreeingPair, signature, signature_across};
pub use testing::{FaultyBackend, FaultyCpu, faulty};
pub use tolerance::{OpClass, TensorTolerancePolicy};
