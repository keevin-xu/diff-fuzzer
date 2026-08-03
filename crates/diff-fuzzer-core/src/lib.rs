//! # diff-fuzzer-core
//!
//! The reusable, **target-agnostic** engine for differential testing.
//!
//! The idea it implements: run the same input through two systems that are supposed
//! to behave identically, and if they disagree, at least one of them is wrong. That
//! sidesteps the usual obstacle to testing complex software — knowing what the
//! correct answer *is* — because a disagreement is evidence on its own.
//!
//! This crate knows nothing about tensors, or databases, or anything else. It knows
//! only the traits in [`traits`]: given a way to generate an input, run it, and
//! canonicalise the results, it can drive the loop and ask an oracle for a verdict.
//! Domain knowledge lives in adapter crates on the far side of those traits.
//!
//! ## Status
//!
//! Trait seams and seeded randomness are in place. Still to come: the driver that
//! runs a case end to end, the tolerance-based comparison that replaces exact
//! equality, shrinking a failure to its smallest form, and writing findings to disk.

pub mod driver;
pub mod minimize;
pub mod oracle;
pub mod report;
pub mod rng;
pub mod runner;
pub mod tolerance;
pub mod traits;

// Re-exported at the crate root so users write `diff_fuzzer_core::Oracle` rather than
// `diff_fuzzer_core::traits::Oracle`. Module structure is our business; the names are
// what callers care about.
pub use driver::{RunOutcome, run_once};
pub use minimize::{Budget, Minimized, Shrink, StopReason, minimize, minimize_within};
pub use oracle::DifferentialOracle;
pub use report::{Divergence, Finding, FindingsLog, read_findings};
pub use rng::SeededRng;
pub use runner::{NormalizedRunner, Runner};
pub use tolerance::{
    Agreement, ApproxEq, Comparison, FixedTolerance, Mismatch, Special, Tolerance, TolerancePolicy,
    compare,
};
pub use traits::{
    Generator, Implementation, Input, NamedOutput, Normalizer, Oracle, RunError, SkipReason,
    Verdict,
};
