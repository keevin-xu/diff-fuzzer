//! The five seams the whole project is built on.
//!
//! Each trait is a question the engine cannot answer for itself, delegated to
//! whoever knows the domain:
//!
//! | Trait            | Question it answers                                  |
//! |------------------|------------------------------------------------------|
//! | [`Generator`]    | what does a valid test case look like?               |
//! | [`Implementation`] | how do I execute one, on this particular system?   |
//! | [`Normalizer`]   | how do I make two results comparable?                |
//! | [`Oracle`]       | do these results count as disagreeing?               |
//! | [`Input`]        | (marker) this type is a test case                    |
//!
//! Nothing here mentions tensors, and nothing here ever will. That is the point: the
//! engine drives the loop, and the domain knowledge lives on the far side of these
//! traits, in an adapter crate. Adding a whole new kind of software to test means
//! writing new implementations of `Generator`, `Implementation` and `Normalizer` —
//! not touching this file.

use crate::report::DivergenceReport;
use crate::rng::SeededRng;

/// Marker for "this type is a test case".
///
/// It requires `Clone` because minimisation repeatedly produces modified copies of a
/// failing input, and `Debug` because an input that cannot be printed cannot be
/// reported.
///
/// A trait with no methods looks pointless, but it earns its place: it lets the other
/// traits below say `type In: Input`, which is what stops someone from accidentally
/// building a generator that produces something unreportable.
pub trait Input: Clone + std::fmt::Debug {}

/// Why an implementation could not produce a result.
///
/// This is distinct from disagreement. If a backend cannot run an input at all, that
/// is not evidence of a bug — comparing "an answer" with "no answer" would be
/// meaningless, so the engine skips the case rather than reporting it.
///
/// `#[derive(thiserror::Error)]` writes the `Display` and `std::error::Error`
/// implementations from the `#[error(...)]` format strings, which is boilerplate that
/// would otherwise be about twenty hand-written lines.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RunError {
    /// The implementation legitimately does not handle this input — an operation it
    /// does not provide, a dtype it does not support. Expected, not a finding.
    #[error("{implementation} does not support this input: {reason}")]
    Unsupported {
        implementation: String,
        reason: String,
    },
    /// The implementation tried and failed.
    #[error("{implementation} failed to run this input: {message}")]
    Failed {
        implementation: String,
        message: String,
    },
}

/// One concrete system under test.
///
/// Every backend is one implementation of this trait. Adding a backend is therefore
/// adding a type and an `impl` block — no new entry point, no change to the engine,
/// no separate harness. That claim is the reason the trait exists in this shape.
pub trait Implementation {
    /// The kind of test case this system accepts.
    type In: Input;
    /// Whatever this system produces natively — a backend's own tensor type, a
    /// database's own row set. Deliberately unconstrained, because it is not yet
    /// comparable to anything; that is [`Normalizer`]'s job.
    type Out;

    /// How this implementation identifies itself in reports. Must be stable, since
    /// findings are grouped by it.
    fn name(&self) -> &str;

    /// Execute one test case.
    fn run(&self, input: &Self::In) -> Result<Self::Out, RunError>;
}

/// Produces valid test cases from a seeded generator.
///
/// "Valid" is the load-bearing word. Inputs are built to satisfy the rules of what
/// they represent, rather than generated blindly and filtered — an input rejected as
/// malformed tests nothing but the validation code.
///
/// Taking `&mut SeededRng` rather than returning random values from nowhere is what
/// makes a run replayable: the same seed walks the generator down the same path.
pub trait Generator {
    type In: Input;

    fn generate(&self, rng: &mut SeededRng) -> Self::In;
}

/// Turns a system's native output into something comparable.
///
/// Two systems can be equally correct and still hand back results that look nothing
/// alike — different internal layouts, different orderings, different ways of
/// spelling the same number. Comparing before canonicalising produces a flood of
/// differences that mean nothing, which is the standard way a project like this
/// drowns in false alarms.
pub trait Normalizer {
    type Out;
    /// The canonical form. Both sides of a comparison are converted to this.
    type Canon;

    /// Note this *takes ownership* of the output rather than borrowing it. Extracting
    /// data from a backend's representation usually consumes it, and consuming avoids
    /// a copy of what may be a large buffer.
    fn normalize(&self, out: Self::Out) -> Self::Canon;
}

/// One implementation's canonicalised result, labelled with who produced it.
///
/// A named struct rather than a bare `(String, C)` tuple, so that reading
/// `outputs[0].implementation` says what it means without looking up which half of
/// the tuple is which.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedOutput<C> {
    pub implementation: String,
    pub output: C,
}

/// Decides whether a set of results constitutes a disagreement.
///
/// This is the pluggable slot. Comparing two implementations against each other is
/// one strategy; checking that a single implementation's results satisfy a
/// relationship that must hold is another. Both are just types implementing this
/// trait, so a second strategy is an addition rather than a rewrite.
///
/// Note what this does *not* do: it never runs anything. The engine runs the
/// implementations and normalises their output, then hands the results here to be
/// judged. Keeping execution out of the oracle means it can be tested against
/// fabricated results, with no backends involved at all.
pub trait Oracle {
    type In: Input;
    /// Must match the [`Normalizer::Canon`] feeding it.
    type Canon;

    fn check(&self, input: &Self::In, outputs: &[NamedOutput<Self::Canon>]) -> Verdict;
}

/// An oracle's judgement on one test case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// The results are consistent. The overwhelmingly common outcome.
    Agree,
    /// The results disagree in a way worth reporting.
    Diverged(DivergenceReport),
    /// This case was not judged, and the reason why.
    ///
    /// Skipping is a first-class outcome, not an error: an input one side cannot run,
    /// or an operation whose result is legitimately allowed to vary, must be excluded
    /// explicitly rather than counted as a disagreement. Carrying the reason keeps
    /// that auditable — silent exclusions are how real bugs get hidden.
    ///
    /// A plain `String` for now; this becomes a proper enum once the real categories
    /// are known from practice rather than guessed at.
    Skipped(String),
}
