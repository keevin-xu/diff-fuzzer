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

use crate::report::Divergence;
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

/// Why a case was not judged.
///
/// Every variant is a case the tool deliberately declined to have an opinion about, and
/// the reason is carried so that decision stays visible. **Silent exclusions are how a
/// fuzzer hides real bugs from its own operator** — a skip that leaves no trace is
/// indistinguishable from a pass, so a growing category of quietly-skipped cases could
/// hollow out a campaign without changing anything anyone looks at.
///
/// These categories come from practice rather than guesswork: the first three were each
/// discovered by hitting them.
// `Eq` was dropped when `Unjudgeable` gained `f64` fields: floats are not `Eq`, and a skip
// reason carrying the bound that made a case unjudgeable is worth more than the trait.
#[derive(Debug, Clone, PartialEq)]
pub enum SkipReason {
    /// Fewer than two results survived, so there was nothing to compare against.
    TooFewResults { available: usize },

    /// One or more implementations could not run the case at all.
    ///
    /// Not evidence of being wrong: being unable to attempt an input is a legitimate
    /// answer, and comparing an answer against no answer would be meaningless.
    CouldNotRun { failures: Vec<String> },

    /// Every element was settled by a special-value rule, so no arithmetic was compared.
    ///
    /// A result that is entirely undefined on both sides *agrees*, but the agreement is
    /// empty — neither implementation was actually exercised. Counting that as a pass
    /// would inflate the evidence a campaign appears to provide, so it is recorded as
    /// what it is: a case that went unjudged.
    NothingComparable { elements: usize },

    /// The derived tolerance is too loose to discriminate, so the case cannot be judged.
    ///
    /// **A bound wide enough to accept any answer is not a passing grade.** Tolerances here
    /// are derived per case from the operation and its values, and for some inputs the
    /// derivation is correct and useless at the same time: a `matmul` whose terms are `1e30`
    /// can cancel to anything within `±1e24`, and no comparison can tell a defect from
    /// arithmetic.
    ///
    /// Reporting those as `Agree` is a false pass, and it was: measured 2026-08-07, **96% of
    /// `matmul` cases and 65% of `softmax` cases** carried a bound that nothing could fail.
    /// Six hours of fuzzing found nothing in `softmax` for that reason — not because it
    /// agreed, but because it could not be judged and said otherwise.
    Unjudgeable { rtol: f64, atol: f64 },

    /// The operation is known to vary legitimately, so a difference here would not be a
    /// bug.
    ///
    /// Nothing uses this yet. It is the slot for behaviour that is genuinely
    /// unspecified — the clearest example being GPU reductions, which use atomics and so
    /// may sum in a different order on every run.
    KnownLegal { class: String, detail: String },
}

impl std::fmt::Display for SkipReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SkipReason::TooFewResults { available } => {
                write!(f, "need at least two results to compare, got {available}")
            }
            SkipReason::Unjudgeable { rtol, atol } => write!(
                f,
                "the derived tolerance (rtol {rtol:.2e}, atol {atol:.2e}) accepts any answer, \
                 so this case cannot be judged"
            ),
            SkipReason::CouldNotRun { failures } => {
                write!(f, "could not run: {}", failures.join("; "))
            }
            SkipReason::NothingComparable { elements } => write!(
                f,
                "all {elements} elements were undefined or infinite on both sides; \
                 no arithmetic was compared"
            ),
            SkipReason::KnownLegal { class, detail } => {
                write!(f, "{class} may legitimately vary: {detail}")
            }
        }
    }
}

/// An oracle's judgement on one test case.
// Not `Eq`: `SkipReason::Unjudgeable` carries the bound that made a case unjudgeable, and
// floats are not `Eq`.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The results are consistent, **and something was actually checked**. The
    /// overwhelmingly common outcome.
    Agree,
    /// The results disagree in a way worth reporting.
    ///
    /// Carries only what disagreed. The seed and other run context are attached by
    /// the driver, which owns those facts — see [`crate::report`].
    Diverged(Divergence),
    /// This case was not judged, and the reason why.
    ///
    /// Skipping is a first-class outcome, not an error.
    Skipped(SkipReason),
}
