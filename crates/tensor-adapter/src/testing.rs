//! Deliberately broken systems, for checking that the detector detects.
//!
//! A tool built to notice disagreement has an unpleasant failure mode: if it silently
//! stops noticing, every run comes back clean, and clean results look exactly like
//! success. Nothing about "no divergences found" distinguishes a healthy fuzzer from
//! one whose comparison has been broken for a week.
//!
//! The remedy is to keep a system around that is *known* to be wrong, and to fail the
//! build if the tool ever fails to catch it. That is what lives here. It is ordinary
//! (not test-only) code because it is used from tests in other crates, and because it
//! is useful for demonstrating a divergence on demand.

use crate::backends::BurnBackend;
use crate::input::TensorOp;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend;
use diff_fuzzer_core::{Implementation, RunError};

/// A backend that computes correctly, then corrupts one number.
///
/// Only the *first* element is altered, on purpose. A fault that changed every element
/// would still be caught by a comparison that only ever looked at one of them — so
/// this shape of fault is the one that actually tests that the whole result is
/// examined.
#[derive(Debug, Clone)]
pub struct FaultyBackend<B: Backend> {
    inner: BurnBackend<B>,
    name: String,
    bias: f32,
}

impl<B: Backend> FaultyBackend<B> {
    /// Wrap a working backend so its first output value is off by `bias`.
    pub fn new(inner: BurnBackend<B>, bias: f32) -> Self {
        let name = format!("{}+fault({bias})", inner.name());
        Self { inner, name, bias }
    }
}

impl<B: Backend> Implementation for FaultyBackend<B> {
    type In = TensorOp;
    type Out = TensorData;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, input: &TensorOp) -> Result<Self::Out, RunError> {
        // Compute the correct answer through the real backend, then spoil it. Doing it
        // this way means the fault is the *only* difference from a correct system,
        // which is what makes a caught divergence attributable to the fault.
        let correct = self.inner.run(input)?;

        let shape = correct.shape.clone();
        let mut values = correct
            .to_vec::<f32>()
            .expect("backends are instantiated with f32 elements");

        if let Some(first) = values.first_mut() {
            *first += self.bias;
        }

        Ok(TensorData::new(values, shape))
    }
}

/// A CPU backend wrong by a known amount.
///
/// Mirrors `flex()` and `libtorch()`, so a caller can construct one without naming
/// `burn`'s types — the same reason those constructors exist. Callers outside this crate
/// should not have to depend on `burn` just to build a backend.
/// Named for its *role*, not its backend. This wraps whichever CPU backend the project
/// currently treats as its reference — it was `ndarray` until PHASE-7A and is now `flex`.
/// Naming it after the backend meant the safeguard silently kept testing a backend that
/// had been swapped out, which is the one thing a fault injector must never do.
pub type FaultyCpu = FaultyBackend<burn::backend::Flex<f32>>;

/// Construct the CPU backend with a deliberate fault of `bias` in its first output value.
pub fn faulty(bias: f32) -> FaultyCpu {
    FaultyBackend::new(crate::backends::flex(), bias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{flex, libtorch};
    use crate::generator::FixedAddGenerator;
    use crate::normalize::{CanonicalTensor, TensorNormalizer};
    use diff_fuzzer_core::{
        DifferentialOracle, FixedTolerance, NormalizedRunner, Runner, Tolerance, Verdict,
        driver::run_once,
    };

    type Oracle = DifferentialOracle<TensorOp, CanonicalTensor, FixedTolerance>;
    type AnyRunner<'a> = &'a dyn Runner<In = TensorOp, Canon = CanonicalTensor>;

    /// The control case: two genuinely correct backends must agree, so that a
    /// divergence in the test below is attributable to the injected fault and not to
    /// the two backends simply disagreeing about everything.
    #[test]
    fn two_correct_backends_agree() {
        let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
        let torch = NormalizedRunner::new(libtorch(), TensorNormalizer);
        let runners: [AnyRunner; 2] = [&cpu, &torch];

        let outcome = run_once(
            1,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );
        assert_eq!(outcome.verdict, Verdict::Agree);
    }

    /// The test this whole file exists for: a system that is wrong by a known amount
    /// must be caught. If this ever fails, the tool has stopped being able to find
    /// anything, and every clean run it reports is worthless.
    #[test]
    fn an_injected_fault_is_caught() {
        let correct = NormalizedRunner::new(flex(), TensorNormalizer);
        let faulty = NormalizedRunner::new(FaultyBackend::new(flex(), 0.5), TensorNormalizer);
        let runners: [AnyRunner; 2] = [&correct, &faulty];

        let outcome = run_once(
            1,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );

        let Verdict::Diverged(divergence) = outcome.verdict else {
            panic!("the injected fault was not caught");
        };
        // The report must name the culprit, and record both results so the difference
        // can be seen rather than taken on trust.
        assert!(
            divergence.summary.contains("fault"),
            "{}",
            divergence.summary
        );
        assert_eq!(divergence.outputs.len(), 2);
    }

    /// A fault across the backend boundary is caught too — the case that most closely
    /// resembles a real finding, where the two sides are genuinely different systems.
    #[test]
    fn a_fault_is_caught_across_different_backends() {
        let cpu = NormalizedRunner::new(flex(), TensorNormalizer);
        let faulty_torch =
            NormalizedRunner::new(FaultyBackend::new(libtorch(), -2.0), TensorNormalizer);
        let runners: [AnyRunner; 2] = [&cpu, &faulty_torch];

        let outcome = run_once(
            1,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );
        assert!(matches!(outcome.verdict, Verdict::Diverged(_)));
    }

    /// A caught divergence must be reproducible from its seed. A finding that cannot be
    /// replayed is a defect in this tool, not a discovery about anything else.
    #[test]
    fn a_caught_divergence_replays_from_its_seed() {
        let correct = NormalizedRunner::new(flex(), TensorNormalizer);
        let faulty = NormalizedRunner::new(FaultyBackend::new(flex(), 0.5), TensorNormalizer);
        let runners: [AnyRunner; 2] = [&correct, &faulty];

        let first = run_once(
            4242,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );
        let replay = run_once(
            4242,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );

        assert!(matches!(first.verdict, Verdict::Diverged(_)));
        assert_eq!(first, replay);
    }

    /// A fault too small to change the result must not be reported. Today this holds
    /// only because the values are integers and `0.0` changes nothing exactly — once
    /// comparison moves to a tolerance, this test becomes the place where "how small is
    /// too small" is actually decided.
    #[test]
    fn a_zero_fault_is_not_a_divergence() {
        let correct = NormalizedRunner::new(flex(), TensorNormalizer);
        let unfaulty = NormalizedRunner::new(FaultyBackend::new(flex(), 0.0), TensorNormalizer);
        let runners: [AnyRunner; 2] = [&correct, &unfaulty];

        let outcome = run_once(
            1,
            &FixedAddGenerator,
            &runners,
            &Oracle::new(FixedTolerance(Tolerance::EXACT)),
        );
        assert_eq!(outcome.verdict, Verdict::Agree);
    }
}
