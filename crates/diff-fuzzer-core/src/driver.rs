//! Running one test case from seed to verdict.
//!
//! This is the loop the whole project is built around, and it is deliberately dull:
//! turn a seed into a case, give that case to every system, convert what comes back
//! into comparable form, and ask the oracle what it thinks. Every interesting decision
//! lives behind one of those traits, which is why this function knows nothing about
//! tensors, tolerances, or backends.
//!
//! It takes the systems as a slice rather than as two parameters, so nothing here
//! assumes the comparison involves exactly two.

use crate::rng::SeededRng;
use crate::runner::Runner;
use crate::traits::{Generator, NamedOutput, Oracle, Verdict};

/// What one test case produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The seed this case came from. Carried because it is the whole of what someone
    /// else needs to reproduce the run — a verdict without it cannot be acted on.
    pub seed: u64,
    pub verdict: Verdict,
}

/// Generate one case from `seed`, run it on every system, and judge the results.
///
/// Systems that cannot run the case are left out of the comparison rather than counted
/// against it: being unable to attempt an input is not evidence of being wrong. If that
/// leaves fewer than two results, the case is skipped and the reasons are kept, since
/// a silently dropped case is indistinguishable from one that passed.
pub fn run_once<G, O>(
    seed: u64,
    generator: &G,
    runners: &[&dyn Runner<In = G::In, Canon = O::Canon>],
    oracle: &O,
) -> RunOutcome
where
    G: Generator,
    O: Oracle<In = G::In>,
{
    // The single source of randomness for this case. Constructed from the seed here
    // and nowhere else, which is what makes the run reproducible.
    let mut rng = SeededRng::from_seed(seed);
    let input = generator.generate(&mut rng);

    tracing::debug!(seed, ?input, "generated case");

    let mut outputs = Vec::with_capacity(runners.len());
    let mut failures = Vec::new();

    for runner in runners {
        match runner.run_and_normalize(&input) {
            Ok(output) => outputs.push(NamedOutput {
                implementation: runner.name().to_string(),
                output,
            }),
            Err(error) => {
                tracing::debug!(seed, implementation = runner.name(), %error, "could not run");
                failures.push(format!("{}: {error}", runner.name()));
            }
        }
    }

    // Report *why* there was nothing to compare. Without this, a run where every
    // system failed would look identical to one where everything agreed.
    let verdict = if outputs.len() < 2 && !failures.is_empty() {
        Verdict::Skipped(failures.join("; "))
    } else {
        oracle.check(&input, &outputs)
    };

    match &verdict {
        // Logged at `info` because a divergence is the thing the whole run exists to
        // find; everything else is at `debug` so a long campaign stays readable.
        Verdict::Diverged(divergence) => {
            tracing::info!(seed, summary = %divergence.summary, "divergence")
        }
        Verdict::Skipped(reason) => tracing::debug!(seed, %reason, "skipped"),
        Verdict::Agree => tracing::debug!(seed, "agreed"),
    }

    RunOutcome { seed, verdict }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::DifferentialOracle;
    use crate::runner::NormalizedRunner;
    use crate::traits::{Implementation, Input, Normalizer, RunError};

    #[derive(Clone, Debug, PartialEq)]
    struct Number(u32);
    impl Input for Number {}

    /// Produces a case that actually depends on the seed, so the determinism test
    /// below is testing something.
    struct RandomNumber;
    impl Generator for RandomNumber {
        type In = Number;
        fn generate(&self, rng: &mut SeededRng) -> Number {
            use rand::RngExt;
            Number(rng.random_range(0..1_000_000))
        }
    }

    /// A system that adds a fixed offset. An offset of zero is "correct"; anything
    /// else stands in for a system that is wrong.
    struct Adder {
        name: &'static str,
        offset: u32,
    }
    impl Implementation for Adder {
        type In = Number;
        type Out = u32;
        fn name(&self) -> &str {
            self.name
        }
        fn run(&self, input: &Number) -> Result<u32, RunError> {
            Ok(input.0 + self.offset)
        }
    }

    /// A system that refuses to run anything.
    struct Refuses;
    impl Implementation for Refuses {
        type In = Number;
        type Out = u32;
        fn name(&self) -> &str {
            "refuses"
        }
        fn run(&self, _input: &Number) -> Result<u32, RunError> {
            Err(RunError::Unsupported {
                implementation: "refuses".to_string(),
                reason: "does not do numbers".to_string(),
            })
        }
    }

    struct Identity;
    impl Normalizer for Identity {
        type Out = u32;
        type Canon = u32;
        fn normalize(&self, out: u32) -> u32 {
            out
        }
    }

    fn runner(name: &'static str, offset: u32) -> NormalizedRunner<Adder, Identity> {
        NormalizedRunner::new(Adder { name, offset }, Identity)
    }

    fn oracle() -> DifferentialOracle<Number, u32> {
        DifferentialOracle::new()
    }

    #[test]
    fn agreeing_systems_produce_agreement() {
        let (a, b) = (runner("a", 0), runner("b", 0));
        let outcome = run_once(1, &RandomNumber, &[&a, &b], &oracle());

        assert_eq!(outcome.verdict, Verdict::Agree);
        assert_eq!(outcome.seed, 1);
    }

    #[test]
    fn a_wrong_system_is_caught() {
        let (correct, wrong) = (runner("correct", 0), runner("wrong", 1));
        let outcome = run_once(1, &RandomNumber, &[&correct, &wrong], &oracle());

        assert!(matches!(outcome.verdict, Verdict::Diverged(_)));
    }

    /// The same seed must produce the same verdict, every time. This is the property
    /// every finding depends on: a divergence that cannot be replayed from its seed is
    /// a defect in this tool, not a discovery about anything else.
    #[test]
    fn the_same_seed_gives_the_same_outcome() {
        let (a, b) = (runner("a", 0), runner("b", 1));
        let first = run_once(99, &RandomNumber, &[&a, &b], &oracle());
        let second = run_once(99, &RandomNumber, &[&a, &b], &oracle());

        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_give_different_cases() {
        let (a, b) = (runner("a", 0), runner("b", 1));
        let first = run_once(1, &RandomNumber, &[&a, &b], &oracle());
        let second = run_once(2, &RandomNumber, &[&a, &b], &oracle());

        // Both diverge, but on different inputs — so the reports differ.
        assert_ne!(first.verdict, second.verdict);
    }

    /// A system that cannot run the case must not be mistaken for one that disagreed.
    #[test]
    fn a_failing_system_leaves_too_little_to_compare() {
        let working = runner("works", 0);
        let broken = NormalizedRunner::new(Refuses, Identity);
        let outcome = run_once(1, &RandomNumber, &[&working, &broken], &oracle());

        let Verdict::Skipped(reason) = outcome.verdict else {
            panic!("expected a skip");
        };
        assert!(reason.contains("does not do numbers"), "{reason}");
    }

    /// With three systems, one refusing still leaves two to compare — so the case is
    /// judged rather than thrown away.
    #[test]
    fn a_failing_system_among_three_still_leaves_a_comparison() {
        let (a, b) = (runner("a", 0), runner("b", 5));
        let broken = NormalizedRunner::new(Refuses, Identity);
        let outcome = run_once(1, &RandomNumber, &[&a, &b, &broken], &oracle());

        assert!(matches!(outcome.verdict, Verdict::Diverged(_)));
    }
}
