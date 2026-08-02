//! Flagging disagreement between implementations.
//!
//! The reasoning is deliberately narrow: several systems were given the same input and
//! are supposed to behave identically, so if their results differ, at least one is
//! wrong. Which one, and why, are separate questions this does not attempt — and does
//! not need to, which is exactly what makes the technique work on software whose
//! correct answers nobody can cheaply compute.
//!
//! "Differ" now means *beyond a tolerance* rather than *not bit-identical*. Two correct
//! implementations routinely disagree in the final bits, so exact comparison reports
//! correct code as broken. The interesting consequence is that the threshold becomes
//! part of the claim: a divergence is only meaningful relative to a stated tolerance,
//! which is why every report carries the one that was in force.

use crate::report::Divergence;
use crate::tolerance::{Agreement, ApproxEq, Comparison, TolerancePolicy};
use crate::traits::{Input, NamedOutput, Oracle, Verdict};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Reports disagreement between two or more implementations, within a tolerance
/// decided per case by a policy.
#[derive(Debug, Clone, Copy)]
pub struct DifferentialOracle<In, C, P> {
    policy: P,
    /// `In` and `C` appear only in the trait's associated types, never in a field, so
    /// this ties the type to them without occupying any memory.
    _types: PhantomData<(In, C)>,
}

impl<In, C, P> DifferentialOracle<In, C, P> {
    pub fn new(policy: P) -> Self {
        Self {
            policy,
            _types: PhantomData,
        }
    }
}

impl<In, C, P> Oracle for DifferentialOracle<In, C, P>
where
    In: Input,
    C: ApproxEq + Debug,
    P: TolerancePolicy<In>,
{
    type In = In;
    type Canon = C;

    fn check(&self, input: &Self::In, outputs: &[NamedOutput<Self::Canon>]) -> Verdict {
        // Fewer than two results means there is nothing to compare against. This is a
        // skip and not a failure: one side legitimately being unable to run an input is
        // expected, and comparing an answer against no answer would be meaningless.
        if outputs.len() < 2 {
            return Verdict::Skipped(format!(
                "need at least two results to compare, got {}",
                outputs.len()
            ));
        }

        let tolerance = self.policy.tolerance_for(input);

        // Compare every result against the first. With two implementations this is just
        // "do they match"; with more, agreeing with the first is enough to conclude all
        // agree, since agreement within a fixed tolerance is transitive enough at this
        // scale — and any disagreement is reported with the specific pair named.
        let reference = &outputs[0];
        let mut complaints: Vec<String> = Vec::new();

        for candidate in &outputs[1..] {
            match reference
                .output
                .approx_compare(&candidate.output, tolerance)
            {
                Agreement::Agree(_) => {}
                Agreement::Structural { reason } => complaints.push(format!(
                    "{} vs {}: {reason}",
                    reference.implementation, candidate.implementation
                )),
                Agreement::Disagree(comparison) => complaints.push(format!(
                    "{} vs {}: {}",
                    reference.implementation,
                    candidate.implementation,
                    describe(&comparison)
                )),
            }
        }

        if complaints.is_empty() {
            return Verdict::Agree;
        }

        Verdict::Diverged(Divergence {
            input: format!("{input:?}"),
            // Every result is recorded, not only the disagreeing ones: knowing what the
            // others produced is what makes a report diagnosable.
            outputs: outputs
                .iter()
                .map(|o| (o.implementation.clone(), format!("{:?}", o.output)))
                .collect(),
            summary: format!(
                "{} (rtol {:e}, atol {:e})",
                complaints.join("; "),
                tolerance.rtol,
                tolerance.atol
            ),
        })
    }
}

/// Turn a comparison into the sentence a report shows.
///
/// The numbers are the point. "These disagree" tells a maintainer nothing they can act
/// on; the size of the error and where it occurred tells them whether to care and where
/// to look.
fn describe(comparison: &Comparison) -> String {
    let location = match &comparison.worst {
        Some(worst) => format!(
            ", worst at element {} ({} vs {})",
            worst.index, worst.left, worst.right
        ),
        None => String::new(),
    };

    format!(
        "{} of {} elements differ, max relative error {:.3e}, max absolute error {:.3e}{location}",
        comparison.mismatches,
        comparison.total,
        comparison.max_relative_error,
        comparison.max_absolute_error
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tolerance::{FixedTolerance, Tolerance};

    /// A stand-in test case. The oracle never inspects an input beyond printing it, so
    /// nothing more is needed — and needing nothing more is the point: this oracle is
    /// tested without a single backend, tensor, or generated case in sight.
    #[derive(Clone, Debug)]
    struct TestInput;
    impl Input for TestInput {}

    fn output(name: &str, values: Vec<f32>) -> NamedOutput<Vec<f32>> {
        NamedOutput {
            implementation: name.to_string(),
            output: values,
        }
    }

    fn check_with(tolerance: Tolerance, outputs: &[NamedOutput<Vec<f32>>]) -> Verdict {
        let oracle: DifferentialOracle<TestInput, Vec<f32>, FixedTolerance> =
            DifferentialOracle::new(FixedTolerance(tolerance));
        oracle.check(&TestInput, outputs)
    }

    fn check(outputs: &[NamedOutput<Vec<f32>>]) -> Verdict {
        check_with(Tolerance::EXACT, outputs)
    }

    #[test]
    fn matching_results_agree() {
        assert_eq!(
            check(&[output("a", vec![1.0, 2.0]), output("b", vec![1.0, 2.0])]),
            Verdict::Agree
        );
    }

    #[test]
    fn differing_results_diverge() {
        let verdict = check(&[output("a", vec![1.0, 2.0]), output("b", vec![1.0, 3.0])]);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence, got {verdict:?}");
        };
        // Naming the pair, not just one side: with more than two implementations,
        // "which two disagreed" is the whole content of the finding.
        assert!(
            divergence.summary.contains("a vs b"),
            "{}",
            divergence.summary
        );
        assert_eq!(divergence.outputs.len(), 2);
        assert_eq!(divergence.input, "TestInput");
    }

    /// The behaviour this step exists for: a difference small enough to be rounding
    /// noise must stop being reported once a tolerance permits it, while remaining a
    /// divergence under exact comparison.
    #[test]
    fn a_small_difference_is_tolerated_but_not_exactly_equal() {
        let outputs = [output("a", vec![1.0]), output("b", vec![1.000_000_1])];

        assert!(matches!(check(&outputs), Verdict::Diverged(_)));
        assert_eq!(
            check_with(Tolerance::new(1e-5, 1e-8), &outputs),
            Verdict::Agree
        );
    }

    /// And the limit of that: a tolerance must not swallow a difference large enough to
    /// matter, or the tool reports clean runs while missing everything.
    #[test]
    fn a_large_difference_is_reported_despite_a_tolerance() {
        let outputs = [output("a", vec![1.0]), output("b", vec![2.0])];
        assert!(matches!(
            check_with(Tolerance::new(1e-5, 1e-8), &outputs),
            Verdict::Diverged(_)
        ));
    }

    /// Structural disagreement must survive *any* tolerance. Results of different sizes
    /// disagree about what the operation produced, which is not a question of degree.
    #[test]
    fn a_structural_difference_is_not_absorbed_by_a_huge_tolerance() {
        let outputs = [
            output("a", vec![1.0, 2.0]),
            output("b", vec![1.0, 2.0, 3.0]),
        ];
        let verdict = check_with(Tolerance::new(1e30, 1e30), &outputs);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("a size difference was absorbed by tolerance");
        };
        assert!(
            divergence.summary.contains("lengths differ"),
            "{}",
            divergence.summary
        );
    }

    /// A report must state the tolerance it was judged against. Without it the claim is
    /// unfalsifiable — nobody can tell whether the difference was meaningful or the
    /// threshold merely tight.
    #[test]
    fn the_report_records_the_tolerance_in_force() {
        let verdict = check_with(
            Tolerance::new(1e-9, 1e-12),
            &[output("a", vec![1.0]), output("b", vec![2.0])],
        );

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence");
        };
        assert!(
            divergence.summary.contains("1e-9"),
            "{}",
            divergence.summary
        );
        assert!(
            divergence.summary.contains("1e-12"),
            "{}",
            divergence.summary
        );
    }

    /// The size and position of the error must reach the report, since that is what
    /// makes a finding actionable rather than merely alarming.
    #[test]
    fn the_report_records_the_size_and_position_of_the_error() {
        let verdict = check(&[
            output("a", vec![1.0, 5.0, 3.0]),
            output("b", vec![1.0, 9.0, 3.0]),
        ]);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence");
        };
        assert!(
            divergence.summary.contains("1 of 3 elements"),
            "{}",
            divergence.summary
        );
        assert!(
            divergence.summary.contains("element 1"),
            "{}",
            divergence.summary
        );
    }

    #[test]
    fn one_result_is_skipped_not_compared() {
        assert!(matches!(
            check(&[output("a", vec![1.0, 2.0])]),
            Verdict::Skipped(_)
        ));
    }

    #[test]
    fn no_results_are_skipped() {
        assert!(matches!(check(&[]), Verdict::Skipped(_)));
    }

    /// With three implementations, one dissenter is enough to report — and it is named
    /// while the agreeing one is not.
    #[test]
    fn a_single_dissenter_among_three_is_reported() {
        let verdict = check(&[
            output("a", vec![1.0]),
            output("b", vec![1.0]),
            output("c", vec![9.0]),
        ]);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence");
        };
        // The dissenting pair is named; the pair that agreed is not mentioned at all.
        // Matching on the whole pair rather than a bare name matters — searching for a
        // single letter would match any word in the error text that happens to contain
        // it, which is how this test previously passed for the wrong reason.
        assert!(
            divergence.summary.contains("a vs c"),
            "{}",
            divergence.summary
        );
        assert!(
            !divergence.summary.contains("a vs b"),
            "{}",
            divergence.summary
        );
    }

    /// Two implementations both producing an undefined result agree. Relying on `==`
    /// would report this as a disagreement, since NaN is not equal to itself.
    #[test]
    fn two_undefined_results_agree() {
        assert_eq!(
            check(&[output("a", vec![f32::NAN]), output("b", vec![f32::NAN])]),
            Verdict::Agree
        );
    }
}
