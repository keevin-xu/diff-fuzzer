//! Flagging disagreement between implementations.
//!
//! The reasoning is deliberately narrow: several systems were given the same input and
//! are supposed to behave identically, so if their results differ, at least one is
//! wrong. Which one, and why, are separate questions this does not attempt — and does
//! not need to, which is exactly what makes the technique work on software whose
//! correct answers nobody can cheaply compute.

use crate::report::Divergence;
use crate::traits::{Input, NamedOutput, Oracle, Verdict};
use std::fmt::Debug;
use std::marker::PhantomData;

/// Reports disagreement between two or more implementations.
///
/// Currently compares results for **exact** equality. That is the right amount of
/// machinery for proving the pipeline detects a difference at all, and the wrong tool
/// for real numeric results: two correct implementations routinely differ in the last
/// bits, because floating-point addition is not associative and different systems sum
/// in different orders. Replacing this with a tolerance is a whole phase of work,
/// which is why it is not smuggled in here.
#[derive(Debug, Clone, Copy)]
pub struct DifferentialOracle<In, C> {
    /// `In` and `C` appear only in the trait's associated types, never in a field, so
    /// this ties the type to them without occupying any memory.
    _types: PhantomData<(In, C)>,
}

impl<In, C> DifferentialOracle<In, C> {
    pub fn new() -> Self {
        Self {
            _types: PhantomData,
        }
    }
}

impl<In, C> Default for DifferentialOracle<In, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<In: Input, C: PartialEq + Debug> Oracle for DifferentialOracle<In, C> {
    type In = In;
    type Canon = C;

    fn check(&self, input: &Self::In, outputs: &[NamedOutput<Self::Canon>]) -> Verdict {
        // Fewer than two results means there is nothing to compare against. This is a
        // skip and not a failure: one side legitimately being unable to run an input
        // is expected, and comparing an answer against no answer would be meaningless.
        if outputs.len() < 2 {
            return Verdict::Skipped(format!(
                "need at least two results to compare, got {}",
                outputs.len()
            ));
        }

        // Compare every result against the first. With two implementations this is
        // just "do they match"; with more, agreeing with the first is enough to
        // conclude all agree, since equality is transitive.
        //
        // A caveat that becomes important later: floating-point equality is *not*
        // reflexive, because NaN does not equal itself. Two implementations that both
        // correctly produce NaN would be reported as disagreeing. Deciding what NaN
        // against NaN should mean is part of making this oracle trustworthy, and is
        // handled when comparison moves to a tolerance.
        let reference = &outputs[0];
        let disagreeing: Vec<&NamedOutput<C>> = outputs[1..]
            .iter()
            .filter(|candidate| candidate.output != reference.output)
            .collect();

        if disagreeing.is_empty() {
            return Verdict::Agree;
        }

        let names: Vec<&str> = disagreeing
            .iter()
            .map(|o| o.implementation.as_str())
            .collect();

        Verdict::Diverged(Divergence {
            input: format!("{input:?}"),
            // Every result is recorded, not only the disagreeing ones: knowing what
            // the others produced is what makes a report diagnosable.
            outputs: outputs
                .iter()
                .map(|o| (o.implementation.clone(), format!("{:?}", o.output)))
                .collect(),
            summary: format!(
                "{} disagreed with {}",
                names.join(", "),
                reference.implementation
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in test case. The oracle never inspects an input beyond printing it, so
    /// nothing more is needed — and needing nothing more is the point: this oracle is
    /// tested without a single backend, tensor, or generated case in sight.
    #[derive(Clone, Debug)]
    struct TestInput;
    impl Input for TestInput {}

    fn output(name: &str, values: Vec<i32>) -> NamedOutput<Vec<i32>> {
        NamedOutput {
            implementation: name.to_string(),
            output: values,
        }
    }

    fn check(outputs: &[NamedOutput<Vec<i32>>]) -> Verdict {
        DifferentialOracle::new().check(&TestInput, outputs)
    }

    #[test]
    fn matching_results_agree() {
        let verdict = check(&[output("a", vec![1, 2]), output("b", vec![1, 2])]);
        assert_eq!(verdict, Verdict::Agree);
    }

    #[test]
    fn differing_results_diverge() {
        let verdict = check(&[output("a", vec![1, 2]), output("b", vec![1, 3])]);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence, got {verdict:?}");
        };
        assert!(divergence.summary.contains('b'), "{}", divergence.summary);
        // Both results are recorded, not only the one that differed.
        assert_eq!(divergence.outputs.len(), 2);
        // The input is recorded too — a report without it could not be acted on.
        assert_eq!(divergence.input, "TestInput");
    }

    #[test]
    fn one_result_is_skipped_not_compared() {
        let verdict = check(&[output("a", vec![1, 2])]);
        assert!(matches!(verdict, Verdict::Skipped(_)));
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
            output("a", vec![1]),
            output("b", vec![1]),
            output("c", vec![9]),
        ]);

        let Verdict::Diverged(divergence) = verdict else {
            panic!("expected a divergence");
        };
        assert!(divergence.summary.contains('c'), "{}", divergence.summary);
        assert!(!divergence.summary.contains('b'), "{}", divergence.summary);
    }

    /// Shapes and sizes differing is a disagreement too, not only differing values.
    #[test]
    fn results_of_different_lengths_diverge() {
        let verdict = check(&[output("a", vec![1, 2]), output("b", vec![1, 2, 3])]);
        assert!(matches!(verdict, Verdict::Diverged(_)));
    }
}
