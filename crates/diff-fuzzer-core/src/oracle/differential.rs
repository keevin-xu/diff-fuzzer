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
use crate::tolerance::{Agreement, ApproxEq, Comparison, Tolerance, TolerancePolicy};
use crate::traits::{Input, NamedOutput, Oracle, SkipReason, Verdict};
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
            return Verdict::Skipped(SkipReason::TooFewResults {
                available: outputs.len(),
            });
        }

        // **Compare every pair, not everything against the first.**
        //
        // This used to compare each result against `outputs[0]`, on the grounds that
        // agreement is transitive enough at this scale. It is not: approximate equality
        // does not compose. If A agrees with B within the tolerance and A agrees with C
        // within it, B and C may still differ by *twice* it — so a genuine B-versus-C
        // divergence sitting just past the threshold would never be looked at.
        //
        // The assumption was also untestable while there were two implementations,
        // because with two there is exactly one pair and the two strategies coincide. A
        // third makes the choice real, and choosing wrongly fails silently: a missed
        // disagreement is indistinguishable from agreement.
        //
        // The cost is on the cheap axis. Running a case costs one execution per
        // implementation; comparing walks arrays already in memory. At five
        // implementations that is five runs against ten array walks.
        let mut complaints: Vec<String> = Vec::new();
        // The tolerance a report cites. With per-pair bounds there is no single number for
        // the case, so the report names the one in force for the pair it describes.
        let mut worst_tolerance: Option<Tolerance> = None;

        // Whether any comparison actually examined arithmetic. A result entirely
        // undefined on both sides agrees without checking anything, and reporting that as
        // a pass would inflate the evidence a campaign appears to provide.
        let mut examined_arithmetic = false;
        let mut vacuous_elements = 0usize;

        // Which implementations agreed with which, so disagreement can be reported as
        // groups rather than as a list of pairs.
        let mut agrees: Vec<Vec<bool>> = vec![vec![false; outputs.len()]; outputs.len()];

        for left in 0..outputs.len() {
            agrees[left][left] = true;
            for right in (left + 1)..outputs.len() {
                let (a, b) = (&outputs[left], &outputs[right]);
                // Asked per pair, not once for the case: a bound can differ by *who* is
                // being compared, and a single tolerance would apply a CPU-versus-CPU
                // assumption to a CPU-versus-GPU comparison.
                let tolerance = self
                    .policy
                    .tolerance_for(input, (&a.implementation, &b.implementation));
                worst_tolerance = Some(tolerance);
                match a.output.approx_compare(&b.output, tolerance) {
                    Agreement::Agree(comparison) => {
                        agrees[left][right] = true;
                        agrees[right][left] = true;
                        if comparison.examined_any_arithmetic() {
                            examined_arithmetic = true;
                        } else {
                            vacuous_elements = vacuous_elements.max(comparison.total);
                        }
                    }
                    Agreement::Structural { reason } => complaints.push(format!(
                        "{} vs {}: {reason}",
                        a.implementation, b.implementation
                    )),
                    Agreement::Disagree(comparison) => complaints.push(format!(
                        "{} vs {}: {}",
                        a.implementation,
                        b.implementation,
                        describe(&comparison)
                    )),
                }
            }
        }

        if complaints.is_empty() {
            // Agreement reached without comparing a single number is not a pass. It is a
            // case that went unjudged, and saying so keeps the distinction visible.
            return if examined_arithmetic {
                Verdict::Agree
            } else {
                Verdict::Skipped(SkipReason::NothingComparable {
                    elements: vacuous_elements,
                })
            };
        }

        Verdict::Diverged(Divergence {
            input: format!("{input:?}"),
            // Every result is recorded, not only the disagreeing ones: knowing what the
            // others produced is what makes a report diagnosable.
            outputs: outputs
                .iter()
                .map(|o| (o.implementation.clone(), format!("{:?}", o.output)))
                .collect(),
            summary: {
                let tolerance = worst_tolerance.unwrap_or(Tolerance::EXACT);
                format!(
                    "{}{} (rtol {:e}, atol {:e})",
                    verdict_line(outputs, &agrees),
                    complaints.join("; "),
                    tolerance.rtol,
                    tolerance.atol
                )
            },
        })
    }
}

/// Partition implementations into groups that agree, and name the outlier if there is one.
///
/// **This is the capability a third implementation buys.** With two, an oracle can only
/// ever say "these disagree" — there is no basis for saying which is wrong, and burn#5284
/// is exactly that limitation in public. With three or more, a lone dissenter against a
/// consensus is visible, and saying so is a materially different claim.
///
/// Reported as a *hint*, never as a verdict. A majority is not proof: implementations can
/// share a bug (two backends built on the same kernel library will agree while both being
/// wrong), and the smaller group may be the correct one. The wording says "differs from",
/// not "is wrong".
///
/// Returns an empty string when everything agrees or when the split is not a clean
/// one-against-the-rest, since a partial claim would be worse than none.
fn verdict_line<C>(outputs: &[NamedOutput<C>], agrees: &[Vec<bool>]) -> String {
    if outputs.len() < 3 {
        return String::new();
    }

    // A lone dissenter agrees with nobody but itself, and every other implementation
    // agrees with every other. Anything more tangled is left to the pairwise complaints.
    for candidate in 0..outputs.len() {
        let alone =
            (0..outputs.len()).all(|other| (other == candidate) == agrees[candidate][other]);
        if !alone {
            continue;
        }

        let rest: Vec<&str> = (0..outputs.len())
            .filter(|&i| i != candidate)
            .map(|i| outputs[i].implementation.as_str())
            .collect();
        let consensus = rest.iter().enumerate().all(|(a, _)| {
            rest.iter().enumerate().all(|(b, _)| {
                let (ia, ib) = (index_of(outputs, rest[a]), index_of(outputs, rest[b]));
                agrees[ia][ib]
            })
        });

        if consensus {
            return format!(
                "{} differs from {} which agree; ",
                outputs[candidate].implementation,
                rest.join(" and ")
            );
        }
    }

    String::new()
}

fn index_of<C>(outputs: &[NamedOutput<C>], name: &str) -> usize {
    outputs
        .iter()
        .position(|o| o.implementation == name)
        .expect("name came from this slice")
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

    /// **The bug the old strategy hid, as a test.**
    ///
    /// Comparing everything against `outputs[0]` meant B-versus-C was never checked. With
    /// a tolerance of 0.15, A agrees with B (0.1 apart) and A agrees with C (0.1 apart),
    /// but B and C are 0.2 apart and genuinely disagree. The old oracle returned `Agree`.
    ///
    /// This is why "transitive enough" was wrong: approximate equality does not compose,
    /// and the error can accumulate to twice the tolerance across a chain.
    #[test]
    fn a_disagreement_that_avoids_the_first_implementation_is_still_caught() {
        let verdict = check_with(
            Tolerance {
                rtol: 0.0,
                atol: 0.15,
            },
            &[
                output("a", vec![1.0]),
                output("b", vec![0.9]),
                output("c", vec![1.1]),
            ],
        );

        match verdict {
            Verdict::Diverged(divergence) => {
                assert!(
                    divergence.summary.contains("b vs c"),
                    "the pair that disagrees must be named: {}",
                    divergence.summary
                );
            }
            other => panic!("b and c differ by more than the tolerance: {other:?}"),
        }
    }

    /// **The capability a third implementation buys.** With two, an oracle can only say
    /// "these disagree"; with three, a lone dissenter against a consensus is visible.
    #[test]
    fn a_lone_dissenter_is_named() {
        let verdict = check(&[
            output("cpu-a", vec![1.0]),
            output("cpu-b", vec![1.0]),
            output("gpu", vec![2.0]),
        ]);

        match verdict {
            Verdict::Diverged(divergence) => assert!(
                divergence.summary.contains("gpu differs from")
                    && divergence.summary.contains("cpu-a")
                    && divergence.summary.contains("cpu-b"),
                "the outlier and the consensus must both be named: {}",
                divergence.summary
            ),
            other => panic!("expected a divergence, got {other:?}"),
        }
    }

    /// A three-way split has no outlier, and claiming one would be worse than saying
    /// nothing. The pairwise complaints still carry the detail.
    #[test]
    fn a_three_way_disagreement_names_no_outlier() {
        let verdict = check(&[
            output("a", vec![1.0]),
            output("b", vec![2.0]),
            output("c", vec![3.0]),
        ]);

        match verdict {
            Verdict::Diverged(divergence) => assert!(
                !divergence.summary.contains("differs from"),
                "no implementation is alone against a consensus here: {}",
                divergence.summary
            ),
            other => panic!("expected a divergence, got {other:?}"),
        }
    }

    /// With two implementations there is no consensus to be alone against, so the
    /// grouping stays silent and the report reads exactly as it did before.
    #[test]
    fn two_implementations_get_no_outlier_claim() {
        let verdict = check(&[output("a", vec![1.0]), output("b", vec![2.0])]);

        match verdict {
            Verdict::Diverged(divergence) => assert!(
                !divergence.summary.contains("differs from"),
                "two implementations cannot establish a majority: {}",
                divergence.summary
            ),
            other => panic!("expected a divergence, got {other:?}"),
        }
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

    /// Two implementations both producing an undefined result do **not** disagree —
    /// relying on `==` would say they did, since NaN is not equal to itself — but nor is
    /// it a pass. Nothing was compared, so the case is recorded as unjudged.
    ///
    /// The distinction matters at volume: a campaign in which every case overflowed
    /// would otherwise report a wall of green while having verified nothing.
    #[test]
    fn an_entirely_undefined_result_is_skipped_rather_than_passed() {
        let verdict = check(&[
            output("a", vec![f32::NAN, f32::INFINITY]),
            output("b", vec![f32::NAN, f32::INFINITY]),
        ]);

        let Verdict::Skipped(SkipReason::NothingComparable { elements }) = verdict else {
            panic!("expected a nothing-comparable skip, got {verdict:?}");
        };
        assert_eq!(elements, 2);
    }

    /// But a result that is only *partly* undefined was still partly checked, and that
    /// counts as a genuine pass. Skipping it would discard real evidence.
    #[test]
    fn a_partly_undefined_result_still_agrees() {
        assert_eq!(
            check(&[
                output("a", vec![f32::NAN, 1.0]),
                output("b", vec![f32::NAN, 1.0]),
            ]),
            Verdict::Agree
        );
    }

    /// An undefined result on one side only remains a divergence — they disagree about
    /// whether an answer exists, which the skip path must not swallow.
    #[test]
    fn an_undefined_result_on_one_side_still_diverges() {
        assert!(matches!(
            check(&[output("a", vec![f32::NAN]), output("b", vec![1.0])]),
            Verdict::Diverged(_)
        ));
    }

    #[test]
    fn too_few_results_names_how_many_there_were() {
        let verdict = check(&[output("a", vec![1.0])]);
        assert_eq!(
            verdict,
            Verdict::Skipped(SkipReason::TooFewResults { available: 1 })
        );
    }
}
