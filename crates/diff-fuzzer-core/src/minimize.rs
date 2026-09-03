//! Shrinking a failing case to the smallest one that still fails.
//!
//! A generated divergence arrives at whatever size the generator happened to produce —
//! possibly a rank-4 tensor of several thousand values, most of which have nothing to do
//! with the failure. Nobody can act on that. A maintainer receiving it has to first
//! work out which part matters, which is work we are better placed to do automatically.
//!
//! The search is a **greedy first-improvement hill climb** over domain-proposed moves: try
//! candidates in order, accept the first that still fails, and restart from it. It stops when
//! no candidate improves, which is a **local** minimum.
//!
//! This is the same predicate-guided idea as delta debugging, but not Zeller's `ddmin`: there is
//! no partition into subsets, no complement testing, no granularity schedule, and therefore no
//! 1-minimality guarantee. Do not describe the result as minimal, only as locally minimal.
//!
//! This module holds the [`Shrink`] capability, which asks a domain "what simpler
//! versions of this are there?". The search that uses it lives alongside, and the moves
//! themselves are necessarily domain knowledge: only the tensor adapter knows that
//! halving a matrix multiplication's inner dimension means changing *both* operands.

/// A value that can propose simpler versions of itself.
///
/// Two obligations, and both matter for the search to terminate and to be trustworthy.
///
/// **Every candidate must be valid.** A shrunk case still has to be something the
/// systems under test will accept — halving one operand of an elementwise operation
/// without halving the other produces a case that cannot run, which wastes a step and
/// teaches nothing. Constraints that held for the generated case must hold for every
/// candidate.
///
/// **Every candidate must be strictly simpler.** If a candidate could be as complex as
/// its parent, the search could cycle forever. "Simpler" here means fewer elements, or
/// values closer to zero — never more of either.
pub trait Shrink: Sized {
    /// Simpler versions of this value, **most aggressive first**.
    ///
    /// Order matters for speed rather than correctness. A greedy search takes the first
    /// candidate that still fails, so offering the biggest reduction first means fewer
    /// rounds to reach the same place: halving a dimension gets there faster than
    /// removing one element at a time.
    fn candidates(&self) -> Vec<Self>;
}

/// Limits on how hard the search may work.
///
/// The `Shrink` contract already implies termination — every candidate is strictly
/// simpler, so the search cannot revisit a case. **Relying on a contract for termination
/// is not the same as enforcing it.** A shrinker with a subtle bug that returns a
/// candidate equal to its parent would loop forever, and an infinite loop inside a
/// reporting path is a far worse failure than an imperfectly shrunk case.
///
/// The step and candidate limits are **deterministic**: the same case shrinks the same
/// way on any machine. The optional time limit is not, and that is discussed on the
/// field itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Most reductions to accept. Generous: a case shrinking a hundred times is
    /// converging, not stuck.
    pub max_steps: usize,

    /// Most candidates to evaluate. This is the limit that actually bites, since each
    /// evaluation is a full run on every system under test.
    pub max_candidates: usize,

    /// Optional wall-clock limit.
    ///
    /// **Off by default, because it costs determinism.** A time limit makes the result
    /// depend on how fast the machine happens to be, so the same case could minimise
    /// differently on two computers — and a minimised reproduction that varies by
    /// machine is not much of a reproduction. The deterministic limits above are the
    /// intended protection; this exists for the case where a single evaluation is
    /// pathologically slow, and when it fires the outcome records that it did.
    pub max_duration: Option<std::time::Duration>,
}

impl Default for Budget {
    fn default() -> Self {
        Self {
            max_steps: 200,
            max_candidates: 10_000,
            max_duration: None,
        }
    }
}

/// Why the search stopped.
///
/// Recorded because **a case that ran out of budget is not minimal**, and claiming
/// otherwise would overstate the result. A report saying "minimised to two elements"
/// means something different from "stopped at two elements with reductions still
/// untried", and the difference is exactly the kind that quietly inflates a claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum StopReason {
    /// Nothing simpler still fails. The result is minimal for the available moves.
    LocalMinimum,
    /// Hit [`Budget::max_steps`].
    StepBudget,
    /// Hit [`Budget::max_candidates`].
    CandidateBudget,
    /// Hit [`Budget::max_duration`]. **The only non-deterministic outcome.**
    TimeBudget,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StopReason::LocalMinimum => write!(f, "no simpler case still fails"),
            StopReason::StepBudget => write!(f, "step budget exhausted; not minimal"),
            StopReason::CandidateBudget => write!(f, "candidate budget exhausted; not minimal"),
            StopReason::TimeBudget => write!(f, "time budget exhausted; not minimal"),
        }
    }
}

/// The outcome of shrinking, with enough detail to say what it achieved.
///
/// The counts are not decoration. "Shrunk to a rank-1 tensor of two elements" invites
/// the question *from what*, and a reproduction is far more convincing when the report
/// can say it came down from several thousand values. The number of candidates tried is
/// the cost side of that: each one is a full execution on every system under test, so it
/// is the figure to watch if minimisation ever becomes the slow part of a campaign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Minimized<T> {
    /// The smallest failing case found.
    pub input: T,
    /// How many reductions were accepted.
    pub steps: usize,
    /// How many candidates were evaluated, accepted or not.
    pub candidates_tried: usize,
    /// Why the search stopped. Only [`StopReason::LocalMinimum`] means the result is
    /// actually minimal.
    pub stopped: StopReason,
}

impl<T> Minimized<T> {
    /// Did the search finish, rather than run out of budget?
    ///
    /// Worth checking before describing a result as minimal in a report.
    pub fn is_minimal(&self) -> bool {
        self.stopped == StopReason::LocalMinimum
    }
}

/// Shrink a failing case to a locally minimal one.
///
/// `still_fails` decides whether a candidate still exhibits the failure — in practice,
/// re-running it on every system and asking the oracle. **It must be deterministic.** If
/// the same candidate can answer differently on two calls, the search wanders and the
/// result is not reproducible, which would defeat the entire purpose.
///
/// The search is **greedy first-improvement**: take the first candidate that still
/// fails, and start again from there. Combined with candidates being ordered
/// most-aggressive-first, that reaches a small case quickly. It finds a *local* minimum,
/// not a global one — a smaller failing case may exist by some other route. That is the
/// right trade: each evaluation costs a full run on every backend, and an exhaustive
/// search would cost far more than the marginal readability is worth.
///
/// Termination rests on `Shrink`'s contract that every candidate is strictly simpler
/// than its parent, so the loop cannot revisit a case. An explicit budget is layered on
/// top separately, because relying on a contract for termination is not the same as
/// enforcing it.
///
/// If the case does not fail to begin with, it is returned untouched with no steps
/// taken. That is a caller error rather than something to assert on: minimisation is
/// often invoked from a reporting path, and crashing there would lose the finding it was
/// called to describe.
pub fn minimize<T, P>(input: T, still_fails: P) -> Minimized<T>
where
    T: Shrink,
    P: FnMut(&T) -> bool,
{
    minimize_within(input, Budget::default(), still_fails)
}

/// [`minimize`], with explicit limits on how hard to work.
pub fn minimize_within<T, P>(input: T, budget: Budget, mut still_fails: P) -> Minimized<T>
where
    T: Shrink,
    P: FnMut(&T) -> bool,
{
    let started = std::time::Instant::now();
    let out_of_time = || {
        budget
            .max_duration
            .is_some_and(|limit| started.elapsed() >= limit)
    };

    let mut current = input;
    let mut steps = 0;
    let mut candidates_tried = 0;

    if !still_fails(&current) {
        return Minimized {
            input: current,
            steps: 0,
            candidates_tried: 1,
            stopped: StopReason::LocalMinimum,
        };
    }

    let stopped = 'shrinking: loop {
        if steps >= budget.max_steps {
            break StopReason::StepBudget;
        }

        for candidate in current.candidates() {
            if candidates_tried >= budget.max_candidates {
                break 'shrinking StopReason::CandidateBudget;
            }
            if out_of_time() {
                break 'shrinking StopReason::TimeBudget;
            }

            candidates_tried += 1;

            if still_fails(&candidate) {
                current = candidate;
                steps += 1;
                // Start again from the smaller case: reductions that were unavailable
                // before may have become possible, and ones already rejected may now
                // succeed on a different shape.
                continue 'shrinking;
            }
        }

        // Nothing simpler still fails, so this is a local minimum.
        break StopReason::LocalMinimum;
    };

    Minimized {
        input: current,
        steps,
        candidates_tried,
        stopped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in for a test case: a list of numbers that shrinks by dropping elements
    /// or halving them. Deliberately unlike a tensor — the search should not know or
    /// care what it is shrinking.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Sample(Vec<u32>);

    impl Shrink for Sample {
        fn candidates(&self) -> Vec<Self> {
            let mut out = Vec::new();

            // Most aggressive first: drop an element.
            for index in 0..self.0.len() {
                let mut smaller = self.0.clone();
                smaller.remove(index);
                out.push(Sample(smaller));
            }
            // Then reduce a value.
            for index in 0..self.0.len() {
                if self.0[index] > 0 {
                    let mut smaller = self.0.clone();
                    smaller[index] /= 2;
                    out.push(Sample(smaller));
                }
            }

            out
        }
    }

    /// The canonical delta-debugging demonstration: a failure caused by one property of
    /// one element, buried in a large input, must shrink to just that element.
    ///
    /// The result is `[112]` rather than `[100]`, and that is worth understanding rather
    /// than working around. Halving takes `900` to `450`, `225`, `112` — and then `56`,
    /// which no longer fails, so `112` is where it stops. **The true minimum is `100`,
    /// and this search does not find it**, because it reaches a local minimum by the
    /// moves available rather than searching exhaustively.
    ///
    /// That is the intended trade. Each evaluation costs a full run on every system
    /// under test, and closing the last 12% of the gap would cost far more than the
    /// marginal readability is worth. What matters is that six values became one.
    #[test]
    fn shrinks_to_the_element_that_causes_the_failure() {
        let large = Sample(vec![3, 7, 900, 2, 41, 8]);

        // Fails only while some element is at least 100.
        let result = minimize(large, |s| s.0.iter().any(|v| *v >= 100));

        assert_eq!(result.input, Sample(vec![112]));
        assert!(result.steps > 0);
    }

    /// A local minimum is genuinely minimal *with respect to the available moves*: no
    /// single candidate of the result still fails. That is the property the search
    /// actually guarantees, and so the one worth asserting.
    #[test]
    fn no_candidate_of_the_result_still_fails() {
        let predicate = |s: &Sample| s.0.iter().any(|v| *v >= 100);
        let result = minimize(Sample(vec![3, 7, 900, 2, 41, 8]), predicate);

        for candidate in result.input.candidates() {
            assert!(
                !predicate(&candidate),
                "{candidate:?} still fails, so {:?} was not minimal",
                result.input
            );
        }
    }

    /// Everything irrelevant must go. A predicate that ignores the input's contents
    /// entirely should leave nothing behind.
    #[test]
    fn removes_everything_the_failure_does_not_need() {
        let result = minimize(Sample(vec![1, 2, 3, 4, 5]), |_| true);
        assert_eq!(result.input, Sample(vec![]));
    }

    /// A case that cannot be reduced further is returned as it is, having tried and
    /// rejected every candidate.
    #[test]
    fn a_minimal_case_is_left_alone() {
        let smallest = Sample(vec![]);
        let result = minimize(smallest.clone(), |s| *s == smallest);

        assert_eq!(result.input, smallest);
        assert_eq!(result.steps, 0);
    }

    /// **The property the whole search depends on.** The same case and predicate must
    /// always shrink to the same result — a minimised reproduction that varied between
    /// runs would be no more actionable than the original.
    #[test]
    fn minimisation_is_deterministic() {
        let case = Sample(vec![9, 4, 250, 17, 3]);
        let predicate = |s: &Sample| s.0.iter().any(|v| *v >= 100);

        let first = minimize(case.clone(), predicate);
        let second = minimize(case, predicate);

        assert_eq!(first, second);
    }

    /// A case that never failed is returned untouched. Minimisation is called from a
    /// reporting path, so this must not crash and lose the finding.
    #[test]
    fn a_case_that_does_not_fail_is_returned_unchanged() {
        let case = Sample(vec![1, 2, 3]);
        let result = minimize(case.clone(), |_| false);

        assert_eq!(result.input, case);
        assert_eq!(result.steps, 0);
    }

    /// The search reports its own cost, since each candidate is a full execution on
    /// every system under test.
    #[test]
    fn the_cost_of_the_search_is_reported() {
        let result = minimize(Sample(vec![5, 200, 6]), |s| s.0.iter().any(|v| *v >= 100));

        assert!(result.candidates_tried >= result.steps);
        assert!(result.candidates_tried > 0);
    }

    /// **The guarantee the budget exists for.** A shrinker that violates its contract by
    /// returning a candidate no simpler than its parent would otherwise loop forever —
    /// and an infinite loop inside a reporting path is a far worse failure than an
    /// imperfectly shrunk case.
    ///
    /// This shrinker is deliberately broken in exactly that way. The test passing at all
    /// is the point: it terminates.
    #[test]
    fn a_shrinker_that_never_makes_progress_still_terminates() {
        #[derive(Debug, Clone, PartialEq, Eq)]
        struct NeverShrinks(u32);

        impl Shrink for NeverShrinks {
            fn candidates(&self) -> Vec<Self> {
                // Contract violation: identical to its parent, forever.
                vec![NeverShrinks(self.0)]
            }
        }

        let budget = Budget {
            max_steps: 50,
            max_candidates: 500,
            max_duration: None,
        };
        let result = minimize_within(NeverShrinks(1), budget, |_| true);

        assert_eq!(result.stopped, StopReason::StepBudget);
        assert!(!result.is_minimal());
        assert_eq!(result.steps, 50);
    }

    /// Running out of budget must be reported, not hidden. A case that stopped early is
    /// **not** minimal, and a report claiming otherwise would overstate the result.
    #[test]
    fn exhausting_the_candidate_budget_is_reported() {
        let budget = Budget {
            max_steps: usize::MAX,
            max_candidates: 3,
            max_duration: None,
        };
        let result = minimize_within(Sample(vec![1; 40]), budget, |_| true);

        assert_eq!(result.stopped, StopReason::CandidateBudget);
        assert!(!result.is_minimal());
        assert!(result.candidates_tried <= 3);
    }

    /// A finished search says so, which is what lets a report distinguish "minimised" from
    /// "gave up here".
    #[test]
    fn a_completed_search_reports_a_local_minimum() {
        let result = minimize(Sample(vec![5, 200, 6]), |s| s.0.iter().any(|v| *v >= 100));

        assert_eq!(result.stopped, StopReason::LocalMinimum);
        assert!(result.is_minimal());
    }

    /// The default budget must be generous enough that ordinary cases finish rather than
    /// being truncated — a limit that fires routinely would silently degrade every
    /// report.
    #[test]
    fn the_default_budget_does_not_truncate_an_ordinary_case() {
        let result = minimize(Sample(vec![900; 30]), |s| s.0.iter().any(|v| *v >= 100));
        assert!(result.is_minimal(), "stopped early: {}", result.stopped);
    }

    /// **Determinism by default.** No time limit is set unless a caller asks for one,
    /// because a wall-clock bound makes the result depend on machine speed — and a
    /// minimised reproduction that differs between computers is not much of a
    /// reproduction.
    #[test]
    fn the_default_budget_sets_no_time_limit() {
        assert_eq!(Budget::default().max_duration, None);
    }

    /// The predicate must never be asked about a case the search would not accept, and
    /// every case it *does* accept must satisfy it. Checking after the fact guards
    /// against a search that returns something which does not actually fail.
    #[test]
    fn the_result_still_fails() {
        let predicate = |s: &Sample| s.0.len() >= 2 && s.0[0] > s.0[1];
        let result = minimize(Sample(vec![8, 1, 4, 9, 2]), predicate);

        assert!(predicate(&result.input), "{:?} does not fail", result.input);
    }
}
