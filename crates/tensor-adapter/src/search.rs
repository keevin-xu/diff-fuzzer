//! Brute-force search for predicates that explain a set of findings.
//!
//! # What it does
//!
//! Given findings (cases that **did** diverge) and a [`Pool`] of negatives (cases that did
//! **not**), enumerate every conjunction of at most three features, score each one, and
//! greedily pick a set of rules that covers the findings.
//!
//! # Why brute force
//!
//! Seventeen features give 833 feature combinations and 6,018 signed predicates. That is
//! nothing — the search runs in milliseconds. Anything cleverer (greedy feature selection,
//! a decision tree, an SAT encoding) would buy no speed that matters and would cost the one
//! property that does: **the developer must be able to describe the algorithm without
//! notes.** Enumerate, score, sort, take the best.
//!
//! # The scoring order, and why the first criterion comes first
//!
//! 1. **Fewest negatives matched.** A rule that fires on a case which did not diverge is
//!    not describing a trigger. This dominates everything else — a rule covering all
//!    findings is worthless if it also covers passing cases.
//! 2. **Most findings covered.** Among rules that stay off the negatives, prefer the one
//!    that explains more.
//! 3. **Fewest features.** Occam, and a two-term rule is readable where a four-term one is
//!    a description of the sample.
//!
//! # The `None` branch is the point
//!
//! When no predicate can cover the remaining findings without also matching negatives, the
//! search says so and lists what it could not explain. **That is a vocabulary gap** — the
//! features do not contain the property that distinguishes those cases — and it is the most
//! informative thing this tool emits, because it is the one output that points at its own
//! blind spot. It is never silently dropped.

use crate::features::{FEATURES, FeatureVec, extract};
use crate::input::TensorOp;
use crate::negatives::{Pool, Source};
use crate::predicate::Predicate;

/// The largest conjunction the search will consider.
///
/// Three is a judgment, not a tuning result: a rule naming four properties of an input has
/// usually stopped describing a mechanism and started describing the sample it was fitted
/// to. Raising it costs nothing in runtime and everything in credibility.
pub const MAX_FEATURES: usize = 3;

/// One rule, with the evidence for and against it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The rule itself.
    pub predicate: Predicate,
    /// Indices into the findings slice that this rule matches.
    pub covered: Vec<usize>,
    /// Negatives matched, broken down by source. **Never summed** — see [`Source`].
    pub negatives_by_source: Vec<(Source, usize, usize)>,
}

impl Candidate {
    /// Total negatives matched. Used *only* for ranking, never for reporting.
    ///
    /// Reporting a single number would conflate "fires on 3 near-misses" with "fires on 3
    /// ordinary cases", and the near-misses were chosen to be hard.
    fn negatives_matched(&self) -> usize {
        self.negatives_by_source.iter().map(|(_, n, _)| n).sum()
    }
}

/// What the search concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    /// Rules committed to, in the order the covering loop chose them.
    pub classes: Vec<Candidate>,
    /// Findings no rule could explain. **A vocabulary gap, and the most useful output.**
    pub unexplained: Vec<usize>,
    /// How many predicates were considered. Reported so the search space is not a mystery.
    pub considered: usize,
}

/// Enumerate every non-vacuous conjunction of at most [`MAX_FEATURES`] features.
///
/// For each choice of *k* distinct feature indices, every one of the 2^k sign assignments:
/// each chosen feature is either required or forbidden. `k = 0` is skipped because the
/// empty predicate matches everything.
///
/// Rust note: this returns an owned `Vec` rather than an iterator. 6,018 predicates at 8
/// bytes each is 48 KB — the simple thing is free here, and a hand-written iterator over
/// combinations-with-signs would be the clever way this module exists to avoid.
pub fn enumerate() -> Vec<Predicate> {
    let n = FEATURES.len();
    let mut out = Vec::new();

    // One feature.
    for a in 0..n {
        push_signed(&mut out, &[a]);
    }
    // Two features.
    for a in 0..n {
        for b in (a + 1)..n {
            push_signed(&mut out, &[a, b]);
        }
    }
    // Three features.
    for a in 0..n {
        for b in (a + 1)..n {
            for c in (b + 1)..n {
                push_signed(&mut out, &[a, b, c]);
            }
        }
    }
    out
}

/// Emit every sign assignment over the given feature indices.
///
/// The `signs` bit loop is the whole trick: bit *i* of `signs` decides whether feature
/// `bits[i]` goes into `required` or into `forbidden`.
fn push_signed(out: &mut Vec<Predicate>, bits: &[usize]) {
    for signs in 0..(1u32 << bits.len()) {
        let mut predicate = Predicate::default();
        for (i, &bit) in bits.iter().enumerate() {
            if signs & (1 << i) == 0 {
                predicate.required |= 1 << bit;
            } else {
                predicate.forbidden |= 1 << bit;
            }
        }
        out.push(predicate);
    }
}

/// Run the search.
///
/// `findings` are cases that diverged; `pool` holds cases that did not. The pool is a
/// [`Pool`] rather than a plain slice because obtaining one required proving the negatives
/// were drawn from the same distribution as the findings — see `negatives::Pool::matched`.
pub fn search(findings: &[TensorOp], pool: &Pool) -> SearchResult {
    let positives: Vec<FeatureVec> = findings.iter().map(extract).collect();
    // Extract once, not once per predicate: 6,018 × |negatives| extractions would be the
    // one place where brute force actually costs something.
    let negatives: Vec<(Source, FeatureVec)> = pool
        .negatives()
        .iter()
        .map(|n| (n.source, extract(&n.case)))
        .collect();

    let predicates = enumerate();
    let mut classes = Vec::new();
    let mut remaining: Vec<usize> = (0..positives.len()).collect();

    // The covering loop. Each pass commits the best rule for whatever is still unexplained,
    // so one run can discover several distinct classes.
    //
    // Termination: every pass either commits a rule covering at least one remaining finding
    // — strictly shrinking `remaining` — or breaks. It cannot loop forever.
    while !remaining.is_empty() {
        let Some(best) = best_for(&predicates, &positives, &remaining, &negatives) else {
            break;
        };
        remaining.retain(|i| !best.covered.contains(i));
        classes.push(best);
    }

    SearchResult {
        classes,
        unexplained: remaining,
        considered: predicates.len(),
    }
}

/// The best rule for the currently unexplained findings, or `None` if nothing qualifies.
///
/// **A predicate matching any negative is rejected outright**, not merely penalised. The
/// scoring order names "fewest negatives" first, and the honest reading of first-and-
/// dominant is a hard filter: a rule that fires on a passing case has been falsified as a
/// trigger claim, and ranking it below a better rule would still leave it eligible to win
/// when nothing better exists. Returning `None` instead is the vocabulary gap, which is a
/// far more useful thing to report than a rule already known to be wrong.
fn best_for(
    predicates: &[Predicate],
    positives: &[FeatureVec],
    remaining: &[usize],
    negatives: &[(Source, FeatureVec)],
) -> Option<Candidate> {
    let mut best: Option<Candidate> = None;

    for &predicate in predicates {
        // Rust note: `debug_assert` rather than `assert` — `enumerate` cannot produce a
        // vacuous predicate, so this documents the invariant without paying for it.
        debug_assert!(!predicate.is_vacuous());

        let covered: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| predicate.matches(positives[i]))
            .collect();
        if covered.is_empty() {
            continue;
        }

        let by_source = negatives_by_source(negatives, predicate);
        let candidate = Candidate {
            predicate,
            covered,
            negatives_by_source: by_source,
        };
        if candidate.negatives_matched() > 0 {
            continue;
        }

        // Ties broken by: more covered, then fewer features.
        let better = match &best {
            None => true,
            Some(current) => {
                (candidate.covered.len(), current.predicate.size())
                    > (current.covered.len(), candidate.predicate.size())
            }
        };
        if better {
            best = Some(candidate);
        }
    }

    best
}

/// `(source, matched, total)` per source, omitting sources with no members.
///
/// Mirrors `Pool::matched_by_source`, but over pre-extracted features so the covering loop
/// does not re-extract on every one of its 6,018 evaluations.
fn negatives_by_source(
    negatives: &[(Source, FeatureVec)],
    predicate: Predicate,
) -> Vec<(Source, usize, usize)> {
    let mut out: Vec<(Source, usize, usize)> = Vec::new();
    for &(source, features) in negatives {
        let entry = match out.iter_mut().find(|(s, _, _)| *s == source) {
            Some(e) => e,
            None => {
                out.push((source, 0, 0));
                out.last_mut().expect("just pushed")
            }
        };
        entry.2 += 1;
        if predicate.matches(features) {
            entry.1 += 1;
        }
    }
    out.sort_by_key(|(s, _, _)| *s);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::TensorValue;
    use crate::negatives::{Negative, Provenance};

    fn matmul(lhs: (&[usize], &[f32]), rhs: (&[usize], &[f32])) -> TensorOp {
        TensorOp::matmul(
            TensorValue::new(lhs.0.to_vec(), lhs.1.to_vec()),
            TensorValue::new(rhs.0.to_vec(), rhs.1.to_vec()),
        )
    }

    /// The burn#5284 shape: opposite-sign products that each overflow.
    fn overflowing() -> TensorOp {
        matmul((&[1, 2], &[1e30, -1e30]), (&[2, 1], &[1e30, 1e30]))
    }

    /// A benign case: small same-sign values, nothing overflows.
    fn benign() -> TensorOp {
        let ones = vec![1.0f32; 4];
        matmul((&[1, 4], &ones), (&[4, 1], &ones))
    }

    fn pool(cases: Vec<(TensorOp, Source)>) -> Pool {
        Pool::unchecked(
            cases
                .into_iter()
                .map(|(case, source)| Negative {
                    case,
                    source,
                    provenance: Provenance::Constructed,
                })
                .collect(),
        )
    }

    /// Every combination of ≤3 of the 17 features, with every sign assignment.
    ///
    /// C(17,1)·2 + C(17,2)·4 + C(17,3)·8 = 34 + 544 + 5440 = 6018. Pinning the number keeps
    /// the enumeration honest: silently dropping the three-feature tier would still produce
    /// plausible-looking output.
    #[test]
    fn the_enumeration_covers_every_signed_combination_of_at_most_three_features() {
        let all = enumerate();

        assert_eq!(all.len(), 6018);
        assert!(all.iter().all(|p| p.size() <= MAX_FEATURES as u32));
        assert!(
            all.iter().all(|p| !p.is_vacuous()),
            "the empty rule matches everything and must never be a candidate"
        );
        // Required and forbidden are disjoint by construction.
        assert!(all.iter().all(|p| p.required & p.forbidden == 0));
    }

    /// **The criterion the whole search rests on.** A rule that fires on a case which did
    /// not diverge is not a trigger claim, and must not be committed however many findings
    /// it happens to cover.
    #[test]
    fn a_predicate_matching_a_committed_negative_is_rejected() {
        // `all_same_sign` matches the benign case too — it is the classic over-broad rule.
        let findings = vec![benign()];
        let negatives = pool(vec![(benign(), Source::NearMiss)]);

        let result = search(&findings, &negatives);

        assert!(
            result.classes.is_empty(),
            "committed a rule that fires on a passing case: {}",
            result.classes[0].predicate.describe()
        );
        assert_eq!(
            result.unexplained,
            vec![0],
            "and the finding must be reported as unexplained, not quietly dropped"
        );
    }

    /// A finding genuinely separable from the negatives gets a rule, and that rule fires on
    /// none of them.
    #[test]
    fn a_separable_finding_gets_a_rule_that_stays_off_the_negatives() {
        let findings = vec![overflowing()];
        let negatives = pool(vec![(benign(), Source::NearMiss)]);

        let result = search(&findings, &negatives);

        assert_eq!(result.classes.len(), 1);
        let class = &result.classes[0];
        assert_eq!(class.covered, vec![0]);
        assert!(
            class.negatives_by_source.iter().all(|(_, n, _)| *n == 0),
            "{} fires on a negative",
            class.predicate.describe()
        );
        assert!(result.unexplained.is_empty());
    }

    /// **The vocabulary gap.** When the features cannot tell a finding apart from a passing
    /// case, the search must say so rather than invent a rule.
    #[test]
    fn a_finding_indistinguishable_from_a_negative_is_reported_as_unexplained() {
        // Identical cases on both sides: no predicate over any vocabulary can separate them.
        let findings = vec![overflowing()];
        let negatives = pool(vec![(overflowing(), Source::NearMiss)]);

        let result = search(&findings, &negatives);

        assert!(result.classes.is_empty());
        assert_eq!(result.unexplained, vec![0]);
    }

    /// The covering loop must halt. Each pass strictly shrinks the remaining set or breaks;
    /// this exercises both exits on a mixed input.
    #[test]
    fn the_covering_loop_terminates() {
        // Two separable findings plus one that duplicates a negative and never can be.
        let findings = vec![overflowing(), benign(), overflowing()];
        let negatives = pool(vec![
            (benign(), Source::NearMiss),
            (overflowing(), Source::Ordinary),
        ]);

        let result = search(&findings, &negatives);

        // Every finding is either covered exactly once or listed as unexplained.
        let covered: usize = result.classes.iter().map(|c| c.covered.len()).sum();
        assert_eq!(covered + result.unexplained.len(), findings.len());
        assert_eq!(result.considered, 6018);
    }

    /// Rules are committed in order of how much they explain, and no finding is claimed
    /// twice — the second pass only ever sees what the first left behind.
    #[test]
    fn the_covering_loop_finds_several_classes_without_double_counting() {
        // Two distinct kinds of finding, separable from the negative by different features.
        let findings = vec![
            overflowing(),
            matmul((&[2, 2, 2], &[1e30; 8]), (&[2, 2, 2], &[1e30; 8])),
        ];
        let negatives = pool(vec![(benign(), Source::Ordinary)]);

        let result = search(&findings, &negatives);

        let mut all: Vec<usize> = result
            .classes
            .iter()
            .flat_map(|c| c.covered.clone())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(
            all.len(),
            result
                .classes
                .iter()
                .map(|c| c.covered.len())
                .sum::<usize>(),
            "a finding was covered by two committed rules"
        );
    }

    /// Negatives are reported per source with their totals, so a reader can see that a rule
    /// survived twelve *near-misses* rather than twelve cases of unstated difficulty.
    #[test]
    fn negatives_are_reported_by_source_with_totals() {
        let findings = vec![overflowing()];
        let negatives = pool(vec![
            (benign(), Source::NearMiss),
            (benign(), Source::Ordinary),
            (benign(), Source::Ordinary),
        ]);

        let result = search(&findings, &negatives);

        assert_eq!(
            result.classes[0].negatives_by_source,
            vec![(Source::NearMiss, 0, 1), (Source::Ordinary, 0, 2)]
        );
    }

    /// With nothing to separate findings from, the search still must not commit a vacuous
    /// or trivially-true rule — but with no negatives it has no evidence either. An empty
    /// pool is refused upstream by `Pool::matched`; `unchecked` is the escape hatch, and
    /// this pins what it does.
    #[test]
    fn with_no_negatives_the_shortest_rule_wins() {
        let findings = vec![overflowing()];
        let result = search(&findings, &Pool::unchecked(Vec::new()));

        assert_eq!(result.classes.len(), 1);
        assert_eq!(
            result.classes[0].predicate.size(),
            1,
            "nothing to exclude, so the fewest-features tiebreak decides"
        );
    }
}
