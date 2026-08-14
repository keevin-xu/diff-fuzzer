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
//! Fifty-five features give 27,775 feature combinations and 215,930 signed predicates. That is
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

use crate::case::OnnxCase;
use crate::features::{FEATURES, FeatureVec, features};
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
    /// Other rules that scored **identically** and lost only on enumeration order.
    ///
    /// # Why this field exists
    ///
    /// The scoring order — fewest negatives, then most covered, then fewest features — can leave
    /// several rules exactly tied, and the loop used to commit whichever the enumeration reached
    /// first. **Measured, on this domain's first real run:**
    ///
    /// | rule | covers | negatives | predicts |
    /// |---|---|---|---|
    /// | `empty_tensor ∧ ¬output_larger_than_input` | 15/44 | 0/30,738 | 322/6,482 = **5%** |
    /// | `empty_tensor ∧ op_reshape` | 15/44 | 0/30,738 | 322/352 = **91%** |
    ///
    /// Identical on every criterion the search has, and one is a coincidence while the other is
    /// a trigger. **Fit cannot see prediction**, so the search has no basis for choosing — and
    /// silently choosing anyway reported the wrong rule. They are all surfaced now and the
    /// caller validates each.
    pub tied_with: Vec<Predicate>,
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
/// Rust note: this returns an owned `Vec` rather than an iterator. 215,930 predicates at 16
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
pub fn search(findings: &[OnnxCase], pool: &Pool) -> SearchResult {
    let positives: Vec<FeatureVec> = findings.iter().map(features).collect();
    // Extract once, not once per predicate: 215,930 × |negatives| computations would be the
    // one place where brute force actually costs something.
    let negatives: Vec<(Source, FeatureVec)> = pool
        .negatives()
        .iter()
        .map(|n| (n.source, features(&n.case)))
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
    let mut tied: Vec<Predicate> = Vec::new();

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
            tied_with: Vec::new(),
        };
        if candidate.negatives_matched() > 0 {
            continue;
        }

        // Ties broken by: more covered, then fewer features — and a genuine tie is **recorded**
        // rather than resolved by enumeration order. See `Candidate::tied_with`.
        match &best {
            None => best = Some(candidate),
            Some(current) => {
                let mine = (candidate.covered.len(), current.predicate.size());
                let theirs = (current.covered.len(), candidate.predicate.size());
                if mine > theirs {
                    tied.clear();
                    best = Some(candidate);
                } else if candidate.covered.len() == current.covered.len()
                    && candidate.predicate.size() == current.predicate.size()
                {
                    tied.push(candidate.predicate);
                }
            }
        }
    }

    best.map(|mut winner| {
        winner.tied_with = tied;
        winner
    })
}

/// `(source, matched, total)` per source, omitting sources with no members.
///
/// Mirrors `Pool::matched_by_source`, but over pre-extracted features so the covering loop
/// does not re-extract on every one of its 215,930 evaluations.
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
    use crate::case::{OpKind, TensorData, TensorValue};
    use crate::negatives::{Negative, Pool, Provenance, SamplingContext, Source};

    const OPSET: i64 = 22;

    fn scalar(op: OpKind, data: TensorData) -> OnnxCase {
        let n = data.len() as i64;
        OnnxCase::new(op, OPSET, vec![TensorValue::new("a", vec![n], data)])
    }

    fn context() -> SamplingContext {
        SamplingContext::new(
            "float-elementwise=on comparisons=on quantized=off special-values=on logic=abc",
            &["tract", "onnxruntime"],
        )
    }

    fn pool(cases: Vec<OnnxCase>) -> Pool {
        let ctx = context();
        Pool::matched(
            cases
                .into_iter()
                .map(|case| Negative {
                    case,
                    source: Source::Interesting,
                    provenance: ctx.provenance(),
                    generator: ctx.generator.clone(),
                    runtimes: ctx.runtimes.clone(),
                })
                .collect(),
            &ctx,
        )
        .expect("fixtures share the context they were built from")
    }

    /// The whole search space is enumerated, and it is small enough to state.
    ///
    /// Fifty-one features give 171,802 signed predicates. Reported so the space is not a mystery,
    /// and asserted so that adding a feature is a visible event rather than a silent one.
    #[test]
    fn the_search_space_is_the_size_it_claims() {
        let all = enumerate();
        assert_eq!(all.len(), 215930);
        assert!(
            all.iter().all(|p| !p.is_vacuous()),
            "the empty rule must never enter the space — it matches everything and claims nothing"
        );
        assert!(all.iter().all(|p| p.size() as usize <= MAX_FEATURES));
    }

    /// A rule is found when the findings share a property the negatives lack.
    ///
    /// The positives are signed zeros — the real F-005 trigger — and the negatives are ordinary
    /// values. `has_negative_zero` separates them.
    #[test]
    fn a_separating_rule_is_found_and_covers_the_findings() {
        let findings = vec![
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0])),
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0, 1.0])),
        ];
        let negatives = pool(vec![
            scalar(OpKind::Sign, TensorData::F32(vec![1.0])),
            scalar(OpKind::Sign, TensorData::F32(vec![2.0, 3.0])),
        ]);

        let result = search(&findings, &negatives);
        assert!(result.unexplained.is_empty(), "{:?}", result.unexplained);
        assert_eq!(result.classes.len(), 1);
        let rule = &result.classes[0];
        assert_eq!(rule.covered.len(), 2);
        assert!(
            rule.predicate.describe().contains("has_negative_zero"),
            "expected the signed-zero atom, got {}",
            rule.predicate.describe()
        );
    }

    /// **A rule matching any negative is rejected outright, not merely ranked lower.**
    ///
    /// Here the findings and the negatives are indistinguishable to the vocabulary: both are
    /// ordinary floats. No rule can separate them, so the honest output is *nothing explained*.
    /// A search that returned its least-bad rule would report a trigger already known to be false.
    #[test]
    fn a_rule_that_fires_on_a_negative_is_refused() {
        let findings = vec![scalar(OpKind::Abs, TensorData::F32(vec![1.0]))];
        let negatives = pool(vec![scalar(OpKind::Abs, TensorData::F32(vec![2.0]))]);

        let result = search(&findings, &negatives);
        assert!(result.classes.is_empty(), "{:?}", result.classes);
        assert_eq!(result.unexplained, vec![0]);
    }

    /// **The vocabulary gap is the most useful output, so it is reported by name** (N11.7).
    ///
    /// One finding is separable and one is not. The search must commit the rule it can justify
    /// *and* say plainly that the other is unexplained — never quietly widen a rule to cover it.
    #[test]
    fn what_cannot_be_explained_is_reported_rather_than_dropped() {
        let findings = vec![
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0])),
            scalar(OpKind::Abs, TensorData::F32(vec![1.0])),
        ];
        let negatives = pool(vec![scalar(OpKind::Abs, TensorData::F32(vec![2.0]))]);

        let result = search(&findings, &negatives);
        assert_eq!(result.classes.len(), 1);
        assert_eq!(result.classes[0].covered, vec![0]);
        assert_eq!(
            result.unexplained,
            vec![1],
            "the inseparable finding must survive as a gap"
        );
        assert_eq!(result.considered, 215930);
    }

    /// The covering loop can commit more than one rule, which is how distinct classes surface.
    #[test]
    fn several_classes_can_be_committed_in_one_run() {
        let findings = vec![
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0])),
            scalar(OpKind::Abs, TensorData::F32(vec![f32::NAN])),
        ];
        let negatives = pool(vec![scalar(OpKind::Abs, TensorData::F32(vec![1.0]))]);

        let result = search(&findings, &negatives);
        assert!(result.unexplained.is_empty(), "{:?}", result.unexplained);
        assert_eq!(
            result.classes.len(),
            2,
            "two unrelated triggers should not collapse into one rule"
        );
    }

    /// Negatives are reported **by source and never summed**.
    ///
    /// A rule firing on three hand-built near-misses and one firing on three ordinary cases are
    /// different claims — the near-misses were chosen to be hard. One total would hide that.
    #[test]
    fn negatives_are_broken_down_by_source() {
        let ctx = context();
        let mixed = Pool::matched(
            vec![
                Negative {
                    case: scalar(OpKind::Abs, TensorData::F32(vec![1.0])),
                    source: Source::Interesting,
                    provenance: ctx.provenance(),
                    generator: ctx.generator.clone(),
                    runtimes: ctx.runtimes.clone(),
                },
                Negative {
                    case: scalar(OpKind::Abs, TensorData::F32(vec![2.0])),
                    source: Source::NearMiss,
                    provenance: Provenance::Constructed,
                    generator: ctx.generator.clone(),
                    runtimes: ctx.runtimes.clone(),
                },
            ],
            &ctx,
        )
        .expect("same conditions");

        let findings = vec![scalar(OpKind::Sign, TensorData::F32(vec![-0.0]))];
        let result = search(&findings, &mixed);
        let by_source = &result.classes[0].negatives_by_source;
        assert_eq!(by_source.len(), 2, "both sources must appear separately");
        assert!(by_source.iter().all(|(_, matched, _)| *matched == 0));
    }

    /// **A tie must be reported, not resolved by enumeration order.**
    ///
    /// Two rules covering the same findings with the same number of terms and the same (empty)
    /// negative record are indistinguishable to the scoring, and the loop used to keep whichever
    /// it saw first. On this domain's first real run that silently discarded a 91%-predictive
    /// rule in favour of a 5% one. The search still cannot choose — fit cannot see prediction —
    /// but it can decline to hide the alternatives.
    #[test]
    fn rules_that_tie_are_all_surfaced() {
        // Two findings that share `has_negative_zero` and `op_sign`; both atoms separate them
        // from the negative equally well, so the two one-term rules tie exactly.
        let findings = vec![
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0])),
            scalar(OpKind::Sign, TensorData::F32(vec![-0.0, -0.0])),
        ];
        let negatives = pool(vec![scalar(OpKind::Abs, TensorData::F32(vec![1.0]))]);

        let result = search(&findings, &negatives);
        let committed = &result.classes[0];
        let all: Vec<String> = std::iter::once(committed.predicate)
            .chain(committed.tied_with.iter().copied())
            .map(|p| p.describe())
            .collect();

        assert!(
            all.len() > 1,
            "two equally-scoring rules exist here; only {} was surfaced",
            all[0]
        );
        assert!(
            all.iter().any(|r| r.contains("op_sign")),
            "the operator-keyed alternative must survive the tie: {all:?}"
        );
    }

    /// Same inputs, same result. A search that cannot be replayed is not evidence.
    #[test]
    fn the_search_is_deterministic() {
        let findings = vec![scalar(OpKind::Sign, TensorData::F32(vec![-0.0]))];
        let negatives = pool(vec![scalar(OpKind::Abs, TensorData::F32(vec![1.0]))]);
        assert_eq!(search(&findings, &negatives), search(&findings, &negatives));
    }
}
