//! Signatures that have already been investigated, and what came of it.
//!
//! # Why this exists
//!
//! A campaign's output is mostly things already known. The `matmul` overflow divergence is
//! reachable within a few hundred thousand executions, so every long run rediscovers it —
//! and if triage presents it beside a genuinely new problem with equal prominence, **the
//! new one is what gets missed.** The scarce resource in triage is attention, not disk.
//!
//! So a signature seen before is recorded here with its outcome, and triage sorts what is
//! new to the top.
//!
//! # What this is not
//!
//! **Not a filter.** Nothing is hidden or discarded — known findings are still counted,
//! still checked for reproduction, still listed. Suppressing them would mean a *change* in
//! a known problem's behaviour went unnoticed, which is exactly the kind of thing worth
//! noticing. This changes the order and the labelling, nothing else.
//!
//! **Not a verdict.** An entry means "we looked", not "it is fine". `Status` records which.

use crate::predicate::Predicate;

/// What was concluded about a signature, and how much that conclusion is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Reported upstream; no answer yet. **The question is open** — this is the weakest
    /// kind of "known", and a reader should treat it as unresolved rather than settled.
    Reported,
    /// Upstream confirmed the behaviour is permitted. Safe to stop re-triaging.
    ConfirmedLegal,
    /// Upstream confirmed a defect. Still open until a fixed version ships.
    ConfirmedBug,
}

impl Status {
    /// A short label for a report.
    pub fn label(self) -> &'static str {
        match self {
            Status::Reported => "reported, awaiting reply",
            Status::ConfirmedLegal => "confirmed legal upstream",
            Status::ConfirmedBug => "confirmed a bug upstream",
        }
    }

    /// Whether this conclusion is settled, or still an open question.
    ///
    /// `Reported` is deliberately **not** settled: filing an issue is not the same as
    /// learning the answer, and a triage report that conflated them would quietly turn an
    /// open question into a closed one.
    pub fn is_settled(self) -> bool {
        !matches!(self, Status::Reported)
    }
}

/// A signature we have already looked at.
#[derive(Debug, Clone, Copy)]
pub struct Known {
    /// The exact signature string, as produced by [`crate::signature`].
    pub signature: &'static str,
    pub status: Status,
    /// Where the investigation is written down — an issue URL, or a local draft.
    pub reference: &'static str,
    /// One line on what it is, so a reader need not open the reference.
    pub note: &'static str,
    /// The trigger this class is believed to have, if one has been ratified.
    ///
    /// **`None` is the honest default and is not a placeholder.** A signature says what a
    /// disagreement *looked like*; a predicate claims what an input must *contain* to cause
    /// it. The second is a much stronger statement and is only earned by a search that has
    /// scored a candidate against cases which did **not** diverge.
    ///
    /// The obvious rule for the one entry here — `overflow_product ∧ mixed_sign_overflow` —
    /// is deliberately **not** recorded: it was measured to match non-diverging cases too,
    /// making it necessary and not sufficient. Writing it down would turn a falsified guess
    /// into a recorded fact.
    pub predicate: Option<Predicate>,
}

/// Every signature investigated so far.
///
/// Deliberately short. An entry is a claim that someone worked through the triage ladder
/// for that signature, so adding one is a decision, not bookkeeping.
pub const KNOWN: &[Known] = &[Known {
    signature: "matmul/rank2/undefined",
    // **Answered by a maintainer on 2026-08-04**, not merely reported: "I don't think
    // inf / NaN should be interchangeable, it's a divergence that then propagates
    // non-uniformly through downstream ops." So the open question this was filed as is
    // closed, and the finding stands.
    status: Status::ConfirmedBug,
    reference: "https://github.com/tracel-ai/burn/issues/5284",
    note: "matmul intermediate products overflow f32; ndarray fuses the multiply-add and \
           yields ±inf where tch rounds first and yields NaN. Root cause confirmed: \
           libtorch's GEMM fuses inside a 4x8 micro-kernel and not in the trailing-corner \
           cleanup, so disagreeing elements number (m mod 4) * (n mod 8). Maintainer \
           notes burn has no explicit cross-backend numerical-agreement contract yet",
    // Left `None` until a search proposes one and a human ratifies it (7B.5–7B.7). The
    // tempting `overflow_product AND mixed_sign_overflow` was falsified by measurement:
    // it matches cases that agree.
    predicate: None,
}];

/// Look up a class by the **trigger** its case carries, rather than by symptom.
///
/// This is the lookup a signature cannot perform: it asks *"does anything already explain
/// an input like this?"*, which is answerable for a case that has never been run.
///
/// Returns `None` while no entry has a ratified predicate — which is the state today, and
/// is why `PHASE-7B`'s search exists.
pub fn known_by_predicate(features: crate::features::FeatureVec) -> Option<&'static Known> {
    KNOWN
        .iter()
        .find(|known| known.predicate.is_some_and(|p| p.matches(features)))
}

/// Look up what is known about a signature, if anything.
pub fn known_issue(signature: &str) -> Option<&'static Known> {
    KNOWN.iter().find(|known| known.signature == signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{TensorOp, TensorValue};
    use crate::normalize::CanonicalTensor;
    use crate::signature::signature;
    use diff_fuzzer_core::Tolerance;

    /// **Guards the signature *rule*, not the backends.** An entry is matched by exact
    /// string, so a change to how signatures are built silently orphans every entry —
    /// triage would report long-settled problems as new, and the registry would look fine
    /// while doing nothing.
    ///
    /// Note the outputs here are **written by hand**, not produced by running anything.
    /// That is deliberate and it is also this test's limit: it cannot tell you whether any
    /// backend still *produces* this class. See
    /// `the_recorded_class_is_still_reachable_from_real_backends` for that half.
    #[test]
    fn the_recorded_matmul_signature_is_what_the_rule_still_produces() {
        let case = TensorOp::matmul(
            TensorValue::new(vec![1, 2], vec![1e30, -1e30]),
            TensorValue::new(vec![2, 1], vec![1e30, 1e30]),
        );

        let canon = |v: f32| CanonicalTensor {
            shape: vec![1, 1],
            dtype: "F32".to_string(),
            values: vec![v],
        };

        let produced = signature(
            &case,
            &canon(f32::INFINITY),
            &canon(f32::NAN),
            Tolerance::EXACT,
        );

        assert!(
            known_issue(&produced).is_some(),
            "the signature rule now produces {produced:?}, which no entry matches — \
             the registry has been orphaned and triage would report this as new"
        );
    }

    /// **The half the rule test cannot cover: is this class still reachable at all?**
    ///
    /// A registry entry claims someone worked the triage ladder for a signature. If no
    /// backend combination produces that signature any more — because a backend was
    /// swapped, or a library fixed it — the entry points at nothing, and **the rule test
    /// above would stay green**, since it supplies its own outputs.
    ///
    /// Added at PHASE-7A step 7A.1, before replacing `ndarray` with `flex`. The case is
    /// chosen deliberately: `[14,4] × [4,27]` diverges under **any** pair including `tch`,
    /// because libtorch alone leaves a non-fusing trailing corner. The originally filed
    /// `[1,2] × [2,1]` does **not** — `flex` and `tch` both return `NaN` there and agree,
    /// so a test built on it would start failing the moment `ndarray` left.
    ///
    /// The distinction that phase turns on: **the minimal reproduction died, the finding
    /// did not.** A case is not a class.
    #[test]
    fn the_recorded_class_is_still_reachable_from_real_backends() {
        use crate::backends::{flex, libtorch};
        use crate::normalize::TensorNormalizer;
        use crate::signature::signature_across;
        use crate::tolerance::TensorTolerancePolicy;
        use diff_fuzzer_core::{Implementation, Normalizer, TolerancePolicy};

        // Sign alternates along the contraction axis, so every dot product contains both a
        // positively- and a negatively-overflowing product. The exact answer is zero.
        let (m, k, n) = (14usize, 4usize, 27usize);
        let lhs: Vec<f32> = (0..m * k)
            .map(|i| {
                if (i % k).is_multiple_of(2) {
                    1e30
                } else {
                    -1e30
                }
            })
            .collect();
        let case = TensorOp::matmul(
            TensorValue::new(vec![m, k], lhs),
            TensorValue::new(vec![k, n], vec![1e30; k * n]),
        );

        let outputs: Vec<(String, CanonicalTensor)> = [
            ("burn-flex", flex().run(&case)),
            ("burn-tch", libtorch().run(&case)),
        ]
        .into_iter()
        .filter_map(|(name, raw)| Some((name.to_string(), TensorNormalizer.normalize(raw.ok()?))))
        .collect();

        assert_eq!(outputs.len(), 2, "both backends must run this case");

        let tolerance = TensorTolerancePolicy
            .tolerance_for(&case, (outputs[0].0.as_str(), outputs[1].0.as_str()));
        let (produced, pair) = signature_across(&case, &outputs, tolerance);

        assert!(
            pair.is_some(),
            "the recorded class is no longer reachable: {produced} — either a backend \
             changed, or this case needs replacing with one that still exhibits it"
        );
        assert!(
            known_issue(&produced).is_some(),
            "real backends produce {produced:?}, which no registry entry matches"
        );
    }

    /// Filing an issue is not the same as getting an answer.
    #[test]
    fn a_merely_reported_signature_is_not_settled() {
        assert!(!Status::Reported.is_settled());
        assert!(Status::ConfirmedLegal.is_settled());
        assert!(Status::ConfirmedBug.is_settled());
    }

    #[test]
    fn an_unrecognised_signature_is_not_known() {
        assert!(known_issue("exp/rank1/numeric/1e-6").is_none());
    }

    /// **Guards a deliberate absence.** No entry has a ratified predicate yet, and that is
    /// a decision rather than an oversight: the obvious rule for the one class here was
    /// *measured* to match cases that agree, so recording it would turn a falsified guess
    /// into a stated fact.
    ///
    /// If this test fails, someone has added one. That is fine — provided it came from the
    /// search in 7B.5 with evidence, and not from writing down the plausible-looking rule.
    #[test]
    fn no_predicate_is_recorded_without_a_search_having_earned_it() {
        for entry in KNOWN {
            assert!(
                entry.predicate.is_none(),
                "{} has a predicate. If a search proposed and a human ratified it, update \
                 this test and record the evidence in DECISIONS.md. If it was written by \
                 hand because it looked right, that is the failure this test exists for.",
                entry.signature
            );
        }
    }

    /// The trigger lookup answers `None` while no predicate is ratified — it must not fall
    /// back to matching on something weaker, which would quietly reintroduce symptom
    /// grouping under a name that promises otherwise.
    #[test]
    fn a_trigger_lookup_finds_nothing_while_no_predicate_is_ratified() {
        use crate::features::extract;

        let case = TensorOp::matmul(
            TensorValue::new(vec![1, 2], vec![1e30, -1e30]),
            TensorValue::new(vec![2, 1], vec![1e30, 1e30]),
        );

        assert!(known_by_predicate(extract(&case)).is_none());
    }

    /// Two entries for one signature would make lookup order-dependent.
    #[test]
    fn signatures_are_unique() {
        let mut seen: Vec<&str> = KNOWN.iter().map(|k| k.signature).collect();
        let count = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), count, "a signature is recorded twice");
    }
}
