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
}];

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

    /// **The test that keeps this file honest.** An entry is matched by exact string, so a
    /// change to the signature *rule* silently orphans every entry — triage would report
    /// long-settled problems as new, and the registry would look fine while doing nothing.
    ///
    /// So the recorded signature is rebuilt from a case that actually produces it.
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
