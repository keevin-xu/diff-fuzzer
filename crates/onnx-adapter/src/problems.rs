//! Which distinct **problems** a set of signatures represents.
//!
//! # Why a campaign must report both numbers
//!
//! The signature key is deliberately fine — `operator / opset / element-type / rank / kind` — and
//! `signature.rs` explains why: merging two problems makes the second invisible, and what that
//! looks like is a shorter, cleaner list.
//!
//! The cost lands here. One defect that manifests at five element types and four ranks produces
//! many signatures. Measured at N7: **27 signatures for 3 problems**. A campaign that reports "27
//! findings" overstates by roughly nine times, and it overstates in the flattering direction,
//! which is the direction nobody checks.
//!
//! So a campaign reports **both**: distinct signatures, and distinct problems after grouping.
//! `PENDING` 2.7.
//!
//! # Why the grouping is declared rather than computed
//!
//! It would be easy to cluster signatures automatically — same operator, same kind, call it one
//! problem. That is exactly the merge the fine key exists to prevent, moved one step later and
//! done without evidence. Two signatures share a cause when somebody has *established* that they
//! do, and F-001 is the proof: `Sign` at integer types and `Sign` at float types looked like one
//! problem, were filed in this table as one problem, and turned out to have **one of them already
//! fixed upstream while the other was untouched**.
//!
//! So the table below is written by hand, each entry naming the finding that established it. A
//! signature matching nothing is reported as **unexplained**, which is the state that demands
//! attention.

use crate::case::OpKind;
use crate::signature::{Kind, Signature};

/// How far along a problem is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// A report exists in `issues/onnx-runtime/final/` and is ready to send.
    Ready,
    /// Real behaviour, but the argument or the value of reporting it is unsettled.
    Candidate,
    /// Confirmed, and deliberately not being reported.
    NotFiling,
}

/// One underlying problem, and the signatures it accounts for.
#[derive(Debug, Clone, Copy)]
pub struct Problem {
    /// Stable identifier for a campaign report to cite.
    pub id: &'static str,
    /// The working draft that established it.
    pub finding: &'static str,
    /// The implementation at fault.
    pub implementation: &'static str,
    pub what: &'static str,
    pub status: Status,

    /// The operator every signature of this problem carries.
    pub operator: OpKind,
    /// The disagreement kind. Part of the key because a wrong answer and a refusal on the same
    /// operator are different problems — `Reshape` produces both in this domain.
    pub kind: Kind,
}

impl Problem {
    /// Does this signature belong to this problem?
    pub fn covers(&self, signature: &Signature) -> bool {
        signature.operator == self.operator.onnx_name() && signature.kind == self.kind
    }
}

/// The known problems.
///
/// **Every entry names a finding draft.** An entry without one would be a claim that two
/// signatures share a cause, made by nobody, recorded nowhere, and load-bearing on a campaign's
/// headline number.
pub const PROBLEMS: &[Problem] = &[
    Problem {
        id: "P-001",
        finding: "F-001 (integer path, fixed upstream) + F-005 (float path, live)",
        implementation: "tract",
        what: "Sign mishandles zero: 1 for integer 0, and -0.0 for -0.0",
        status: Status::Ready,
        operator: OpKind::Sign,
        kind: Kind::Value,
    },
    Problem {
        id: "P-002",
        finding: "F-004",
        implementation: "onnxruntime",
        what: "Where returns +0.0 for a -0.0 selected from X; the Y branch is correct",
        status: Status::Ready,
        operator: OpKind::Where,
        kind: Kind::Value,
    },
    Problem {
        id: "P-003",
        finding: "F-002",
        implementation: "tract and candle",
        what: "Reshape of a zero-size tensor is refused, while the reference and ONNX Runtime execute it",
        status: Status::Candidate,
        operator: OpKind::Reshape,
        kind: Kind::RejectedVersusOk,
    },
    Problem {
        id: "P-004",
        finding: "F-006",
        implementation: "tract",
        what: "Div panics on int32::MIN / -1, which ONNX leaves undetermined",
        status: Status::Candidate,
        operator: OpKind::Div,
        kind: Kind::Crash,
    },
];

/// A campaign's grouping of signatures into problems.
#[derive(Debug, Default)]
pub struct Grouping {
    /// Problems seen, with how many signatures and occurrences each accounted for.
    pub matched: Vec<(&'static Problem, usize, usize)>,
    /// Signatures matching no known problem, with their occurrence counts.
    ///
    /// **The number that matters.** Everything else is already understood; this is what a campaign
    /// exists to produce.
    pub unexplained: Vec<(String, usize)>,
}

impl Grouping {
    /// Distinct problems, which is the number a campaign should lead with.
    pub fn problems(&self) -> usize {
        self.matched.len() + self.unexplained.len()
    }

    /// Whether anything was seen that nobody has explained.
    pub fn has_unexplained(&self) -> bool {
        !self.unexplained.is_empty()
    }
}

/// Group signatures, each paired with how many times it occurred.
pub fn group(signatures: &[(Signature, usize)]) -> Grouping {
    let mut grouping = Grouping::default();

    for problem in PROBLEMS {
        let covered: Vec<&(Signature, usize)> = signatures
            .iter()
            .filter(|(signature, _)| problem.covers(signature))
            .collect();
        if covered.is_empty() {
            continue;
        }
        let occurrences = covered.iter().map(|(_, count)| *count).sum();
        grouping.matched.push((problem, covered.len(), occurrences));
    }

    for (signature, count) in signatures {
        if !PROBLEMS.iter().any(|p| p.covers(signature)) {
            grouping.unexplained.push((signature.key(), *count));
        }
    }
    // Sorted so a campaign report does not depend on discovery order.
    grouping.unexplained.sort();
    grouping
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::ElemType;

    fn signature(operator: &str, elem: ElemType, rank: usize, kind: Kind) -> Signature {
        Signature {
            operator: operator.to_string(),
            opset: 22,
            elem_type: elem,
            rank,
            kind,
            participants: vec![("tract".into(), "ok".into())],
        }
    }

    /// **The measurement that opened `PENDING` 2.7.** The N7 hunt's signature set — `Sign` at many
    /// types and ranks, `Where` likewise, `Reshape` as refusals — must collapse to three.
    #[test]
    fn the_n7_signature_set_collapses_to_three_problems() {
        let mut signatures = Vec::new();
        for elem in [ElemType::F32, ElemType::F64, ElemType::I32, ElemType::I64] {
            for rank in 1..=4 {
                signatures.push((signature("Sign", elem, rank, Kind::Value), 2));
            }
        }
        for rank in 1..=4 {
            signatures.push((signature("Where", ElemType::F32, rank, Kind::Value), 3));
            signatures.push((
                signature("Reshape", ElemType::Bool, rank, Kind::RejectedVersusOk),
                1,
            ));
        }

        let grouping = group(&signatures);
        assert_eq!(grouping.problems(), 3, "24 signatures must be 3 problems");
        assert!(!grouping.has_unexplained(), "{:?}", grouping.unexplained);

        let sign = grouping
            .matched
            .iter()
            .find(|(p, _, _)| p.id == "P-001")
            .expect("Sign must be grouped");
        assert_eq!(sign.1, 16, "signatures accounted for");
        assert_eq!(sign.2, 32, "occurrences accounted for");
    }

    /// **The number a campaign exists to produce.** A signature nobody has explained must be
    /// reported as unexplained rather than absorbed into the nearest problem.
    #[test]
    fn an_unrecognised_signature_is_reported_not_absorbed() {
        let signatures = vec![
            (signature("Sign", ElemType::I32, 1, Kind::Value), 5),
            (signature("Sqrt", ElemType::F32, 2, Kind::Value), 1),
        ];
        let grouping = group(&signatures);
        assert!(grouping.has_unexplained());
        assert_eq!(grouping.unexplained.len(), 1);
        assert!(grouping.unexplained[0].0.starts_with("Sqrt/"));
        assert_eq!(
            grouping.problems(),
            2,
            "an unexplained signature counts as its own problem until somebody explains it"
        );
    }

    /// A wrong answer and a refusal on the same operator are **different problems**. `Reshape`
    /// produces both in this domain, and merging them would hide one.
    #[test]
    fn the_disagreement_kind_separates_problems_on_one_operator() {
        let refusal = signature("Reshape", ElemType::F32, 2, Kind::RejectedVersusOk);
        let wrong_answer = signature("Reshape", ElemType::F32, 2, Kind::Value);

        let reshape = PROBLEMS.iter().find(|p| p.id == "P-003").unwrap();
        assert!(reshape.covers(&refusal));
        assert!(
            !reshape.covers(&wrong_answer),
            "a wrong answer from Reshape would be a different problem and must not be absorbed"
        );
    }

    /// Every problem must name the finding that established it — the grouping is a claim, and a
    /// claim needs a source. Same rule as the legal-difference catalog's mandatory citation.
    #[test]
    fn every_problem_names_its_finding() {
        for problem in PROBLEMS {
            assert!(
                problem.finding.contains("F-"),
                "problem {:?} names no finding",
                problem.id
            );
            assert!(!problem.what.is_empty());
            assert!(!problem.implementation.is_empty());
        }
    }

    /// Identifiers must be unique, since a campaign report cites them.
    #[test]
    fn problem_identifiers_are_unique() {
        let mut ids: Vec<&str> = PROBLEMS.iter().map(|p| p.id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    /// No two problems may claim the same signature, or the grouping double-counts.
    #[test]
    fn problems_do_not_overlap() {
        for (i, a) in PROBLEMS.iter().enumerate() {
            for b in &PROBLEMS[i + 1..] {
                assert!(
                    !(a.operator == b.operator && a.kind == b.kind),
                    "{:?} and {:?} would both claim the same signatures",
                    a.id,
                    b.id
                );
            }
        }
    }
}
