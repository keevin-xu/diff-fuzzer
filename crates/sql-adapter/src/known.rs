//! Differences the engines are *documented* to have, which are therefore not bugs.
//!
//! # The rule that governs this file
//!
//! An entry here **suppresses** a divergence. That makes it the most dangerous kind of code
//! in the project: too broad, and it eats a real finding, and the campaign that should have
//! reported the bug instead reports a clean run. Nothing else here fails so quietly.
//!
//! So every entry must:
//!
//! 1. **cite `SPECS.md` on both sides** — what each engine *promises*, not what it was seen
//!    to do, because a behaviour can change between releases while a contract cannot;
//! 2. **be as narrow as the evidence**, matching the specific mechanism and nothing near it;
//! 3. **leave a trace** — a skipped case says which entry skipped it, so a filter can be
//!    audited rather than believed.
//!
//! This file was deliberately empty until S5. The catalog was not built at S4 because every
//! documented difference was unreachable by construction, and an empty filter is machinery
//! built before the thing it filters. The first entry arrives now because widening the
//! generator made one reachable — and 22 of 22 divergences in a 3,000-case run were it.

use crate::ast::SqlCase;
use crate::normalize::CanonicalResult;
use crate::outcome::ErrorClass;
use crate::signature::{DisagreementKind, clause_shape};

/// One documented, legal cross-engine difference.
#[derive(Debug, Clone, Copy)]
pub struct LegalDifference {
    /// Short stable name, recorded on every case it skips.
    pub name: &'static str,
    /// Where the evidence lives. Both sides, or it does not belong here.
    pub citation: &'static str,
}

/// Integer overflow: SQLite falls back to floating point, DuckDB raises.
///
/// **SQLite** (`SPECS.md` §2.3): *"The other arithmetic operators perform integer arithmetic
/// if both operands are integers and no overflow would result, or floating point arithmetic,
/// per IEEE Standard 754, if either operand is a real value or integer arithmetic would
/// produce an overflow."*
///
/// **DuckDB** (`SPECS.md` §3.7): *"Attempts to store values outside of the allowed range will
/// result in an error."*
///
/// Two promises, kept by both engines, that cannot both be satisfied by one answer. Neither
/// engine is wrong, so a difference here is legal by construction rather than by judgment —
/// which is the standard every entry in this file has to meet.
pub const INTEGER_OVERFLOW: LegalDifference = LegalDifference {
    name: "integer-overflow-real-vs-error",
    citation: "SPECS.md §2.3 (SQLite) + §3.7 (DuckDB) + §4.9 (the pair)",
};

/// Is this divergence one the engines are documented to be allowed to have?
///
/// Returns the entry that covers it, or `None` — in which case the divergence stands and
/// gets reported. `None` is the default for everything: an unrecognised difference is a
/// candidate finding, never a quiet skip.
pub fn legal_difference(
    case: &SqlCase,
    kind: DisagreementKind,
    outputs: &[&CanonicalResult],
) -> Option<LegalDifference> {
    // Narrow on three independent facts, all of which the documented mechanism implies:
    //
    //   1. one engine answered and the other refused,
    //   2. the refusal was specifically an out-of-range complaint,
    //   3. the query actually contains arithmetic — without it, overflow cannot arise, and
    //      an out-of-range error would be something else entirely.
    //
    // Any one of the three alone would be too broad. Together they describe the mechanism
    // and little else.
    if kind != DisagreementKind::RowsVersusError {
        return None;
    }

    let out_of_range = outputs
        .iter()
        .any(|output| matches!(output, CanonicalResult::Error(ErrorClass::OutOfRange)));
    let some_rows = outputs
        .iter()
        .any(|output| matches!(output, CanonicalResult::Rows(_)));
    let has_arithmetic = clause_shape(case).contains(&"arithmetic");

    (out_of_range && some_rows && has_arithmetic).then_some(INTEGER_OVERFLOW)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{BinaryOp, Expr, Literal};

    fn rows() -> CanonicalResult {
        CanonicalResult::Rows(vec![vec!["2147483649".to_string()]])
    }

    fn overflow_case() -> SqlCase {
        let mut case = SqlCase::fixed_example();
        case.query.projection = vec![Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Literal(Literal::Integer(2))),
            right: Box::new(Expr::Literal(Literal::Integer(i32::MAX as i64))),
        }];
        case
    }

    #[test]
    fn the_documented_overflow_difference_is_recognised() {
        let case = overflow_case();
        let error = CanonicalResult::Error(ErrorClass::OutOfRange);
        let found = legal_difference(&case, DisagreementKind::RowsVersusError, &[&rows(), &error]);
        assert_eq!(found.map(|entry| entry.name), Some(INTEGER_OVERFLOW.name));
    }

    #[test]
    fn it_is_symmetric_in_which_engine_refused() {
        // Whichever side raises, it is the same documented difference. An entry that only
        // matched one ordering would silently report half of them as findings.
        let case = overflow_case();
        let error = CanonicalResult::Error(ErrorClass::OutOfRange);
        assert!(
            legal_difference(&case, DisagreementKind::RowsVersusError, &[&error, &rows()])
                .is_some()
        );
    }

    #[test]
    fn a_query_without_arithmetic_is_not_covered() {
        // The narrowing that matters most: an out-of-range error from somewhere other than
        // arithmetic is a different phenomenon, and this entry must not absorb it.
        let case = SqlCase::fixed_example();
        assert!(!clause_shape(&case).contains(&"arithmetic"));

        let error = CanonicalResult::Error(ErrorClass::OutOfRange);
        assert!(
            legal_difference(&case, DisagreementKind::RowsVersusError, &[&rows(), &error])
                .is_none(),
            "without arithmetic this is not the documented overflow difference"
        );
    }

    #[test]
    fn a_different_error_class_is_not_covered() {
        let case = overflow_case();
        let error = CanonicalResult::Error(ErrorClass::TypeMismatch);
        assert!(
            legal_difference(&case, DisagreementKind::RowsVersusError, &[&rows(), &error])
                .is_none()
        );
    }

    #[test]
    fn a_row_content_difference_is_never_covered() {
        // The entry describes rows-versus-error. Two engines returning *different rows* for
        // an arithmetic query is a real finding, and must survive the filter.
        let case = overflow_case();
        let other = CanonicalResult::Rows(vec![vec!["7".to_string()]]);
        assert!(
            legal_difference(&case, DisagreementKind::RowContent, &[&rows(), &other]).is_none(),
            "a content difference must never be filtered as an overflow"
        );
    }

    #[test]
    fn nothing_is_covered_by_default() {
        // The default has to be "report it". A catalog whose default was to skip would turn
        // every unanticipated difference into silence.
        let case = SqlCase::fixed_example();
        for kind in [
            DisagreementKind::RowCount,
            DisagreementKind::RowContent,
            DisagreementKind::Ordering,
            DisagreementKind::ErrorClassMismatch,
        ] {
            assert!(legal_difference(&case, kind, &[&rows(), &rows()]).is_none());
        }
    }
}
