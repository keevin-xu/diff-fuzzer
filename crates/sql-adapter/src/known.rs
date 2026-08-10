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

/// Comma-join precedence: SQLite binds all join operators equally, left to right.
///
/// **SQLite** (`SPECS.md` §2.11, quoting its own quirks page): *"SQLite gives all join operators
/// equal precedence and processes them from left to right. But this is not quite correct…
/// comma-joins have lower precedence than all others join operators."*
///
/// **Measured** (`examples/join_precedence_probe.rs`): `FROM a, b RIGHT JOIN c ON b.y = c.z` with
/// no matching key returns `(NULL, NULL, 3)` on SQLite — consistent with `(a, b) RIGHT JOIN c` —
/// and `(1, NULL, 3)` on DuckDB, consistent with `a, (b RIGHT JOIN c)`.
///
/// # Why this is catalogued despite being a defect rather than a licensed difference
///
/// Every other entry here is legal *by construction*: two engines keeping two documented
/// promises that cannot both be satisfied. **This one is not.** SQLite says it is wrong. It is
/// catalogued for a different reason: the mechanism is fully understood and would otherwise
/// swamp every campaign that enables comma-joins, exactly as chained set operations do at 12.5%
/// — thousands of findings, one cause, and a run that has to be thrown away.
///
/// **The distinction matters and must not be lost.** `INTEGER_OVERFLOW` means *"not a bug"*;
/// this means *"a known bug, already understood, do not report it again"*. Suppressing them
/// through the same mechanism is a convenience, not a claim that they are the same kind of thing.
pub const COMMA_JOIN_PRECEDENCE: LegalDifference = LegalDifference {
    name: "comma-join-precedence-known-sqlite-defect",
    citation: "SPECS.md §2.11 (SQLite documents its own parser as incorrect) + measured",
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

/// Is this divergence the known comma-join precedence defect?
///
/// Kept as a separate function rather than another branch of [`legal_difference`], because it
/// answers a different question: not *"are the engines permitted to differ here?"* but *"is this
/// the defect we already understand?"* Folding them together would let a genuine finding be
/// suppressed by a rule written for a known one.
///
/// Narrow on two independent facts, both of which the mechanism requires:
///
///   1. the `FROM` clause lists **more than one** table — a comma-join, and
///   2. the query also has an **explicit** join, since the defect is about how the two bind
///      relative to each other. A comma-join alone has no precedence question to get wrong.
///
/// Either alone would be far too broad: most queries in a two-table schema have a join.
pub fn known_comma_join_defect(case: &SqlCase) -> Option<LegalDifference> {
    let comma_joined = case.query.from.len() > 1;
    let explicit_join = case.query.join.is_some();

    (comma_joined && explicit_join).then_some(COMMA_JOIN_PRECEDENCE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SqlCase;

    /// The known comma-join defect is recognised, and **only** when both halves are present.
    ///
    /// The narrowness is the whole point. Most queries in a two-table schema have a join, and
    /// most have a single-table `FROM`; a rule keying on either alone would suppress a large
    /// share of genuine findings under the name of a known defect. That is the failure mode a
    /// legal-difference catalog exists to avoid — `POLICY.md`'s note that a filter fails
    /// *silently* when it is wrong.
    #[test]
    fn the_comma_join_defect_needs_both_a_comma_join_and_an_explicit_join() {
        let base = SqlCase::fixed_example();

        // Neither: an ordinary single-table query.
        assert!(known_comma_join_defect(&base).is_none());

        // A comma-join alone — no precedence question to get wrong, because there is no other
        // join operator to bind against.
        let mut comma_only = base.clone();
        comma_only.query.from = vec!["t0".to_string(), "t1".to_string()];
        assert!(
            known_comma_join_defect(&comma_only).is_none(),
            "a comma-join with no explicit join has no precedence to resolve"
        );

        // An explicit join alone — the ordinary case, and by far the most common shape in this
        // corpus. Suppressing these would blind the oracle to most of what it can see.
        let mut join_only = base.clone();
        join_only.query.join = Some(crate::schema::Join {
            kind: crate::schema::JoinKind::Inner,
            table: "t1".to_string(),
            on: Expr::Literal(Literal::Integer(1)),
        });
        assert!(
            known_comma_join_defect(&join_only).is_none(),
            "an ordinary join must not be mistaken for the known defect"
        );

        // Both — the documented shape.
        let mut both = join_only.clone();
        both.query.from = vec!["t0".to_string(), "t1".to_string()];
        assert_eq!(
            known_comma_join_defect(&both).map(|entry| entry.name),
            Some(COMMA_JOIN_PRECEDENCE.name)
        );
    }

    /// The two catalog entries are distinguishable, and describe different kinds of thing.
    ///
    /// `INTEGER_OVERFLOW` means *"not a bug"* — two documented promises that cannot both hold.
    /// `COMMA_JOIN_PRECEDENCE` means *"a known bug, already understood"* — SQLite says it is
    /// wrong. They are suppressed by the same mechanism as a convenience; the names must keep
    /// the distinction visible, because a reader who conflates them would conclude this project
    /// found nothing when it found one thing.
    #[test]
    fn the_two_entries_are_not_the_same_kind_of_claim() {
        assert_ne!(INTEGER_OVERFLOW.name, COMMA_JOIN_PRECEDENCE.name);
        assert!(
            COMMA_JOIN_PRECEDENCE.name.contains("defect"),
            "the name must say this is a defect, not a licensed difference: {}",
            COMMA_JOIN_PRECEDENCE.name
        );
        assert!(
            COMMA_JOIN_PRECEDENCE.citation.contains("§2.11"),
            "every entry cites its evidence"
        );
    }

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
