//! When are two findings the same problem?
//!
//! A campaign that finds one bug a thousand times should report it once. A campaign that
//! finds two bugs must never report them as one. The signature is the key that decides,
//! and the asymmetry between those two mistakes decides how it is built.
//!
//! # Err finer, never coarser
//!
//! Too fine, and one problem is split across two groups: an investigation is wasted, **and
//! you notice**. Too coarse, and a second, genuinely different bug is folded into the first
//! and never investigated — **and it looks exactly like success**. So when in doubt, split.
//!
//! # What goes in the key, and what deliberately does not
//!
//! - **In:** the shape of the query (which clauses and operators it used) and the *kind* of
//!   disagreement. Both are properties of the **problem**.
//! - **Out: the engine names.** Which implementations were running is a fact about the
//!   experiment, not about the bug. Putting them in the key would give the same problem one
//!   string in a two-engine campaign and another in a three-engine one, orphaning a registry
//!   of investigated problems the moment an engine is added. The disagreeing pair is
//!   recorded *beside* the signature instead, so precision is kept without paying for it in
//!   the key.
//! - **Out: the data, the literals, the table names.** They are the *input*, not the cause.
//!   Including them gives nearly every case its own signature, which is the same as having
//!   no de-duplication at all.
//!
//! **One case, one signature.** Tempting with several engines to emit one finding per
//! disagreeing pair — but a lone dissenter disagrees with everyone, so one obvious problem
//! would become N−1 findings and every count would inflate.

use crate::ast::SqlCase;
use crate::normalize::CanonicalResult;
use crate::schema::{BinaryOp, Expr, UnaryOp};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The kind of disagreement, independent of the values involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DisagreementKind {
    /// Different numbers of rows.
    RowCount,
    /// Same shape, different values.
    RowContent,
    /// The same rows in a different order, on a query whose order was meant to be fixed.
    Ordering,
    /// One engine answered, the other refused the query.
    RowsVersusError,
    /// Both refused, for different stated reasons.
    ErrorClassMismatch,
}

impl DisagreementKind {
    /// Classify how two canonical results differ.
    ///
    /// Returns `None` when they do not — which is not a formality: a caller that assumed a
    /// difference and got a kind anyway would be recording a signature for a case that
    /// agreed.
    pub fn between(left: &CanonicalResult, right: &CanonicalResult) -> Option<DisagreementKind> {
        match (left, right) {
            (CanonicalResult::Rows(left_rows), CanonicalResult::Rows(right_rows)) => {
                if left_rows == right_rows {
                    None
                } else if left_rows.len() != right_rows.len() {
                    Some(DisagreementKind::RowCount)
                } else {
                    // Same rows, different sequence: an ordering disagreement. Reachable
                    // only for a query whose order was total, since anything else was
                    // sorted before it got here.
                    let mut left_sorted = left_rows.clone();
                    let mut right_sorted = right_rows.clone();
                    left_sorted.sort();
                    right_sorted.sort();
                    if left_sorted == right_sorted {
                        Some(DisagreementKind::Ordering)
                    } else {
                        Some(DisagreementKind::RowContent)
                    }
                }
            }
            (CanonicalResult::Rows(_), CanonicalResult::Error(_))
            | (CanonicalResult::Error(_), CanonicalResult::Rows(_)) => {
                Some(DisagreementKind::RowsVersusError)
            }
            (CanonicalResult::Error(left_class), CanonicalResult::Error(right_class)) => {
                if left_class == right_class {
                    None
                } else {
                    Some(DisagreementKind::ErrorClassMismatch)
                }
            }
        }
    }

    /// A short stable name, for use inside a signature string.
    pub fn as_str(self) -> &'static str {
        match self {
            DisagreementKind::RowCount => "row-count",
            DisagreementKind::RowContent => "row-content",
            DisagreementKind::Ordering => "ordering",
            DisagreementKind::RowsVersusError => "rows-vs-error",
            DisagreementKind::ErrorClassMismatch => "error-class",
        }
    }
}

/// The clauses and operators a query used, as sorted, de-duplicated tags.
///
/// Sorted so the signature does not depend on the order features happened to be
/// discovered in — two cases using the same constructs must produce the same key.
pub fn clause_shape(case: &SqlCase) -> Vec<&'static str> {
    let mut tags: BTreeSet<&'static str> = BTreeSet::new();

    if case.query.filter.is_some() {
        tags.insert("where");
    }
    if !case.query.order_by.is_empty() {
        tags.insert("order-by");
    }
    if !case.query.group_by.is_empty() {
        tags.insert("group-by");
    }
    if let Some(branch) = &case.query.set_op {
        tags.insert(match branch.op {
            crate::schema::SetOp::Union => "union",
            crate::schema::SetOp::UnionAll => "union-all",
            crate::schema::SetOp::Intersect => "intersect",
            crate::schema::SetOp::Except => "except",
        });
        if let Some(filter) = &branch.right.filter {
            collect_expr_tags(filter, &mut tags);
        }
        if let Some(inner) = &branch.right.set_op {
            // Chained, and the *pair* of operators is what precedence turns on — so the tag
            // names both rather than just saying "chained".
            tags.insert("chained-set-op");
            tags.insert(match inner.op {
                crate::schema::SetOp::Union => "union",
                crate::schema::SetOp::UnionAll => "union-all",
                crate::schema::SetOp::Intersect => "intersect",
                crate::schema::SetOp::Except => "except",
            });
            if let Some(filter) = &inner.right.filter {
                collect_expr_tags(filter, &mut tags);
            }
        }
    }
    // An aggregate over an empty table is its own phenomenon — the row count collapses to
    // one whatever the data — so it earns a tag rather than hiding inside "empty-table".
    if case.aggregates() && case.queried_rows().is_empty() {
        tags.insert("aggregate-over-empty");
    }
    if case.query.limit.is_some() {
        tags.insert("limit");
    }
    if case.schema.len() > 1 {
        tags.insert("multi-table");
    }
    if case.queried_rows().is_empty() {
        tags.insert("empty-table");
    }

    for expression in &case.query.projection {
        collect_expr_tags(expression, &mut tags);
    }
    if let Some(filter) = &case.query.filter {
        collect_expr_tags(filter, &mut tags);
    }

    tags.into_iter().collect()
}

/// Tags from a nested statement — a subquery's own constructs are part of the query's shape.
fn collect_stmt_tags(statement: &crate::schema::SelectStmt, tags: &mut BTreeSet<&'static str>) {
    for expression in &statement.projection {
        collect_expr_tags(expression, tags);
    }
    if let Some(filter) = &statement.filter {
        collect_expr_tags(filter, tags);
    }
    if !statement.group_by.is_empty() {
        tags.insert("group-by");
    }
}

fn collect_expr_tags(expression: &Expr, tags: &mut BTreeSet<&'static str>) {
    match expression {
        Expr::Column(_) => {}
        Expr::Literal(crate::schema::Literal::Null) => {
            tags.insert("null-literal");
        }
        Expr::Literal(_) => {}
        Expr::Cast { expr, .. } => {
            tags.insert("cast");
            collect_expr_tags(expr, tags);
        }
        Expr::Unary { op, operand } => {
            tags.insert(match op {
                UnaryOp::Not => "not",
                UnaryOp::Negate => "negate",
                UnaryOp::IsNull => "is-null",
                UnaryOp::IsNotNull => "is-not-null",
            });
            collect_expr_tags(operand, tags);
        }
        Expr::Exists { not, query } => {
            tags.insert(if *not { "not-exists" } else { "exists" });
            collect_stmt_tags(query, tags);
        }
        Expr::ScalarSubquery { left, query, .. } => {
            tags.insert("scalar-subquery");
            collect_expr_tags(left, tags);
            collect_stmt_tags(query, tags);
        }
        // `IN` and `NOT IN` get **separate** tags, because they are not the same construct
        // for our purposes: the three-valued-logic trap is asymmetric and lives entirely on
        // the negated side. Collapsing them would group a finding with cases that cannot
        // exhibit it.
        Expr::InSubquery { not, left, query } => {
            tags.insert(if *not {
                "not-in-subquery"
            } else {
                "in-subquery"
            });
            collect_expr_tags(left, tags);
            collect_stmt_tags(query, tags);
        }
        Expr::Aggregate { func, arg } => {
            tags.insert(match func {
                crate::schema::AggregateFunc::CountRows => "count-rows",
                crate::schema::AggregateFunc::Count => "count",
                crate::schema::AggregateFunc::Min => "min",
                crate::schema::AggregateFunc::Max => "max",
                crate::schema::AggregateFunc::Sum => "sum",
            });
            if let Some(inner) = arg {
                collect_expr_tags(inner, tags);
            }
        }
        Expr::Binary { op, left, right } => {
            tags.insert(match op {
                BinaryOp::Equal | BinaryOp::NotEqual => "equality",
                BinaryOp::Less
                | BinaryOp::LessOrEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterOrEqual => "comparison",
                BinaryOp::And => "and",
                BinaryOp::Or => "or",
                BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply => "arithmetic",
            });
            collect_expr_tags(left, tags);
            collect_expr_tags(right, tags);
        }
    }
}

/// The de-duplication key for a finding: what the query was made of, and how they differed.
pub fn signature(case: &SqlCase, kind: DisagreementKind) -> String {
    format!("{}/{}", clause_shape(case).join("+"), kind.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::ErrorClass;

    fn rows(values: &[&[&str]]) -> CanonicalResult {
        CanonicalResult::Rows(
            values
                .iter()
                .map(|row| row.iter().map(|cell| cell.to_string()).collect())
                .collect(),
        )
    }

    #[test]
    fn identical_results_have_no_disagreement_kind() {
        assert_eq!(
            DisagreementKind::between(&rows(&[&["1"]]), &rows(&[&["1"]])),
            None
        );
        assert_eq!(
            DisagreementKind::between(
                &CanonicalResult::Error(ErrorClass::Other),
                &CanonicalResult::Error(ErrorClass::Other)
            ),
            None
        );
    }

    #[test]
    fn each_kind_is_reachable() {
        // The rule the tensor domain paid for: a classifier must be able to return every
        // class it claims to have. Each of the five is produced here from real inputs.
        assert_eq!(
            DisagreementKind::between(&rows(&[&["1"], &["2"]]), &rows(&[&["1"]])),
            Some(DisagreementKind::RowCount)
        );
        assert_eq!(
            DisagreementKind::between(&rows(&[&["1"]]), &rows(&[&["2"]])),
            Some(DisagreementKind::RowContent)
        );
        assert_eq!(
            DisagreementKind::between(&rows(&[&["1"], &["2"]]), &rows(&[&["2"], &["1"]])),
            Some(DisagreementKind::Ordering)
        );
        assert_eq!(
            DisagreementKind::between(&rows(&[&["1"]]), &CanonicalResult::Error(ErrorClass::Other)),
            Some(DisagreementKind::RowsVersusError)
        );
        assert_eq!(
            DisagreementKind::between(
                &CanonicalResult::Error(ErrorClass::OutOfRange),
                &CanonicalResult::Error(ErrorClass::TypeMismatch)
            ),
            Some(DisagreementKind::ErrorClassMismatch)
        );
    }

    #[test]
    fn ordering_is_distinguished_from_content() {
        // The distinction that matters most in this domain: the same rows rearranged is a
        // different problem from different rows, and folding them together would send an
        // ordering bug to whatever group content bugs land in.
        let reordered =
            DisagreementKind::between(&rows(&[&["a"], &["b"]]), &rows(&[&["b"], &["a"]]));
        let different =
            DisagreementKind::between(&rows(&[&["a"], &["b"]]), &rows(&[&["a"], &["c"]]));
        assert_ne!(reordered, different);
    }

    #[test]
    fn the_signature_excludes_engine_names_and_data() {
        let case = SqlCase::fixed_example();
        let key = signature(&case, DisagreementKind::RowContent);

        assert!(
            !key.contains("sqlite"),
            "engine names must not be in the key"
        );
        assert!(
            !key.contains("duckdb"),
            "engine names must not be in the key"
        );
        // Literals from the data must not appear either — including them would give nearly
        // every case its own signature.
        assert!(!key.contains("one"));
        assert!(!key.contains('3'));
    }

    #[test]
    fn the_same_shape_gives_the_same_signature_whatever_the_values() {
        use crate::schema::Literal;

        let mut first = SqlCase::fixed_example();
        let mut second = SqlCase::fixed_example();
        for row in &mut second.data[0].rows {
            row[0] = Literal::Integer(999);
        }

        assert_eq!(
            signature(&first, DisagreementKind::RowContent),
            signature(&second, DisagreementKind::RowContent)
        );

        // But a different *construct* must give a different signature.
        first.query.filter = None;
        assert_ne!(
            signature(&first, DisagreementKind::RowContent),
            signature(&second, DisagreementKind::RowContent)
        );
    }

    #[test]
    fn the_kind_is_part_of_the_key() {
        let case = SqlCase::fixed_example();
        assert_ne!(
            signature(&case, DisagreementKind::RowCount),
            signature(&case, DisagreementKind::Ordering)
        );
    }

    #[test]
    fn clause_shape_is_stable_and_sorted() {
        let case = SqlCase::fixed_example();
        let shape = clause_shape(&case);
        let mut sorted = shape.clone();
        sorted.sort();
        assert_eq!(shape, sorted, "tags must be in a deterministic order");
        assert_eq!(clause_shape(&case), shape, "and stable across calls");
    }
}
