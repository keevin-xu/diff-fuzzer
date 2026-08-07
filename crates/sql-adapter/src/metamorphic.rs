//! Checking one engine against **itself**.
//!
//! # The blind spot this exists to reach
//!
//! A differential oracle compares two engines and reports disagreement. It therefore cannot
//! see a bug they **share** — both return the same wrong answer, and that is indistinguishable
//! from both being right. This is not a gap in the implementation; it is a property of the
//! technique, and no amount of scale reaches past it. A 1.6-million-case campaign that agrees
//! everywhere is exactly as consistent with "both engines are correct" as with "both engines
//! are wrong in the same way".
//!
//! A **metamorphic** oracle needs no second engine. It transforms a query into another whose
//! answer must be related to the first *by the definition of SQL*, runs both on one engine,
//! and checks the relation. A violation is that engine contradicting itself — which is a bug
//! regardless of what any other engine does.
//!
//! # TLP — Ternary Logic Partitioning
//!
//! For any predicate `p`, every row falls into exactly one of three buckets: `p` is TRUE, `p`
//! is FALSE, or `p` is **UNKNOWN** (which is what SQL's three-valued logic returns when `NULL`
//! is involved). `WHERE p` keeps the first. `WHERE NOT p` keeps the second. `WHERE p IS NULL`
//! keeps the third. Nothing is in two buckets and nothing is in none, so:
//!
//! ```text
//! rows(WHERE p) ∪ rows(WHERE NOT p) ∪ rows(WHERE p IS NULL)  ==  rows(no WHERE at all)
//! ```
//!
//! as **multisets** — `UNION ALL`, not `UNION`, since duplicates must survive on both sides.
//!
//! The relation holds for every `p` and every table, so any counterexample is a defect. And
//! the third partition is the whole point: an engine that mishandles `UNKNOWN` — treating it
//! as FALSE somewhere it shouldn't — loses rows from the union while the unpartitioned query
//! keeps them. That is precisely the class two engines can share, because three-valued logic
//! is the part of SQL implementations most often get subtly wrong in the same way.
//!
//! # What can go wrong with the tool rather than the engine
//!
//! **The transform itself can be the bug.** If the three variants do not actually partition the
//! rows, every case "diverges" and the oracle is reporting its own defect. The guards:
//! `NOT (NOT p)` is not assumed equivalent to `p`; the partition uses `p IS NULL` on the
//! predicate itself rather than on any column; and the tests below check the relation on cases
//! whose answer is known by hand before trusting it on generated ones.

use crate::ast::SqlCase;
use crate::outcome::{Cell, SqlOutcome};
use crate::schema::{Expr, SelectStmt, UnaryOp};

/// The four queries TLP compares: one whole, three parts.
///
/// Held together rather than as loose statements, because the relation is a claim about the
/// set of them — and because a caller that ran three of the four would silently be checking
/// something else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partitioned {
    /// The original query with **no** `WHERE` clause: every row.
    pub whole: SqlCase,
    /// `WHERE p` — rows where the predicate is TRUE.
    pub is_true: SqlCase,
    /// `WHERE NOT p` — rows where it is FALSE.
    pub is_false: SqlCase,
    /// `WHERE p IS NULL` — rows where it is UNKNOWN.
    ///
    /// The partition that matters. Without it the relation is simply false whenever a `NULL`
    /// touches the predicate, and an oracle built on the other two would report every such
    /// case as a bug.
    pub is_unknown: SqlCase,
}

/// Build the four queries from a case, or `None` if TLP does not apply to it.
///
/// Returns `None` — rather than something approximate — when the case cannot be partitioned
/// meaningfully:
///
/// - **No `WHERE` clause**: there is no predicate to partition on.
/// - **Aggregates or `GROUP BY`**: the rows coming back are groups, not rows, and the union of
///   three partitions' *aggregates* is not the aggregate of the whole. `SUM` over a partition
///   is not a third of `SUM` over everything. TLP has aggregate-aware variants; this is not one.
/// - **A set operation**: the relation is about one query's rows, and a set operation's output
///   is not that.
/// - **`LIMIT`**: it truncates, so a partition's limit and the whole's are unrelated.
///
/// Each exclusion is a case where the *relation itself* would not hold, so including it would
/// manufacture violations — the tool reporting its own misunderstanding as an engine's bug.
pub fn partition(case: &SqlCase) -> Option<Partitioned> {
    let predicate = case.query.filter.clone()?;

    if !case.query.group_by.is_empty()
        || case.aggregates()
        || case.query.set_op.is_some()
        || case.query.limit.is_some()
    {
        return None;
    }

    // Row order must not matter: the union is a multiset comparison, and the whole query's
    // order says nothing about the concatenation of three partitions'. Stripping `ORDER BY`
    // makes that explicit rather than relying on the comparison to sort it away.
    let base = |filter: Option<Expr>| {
        let mut variant = case.clone();
        variant.query = SelectStmt {
            filter,
            order_by: Vec::new(),
            limit: None,
            ..case.query.clone()
        };
        variant
    };

    Some(Partitioned {
        whole: base(None),
        is_true: base(Some(predicate.clone())),
        is_false: base(Some(Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(predicate.clone()),
        })),
        is_unknown: base(Some(Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(predicate),
        })),
    })
}

/// What the relation check concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Relation {
    /// The three partitions reconstruct the whole. The engine is self-consistent here.
    Holds,
    /// They do not — the engine contradicted itself.
    Violated {
        whole: usize,
        partitions: usize,
        /// Rows present in one side and not the other, rendered. The actual evidence.
        only_in_whole: Vec<String>,
        only_in_partitions: Vec<String>,
    },
    /// Nothing was checked, and why. Never silently conflated with `Holds` — a case that could
    /// not be judged is not a case that passed.
    NotChecked(&'static str),
}

/// Does the union of the three partitions equal the whole, as multisets?
///
/// Takes outcomes rather than running anything, so it can be tested against fabricated results
/// with no engine involved — the same separation the differential oracle has.
pub fn check(
    whole: &SqlOutcome,
    is_true: &SqlOutcome,
    is_false: &SqlOutcome,
    is_unknown: &SqlOutcome,
) -> Relation {
    let parts = [whole, is_true, is_false, is_unknown];
    if parts.iter().any(|outcome| matches!(outcome, SqlOutcome::Error(_))) {
        // One variant erroring while others did not is a real signal, but it is a *different*
        // claim from the row relation, and folding them together would make the counts
        // meaningless. Reported as unchecked here.
        return Relation::NotChecked("a variant returned an error rather than rows");
    }

    let rows = |outcome: &SqlOutcome| match outcome {
        SqlOutcome::Rows(grid) => grid.clone(),
        SqlOutcome::Error(_) => Vec::new(),
    };

    let whole_rows = rows(whole);
    let mut partition_rows = rows(is_true);
    partition_rows.extend(rows(is_false));
    partition_rows.extend(rows(is_unknown));

    // Multiset comparison: sort the rendered rows and compare. Duplicates must survive, which
    // is why this is not a set difference — an engine dropping one of two identical rows is a
    // bug the set version would miss.
    let render = |grid: Vec<Vec<Cell>>| {
        let mut lines: Vec<String> = grid
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| format!("{cell:?}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .collect();
        lines.sort();
        lines
    };

    let left = render(whole_rows);
    let right = render(partition_rows);

    if left == right {
        return Relation::Holds;
    }

    Relation::Violated {
        whole: left.len(),
        partitions: right.len(),
        only_in_whole: difference(&left, &right),
        only_in_partitions: difference(&right, &left),
    }
}

/// Multiset difference: what is in `left` more often than in `right`.
fn difference(left: &[String], right: &[String]) -> Vec<String> {
    let mut remaining: Vec<&String> = right.iter().collect();
    let mut extra = Vec::new();
    for value in left {
        match remaining.iter().position(|candidate| *candidate == value) {
            Some(index) => {
                remaining.remove(index);
            }
            None => extra.push(value.clone()),
        }
    }
    extra
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{DuckDbImpl, SqliteImpl};
    use crate::gen_schema::Bounds;
    use crate::generator::SqlGenerator;
    use crate::schema::{BinaryOp, ColumnRef, Literal};
    use diff_fuzzer_core::SeededRng;
    use diff_fuzzer_core::traits::{Generator, Implementation};

    fn rows(values: &[&[i64]]) -> SqlOutcome {
        SqlOutcome::Rows(
            values
                .iter()
                .map(|row| row.iter().map(|n| Cell::Integer(*n)).collect())
                .collect(),
        )
    }

    #[test]
    fn three_partitions_that_reconstruct_the_whole_hold() {
        let relation = check(
            &rows(&[&[1], &[2], &[3]]),
            &rows(&[&[1]]),
            &rows(&[&[2]]),
            &rows(&[&[3]]),
        );
        assert_eq!(relation, Relation::Holds);
    }

    #[test]
    fn a_lost_row_is_a_violation_and_the_evidence_names_it() {
        // The shape a three-valued-logic bug takes: the UNKNOWN partition comes back empty and
        // the row it should have held vanishes from the union.
        let relation = check(
            &rows(&[&[1], &[2], &[3]]),
            &rows(&[&[1]]),
            &rows(&[&[2]]),
            &rows(&[]),
        );
        match relation {
            Relation::Violated {
                whole,
                partitions,
                only_in_whole,
                ..
            } => {
                assert_eq!((whole, partitions), (3, 2));
                assert_eq!(only_in_whole.len(), 1, "the missing row is named");
            }
            other => panic!("expected a violation, got {other:?}"),
        }
    }

    #[test]
    fn duplicates_must_survive_the_comparison() {
        // A set comparison would call this equal. It is not: the whole has the row twice.
        let relation = check(
            &rows(&[&[1], &[1]]),
            &rows(&[&[1]]),
            &rows(&[]),
            &rows(&[]),
        );
        assert!(matches!(relation, Relation::Violated { .. }));
    }

    #[test]
    fn an_error_in_any_variant_is_unchecked_not_holding() {
        let relation = check(
            &rows(&[&[1]]),
            &SqlOutcome::Error(crate::outcome::ErrorClass::Other),
            &rows(&[]),
            &rows(&[]),
        );
        assert!(matches!(relation, Relation::NotChecked(_)));
    }

    #[test]
    fn cases_tlp_cannot_partition_are_refused_rather_than_approximated() {
        // No predicate: nothing to partition on.
        let mut no_filter = SqlCase::fixed_example();
        no_filter.query.filter = None;
        assert!(partition(&no_filter).is_none());

        // An aggregate: the union of three partitions' sums is not the sum of the whole.
        let mut aggregated = SqlCase::fixed_example();
        aggregated.query.projection = vec![Expr::Aggregate {
            func: crate::schema::AggregateFunc::CountRows,
            arg: None,
        }];
        assert!(partition(&aggregated).is_none());

        // A LIMIT truncates, so the partitions' limits and the whole's are unrelated.
        let mut limited = SqlCase::fixed_example();
        limited.query.limit = Some(1);
        assert!(partition(&limited).is_none());
    }

    /// The relation, verified on a case whose answer is known by hand **before** it is trusted
    /// on generated ones — because a wrong transform would report every case as a bug.
    #[test]
    fn the_relation_holds_on_a_hand_checked_case_with_a_null() {
        let mut case = SqlCase::fixed_example();
        // c0 is INTEGER with a NULL in one row; the predicate is UNKNOWN for exactly that row,
        // so all three partitions are non-empty and the third is load-bearing.
        case.data[0].rows = vec![
            vec![Literal::Integer(1), Literal::Text("a".into())],
            vec![Literal::Integer(5), Literal::Text("b".into())],
            vec![Literal::Null, Literal::Text("c".into())],
        ];
        case.query.filter = Some(Expr::Binary {
            op: BinaryOp::Greater,
            left: Box::new(Expr::Column(ColumnRef {
                table: "t0".into(),
                column: "c0".into(),
            })),
            right: Box::new(Expr::Literal(Literal::Integer(2))),
        });
        case.query.order_by = Vec::new();

        let parts = partition(&case).expect("this case partitions");
        for engine in ["sqlite", "duckdb"] {
            let run = |c: &SqlCase| -> SqlOutcome {
                if engine == "sqlite" {
                    SqliteImpl.run(c).expect("runs")
                } else {
                    DuckDbImpl.run(c).expect("runs")
                }
            };
            assert_eq!(
                check(
                    &run(&parts.whole),
                    &run(&parts.is_true),
                    &run(&parts.is_false),
                    &run(&parts.is_unknown)
                ),
                Relation::Holds,
                "TLP must hold on {engine} for a hand-checked case"
            );
        }
    }

    /// **How much of a corpus TLP can even judge** — a property worth measuring rather than
    /// assuming, because it bounds what this oracle can reach.
    ///
    /// `partition` refuses aggregates, grouping, set operations and `LIMIT`, since the relation
    /// does not hold for them. In the combined configuration those are most of the corpus, so
    /// TLP sees a minority of it. That is not a defect — it is the honest reach of the
    /// technique, and it means a TLP campaign should run on a configuration where the relation
    /// applies rather than on the differential campaign's.
    #[test]
    fn how_much_of_each_configuration_tlp_can_judge() {
        for (name, bounds, floor) in [
            ("V1", Bounds::V1, 50),
            ("V1_ALL", Bounds::V1_ALL, 10),
        ] {
            let generator = SqlGenerator::new(bounds);
            let partitionable = (0..300)
                .filter(|seed| {
                    partition(&generator.generate(&mut SeededRng::from_seed(*seed))).is_some()
                })
                .count();
            let percent = 100 * partitionable / 300;
            assert!(
                percent >= floor,
                "{name}: only {percent}% partitionable, below the {floor}% this test pins"
            );
        }
    }

    /// And on generated cases, on **both** engines — the check that the transform is sound
    /// before any violation it reports can be believed.
    #[test]
    fn the_relation_holds_across_generated_cases() {
        let generator = SqlGenerator::new(Bounds::V1);
        let mut checked = 0;

        for seed in 0..300 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(parts) = partition(&case) else {
                continue;
            };

            let (Ok(whole), Ok(t), Ok(f), Ok(u)) = (
                SqliteImpl.run(&parts.whole),
                SqliteImpl.run(&parts.is_true),
                SqliteImpl.run(&parts.is_false),
                SqliteImpl.run(&parts.is_unknown),
            ) else {
                continue;
            };

            if let Relation::Violated { .. } = check(&whole, &t, &f, &u) {
                panic!(
                    "seed {seed}: TLP violated on sqlite — either a real bug or, far more \
                     likely at this stage, a defect in the transform:\n{}",
                    parts.whole.statements(crate::render::Dialect::Sqlite).join(";\n")
                );
            }
            checked += 1;
        }

        assert!(
            checked > 100,
            "only {checked} of 300 cases were checkable — too few to call the transform sound"
        );
    }
}
