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
use crate::schema::{AggregateFunc, Expr, SelectStmt, UnaryOp};
use std::collections::HashMap;

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

/// A whole-table aggregate, partitioned the same way.
///
/// The aggregate-aware half of TLP. Without it the oracle refuses every aggregate query, which
/// on the combined configuration is over half the corpus — and aggregates over `NULL`-heavy
/// partitions are exactly where an engine would plausibly slip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedAggregate {
    pub func: AggregateFunc,
    pub whole: SqlCase,
    pub is_true: SqlCase,
    pub is_false: SqlCase,
    pub is_unknown: SqlCase,
}

/// Partition a **whole-table aggregate** query, or `None` if the case is not one.
///
/// Requires a single aggregate in the projection and no `GROUP BY`. Both restrictions are
/// about keeping recombination unambiguous rather than about difficulty:
///
/// - **One aggregate**, because with several the relation is a claim about each independently
///   and a single verdict could not say which failed.
/// - **No `GROUP BY`**, because grouping partitions the *output* as well as the input — a
///   group present in two partitions must be combined, not concatenated, and combining
///   depends on the aggregate. That is a further variant, not this one.
pub fn partition_aggregate(case: &SqlCase) -> Option<PartitionedAggregate> {
    let predicate = case.query.filter.clone()?;

    if !case.query.group_by.is_empty() || case.query.set_op.is_some() || case.query.limit.is_some()
    {
        return None;
    }
    let [Expr::Aggregate { func, .. }] = case.query.projection.as_slice() else {
        return None;
    };

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

    Some(PartitionedAggregate {
        func: *func,
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

/// Does the whole's aggregate equal the recombination of the partitions' aggregates?
///
/// **The recombination rule differs per aggregate, and `NULL` is why.** Measured on DuckDB:
/// over an empty partition `COUNT` returns `0` while `SUM` and `MIN`/`MAX` return `NULL`. So
/// `NULL` here means *"no rows contributed"*, not *"the answer is unknown"* — and a rule that
/// added the partitions naively would report every case with an empty partition as a bug.
///
/// - **`COUNT`** — the whole is the **sum** of the partitions. Empty contributes `0`.
/// - **`SUM`** — the sum of the partitions that returned a value. If *all three* returned
///   `NULL`, the whole must be `NULL` too: summing nothing is not zero.
/// - **`MIN`/`MAX`** — the min/max over the partitions that returned a value, `NULL` if none.
pub fn check_aggregate(
    func: AggregateFunc,
    whole: &SqlOutcome,
    is_true: &SqlOutcome,
    is_false: &SqlOutcome,
    is_unknown: &SqlOutcome,
) -> Relation {
    let single = |outcome: &SqlOutcome| -> Option<Option<i64>> {
        match outcome {
            SqlOutcome::Rows(grid) if grid.len() == 1 && grid[0].len() == 1 => match &grid[0][0] {
                Cell::Integer(number) => Some(Some(*number)),
                Cell::Null => Some(None),
                // A `MIN`/`MAX` over a text column is well-defined but not comparable as an
                // integer; refusing beats inventing an ordering.
                Cell::Text(_) => None,
            },
            _ => None,
        }
    };

    let (Some(whole), Some(t), Some(f), Some(u)) = (
        single(whole),
        single(is_true),
        single(is_false),
        single(is_unknown),
    ) else {
        return Relation::NotChecked("an aggregate variant did not return one integer cell");
    };

    let parts: Vec<i64> = [t, f, u].into_iter().flatten().collect();

    // Shared with the grouped check below — the recombination rule is a property of the
    // aggregate, not of whether the query groups.
    let expected: Option<i64> = recombine(func, &parts);

    if whole == expected {
        return Relation::Holds;
    }

    Relation::Violated {
        whole: whole.map_or(0, |value| value as usize),
        partitions: expected.map_or(0, |value| value as usize),
        only_in_whole: vec![format!("{func:?} over the whole table = {whole:?}")],
        only_in_partitions: vec![format!(
            "recombined from partitions = {expected:?} (true={t:?}, false={f:?}, unknown={u:?})"
        )],
    }
}

/// A **grouped** aggregate query, partitioned the same way.
///
/// The last of the three TLP forms. `partition` handles row queries, `partition_aggregate`
/// whole-table aggregates, and this one `GROUP BY` — which between them is most of what the
/// generator produces.
///
/// Grouping is harder than the other two for one reason: it partitions the **output** as well
/// as the input. A row query's three partitions concatenate; a whole-table aggregate's three
/// results combine into one. Here each partition returns a *set of groups*, and a group that
/// appears in two partitions must have its aggregate combined across them — while a group that
/// appears in only one carries straight through. So the check is per group key, not per result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedGroups {
    /// One entry per aggregate column, in projection order after the group key.
    pub funcs: Vec<AggregateFunc>,
    pub whole: SqlCase,
    pub is_true: SqlCase,
    pub is_false: SqlCase,
    pub is_unknown: SqlCase,
}

/// Partition a `GROUP BY` query, or `None` if the case is not one this relation covers.
///
/// Requires exactly one grouping key, a projection of that key followed by one or more
/// aggregates, and no set operation or `LIMIT`. The projection shape is checked rather than
/// assumed: the relation reads column 0 as the key and the rest as aggregates, so a projection
/// that did not look like that would be silently misread.
pub fn partition_grouped(case: &SqlCase) -> Option<PartitionedGroups> {
    let predicate = case.query.filter.clone()?;

    // One key, because the check buckets by a single cell. Several keys is the same idea with
    // a tuple key and no new insight, so it is left out rather than built speculatively.
    if case.query.group_by.len() != 1 || case.query.set_op.is_some() || case.query.limit.is_some() {
        return None;
    }

    // `[first, rest @ ..]` is a **slice pattern** — it binds the first element and borrows the
    // remainder as a slice in one step, and fails to match on an empty slice.
    let [Expr::Column(key), rest @ ..] = case.query.projection.as_slice() else {
        return None;
    };
    if rest.is_empty() || *key != case.query.group_by[0] {
        return None;
    }

    // `collect` into `Option<Vec<_>>` short-circuits: one non-aggregate makes the whole thing
    // `None`. That is the idiomatic way to say "all of these must succeed" over an iterator.
    let funcs: Vec<AggregateFunc> = rest
        .iter()
        .map(|column| match column {
            Expr::Aggregate { func, .. } => Some(*func),
            _ => None,
        })
        .collect::<Option<_>>()?;

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

    Some(PartitionedGroups {
        funcs,
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

/// Recombine one aggregate from the partitions that produced a value for it.
///
/// `parts` holds only the non-`NULL` results. A partition that returned `NULL` — or that did
/// not contain the group at all — contributed no rows, so it contributes nothing here. See
/// [`check_aggregate`] for why `NULL` means "no rows" rather than "unknown".
fn recombine(func: AggregateFunc, parts: &[i64]) -> Option<i64> {
    match func {
        // Counting never yields `NULL`, and an absent group counts zero, so an empty `parts`
        // correctly gives `Some(0)`.
        AggregateFunc::CountRows | AggregateFunc::Count => Some(parts.iter().sum()),
        // Summing nothing is `NULL`, not zero — the one place the two differ.
        AggregateFunc::Sum => (!parts.is_empty()).then(|| parts.iter().sum()),
        AggregateFunc::Min => parts.iter().copied().min(),
        AggregateFunc::Max => parts.iter().copied().max(),
    }
}

/// Does every group's aggregate in the whole equal the recombination of that group's
/// aggregates across the three partitions?
///
/// Three ways this reports a problem, all of them defects in the engine:
///
/// - A group in the whole that no partition produced, or vice versa. Every row is in exactly
///   one partition, so the key sets must match.
/// - A group whose recombined aggregate differs from the whole's.
/// - A variant returning two rows for one group key, which `GROUP BY` forbids by definition.
///
/// Returns `NotChecked` — never a violation — when a variant errored, returned an unexpected
/// row width, or produced a text aggregate, since none of those are judgements about the
/// relation.
/// One `GROUP BY` result read into a lookup: group key → one value per aggregate column, where
/// `None` is a `NULL` aggregate. Paired with the row count it came from, so a key that appeared
/// twice can be told from one that appeared once.
///
/// A **type alias** — it introduces no new type, just a shorter name for an existing one.
type GroupedResult = (HashMap<Cell, Vec<Option<i64>>>, usize);

pub fn check_grouped(
    funcs: &[AggregateFunc],
    whole: &SqlOutcome,
    is_true: &SqlOutcome,
    is_false: &SqlOutcome,
    is_unknown: &SqlOutcome,
) -> Relation {
    // Read one result into `key -> aggregate values`, plus the row count so the caller can
    // tell a duplicated key from a merged one.
    let read = |outcome: &SqlOutcome| -> Result<GroupedResult, &'static str> {
        let SqlOutcome::Rows(rows) = outcome else {
            return Err("a variant returned an error rather than rows");
        };
        let mut map = HashMap::new();
        for row in rows {
            let [key, aggregates @ ..] = row.as_slice() else {
                return Err("a result row had no columns");
            };
            if aggregates.len() != funcs.len() {
                return Err("a result row had an unexpected number of columns");
            }
            let mut values = Vec::with_capacity(aggregates.len());
            for cell in aggregates {
                match cell {
                    Cell::Integer(number) => values.push(Some(*number)),
                    Cell::Null => values.push(None),
                    // A `MIN`/`MAX` over text is well-defined but not comparable as an
                    // integer; refusing beats inventing an ordering.
                    Cell::Text(_) => return Err("an aggregate column was not an integer"),
                }
            }
            map.insert(key.clone(), values);
        }
        let rows = rows.len();
        Ok((map, rows))
    };

    let mut maps = Vec::with_capacity(4);
    for (name, outcome) in [
        ("whole", whole),
        ("p is true", is_true),
        ("p is false", is_false),
        ("p is unknown", is_unknown),
    ] {
        match read(outcome) {
            Err(reason) => return Relation::NotChecked(reason),
            // Fewer distinct keys than rows means the same group came back twice.
            Ok((map, rows)) if map.len() != rows => {
                return Relation::Violated {
                    whole: rows,
                    partitions: map.len(),
                    only_in_whole: vec![format!(
                        "the `{name}` variant returned {rows} rows for only {} distinct group keys",
                        map.len()
                    )],
                    only_in_partitions: vec![
                        "`GROUP BY` returns exactly one row per group".to_string(),
                    ],
                };
            }
            Ok((map, _)) => maps.push(map),
        }
    }

    let (whole_groups, partitions) = maps.split_first().expect("four maps were pushed");

    // Every key mentioned anywhere. Sorted by rendering so the report is deterministic —
    // `HashMap` iteration order is not, and a finding that reorders between runs is not
    // reproducible evidence.
    let mut keys: Vec<&Cell> = whole_groups.keys().collect();
    for partition in partitions {
        for key in partition.keys() {
            if !whole_groups.contains_key(key) {
                keys.push(key);
            }
        }
    }
    keys.sort_by_key(|key| format!("{key:?}"));
    keys.dedup();

    let mut only_in_whole = Vec::new();
    let mut only_in_partitions = Vec::new();

    for key in keys {
        let expected: Vec<Option<i64>> = (0..funcs.len())
            .map(|column| {
                let present: Vec<i64> = partitions
                    .iter()
                    .filter_map(|partition| partition.get(key))
                    .filter_map(|values| values[column])
                    .collect();
                recombine(funcs[column], &present)
            })
            .collect();

        match whole_groups.get(key) {
            // A group the partitions produced and the whole did not, or the reverse. Both are
            // reported, because which side lost the group is the first thing triage needs.
            None => only_in_partitions.push(format!(
                "group {key:?}: {expected:?}, absent from the whole"
            )),
            Some(actual) if *actual != expected => {
                // A group present in no partition recombines to all-`Some(0)`/`None` rather
                // than to nothing, so this arm also covers "the whole has a group the
                // partitions lost" — the values will not match.
                only_in_whole.push(format!("group {key:?}: whole = {actual:?}"));
                only_in_partitions.push(format!("group {key:?}: recombined = {expected:?}"));
            }
            Some(_) => {}
        }
    }

    if only_in_whole.is_empty() && only_in_partitions.is_empty() {
        return Relation::Holds;
    }

    Relation::Violated {
        whole: whole_groups.len(),
        partitions: partitions.iter().map(HashMap::len).sum(),
        only_in_whole,
        only_in_partitions,
    }
}

/// The NoREC pair: the same question asked in a way the optimizer can use, and a way it cannot.
///
/// # A different failure mode from TLP, which is the point of having both
///
/// TLP tests the engine's **three-valued logic** — whether TRUE/FALSE/UNKNOWN partition the
/// rows. NoREC tests its **optimizer**: it asks the same question twice, once as a `WHERE`
/// clause the planner can push down, index, or reorder, and once as a projected expression it
/// can do nothing with except evaluate per row. If the optimized path disagrees with the
/// unoptimized one, the optimization is wrong.
///
/// Two oracles that fail differently is a far stronger claim than one oracle run longer — and
/// an optimizer bug is a class TLP structurally cannot reach, because both of TLP's sides are
/// equally optimizable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoRec {
    /// `SELECT COUNT(*) FROM t WHERE p` — the optimizable form.
    pub filtered: SqlCase,
    /// `SELECT (p) FROM t` — one truth value per row, nothing to optimize away.
    pub projected: SqlCase,
}

/// Build the NoREC pair, or `None` where the equivalence would not hold.
pub fn norec(case: &SqlCase) -> Option<NoRec> {
    let predicate = case.query.filter.clone()?;

    // Same exclusions as TLP, and for the same reason: with grouping, a set operation or a
    // `LIMIT`, "the number of rows the predicate selects" is not what either query returns.
    if !case.query.group_by.is_empty()
        || case.aggregates()
        || case.query.set_op.is_some()
        || case.query.limit.is_some()
    {
        return None;
    }

    let mut filtered = case.clone();
    filtered.query = SelectStmt {
        projection: vec![Expr::Aggregate {
            func: AggregateFunc::CountRows,
            arg: None,
        }],
        filter: Some(predicate.clone()),
        order_by: Vec::new(),
        limit: None,
        ..case.query.clone()
    };

    let mut projected = case.clone();
    projected.query = SelectStmt {
        // The predicate itself, per row. An engine cannot use it to skip work, so this side is
        // the reference the optimized side is judged against.
        projection: vec![predicate],
        filter: None,
        order_by: Vec::new(),
        limit: None,
        ..case.query.clone()
    };

    Some(NoRec {
        filtered,
        projected,
    })
}

/// Does the filtered count equal the number of rows whose projected predicate is true?
///
/// **Only TRUE counts.** A predicate that is UNKNOWN — `NULL` involved — is excluded by `WHERE`
/// and comes back as `NULL` in the projection, so it must be excluded on both sides. Counting
/// it on either side would make every `NULL`-touching case look like a violation, which is the
/// mistake that would turn this oracle into a noise generator.
pub fn check_norec(filtered: &SqlOutcome, projected: &SqlOutcome) -> Relation {
    let (SqlOutcome::Rows(count_grid), SqlOutcome::Rows(rows)) = (filtered, projected) else {
        return Relation::NotChecked("a NoREC variant returned an error rather than rows");
    };

    let [row] = count_grid.as_slice() else {
        return Relation::NotChecked("the filtered side did not return exactly one row");
    };
    let [Cell::Integer(counted)] = row.as_slice() else {
        return Relation::NotChecked("the filtered side did not return a count");
    };

    // A boolean reads back as 0/1 on both engines — see `backends.rs`. Anything else means the
    // projected expression was not a truth value, which is a case this relation does not cover.
    let mut truths = 0i64;
    for row in rows {
        match row.as_slice() {
            [Cell::Integer(0)] | [Cell::Null] => {}
            [Cell::Integer(1)] => truths += 1,
            _ => return Relation::NotChecked("the projected side was not a truth value"),
        }
    }

    if *counted == truths {
        return Relation::Holds;
    }

    Relation::Violated {
        whole: *counted as usize,
        partitions: truths as usize,
        only_in_whole: vec![format!("WHERE p counted {counted} rows")],
        only_in_partitions: vec![format!("projecting p found {truths} true rows")],
    }
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
    if parts
        .iter()
        .any(|outcome| matches!(outcome, SqlOutcome::Error(_)))
    {
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

    /// Build a grouped result: each entry is a group key followed by its aggregate values,
    /// where `None` renders as `NULL`.
    fn groups(values: &[(i64, &[Option<i64>])]) -> SqlOutcome {
        SqlOutcome::Rows(
            values
                .iter()
                .map(|(key, aggregates)| {
                    let mut row = vec![Cell::Integer(*key)];
                    row.extend(aggregates.iter().map(|value| match value {
                        Some(number) => Cell::Integer(*number),
                        None => Cell::Null,
                    }));
                    row
                })
                .collect(),
        )
    }

    #[test]
    fn grouped_counts_that_split_across_partitions_recombine() {
        // Worked out by hand. Table: group 1 has 5 rows, group 2 has 3. The predicate is TRUE
        // for 3 of group 1 and 1 of group 2, FALSE for 2 and 0, UNKNOWN for 0 and 2.
        // Group 1: 3 + 2 + 0 = 5. Group 2: 1 + 0 + 2 = 3. Both match the whole.
        let relation = check_grouped(
            &[AggregateFunc::CountRows],
            &groups(&[(1, &[Some(5)]), (2, &[Some(3)])]),
            &groups(&[(1, &[Some(3)]), (2, &[Some(1)])]),
            &groups(&[(1, &[Some(2)])]),
            &groups(&[(2, &[Some(2)])]),
        );
        assert_eq!(relation, Relation::Holds);
    }

    #[test]
    fn a_group_that_lives_in_only_one_partition_still_holds() {
        // Group 7's rows all satisfy the predicate, so it appears in the TRUE partition and
        // nowhere else. Its count must carry through untouched — an implementation that
        // required every group in every partition would call this a violation.
        let relation = check_grouped(
            &[AggregateFunc::CountRows],
            &groups(&[(7, &[Some(4)])]),
            &groups(&[(7, &[Some(4)])]),
            &groups(&[]),
            &groups(&[]),
        );
        assert_eq!(relation, Relation::Holds);
    }

    #[test]
    fn grouped_sum_over_a_null_group_recombines_to_null() {
        // `SUM` of a column that is entirely `NULL` within a group is `NULL`, not 0. Every
        // partition holding that group returns `NULL`, so the recombination must too —
        // summing an empty list of contributions is `NULL`.
        let relation = check_grouped(
            &[AggregateFunc::Sum],
            &groups(&[(1, &[None])]),
            &groups(&[(1, &[None])]),
            &groups(&[(1, &[None])]),
            &groups(&[]),
        );
        assert_eq!(relation, Relation::Holds);
    }

    #[test]
    fn grouped_min_takes_the_smallest_across_partitions() {
        // MIN over group 1 is 3 in TRUE, 9 in FALSE, absent in UNKNOWN. The whole must be 3.
        let held = check_grouped(
            &[AggregateFunc::Min],
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(9)])]),
            &groups(&[]),
        );
        assert_eq!(held, Relation::Holds);

        // The same partitions with the whole claiming 9 — an engine that lost the smaller row.
        let violated = check_grouped(
            &[AggregateFunc::Min],
            &groups(&[(1, &[Some(9)])]),
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(9)])]),
            &groups(&[]),
        );
        assert!(matches!(violated, Relation::Violated { .. }));
    }

    #[test]
    fn a_group_the_whole_lost_is_a_violation_naming_it() {
        // The three-valued-logic shape again, this time in grouped form: group 2 exists only
        // because of rows where the predicate is UNKNOWN, and the unpartitioned query does
        // not return it.
        let relation = check_grouped(
            &[AggregateFunc::CountRows],
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(3)])]),
            &groups(&[]),
            &groups(&[(2, &[Some(2)])]),
        );
        match relation {
            Relation::Violated {
                only_in_partitions, ..
            } => assert!(
                only_in_partitions
                    .iter()
                    .any(|line| line.contains("Integer(2)")),
                "the evidence names the lost group, got {only_in_partitions:?}"
            ),
            other => panic!("expected a violation, got {other:?}"),
        }
    }

    #[test]
    fn a_duplicated_group_key_is_a_violation() {
        // `GROUP BY` returns one row per group by definition, so two rows for key 1 is the
        // engine contradicting itself regardless of what the partitions say.
        let relation = check_grouped(
            &[AggregateFunc::CountRows],
            &groups(&[(1, &[Some(2)]), (1, &[Some(3)])]),
            &groups(&[(1, &[Some(5)])]),
            &groups(&[]),
            &groups(&[]),
        );
        assert!(matches!(relation, Relation::Violated { .. }));
    }

    #[test]
    fn a_text_aggregate_is_unchecked_rather_than_guessed() {
        let with_text =
            SqlOutcome::Rows(vec![vec![Cell::Integer(1), Cell::Text("'a'".to_string())]]);
        let relation = check_grouped(
            &[AggregateFunc::Min],
            &with_text,
            &groups(&[(1, &[Some(1)])]),
            &groups(&[]),
            &groups(&[]),
        );
        assert!(matches!(relation, Relation::NotChecked(_)));
    }

    #[test]
    fn partition_grouped_refuses_a_row_query_and_accepts_a_grouped_one() {
        // Over generated cases: whatever `partition_grouped` accepts must have a `GROUP BY`,
        // and the four variants must differ only in their `WHERE`.
        let generator = SqlGenerator::new(Bounds::V1_ALL);
        let mut accepted = 0;
        for seed in 0..2_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(parts) = partition_grouped(&case) else {
                continue;
            };
            accepted += 1;
            assert!(
                !case.query.group_by.is_empty(),
                "seed {seed} was not grouped"
            );
            assert!(parts.whole.query.filter.is_none());
            assert!(parts.is_true.query.filter.is_some());
            assert_eq!(parts.funcs.len(), case.query.projection.len() - 1);
            // The three forms are mutually exclusive: a case is at most one of them.
            assert!(partition(&case).is_none() && partition_aggregate(&case).is_none());
        }
        assert!(accepted > 0, "no grouped case in 2000 seeds");
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
        let relation = check(&rows(&[&[1], &[1]]), &rows(&[&[1]]), &rows(&[]), &rows(&[]));
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

    /// The recombination rules, one per aggregate, checked against fabricated results.
    ///
    /// These encode the `NULL`-means-no-rows reading measured on the engines, and getting any
    /// of them wrong would report correct engines as broken on every case with an empty
    /// partition.
    #[test]
    fn each_aggregate_recombines_by_its_own_rule() {
        let cell = |value: Option<i64>| {
            SqlOutcome::Rows(vec![vec![match value {
                Some(number) => Cell::Integer(number),
                None => Cell::Null,
            }]])
        };

        // COUNT: the whole is the sum, and an empty partition contributes zero — not NULL.
        assert_eq!(
            check_aggregate(
                AggregateFunc::CountRows,
                &cell(Some(5)),
                &cell(Some(3)),
                &cell(Some(2)),
                &cell(Some(0))
            ),
            Relation::Holds
        );

        // SUM: partitions that returned NULL contributed no rows and are skipped.
        assert_eq!(
            check_aggregate(
                AggregateFunc::Sum,
                &cell(Some(10)),
                &cell(Some(7)),
                &cell(None),
                &cell(Some(3))
            ),
            Relation::Holds
        );

        // SUM over nothing at all is NULL, **not zero** — the distinction a naive rule loses.
        assert_eq!(
            check_aggregate(
                AggregateFunc::Sum,
                &cell(None),
                &cell(None),
                &cell(None),
                &cell(None)
            ),
            Relation::Holds
        );
        assert!(matches!(
            check_aggregate(
                AggregateFunc::Sum,
                &cell(Some(0)),
                &cell(None),
                &cell(None),
                &cell(None)
            ),
            Relation::Violated { .. }
        ));

        // MIN/MAX: over the partitions that returned a value.
        assert_eq!(
            check_aggregate(
                AggregateFunc::Min,
                &cell(Some(1)),
                &cell(Some(4)),
                &cell(Some(1)),
                &cell(None)
            ),
            Relation::Holds
        );
        assert_eq!(
            check_aggregate(
                AggregateFunc::Max,
                &cell(Some(9)),
                &cell(Some(4)),
                &cell(Some(9)),
                &cell(None)
            ),
            Relation::Holds
        );
    }

    #[test]
    fn a_wrong_aggregate_total_is_caught() {
        let cell = |value: i64| SqlOutcome::Rows(vec![vec![Cell::Integer(value)]]);
        assert!(matches!(
            check_aggregate(
                AggregateFunc::CountRows,
                &cell(5),
                &cell(3),
                &cell(1),
                &cell(0)
            ),
            Relation::Violated { .. }
        ));
    }

    /// The aggregate relation on real engines, before it is trusted to report anything.
    #[test]
    fn the_aggregate_relation_holds_across_generated_cases() {
        let generator = SqlGenerator::new(Bounds::V1_AGGREGATES);
        let mut checked = 0;

        for seed in 0..400 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(parts) = partition_aggregate(&case) else {
                continue;
            };

            for engine in ["sqlite", "duckdb"] {
                let run = |c: &SqlCase| -> Option<SqlOutcome> {
                    if engine == "sqlite" {
                        SqliteImpl.run(c).ok()
                    } else {
                        DuckDbImpl.run(c).ok()
                    }
                };
                let (Some(w), Some(t), Some(f), Some(u)) = (
                    run(&parts.whole),
                    run(&parts.is_true),
                    run(&parts.is_false),
                    run(&parts.is_unknown),
                ) else {
                    continue;
                };

                match check_aggregate(parts.func, &w, &t, &f, &u) {
                    Relation::Violated {
                        only_in_whole,
                        only_in_partitions,
                        ..
                    } => panic!(
                        "seed {seed} on {engine}: aggregate TLP violated — far more likely a \
                         defect in the recombination rule than an engine bug at this stage.\n\
                         {}\n{}\n{}",
                        only_in_whole.join(" "),
                        only_in_partitions.join(" "),
                        parts
                            .whole
                            .statements(crate::render::Dialect::Sqlite)
                            .join(";\n")
                    ),
                    Relation::Holds => checked += 1,
                    Relation::NotChecked(_) => {}
                }
            }
        }

        assert!(checked > 50, "only {checked} aggregate checks ran");
    }

    /// The grouped relation against **real engines**, which is the only way to find out
    /// whether the recombination rule matches what SQL actually does. The fabricated tests
    /// above check the arithmetic; this checks the premise.
    #[test]
    fn the_grouped_relation_holds_across_generated_cases() {
        let generator = SqlGenerator::new(Bounds::V1_AGGREGATES);
        let mut checked = 0;

        for seed in 0..400 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(parts) = partition_grouped(&case) else {
                continue;
            };

            for engine in ["sqlite", "duckdb"] {
                let run = |c: &SqlCase| -> Option<SqlOutcome> {
                    if engine == "sqlite" {
                        SqliteImpl.run(c).ok()
                    } else {
                        DuckDbImpl.run(c).ok()
                    }
                };
                let (Some(w), Some(t), Some(f), Some(u)) = (
                    run(&parts.whole),
                    run(&parts.is_true),
                    run(&parts.is_false),
                    run(&parts.is_unknown),
                ) else {
                    continue;
                };

                match check_grouped(&parts.funcs, &w, &t, &f, &u) {
                    Relation::Violated {
                        only_in_whole,
                        only_in_partitions,
                        ..
                    } => panic!(
                        "seed {seed} on {engine}: grouped TLP violated — far more likely a \
                         defect in the recombination rule than an engine bug at this stage.\n\
                         {}\n{}\n{}",
                        only_in_whole.join(" "),
                        only_in_partitions.join(" "),
                        parts
                            .whole
                            .statements(crate::render::Dialect::Sqlite)
                            .join(";\n")
                    ),
                    Relation::Holds => checked += 1,
                    Relation::NotChecked(_) => {}
                }
            }
        }

        assert!(checked > 50, "only {checked} grouped checks ran");
    }

    /// NoREC's recombination, against fabricated results.
    #[test]
    fn norec_counts_only_true_rows() {
        let count = |n: i64| SqlOutcome::Rows(vec![vec![Cell::Integer(n)]]);
        let projected = |values: &[Option<i64>]| {
            SqlOutcome::Rows(
                values
                    .iter()
                    .map(|value| {
                        vec![match value {
                            Some(number) => Cell::Integer(*number),
                            None => Cell::Null,
                        }]
                    })
                    .collect(),
            )
        };

        // Two true, one false, one unknown: `WHERE p` returns 2.
        assert_eq!(
            check_norec(&count(2), &projected(&[Some(1), Some(0), Some(1), None])),
            Relation::Holds
        );

        // **`NULL` must not count.** Counting it would make every case involving a `NULL` in
        // the predicate look like a violation — the mistake that turns this into a noise
        // generator rather than an oracle.
        assert!(matches!(
            check_norec(&count(3), &projected(&[Some(1), Some(0), Some(1), None])),
            Relation::Violated { .. }
        ));

        // An empty table: nothing on either side.
        assert_eq!(check_norec(&count(0), &projected(&[])), Relation::Holds);
    }

    /// NoREC against the real engines, before it is trusted to report anything.
    #[test]
    fn norec_holds_across_generated_cases() {
        let generator = SqlGenerator::new(Bounds::V1);
        let mut checked = 0;

        for seed in 0..300 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(pair) = norec(&case) else { continue };

            for engine in ["sqlite", "duckdb"] {
                let run = |c: &SqlCase| -> Option<SqlOutcome> {
                    if engine == "sqlite" {
                        SqliteImpl.run(c).ok()
                    } else {
                        DuckDbImpl.run(c).ok()
                    }
                };
                let (Some(filtered), Some(projected)) = (run(&pair.filtered), run(&pair.projected))
                else {
                    continue;
                };

                match check_norec(&filtered, &projected) {
                    Relation::Violated {
                        only_in_whole,
                        only_in_partitions,
                        ..
                    } => panic!(
                        "seed {seed} on {engine}: NoREC violated — at this stage far more \
                         likely a defect in the rewrite than an optimizer bug.\n{}\n{}\n{}",
                        only_in_whole.join(" "),
                        only_in_partitions.join(" "),
                        pair.filtered
                            .statements(crate::render::Dialect::Sqlite)
                            .join(";\n")
                    ),
                    Relation::Holds => checked += 1,
                    Relation::NotChecked(_) => {}
                }
            }
        }

        assert!(checked > 100, "only {checked} NoREC checks ran");
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
        // Measured across **all three** forms, not just the row one. Pinning only `partition`
        // was how the grouped form's contribution went unmeasured: the row figure is unchanged
        // by adding a second or third form, so it cannot show whether they earn their place.
        for (name, bounds, floor) in [("V1", Bounds::V1, 50), ("V1_ALL", Bounds::V1_ALL, 40)] {
            let generator = SqlGenerator::new(bounds);
            let (mut rows, mut aggregate, mut grouped) = (0, 0, 0);
            for seed in 0..300 {
                let case = generator.generate(&mut SeededRng::from_seed(seed));
                if partition(&case).is_some() {
                    rows += 1;
                }
                if partition_aggregate(&case).is_some() {
                    aggregate += 1;
                }
                if partition_grouped(&case).is_some() {
                    grouped += 1;
                }
            }
            let percent = 100 * (rows + aggregate + grouped) / 300;
            assert!(
                percent >= floor,
                "{name}: only {percent}% partitionable ({rows} rows, {aggregate} aggregate, \
                 {grouped} grouped), below the {floor}% this test pins"
            );
        }
    }

    /// The grouped form must actually reach cases the other two cannot, or it is dead weight
    /// dressed up as coverage.
    #[test]
    fn the_grouped_form_reaches_cases_the_others_refuse() {
        let generator = SqlGenerator::new(Bounds::V1_ALL);
        let mut only_grouped = 0;
        for seed in 0..1_000 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if partition_grouped(&case).is_some() {
                assert!(
                    partition(&case).is_none() && partition_aggregate(&case).is_none(),
                    "seed {seed} matched more than one form"
                );
                only_grouped += 1;
            }
        }
        assert!(
            only_grouped > 30,
            "the grouped form judged only {only_grouped} of 1000 cases — too few to be worth its \
             recombination rules"
        );
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
                    parts
                        .whole
                        .statements(crate::render::Dialect::Sqlite)
                        .join(";\n")
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
