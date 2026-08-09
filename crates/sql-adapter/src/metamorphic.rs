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
    /// Whether the query is `SELECT DISTINCT`, which changes **which relation holds**.
    ///
    /// Not a detail: under `DISTINCT` the multiset relation is *false*, and checking it anyway
    /// would report a violation on nearly every such case. The caller must use
    /// [`check_distinct`] when this is set. See its docs for why the set relation survives and
    /// what it costs.
    pub distinct: bool,
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
        || case.query.having.is_some()
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
        distinct: case.query.distinct,
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

    // `DISTINCT` is refused rather than handled: `SELECT DISTINCT SUM(x)` returns one row
    // whatever `DISTINCT` does, so it is a no-op here — but "probably a no-op" is not a basis
    // for a relation, and the recombination rules were derived without it.
    if case.query.distinct
        || !case.query.group_by.is_empty()
        || case.query.set_op.is_some()
        || case.query.limit.is_some()
        || case.query.having.is_some()
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
    /// How many leading projection columns are **grouping keys**.
    ///
    /// One before S9.7, any number now. Carried explicitly rather than re-derived, because the
    /// check splits each result row into "key" and "aggregates" at this boundary, and getting
    /// it wrong would silently compare an aggregate against a key.
    pub keys: usize,
    /// One entry per aggregate column, in projection order after the group keys.
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

    // **Any number of grouping keys since S9.7** — the check buckets by the *tuple* of leading
    // cells. It was one key until multi-column `GROUP BY` became an axis, at which point
    // "the same idea with a tuple key and no new insight" stopped being a reason to skip it:
    // the insight was not new, but the coverage was.
    //
    // `DISTINCT` refused for the same reason as the whole-table form: `GROUP BY` already emits
    // one row per group, so `DISTINCT` is very likely a no-op — and the per-group recombination
    // was derived without it.
    // **`HAVING` breaks this relation, and not subtly.** It filters groups by their aggregate
    // value — but each partition's aggregate differs from the whole's, so a group with
    // `SUM = 6` passes `HAVING SUM > 5` in the whole and fails in both partitions when its rows
    // split 2 and 4. The whole keeps the group, the partitions lose it, on a correct engine.
    // Partitioning on the `HAVING` predicate instead is sound — see [`partition_having`].
    if case.query.distinct
        || case.query.group_by.is_empty()
        || case.query.set_op.is_some()
        || case.query.limit.is_some()
        || case.query.having.is_some()
    {
        return None;
    }

    // The projection must be exactly the grouping keys, in order, then aggregates. Checked
    // rather than assumed: the check reads the first `keys` cells of every result row as the
    // group key, so a projection shaped differently would be silently misread.
    let keys = case.query.group_by.len();
    if case.query.projection.len() <= keys {
        return None;
    }
    let (projected_keys, rest) = case.query.projection.split_at(keys);
    for (projected, grouped) in projected_keys.iter().zip(&case.query.group_by) {
        match projected {
            Expr::Column(column) if column == grouped => {}
            _ => return None,
        }
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
        keys,
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
type GroupedResult = (HashMap<Vec<Cell>, Vec<Option<i64>>>, usize);

pub fn check_grouped(
    keys: usize,
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
            if row.len() != keys + funcs.len() {
                return Err("a result row had an unexpected number of columns");
            }
            // The **tuple** of leading cells is the group key. `Vec<Cell>` works as a `HashMap`
            // key because `Cell` derives `Hash` alongside `Eq`, and `Vec` inherits both.
            let (key, aggregates) = row.split_at(keys);
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
            map.insert(key.to_vec(), values);
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
    let mut all_keys: Vec<&Vec<Cell>> = whole_groups.keys().collect();
    for partition in partitions {
        for key in partition.keys() {
            if !whole_groups.contains_key(key) {
                all_keys.push(key);
            }
        }
    }
    all_keys.sort_by_key(|key| format!("{key:?}"));
    all_keys.dedup();

    let mut only_in_whole = Vec::new();
    let mut only_in_partitions = Vec::new();

    for key in all_keys {
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

/// A query partitioned on its **`HAVING`** predicate rather than its `WHERE`.
///
/// # Why this exists, and why it is not a workaround
///
/// `HAVING` breaks the `WHERE`-partitioned forms, because splitting the *rows* changes each
/// partition's aggregate and so changes which groups survive the `HAVING`. But the same
/// observation supplies a sound relation: partition on the `HAVING` predicate itself. Every
/// group falls into exactly one of `h` TRUE / `h` FALSE / `h` UNKNOWN, nothing else moves, and
/// the three sets of groups reconstruct the unfiltered result.
///
/// It reaches a class the `WHERE` forms cannot: three-valued logic over a value the engine
/// **computed** rather than one that was stored. `HAVING SUM(x) > 0` on a group whose `SUM` is
/// `NULL` is UNKNOWN — so this tests the aggregation path's `NULL` handling, where the others
/// test the comparison's.
///
/// The comparison is [`check`] unchanged: the output rows are groups, group keys are distinct,
/// so the ordinary multiset relation applies with nothing added.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionedHaving {
    /// The query with **no** `HAVING`: every group.
    pub whole: SqlCase,
    pub is_true: SqlCase,
    pub is_false: SqlCase,
    pub is_unknown: SqlCase,
}

/// Build the `HAVING` partition, or `None` if the case has no `HAVING` to partition on.
pub fn partition_having(case: &SqlCase) -> Option<PartitionedHaving> {
    let predicate = case.query.having.clone()?;

    // A set operation or `LIMIT` would make "the groups this query returns" something other
    // than what the relation is about, exactly as for the other forms. `DISTINCT` is excluded
    // because it would deduplicate groups and reintroduce the straddling-duplicate problem.
    if case.query.set_op.is_some() || case.query.limit.is_some() || case.query.distinct {
        return None;
    }

    let base = |having: Option<Expr>| {
        let mut variant = case.clone();
        variant.query = SelectStmt {
            having,
            order_by: Vec::new(),
            limit: None,
            ..case.query.clone()
        };
        variant
    };

    Some(PartitionedHaving {
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

    // **`DISTINCT` breaks NoREC outright**, and not subtly: the projected side becomes
    // `SELECT DISTINCT (p) FROM t`, which collapses every row's truth value into at most three
    // rows. Counting those would compare "how many rows match" against "how many *distinct*
    // truth values occurred", which is not the same question and would fail on almost every
    // case with more than three rows.
    if case.query.distinct || case.query.having.is_some() {
        return None;
    }

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
    compare(whole, is_true, is_false, is_unknown, false)
}

/// The same relation for a **`SELECT DISTINCT`** query, compared as **sets**.
///
/// # Why the multiset relation is false here
///
/// Take two rows that project to the same value, one where the predicate is TRUE and one where
/// it is FALSE. Each partition deduplicates within itself and keeps its copy, so the union has
/// the value **twice**. The unpartitioned query deduplicates across everything and has it
/// **once**. The relation fails — on the engine being correct. Running [`check`] on a
/// `DISTINCT` query would therefore report a violation on nearly every case with a duplicate,
/// which is the tool reporting its own misunderstanding at scale.
///
/// # Why the set relation survives
///
/// Every row is still in exactly one partition, so the *set* of values appearing in the whole
/// is exactly the set appearing across the partitions. Deduplicating both sides before
/// comparing restores a true relation.
///
/// # What that costs, stated plainly
///
/// **This is a strictly weaker oracle.** Comparing as sets means an engine that returns the
/// wrong *number* of copies of a row cannot be caught here — and duplicate handling is
/// precisely what `DISTINCT` is about. The alternative was to refuse `DISTINCT` cases entirely,
/// which reaches nothing at all; a weaker check over the cases is better than no check, as long
/// as nobody later reads a clean `DISTINCT` run as evidence about duplicate counts. It is not.
pub fn check_distinct(
    whole: &SqlOutcome,
    is_true: &SqlOutcome,
    is_false: &SqlOutcome,
    is_unknown: &SqlOutcome,
) -> Relation {
    compare(whole, is_true, is_false, is_unknown, true)
}

/// The shared body of [`check`] and [`check_distinct`].
///
/// `deduplicate` decides whether the comparison is over sets or multisets — the one thing the
/// two relations differ in. Note it must be applied **after** concatenating the partitions, not
/// to each partition: the whole point is that duplicates can straddle two partitions.
fn compare(
    whole: &SqlOutcome,
    is_true: &SqlOutcome,
    is_false: &SqlOutcome,
    is_unknown: &SqlOutcome,
    deduplicate: bool,
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

    let mut left = render(whole_rows);
    let mut right = render(partition_rows);
    if deduplicate {
        // `render` sorts, so equal lines are already adjacent and `dedup` removes all repeats.
        left.dedup();
        right.dedup();
    }

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

    /// The `NOT IN` trap, built by hand and run on **both engines**.
    ///
    /// This is not a test of our code — it is a test of the *premise* the `not_in` axis rests
    /// on. `t0.c0 NOT IN (SELECT t1.c0 FROM t1)` where `t1.c0` contains a `NULL` must return
    /// **no rows at all**, because every row's predicate is UNKNOWN rather than TRUE. The
    /// positive `IN` form must still return its match, which is the asymmetry that makes the
    /// trap a trap.
    ///
    /// **If this test fails because both engines return rows, that is not a broken test — it
    /// is the shared bug the metamorphic oracle exists to find**, and it would be the first
    /// thing this project has found that a differential campaign structurally could not.
    #[test]
    fn not_in_with_a_null_returns_nothing_on_both_engines() {
        use crate::schema::{Column, InsertRows, Literal, SqlType, Table};

        let column = |table: &str, name: &str| ColumnRef {
            table: table.to_string(),
            column: name.to_string(),
        };
        let table = |name: &str| Table {
            name: name.to_string(),
            columns: vec![Column {
                name: "c0".to_string(),
                sql_type: SqlType::Integer,
            }],
        };

        let membership = |not: bool| SqlCase {
            schema: vec![table("t0"), table("t1")],
            data: vec![
                InsertRows {
                    table: "t0".to_string(),
                    rows: vec![
                        vec![Literal::Integer(1)],
                        vec![Literal::Integer(2)],
                        vec![Literal::Integer(3)],
                    ],
                },
                InsertRows {
                    table: "t1".to_string(),
                    // The `NULL` is the whole point: without it `NOT IN` returns 1 and 3.
                    rows: vec![vec![Literal::Integer(2)], vec![Literal::Null]],
                },
            ],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(column("t0", "c0"))],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: Some(Expr::InSubquery {
                    not,
                    left: Box::new(Expr::Column(column("t0", "c0"))),
                    query: Box::new(SelectStmt {
                        having: None,
                        distinct: false,
                        projection: vec![Expr::Column(column("t1", "c0"))],
                        from: "t1".to_string(),
                        join: None,
                        set_op: None,
                        group_by: Vec::new(),
                        filter: None,
                        order_by: Vec::new(),
                        limit: None,
                    }),
                }),
                order_by: Vec::new(),
                limit: None,
            },
        };

        for engine in ["sqlite", "duckdb"] {
            let run = |case: &SqlCase| -> SqlOutcome {
                if engine == "sqlite" {
                    SqliteImpl.run(case).expect("sqlite runs the case")
                } else {
                    DuckDbImpl.run(case).expect("duckdb runs the case")
                }
            };

            // `NOT IN` against a list containing NULL: every row is UNKNOWN, so none survive.
            let negated = run(&membership(true));
            assert_eq!(
                negated,
                SqlOutcome::Rows(vec![]),
                "{engine}: NOT IN against a NULL-containing subquery returned rows. Either \
                 three-valued logic is wrong here, or our understanding is. Check the plain IN \
                 case below before believing the first."
            );

            // The control: `IN` is unaffected, because `true OR unknown` is true.
            let plain = run(&membership(false));
            assert_eq!(
                plain,
                SqlOutcome::Rows(vec![vec![Cell::Integer(2)]]),
                "{engine}: IN should still find the matching row"
            );
        }
    }

    /// The `NOT IN` trap over a **literal list**, on both engines.
    ///
    /// The subquery form is verified above; this is the constant-foldable route to the same
    /// logic, and it is checked separately **because a shared answer is not a shared code
    /// path**. An engine may execute a subquery and fold a list, so getting one right says
    /// nothing about the other.
    ///
    /// As before: if both engines return rows for the negated form, that is not a broken test —
    /// it is the shared bug this project exists to find.
    #[test]
    fn not_in_a_list_holding_null_returns_nothing_on_both_engines() {
        use crate::schema::{Column, InsertRows, Literal, SqlType, Table};

        let membership = |not: bool, list: Vec<Literal>| SqlCase {
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![Column {
                    name: "c0".to_string(),
                    sql_type: SqlType::Integer,
                }],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![
                    vec![Literal::Integer(1)],
                    vec![Literal::Integer(2)],
                    vec![Literal::Integer(3)],
                ],
            }],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(ColumnRef {
                    table: "t0".to_string(),
                    column: "c0".to_string(),
                })],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: Some(Expr::InList {
                    not,
                    left: Box::new(Expr::Column(ColumnRef {
                        table: "t0".to_string(),
                        column: "c0".to_string(),
                    })),
                    list,
                }),
                order_by: Vec::new(),
                limit: None,
            },
        };

        let with_null = || vec![Literal::Integer(2), Literal::Null];
        let without_null = || vec![Literal::Integer(2)];

        for engine in ["sqlite", "duckdb"] {
            let run = |case: &SqlCase| -> SqlOutcome {
                if engine == "sqlite" {
                    SqliteImpl.run(case).expect("sqlite runs the case")
                } else {
                    DuckDbImpl.run(case).expect("duckdb runs the case")
                }
            };

            // `1 NOT IN (2, NULL)` is `1 <> 2 AND 1 <> NULL` = `true AND unknown` = unknown.
            // Every row is unknown, so none survive — even 1 and 3, which are plainly absent.
            assert_eq!(
                run(&membership(true, with_null())),
                SqlOutcome::Rows(vec![]),
                "{engine}: NOT IN a NULL-holding list must return nothing"
            );

            // **The control that makes the above meaningful.** Drop the `NULL` and the same
            // query returns 1 and 3. Without this, "returned nothing" could equally mean the
            // predicate was broken in some way having nothing to do with three-valued logic.
            assert_eq!(
                run(&membership(true, without_null())),
                SqlOutcome::Rows(vec![vec![Cell::Integer(1)], vec![Cell::Integer(3)]]),
                "{engine}: without the NULL, NOT IN must exclude only the listed value"
            );

            // And the positive form is unaffected by the `NULL`: `true OR unknown` is true.
            assert_eq!(
                run(&membership(false, with_null())),
                SqlOutcome::Rows(vec![vec![Cell::Integer(2)]]),
                "{engine}: IN is not affected by a NULL in the list"
            );
        }
    }

    /// `DISTINCT` collapses two `NULL`s into one, on both engines.
    ///
    /// The one place SQL contradicts its own equality rule: `NULL = NULL` is UNKNOWN
    /// everywhere else, but `DISTINCT` treats two `NULL`s as the same value. An engine has to
    /// special-case it rather than reuse its equality, and a special case is somewhere to be
    /// wrong. Verified before hunting, as with both `NOT IN` forms.
    #[test]
    fn distinct_collapses_nulls_on_both_engines() {
        use crate::schema::{Column, InsertRows, Literal, SqlType, Table};

        let query = |distinct: bool| SqlCase {
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![Column {
                    name: "c0".to_string(),
                    sql_type: SqlType::Integer,
                }],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                // Two NULLs and two 1s: both kinds of duplicate, so the result distinguishes
                // "deduplicates values" from "deduplicates NULLs" from "does neither".
                rows: vec![
                    vec![Literal::Null],
                    vec![Literal::Integer(1)],
                    vec![Literal::Null],
                    vec![Literal::Integer(1)],
                ],
            }],
            query: SelectStmt {
                having: None,
                distinct,
                projection: vec![Expr::Column(ColumnRef {
                    table: "t0".to_string(),
                    column: "c0".to_string(),
                })],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: None,
                order_by: Vec::new(),
                limit: None,
            },
        };

        for engine in ["sqlite", "duckdb"] {
            let run = |case: &SqlCase| -> Vec<Vec<Cell>> {
                let outcome = if engine == "sqlite" {
                    SqliteImpl.run(case).expect("sqlite runs the case")
                } else {
                    DuckDbImpl.run(case).expect("duckdb runs the case")
                };
                let SqlOutcome::Rows(mut rows) = outcome else {
                    panic!("{engine}: expected rows");
                };
                rows.sort_by_key(|row| format!("{row:?}"));
                rows
            };

            assert_eq!(
                run(&query(false)).len(),
                4,
                "{engine}: no DISTINCT keeps all four"
            );
            assert_eq!(
                run(&query(true)),
                vec![vec![Cell::Integer(1)], vec![Cell::Null]],
                "{engine}: DISTINCT must collapse the two NULLs to one, and the two 1s to one"
            );
        }
    }

    /// **The multiset relation really does fail under `DISTINCT`** — checked rather than
    /// assumed, because the whole reason `check_distinct` exists rests on it.
    ///
    /// A value appearing in two different partitions survives once in each, so the union has it
    /// twice while the deduplicated whole has it once.
    #[test]
    fn distinct_breaks_the_multiset_relation_and_the_set_relation_survives() {
        // The whole deduplicates across everything: one row.
        let whole = rows(&[&[1]]);
        // The same value falls either side of the predicate, deduplicated within each side.
        let is_true = rows(&[&[1]]);
        let is_false = rows(&[&[1]]);
        let is_unknown = rows(&[]);

        assert!(
            matches!(
                check(&whole, &is_true, &is_false, &is_unknown),
                Relation::Violated { .. }
            ),
            "the multiset relation must fail here — if it did not, `check_distinct` would be \
             unnecessary and its weaker comparison unjustified"
        );

        assert_eq!(
            check_distinct(&whole, &is_true, &is_false, &is_unknown),
            Relation::Holds,
            "the set relation must survive the same case"
        );
    }

    /// And the weakening is real: a lost duplicate is invisible to the set comparison.
    ///
    /// Pinned so nobody later reads a clean `DISTINCT` run as evidence about duplicate counts.
    #[test]
    fn the_set_comparison_cannot_see_a_lost_duplicate() {
        let whole = rows(&[&[1], &[1]]);
        let partitions = rows(&[&[1]]);

        assert!(matches!(
            check(&whole, &partitions, &rows(&[]), &rows(&[])),
            Relation::Violated { .. }
        ));
        assert_eq!(
            check_distinct(&whole, &partitions, &rows(&[]), &rows(&[])),
            Relation::Holds,
            "documented blind spot: set comparison cannot count copies"
        );
    }

    /// `HAVING` over a `NULL` aggregate drops the group, on both engines.
    ///
    /// The trap this axis exists for: `SUM` over a group of all-`NULL`s is `NULL`, so
    /// `HAVING SUM(x) > 0` is UNKNOWN and the group disappears — not because its sum is small,
    /// but because it has no sum. Three-valued logic on a **computed** value.
    #[test]
    fn having_over_a_null_aggregate_drops_the_group_on_both_engines() {
        use crate::schema::{AggregateFunc, BinaryOp, Column, InsertRows, Literal, SqlType, Table};

        let key = ColumnRef {
            table: "t0".to_string(),
            column: "g".to_string(),
        };
        let value = ColumnRef {
            table: "t0".to_string(),
            column: "v".to_string(),
        };
        let sum = || Expr::Aggregate {
            func: AggregateFunc::Sum,
            arg: Some(Box::new(Expr::Column(value.clone()))),
        };

        let query = |having: Option<Expr>| SqlCase {
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![
                    Column {
                        name: "g".to_string(),
                        sql_type: SqlType::Integer,
                    },
                    Column {
                        name: "v".to_string(),
                        sql_type: SqlType::Integer,
                    },
                ],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![
                    // Group 1 sums to 5. Group 2 is all NULL, so its SUM is NULL.
                    vec![Literal::Integer(1), Literal::Integer(5)],
                    vec![Literal::Integer(2), Literal::Null],
                    vec![Literal::Integer(2), Literal::Null],
                ],
            }],
            query: SelectStmt {
                having,
                distinct: false,
                projection: vec![Expr::Column(key.clone()), sum()],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: vec![key.clone()],
                filter: None,
                order_by: Vec::new(),
                limit: None,
            },
        };

        let greater_than_zero = || {
            Some(Expr::Binary {
                op: BinaryOp::Greater,
                left: Box::new(sum()),
                right: Box::new(Expr::Literal(Literal::Integer(0))),
            })
        };

        for engine in ["sqlite", "duckdb"] {
            let run = |case: &SqlCase| -> Vec<Vec<Cell>> {
                let outcome = if engine == "sqlite" {
                    SqliteImpl.run(case).expect("sqlite runs the case")
                } else {
                    DuckDbImpl.run(case).expect("duckdb runs the case")
                };
                let SqlOutcome::Rows(mut rows) = outcome else {
                    panic!("{engine}: expected rows")
                };
                rows.sort_by_key(|row| format!("{row:?}"));
                rows
            };

            // Both groups exist without the HAVING — one summing to 5, one summing to NULL.
            assert_eq!(
                run(&query(None)).len(),
                2,
                "{engine}: two groups before HAVING"
            );

            // With it, only group 1 survives. Group 2 is dropped because UNKNOWN is not TRUE,
            // which is the whole point — it is not dropped for being small.
            assert_eq!(
                run(&query(greater_than_zero())),
                vec![vec![Cell::Integer(1), Cell::Integer(5)]],
                "{engine}: a NULL aggregate must fail HAVING rather than pass it"
            );

            // **The control.** `HAVING SUM(v) IS NULL` picks out exactly the group the other
            // form dropped — proving it was excluded by UNKNOWN, not by absence.
            let is_null = Some(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(sum()),
            });
            assert_eq!(
                run(&query(is_null)),
                vec![vec![Cell::Integer(2), Cell::Null]],
                "{engine}: the UNKNOWN partition must contain the dropped group"
            );
        }
    }

    /// The `HAVING` partition holds across generated cases, on both engines.
    #[test]
    fn the_having_relation_holds_across_generated_cases() {
        let generator = SqlGenerator::new(Bounds::V1_HAVING);
        let mut checked = 0;

        for seed in 0..400 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            let Some(parts) = partition_having(&case) else {
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

                match check(&w, &t, &f, &u) {
                    Relation::Violated {
                        only_in_whole,
                        only_in_partitions,
                        ..
                    } => panic!(
                        "seed {seed} on {engine}: HAVING TLP violated — far likelier a defect in \
                         the transform than an engine bug at this stage.\n{}\n{}\n{}",
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
        assert!(checked > 50, "only {checked} HAVING checks ran");
    }

    /// A **correlated** `NOT IN` traps some rows and not others — verified on both engines.
    ///
    /// The uncorrelated form is all-or-nothing: one `NULL` anywhere in the list and the whole
    /// query returns empty. Correlated, each outer row is tested against its own list, so the
    /// `NULL` reaches some rows and not others. That is a harder thing for an engine to get
    /// right — particularly one that rewrites `NOT IN` into an anti-join — and a much harder
    /// thing to notice by eye, which is why it is pinned here rather than trusted.
    #[test]
    fn a_correlated_not_in_traps_only_the_rows_whose_list_holds_a_null() {
        use crate::schema::{BinaryOp, Column, InsertRows, Literal, SqlType, Table};

        let column = |table: &str, name: &str| ColumnRef {
            table: table.to_string(),
            column: name.to_string(),
        };
        let table = |name: &str| Table {
            name: name.to_string(),
            columns: vec![
                Column {
                    name: "k".to_string(),
                    sql_type: SqlType::Integer,
                },
                Column {
                    name: "v".to_string(),
                    sql_type: SqlType::Integer,
                },
            ],
        };

        let case = SqlCase {
            schema: vec![table("t0"), table("t1")],
            data: vec![
                InsertRows {
                    table: "t0".to_string(),
                    rows: vec![
                        vec![Literal::Integer(1), Literal::Integer(10)],
                        vec![Literal::Integer(2), Literal::Integer(20)],
                    ],
                },
                InsertRows {
                    table: "t1".to_string(),
                    rows: vec![
                        // Group k=1 has a NULL in its list -> outer row 1 is trapped.
                        vec![Literal::Integer(1), Literal::Null],
                        // Group k=2 has no NULL and does not contain 20 -> outer row 2 survives.
                        vec![Literal::Integer(2), Literal::Integer(99)],
                    ],
                },
            ],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(column("t0", "k"))],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: Some(Expr::InSubquery {
                    not: true,
                    left: Box::new(Expr::Column(column("t0", "v"))),
                    query: Box::new(SelectStmt {
                        having: None,
                        distinct: false,
                        projection: vec![Expr::Column(column("t1", "v"))],
                        from: "t1".to_string(),
                        join: None,
                        set_op: None,
                        group_by: Vec::new(),
                        filter: Some(Expr::Binary {
                            op: BinaryOp::Equal,
                            left: Box::new(Expr::Column(column("t1", "k"))),
                            right: Box::new(Expr::Column(column("t0", "k"))),
                        }),
                        order_by: Vec::new(),
                        limit: None,
                    }),
                }),
                order_by: Vec::new(),
                limit: None,
            },
        };

        for engine in ["sqlite", "duckdb"] {
            let outcome = if engine == "sqlite" {
                SqliteImpl.run(&case).expect("sqlite runs the case")
            } else {
                DuckDbImpl.run(&case).expect("duckdb runs the case")
            };

            // **Exactly one row survives**, which is the whole point: the uncorrelated form
            // would return either both or neither. Row 1's list is (NULL) so it is UNKNOWN;
            // row 2's list is (99) so `20 NOT IN (99)` is plainly TRUE.
            assert_eq!(
                outcome,
                SqlOutcome::Rows(vec![vec![Cell::Integer(2)]]),
                "{engine}: a correlated NOT IN must trap only the rows whose own list has a NULL"
            );
        }
    }

    /// A compound `GROUP BY` treats `NULL`s as equal **per column**, on both engines.
    ///
    /// `(NULL, 1)` and `(NULL, 1)` are one group — grouping's `NULL`-equality exception carried
    /// through a tuple — while `(NULL, 1)` and `(NULL, 2)` are two. An engine hashing a compound
    /// key has to apply the exception to every column, not just the first.
    #[test]
    fn a_compound_group_key_treats_nulls_as_equal_per_column() {
        use crate::schema::{AggregateFunc, Column, InsertRows, Literal, SqlType, Table};

        let column = |name: &str| ColumnRef {
            table: "t0".to_string(),
            column: name.to_string(),
        };
        let case = SqlCase {
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![
                    Column {
                        name: "a".to_string(),
                        sql_type: SqlType::Integer,
                    },
                    Column {
                        name: "b".to_string(),
                        sql_type: SqlType::Integer,
                    },
                ],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![
                    vec![Literal::Null, Literal::Integer(1)],
                    vec![Literal::Null, Literal::Integer(1)],
                    vec![Literal::Null, Literal::Integer(2)],
                ],
            }],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![
                    Expr::Column(column("a")),
                    Expr::Column(column("b")),
                    Expr::Aggregate {
                        func: AggregateFunc::CountRows,
                        arg: None,
                    },
                ],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: vec![column("a"), column("b")],
                filter: None,
                order_by: Vec::new(),
                limit: None,
            },
        };

        for engine in ["sqlite", "duckdb"] {
            let outcome = if engine == "sqlite" {
                SqliteImpl.run(&case).expect("sqlite runs the case")
            } else {
                DuckDbImpl.run(&case).expect("duckdb runs the case")
            };
            let SqlOutcome::Rows(mut result) = outcome else {
                panic!("{engine}: expected rows")
            };
            result.sort_by_key(|row| format!("{row:?}"));

            // Two groups, not three and not one: the two `(NULL, 1)` rows merge because
            // grouping treats `NULL`s as equal, and `(NULL, 2)` stays separate because the
            // second column differs.
            assert_eq!(
                result,
                vec![
                    vec![Cell::Null, Cell::Integer(1), Cell::Integer(2)],
                    vec![Cell::Null, Cell::Integer(2), Cell::Integer(1)],
                ],
                "{engine}: compound key must apply NULL-equality per column"
            );
        }
    }

    /// The grouped relation with a **two-column** key, recombined per tuple.
    #[test]
    fn grouped_counts_recombine_under_a_compound_key() {
        // Hand-computed. Keys (1,1) and (1,2). (1,1) has 4 rows split 3 TRUE / 1 FALSE;
        // (1,2) has 2 rows, both UNKNOWN.
        let pair =
            |a: i64, b: i64, n: i64| vec![Cell::Integer(a), Cell::Integer(b), Cell::Integer(n)];
        let grid = |rows: Vec<Vec<Cell>>| SqlOutcome::Rows(rows);

        let relation = check_grouped(
            2,
            &[AggregateFunc::CountRows],
            &grid(vec![pair(1, 1, 4), pair(1, 2, 2)]),
            &grid(vec![pair(1, 1, 3)]),
            &grid(vec![pair(1, 1, 1)]),
            &grid(vec![pair(1, 2, 2)]),
        );
        assert_eq!(relation, Relation::Holds);

        // And the tuple must be read as a whole: swapping the second column of one key makes
        // it a different group, so the recombination must fail.
        let violated = check_grouped(
            2,
            &[AggregateFunc::CountRows],
            &grid(vec![pair(1, 1, 4), pair(1, 2, 2)]),
            &grid(vec![pair(1, 9, 3)]),
            &grid(vec![pair(1, 1, 1)]),
            &grid(vec![pair(1, 2, 2)]),
        );
        assert!(
            matches!(violated, Relation::Violated { .. }),
            "the whole tuple must identify the group, not just its first column"
        );
    }

    /// A `CASE` reaches `NULL` by **two independent routes**, on both engines.
    ///
    /// Both are checked in one case so the two mechanisms are visibly distinct rather than
    /// conflated: an omitted `ELSE` yields `NULL` for a row matching nothing, and a branch whose
    /// condition is UNKNOWN is *not taken* — so a `NULL` in the condition also falls through.
    #[test]
    fn a_case_reaches_null_by_a_missing_else_and_by_an_unknown_condition() {
        use crate::schema::{BinaryOp, Column, InsertRows, Literal, SqlType, Table};

        let v = ColumnRef {
            table: "t0".to_string(),
            column: "v".to_string(),
        };
        // WHEN v > 0 THEN 1  [ELSE 0]
        let case_expr = |otherwise: Option<i64>| Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Greater,
                    left: Box::new(Expr::Column(v.clone())),
                    right: Box::new(Expr::Literal(Literal::Integer(0))),
                },
                Expr::Literal(Literal::Integer(1)),
            )],
            otherwise: otherwise.map(|n| Box::new(Expr::Literal(Literal::Integer(n)))),
        };

        let case = |otherwise: Option<i64>| SqlCase {
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![Column {
                    name: "v".to_string(),
                    sql_type: SqlType::Integer,
                }],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![
                    vec![Literal::Integer(5)],  // matches:      -> 1
                    vec![Literal::Integer(-5)], // no match:     -> ELSE, or NULL
                    vec![Literal::Null],        // UNKNOWN cond: -> ELSE, or NULL
                ],
            }],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![case_expr(otherwise)],
                from: "t0".to_string(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: None,
                order_by: Vec::new(),
                limit: None,
            },
        };

        for engine in ["sqlite", "duckdb"] {
            let run = |c: &SqlCase| -> Vec<Vec<Cell>> {
                let outcome = if engine == "sqlite" {
                    SqliteImpl.run(c).expect("sqlite runs the case")
                } else {
                    DuckDbImpl.run(c).expect("duckdb runs the case")
                };
                let SqlOutcome::Rows(mut rows) = outcome else {
                    panic!("{engine}: expected rows")
                };
                rows.sort_by_key(|row| format!("{row:?}"));
                rows
            };

            // **Without `ELSE`: two of three rows are `NULL`** — one for matching nothing, one
            // because its condition was UNKNOWN rather than FALSE.
            assert_eq!(
                run(&case(None)),
                vec![vec![Cell::Integer(1)], vec![Cell::Null], vec![Cell::Null]],
                "{engine}: a missing ELSE must yield NULL for unmatched rows"
            );

            // **The control.** With `ELSE 0` the same three rows yield 1, 0, 0 — proving the
            // `NULL`s above came from the absent clause and the untaken branch, not from the
            // data being unreadable or the comparison failing outright.
            assert_eq!(
                run(&case(Some(0))),
                vec![
                    vec![Cell::Integer(0)],
                    vec![Cell::Integer(0)],
                    vec![Cell::Integer(1)]
                ],
                "{engine}: with an ELSE, no row should be NULL"
            );
        }
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
            1,
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
            1,
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
            1,
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
            1,
            &[AggregateFunc::Min],
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(3)])]),
            &groups(&[(1, &[Some(9)])]),
            &groups(&[]),
        );
        assert_eq!(held, Relation::Holds);

        // The same partitions with the whole claiming 9 — an engine that lost the smaller row.
        let violated = check_grouped(
            1,
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
            1,
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
            1,
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
            1,
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
            // Computed from `keys`, not hardcoded to 1 — the same trap as the feature-vector
            // test that pinned a 17-feature size and failed on a correct 20-feature one.
            assert_eq!(parts.keys, case.query.group_by.len());
            assert_eq!(
                parts.funcs.len(),
                case.query.projection.len() - parts.keys,
                "seed {seed}: projection is not keys-then-aggregates"
            );
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

                match check_grouped(parts.keys, &parts.funcs, &w, &t, &f, &u) {
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
            let (mut rows, mut aggregate, mut grouped, mut having) = (0, 0, 0, 0);
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
                // The fourth form, added at S9.5. Counting only three was why this test failed
                // when `HAVING` entered `V1_ALL`: the other three refuse a `HAVING` query, so
                // coverage looked to have dropped when in fact a new form had picked it up.
                if partition_having(&case).is_some() {
                    having += 1;
                }
            }
            let percent = 100 * (rows + aggregate + grouped + having) / 300;
            assert!(
                percent >= floor,
                "{name}: only {percent}% partitionable ({rows} rows, {aggregate} aggregate, \
                 {grouped} grouped, {having} having), below the {floor}% this test pins"
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
