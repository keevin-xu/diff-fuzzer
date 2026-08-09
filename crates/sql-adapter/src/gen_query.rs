//! Generating the query, against a schema and data that already exist.
//!
//! This is the second half of SQLancer's split: the state was built first, so every choice
//! here can consult it. A column reference is chosen *from the table's columns*, and the
//! literal it is compared against is drawn *at the column's own type*. Validity is therefore
//! not checked — it is the only thing that can be built.
//!
//! # The three ways a case could be invalid, and how each is made impossible
//!
//! | Invalidity | How it is prevented |
//! |---|---|
//! | A column that does not exist | references are picked from the table, never spelled |
//! | A type mismatch (`c0 = 'x'` where `c0` is an integer) | both sides are generated at one chosen type |
//! | Ambiguity about what a query even means | the constructs that make it ambiguous are not generated |
//!
//! The third is the interesting one, and it is where `POLICY.md`'s Lever 1 lives: a `LIMIT`
//! without a total order does not ask a well-defined question, so it is not produced.

use crate::gen_schema::Bounds;
use crate::ordering::orders_rows_totally;
use crate::schema::{
    AggregateFunc, BinaryOp, ColumnRef, Direction, Expr, InsertRows, Join, JoinKind, Literal,
    OrderKey, SelectStmt, SetBranch, SetOp, SqlType, Table, UnaryOp,
};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Build one `SELECT` over one of the schema's tables.
///
/// Takes the **data** as well as the schema, which looks unnecessary until you ask whether a
/// `LIMIT` is allowed: that depends on whether the `ORDER BY` totally orders *these rows*,
/// which is a fact about the data (see [`crate::ordering`]).
pub fn generate_query(
    rng: &mut SeededRng,
    tables: &[Table],
    data: &[InsertRows],
    bounds: Bounds,
) -> SelectStmt {
    let table = &tables[rng.random_range(0..tables.len())];

    // A join, when enabled and when there is another table to join to. Chosen first because
    // it puts a second table in scope, which every later choice may then reference.
    //
    // **Only some of the time**, and that matters: a joined query is treated as unordered, so
    // joining *every* query would silently strip ordered queries out of the corpus — the same
    // confound the set-op axis produced, where a run reporting clean agreement had quietly
    // stopped testing ordering. Enabling an axis must add cases, never remove them.
    let join = if bounds.joins && tables.len() > 1 && rng.random_range(0..100) < 60 {
        let other = tables
            .iter()
            .find(|candidate| candidate.name != table.name)
            .expect("more than one table");
        generate_join(rng, table, other)
    } else {
        None
    };
    // A joined query is treated as unordered — an outer join manufactures rows that are in no
    // table — so it gets no `ORDER BY` and no `LIMIT`, the same reasoning as grouping.
    let joined = join.is_some();
    let rows = data
        .iter()
        .find(|insert| insert.table == table.name)
        .map(|insert| insert.rows.as_slice())
        .unwrap_or_default();

    // Three shapes of query, when aggregates are enabled: plain rows, a whole-table
    // aggregate, and a grouped aggregate. Chosen up front because the choice constrains
    // what the projection may contain — a grouped query may project only its grouping
    // columns and aggregates, and getting that wrong produces SQL DuckDB refuses.
    let shape = if bounds.aggregates {
        match rng.random_range(0..100) {
            0..=49 => QueryShape::Rows,
            50..=69 => QueryShape::WholeTableAggregate,
            _ => QueryShape::Grouped,
        }
    } else {
        QueryShape::Rows
    };

    let (projection, group_by) = match shape {
        QueryShape::Rows => (generate_projection(rng, table, bounds), Vec::new()),
        QueryShape::WholeTableAggregate => {
            let count = rng.random_range(1..=2);
            let projection = (0..count)
                .map(|_| generate_aggregate(rng, table, bounds))
                .collect();
            (projection, Vec::new())
        }
        QueryShape::Grouped => {
            // One key, or two when the axis allows and the table has two columns to give.
            // Picked **without replacement**: `GROUP BY c0, c0` is legal but degenerate — it
            // groups exactly as `GROUP BY c0` does, so it would dilute the axis with cases
            // that do not exercise a compound key at all.
            let wants_two =
                bounds.multi_group_by && table.columns.len() > 1 && rng.random_range(0..100) < 60;

            let first = rng.random_range(0..table.columns.len());
            let mut chosen = vec![first];
            if wants_two {
                let mut second = rng.random_range(0..table.columns.len() - 1);
                if second >= first {
                    second += 1;
                }
                chosen.push(second);
            }

            let keys: Vec<ColumnRef> = chosen
                .iter()
                .map(|index| reference(table, &table.columns[*index].name))
                .collect();

            let mut projection: Vec<Expr> = keys.iter().cloned().map(Expr::Column).collect();
            for _ in 0..rng.random_range(1..=2) {
                projection.push(generate_aggregate(rng, table, bounds));
            }
            (projection, keys)
        }
    };

    let mut filter =
        (rng.random_range(0..100) < 70).then(|| generate_predicate(rng, table, bounds, 0));

    // An `IN`/`NOT IN` membership test in the `WHERE` clause. **Conjoined with `AND`
    // specifically, never `OR`** — unlike the correlated subquery below, which picks either.
    // `OR` would let the other side of the disjunction return rows on its own, which is
    // exactly what would mask the trap: the whole signal here is a predicate that returns
    // *nothing* when it should return something.
    // The literal-list membership test. Needs no second table, so it fires in single-table
    // schemas too — and is conjoined with `AND` for the same reason as the subquery form: `OR`
    // would let the other side return rows and mask a predicate that should return none.
    if bounds.not_in_list && rng.random_range(0..100) < 45 {
        let membership = generate_in_list(rng, table);
        filter = Some(match filter {
            Some(existing) => Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(existing),
                right: Box::new(membership),
            },
            None => membership,
        });
    }

    // The correlated membership test. Same `AND`-only conjunction as the other two, for the
    // same reason: a disjunction could return rows on its own and mask a predicate that should
    // return none.
    if bounds.not_in_correlated && tables.len() > 1 && rng.random_range(0..100) < 45 {
        let inner = tables
            .iter()
            .find(|candidate| candidate.name != table.name)
            .expect("more than one table");
        if let Some(membership) = generate_correlated_not_in(rng, table, inner) {
            filter = Some(match filter {
                Some(existing) => Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(existing),
                    right: Box::new(membership),
                },
                None => membership,
            });
        }
    }

    if bounds.not_in && tables.len() > 1 && rng.random_range(0..100) < 45 {
        let inner = tables
            .iter()
            .find(|candidate| candidate.name != table.name)
            .expect("more than one table");
        if let Some(membership) = generate_not_in(rng, table, inner) {
            filter = Some(match filter {
                Some(existing) => Expr::Binary {
                    op: BinaryOp::And,
                    left: Box::new(existing),
                    right: Box::new(membership),
                },
                None => membership,
            });
        }
    }

    // A correlated subquery in the `WHERE` clause. Conjoined with whatever predicate is
    // already there rather than replacing it, so enabling this axis **adds** to a case
    // instead of substituting for part of it — the rule three earlier confounds taught.
    if bounds.subqueries && tables.len() > 1 && rng.random_range(0..100) < 45 {
        let inner = tables
            .iter()
            .find(|candidate| candidate.name != table.name)
            .expect("more than one table");
        if let Some(subquery) = generate_subquery(rng, table, inner) {
            filter = Some(match filter {
                Some(existing) => Expr::Binary {
                    op: if rng.random_range(0..2) == 0 {
                        BinaryOp::And
                    } else {
                        BinaryOp::Or
                    },
                    left: Box::new(existing),
                    right: Box::new(subquery),
                },
                None => subquery,
            });
        }
    }

    // Decided **before** the ordering, because it constrains it. Getting this order wrong is
    // not hypothetical: an earlier version suppressed `ORDER BY` for every row query whenever
    // set operations were *enabled*, so a run meant to add one axis silently removed another
    // and produced a corpus with **no ordered queries at all**.
    let wants_set_op = bounds.set_ops && shape == QueryShape::Rows && rng.random_range(0..100) < 55;

    // **`ORDER BY` is only generated for row queries.** A grouped query may order only by
    // its grouping columns or by an aggregate — SQLite tolerates more, DuckDB refuses — and
    // an aggregate with no `GROUP BY` returns a single row, so ordering it says nothing.
    // Generating the strict form is what both engines accept.
    let order_by = match shape {
        // No ordering when *this query* has a set operation: an `ORDER BY` would attach to a
        // branch rather than to the combined result. Note the condition is `wants_set_op`,
        // not `bounds.set_ops` — a row query that did not get one still gets ordered.
        QueryShape::Rows if wants_set_op || joined => Vec::new(),
        QueryShape::Rows => generate_order_by(rng, table),
        QueryShape::WholeTableAggregate => Vec::new(),
        // A grouped query may order by its grouping columns — legal on both engines, and
        // provably total (one row per group). Generated often, because otherwise the whole
        // grouped third of the corpus is compared with its row order sorted away.
        QueryShape::Grouped if rng.random_range(0..100) < 70 => group_by
            .iter()
            .map(|column| OrderKey {
                column: column.clone(),
                direction: if rng.random_range(0..2) == 0 {
                    Direction::Ascending
                } else {
                    Direction::Descending
                },
                nulls_first: rng.random_range(0..2) == 0,
            })
            .collect(),
        QueryShape::Grouped => Vec::new(),
    };

    // **The rule that needs the data.** A `LIMIT` on a query whose order is not total lets
    // two engines return different *rows*, both legally — a difference no normalization can
    // repair and no catalog entry could honestly excuse. So the limit is only offered when
    // the order has been shown to be total for this case's rows.
    // The same restriction, for the same reason plus one more: a grouped query's output rows
    // are *groups*, not seeded rows, so `orders_rows_totally` — which inspects the seeded
    // rows — is answering a different question entirely. Rather than teach it to compute
    // groups, only row queries are eligible for a `LIMIT`.
    let limit = if shape == QueryShape::Rows
        && !joined
        && orders_rows_totally(&order_by, table, rows)
        && rng.random_range(0..100) < 30
    {
        Some(rng.random_range(0..=rows.len() as u32))
    } else {
        None
    };

    // A set operation, when enabled and when the query is the plain row shape. The right
    // branch projects **the same expressions** as the left, which guarantees identical arity
    // and identical types — the two things a set operation requires — while a different
    // `WHERE` is what makes the two sides actually differ. That difference is the point:
    // `INTERSECT` and `EXCEPT` say nothing interesting about two identical row sets.
    // Note what this permits: a set operation whose branches are joined queries, and a
    // grouped query over a join. Those combinations are the point of the combined
    // configuration — each axis alone came back clean.
    let set_op = if wants_set_op {
        let op = match rng.random_range(0..4) {
            0 => SetOp::Union,
            1 => SetOp::UnionAll,
            2 => SetOp::Intersect,
            _ => SetOp::Except,
        };
        // Chaining, when enabled: a third branch under a **different** operator, because
        // precedence is only observable when the operators differ. `A UNION B UNION C` groups
        // the same way whichever rule applies; `A UNION B INTERSECT C` does not — it is
        // `(A UNION B) INTERSECT C` under SQLite's documented left-to-right rule and
        // `A UNION (B INTERSECT C)` under SQL92's. Nothing is parenthesized, deliberately:
        // the rendered text is the probe, and each engine parses it by its own rule.
        //
        // Note the AST nests `A op (B op2 C)` while the text is flat. Here the *text* is the
        // meaning, not the tree — the one place in this crate where that is true, and the
        // reason the renderer must never start adding parentheses to set operations.
        let inner = if bounds.chained_set_ops && rng.random_range(0..100) < 70 {
            let second = match (op, rng.random_range(0..2)) {
                // Pair a deduplicating union or difference with an intersection, which is
                // exactly the pairing the two precedence rules disagree about.
                (SetOp::Intersect, _) => {
                    if rng.random_range(0..2) == 0 {
                        SetOp::Union
                    } else {
                        SetOp::Except
                    }
                }
                (_, _) => SetOp::Intersect,
            };
            Some(SetBranch {
                op: second,
                right: Box::new(SelectStmt {
                    having: None,
                    distinct: false,
                    projection: projection.clone(),
                    from: table.name.clone(),
                    join: None,
                    set_op: None,
                    group_by: Vec::new(),
                    filter: (rng.random_range(0..100) < 80)
                        .then(|| generate_predicate(rng, table, bounds, 0)),
                    order_by: Vec::new(),
                    limit: None,
                }),
            })
        } else {
            None
        };

        Some(SetBranch {
            op,
            right: Box::new(SelectStmt {
                having: None,
                distinct: false,
                projection: projection.clone(),
                from: table.name.clone(),
                join: None,
                set_op: inner,
                group_by: Vec::new(),
                filter: (rng.random_range(0..100) < 80)
                    .then(|| generate_predicate(rng, table, bounds, 0)),
                order_by: Vec::new(),
                limit: None,
            }),
        })
    } else {
        None
    };

    // `HAVING`, on aggregating queries only — it filters groups, so there must be groups.
    //
    // The predicate compares an **aggregate** against a small literal, which is the whole
    // point: it applies three-valued logic to a value the engine *computed*. A `SUM` over a
    // group of all-`NULL`s is `NULL`, so `HAVING SUM(x) > 0` is UNKNOWN and the group vanishes
    // — the same trap as a `NULL` in a `WHERE`, one level up.
    let having = if bounds.having && shape != QueryShape::Rows && rng.random_range(0..100) < 55 {
        // Reuse a projected aggregate rather than inventing one, so the `HAVING` refers to
        // something the reader can see in the result. Both engines also accept an aggregate in
        // `HAVING` that is not projected, but a repro is easier to read when it is.
        // **Only aggregates whose result is numeric**, and this restriction was added after a
        // campaign found it missing. `MAX` over a `TEXT` column is `TEXT`, and comparing that
        // against an integer literal is the documented text-versus-integer difference the
        // subset keeps unrepresentable — SQLite coerces, DuckDB refuses. Generating it produced
        // 825 `rows-vs-error` findings that were all our own invalid SQL.
        //
        // `COUNT` is always numeric; `MIN`/`MAX`/`SUM` are numeric only when their argument is.
        let aggregate = projection
            .iter()
            .find(|expression| numeric_aggregate(expression, table))
            .cloned();
        aggregate.map(|aggregate| Expr::Binary {
            op: match rng.random_range(0..4) {
                0 => BinaryOp::Greater,
                1 => BinaryOp::Less,
                2 => BinaryOp::NotEqual,
                _ => BinaryOp::Equal,
            },
            left: Box::new(aggregate),
            right: Box::new(Expr::Literal(Literal::Integer(rng.random_range(-2..=2)))),
        })
    } else {
        None
    };

    // **`DISTINCT` is decided last, and only when it cannot invalidate the query.** With
    // `DISTINCT`, every `ORDER BY` key must appear in the projection or DuckDB refuses — the
    // same shape of mistake the aggregate widening made, where generated `ORDER BY` on grouped
    // queries was rejected for 24.5% of cases. Deciding after the projection and ordering are
    // fixed means the axis *adds* `DISTINCT` to queries that already permit it, rather than
    // forcing a query shape to accommodate it.
    //
    // Also excluded on set operations: `UNION` already deduplicates, so `SELECT DISTINCT` on a
    // branch tests nothing new while confounding two dedup mechanisms in one case.
    let ordering_is_projected = order_by.iter().all(|key| {
        projection
            .iter()
            .any(|expression| matches!(expression, Expr::Column(column) if *column == key.column))
    });
    // **Row queries only**, and this was caught by a test rather than foreseen. Allowing
    // `DISTINCT` on aggregate and grouped queries dropped the metamorphic oracle's coverage of
    // `V1_ALL` from 40% to 36%, because those two partition forms refuse `DISTINCT` — so the
    // axis was *removing* judgeable cases, which is exactly rule 3.
    //
    // It also loses nothing: `GROUP BY` already emits one row per group and a whole-table
    // aggregate returns a single row, so `DISTINCT` there is a no-op wearing the costume of a
    // construct. The restriction makes the axis additive and costs no coverage of its own.
    let distinct = bounds.distinct
        && shape == QueryShape::Rows
        && set_op.is_none()
        && ordering_is_projected
        && rng.random_range(0..100) < 50;

    SelectStmt {
        having,
        distinct,
        projection,
        from: table.name.clone(),
        join,
        set_op,
        group_by,
        filter,
        order_by,
        limit,
    }
}

/// What kind of query to build. The shape decides what a legal projection looks like.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueryShape {
    /// Ordinary row-returning `SELECT`.
    Rows,
    /// Aggregates with no `GROUP BY`: one row from the whole table — including from an
    /// **empty** table, which is where aggregates are most likely to differ.
    WholeTableAggregate,
    /// `GROUP BY` one column, projecting it plus aggregates.
    Grouped,
}

/// A correlated subquery over `inner`, referencing `outer`'s row.
///
/// Two forms, chosen because they fail differently:
///
/// - **`EXISTS (SELECT ... WHERE inner.c = outer.c)`** — a truth value. Interesting because
///   it is defined even when the inner query returns nothing, and because `NOT EXISTS` over
///   an empty result is the natural way to write "no match", where `NOT IN` famously is not.
/// - **`outer.c <op> (SELECT MAX(inner.c) ...)`** — a comparison against a scalar. Interesting
///   because when the inner query returns **no rows** the scalar is `NULL` and the comparison
///   is unknown rather than false. An aggregate is used so the subquery cannot accidentally
///   return several rows, which is a runtime error rather than a divergence.
///
/// The correlation — `= outer.c` — is the point. Without it the subquery is a constant the
/// optimizer can hoist, and both engines would compute it once and agree.
/// Does this projected expression aggregate to a **number**?
///
/// The question a `HAVING` comparison against an integer literal has to ask first. `COUNT`
/// counts, so it is numeric whatever it counted; `MIN`/`MAX`/`SUM` take the type of their
/// argument, so over a `TEXT` column they are `TEXT` and comparing them to an integer is the
/// cross-type comparison this subset exists to avoid.
fn numeric_aggregate(expression: &Expr, table: &Table) -> bool {
    let Expr::Aggregate { func, arg } = expression else {
        return false;
    };
    match func {
        // A count is a number however it was reached, including `COUNT(text_column)`.
        AggregateFunc::CountRows | AggregateFunc::Count => true,
        AggregateFunc::Min | AggregateFunc::Max | AggregateFunc::Sum => match arg.as_deref() {
            Some(Expr::Column(reference)) => {
                table.column(&reference.column).is_some_and(|(_, column)| {
                    matches!(column.sql_type, SqlType::Integer | SqlType::BigInt)
                })
            }
            // Anything else is not a shape this generator produces; refusing is the safe answer.
            _ => false,
        },
    }
}

/// `c IN (1, 2, NULL)` — or, mostly, `NOT IN`.
///
/// The constant-foldable sibling of [`generate_not_in`]. Three choices carry the design:
///
/// - **A `NULL` in the list at 70%.** Without one, `NOT IN` is perfectly well-behaved and the
///   case tests nothing this axis was added for. The other 30% are the control: if something
///   fires on a `NULL`-free list too, it is not the three-valued-logic trap.
/// - **Values drawn from the column's own pool**, so a match is likely. `x NOT IN (…)` where
///   `x` matches nothing is uninteresting whatever the `NULL` does — the trap is visible only
///   when some rows *would* have been excluded and some *would not*.
/// - **Two to four literals.** Long enough that an engine may switch strategy (a chain of `OR`s
///   versus a hash set), short enough to read in a bug report.
///
/// Returns `Expr` rather than `Option<Expr>`, unlike [`generate_not_in`]: a literal list needs
/// no compatible column *pair* across two tables, so there is no configuration in which it
/// cannot be built. An `Option` here would be a `None` no caller could ever observe.
fn generate_in_list(rng: &mut SeededRng, table: &Table) -> Expr {
    let column = &table.columns[rng.random_range(0..table.columns.len())];

    let length = rng.random_range(2..=4);
    let mut list: Vec<Literal> = (0..length)
        // The same value pool the *data* is drawn from — which is what makes a match likely.
        // `generate_literal` favours boundaries and repeats over uniform sampling, so a list
        // built from it collides with the seeded rows far more often than random values would.
        .map(|_| crate::gen_schema::generate_literal(rng, column.sql_type))
        .collect();

    // The `NULL` goes at a random position rather than always last: an engine that folds the
    // list may treat the first element specially, and a fixed position would never find out.
    if rng.random_range(0..100) < 70 {
        let at = rng.random_range(0..list.len());
        list[at] = Literal::Null;
    }

    Expr::InList {
        not: rng.random_range(0..100) < 80,
        left: Box::new(Expr::Column(reference(table, &column.name))),
        list,
    }
}

/// `outer.a NOT IN (SELECT inner.b FROM inner WHERE inner.c = outer.d)` — the **correlated**
/// membership test.
///
/// The list is recomputed per outer row, so each row is tested against a *different* list — and
/// crucially, whether that list contains a `NULL` can differ row by row. The uncorrelated form
/// is all-or-nothing: one `NULL` anywhere and the whole query returns empty. Here the trap
/// fires for some rows and not others, which is both a harder thing to implement and a harder
/// thing to spot by eye.
///
/// Two column pairs are needed, and they are chosen independently: one for the membership
/// comparison and one for the correlation. Reusing a single pair would restrict the shape for
/// no reason and would make every case look the same to the signature.
fn generate_correlated_not_in(rng: &mut SeededRng, outer: &Table, inner: &Table) -> Option<Expr> {
    let mut pairs = Vec::new();
    for outer_column in &outer.columns {
        for inner_column in &inner.columns {
            if outer_column.sql_type.accepts(inner_column.sql_type) {
                pairs.push((outer_column, inner_column));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }

    let (member_outer, member_inner) = pairs[rng.random_range(0..pairs.len())];
    let (correlate_outer, correlate_inner) = pairs[rng.random_range(0..pairs.len())];

    Some(Expr::InSubquery {
        not: rng.random_range(0..100) < 80,
        left: Box::new(Expr::Column(reference(outer, &member_outer.name))),
        query: Box::new(SelectStmt {
            having: None,
            distinct: false,
            projection: vec![Expr::Column(reference(inner, &member_inner.name))],
            from: inner.name.clone(),
            join: None,
            set_op: None,
            group_by: Vec::new(),
            // **The correlation.** Referencing the outer row here is what makes the subquery
            // re-evaluated per row; `NotEqual` is offered as well as `Equal` so the inner list
            // is sometimes large and sometimes a single row.
            filter: Some(Expr::Binary {
                op: if rng.random_range(0..100) < 75 {
                    BinaryOp::Equal
                } else {
                    BinaryOp::NotEqual
                },
                left: Box::new(Expr::Column(reference(inner, &correlate_inner.name))),
                right: Box::new(Expr::Column(reference(outer, &correlate_outer.name))),
            }),
            order_by: Vec::new(),
            limit: None,
        }),
    })
}

/// `outer.c IN (SELECT inner.d FROM inner)` — or, mostly, `NOT IN`.
///
/// **Uncorrelated, deliberately.** The subquery has no `WHERE` referencing the outer row, so
/// it is evaluated once and yields a fixed list. That is the canonical shape of the trap: the
/// question "does this value appear in that column?" is one every reader believes they can
/// answer, and `NULL` makes the negated form answer *nothing* instead. A correlated variant is
/// also interesting, but it confounds two mechanisms — correlation is already its own axis.
///
/// **`NOT IN` is favoured 4:1 over `IN`**, because the asymmetry is the point: `IN` is
/// well-behaved with `NULL` (`true OR unknown` is true), while `NOT IN` is not
/// (`true AND unknown` is unknown). `IN` is still generated, at a low rate, as a control — if
/// something fires on both forms it is not the trap.
fn generate_not_in(rng: &mut SeededRng, outer: &Table, inner: &Table) -> Option<Expr> {
    // Same type discipline as everywhere else: comparing text against integer is a documented
    // engine difference and is kept unrepresentable rather than catalogued.
    let mut pairs = Vec::new();
    for outer_column in &outer.columns {
        for inner_column in &inner.columns {
            if outer_column.sql_type.accepts(inner_column.sql_type) {
                pairs.push((outer_column, inner_column));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    let (outer_column, inner_column) = pairs[rng.random_range(0..pairs.len())];

    Some(Expr::InSubquery {
        not: rng.random_range(0..100) < 80,
        left: Box::new(Expr::Column(reference(outer, &outer_column.name))),
        query: Box::new(SelectStmt {
            having: None,
            distinct: false,
            projection: vec![Expr::Column(reference(inner, &inner_column.name))],
            from: inner.name.clone(),
            join: None,
            set_op: None,
            group_by: Vec::new(),
            filter: None,
            order_by: Vec::new(),
            limit: None,
        }),
    })
}

fn generate_subquery(rng: &mut SeededRng, outer: &Table, inner: &Table) -> Option<Expr> {
    // The correlation needs one column of each table with compatible types.
    let mut pairs = Vec::new();
    for outer_column in &outer.columns {
        for inner_column in &inner.columns {
            if outer_column.sql_type.accepts(inner_column.sql_type) {
                pairs.push((outer_column, inner_column));
            }
        }
    }
    if pairs.is_empty() {
        return None;
    }
    let (outer_column, inner_column) = pairs[rng.random_range(0..pairs.len())];

    let correlation = Expr::Binary {
        op: if rng.random_range(0..100) < 80 {
            BinaryOp::Equal
        } else {
            BinaryOp::NotEqual
        },
        left: Box::new(Expr::Column(reference(inner, &inner_column.name))),
        right: Box::new(Expr::Column(reference(outer, &outer_column.name))),
    };

    if rng.random_range(0..100) < 55 {
        Some(Expr::Exists {
            not: rng.random_range(0..2) == 0,
            query: Box::new(SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(reference(inner, &inner_column.name))],
                from: inner.name.clone(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: Some(correlation),
                order_by: Vec::new(),
                limit: None,
            }),
        })
    } else {
        // `MIN`/`MAX` keep the subquery scalar *and* keep its type equal to the column's, so
        // the outer comparison stays well-typed. `COUNT` would be an integer regardless of the
        // column, which would break the type discipline for a `TEXT` column.
        let func = if rng.random_range(0..2) == 0 {
            AggregateFunc::Min
        } else {
            AggregateFunc::Max
        };
        Some(Expr::ScalarSubquery {
            op: match rng.random_range(0..4) {
                0 => BinaryOp::Equal,
                1 => BinaryOp::NotEqual,
                2 => BinaryOp::Less,
                _ => BinaryOp::Greater,
            },
            left: Box::new(Expr::Column(reference(outer, &outer_column.name))),
            query: Box::new(SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Aggregate {
                    func,
                    arg: Some(Box::new(Expr::Column(reference(inner, &inner_column.name)))),
                }],
                from: inner.name.clone(),
                join: None,
                set_op: None,
                group_by: Vec::new(),
                filter: Some(correlation),
                order_by: Vec::new(),
                limit: None,
            }),
        })
    }
}

/// A join between two tables, with an `ON` predicate over type-compatible columns.
///
/// The `ON` predicate compares one column from each side, chosen so their types match — which
/// is what makes the join meaningful rather than a filtered cross product. If no pair of
/// compatible columns exists, no join is generated: a join on nothing would be testing the
/// cross product, which is a different construct.
///
/// **Equality is favoured heavily.** An outer join's interest is in the rows that *fail* to
/// match and get padded with `NULL`s, and equality is what produces a realistic mix of matched
/// and unmatched rows. Inequality predicates tend to match everything or nothing.
fn generate_join(rng: &mut SeededRng, left: &Table, right: &Table) -> Option<Join> {
    let mut pairs = Vec::new();
    for left_column in &left.columns {
        for right_column in &right.columns {
            if left_column.sql_type.accepts(right_column.sql_type) {
                pairs.push((left_column, right_column));
            }
        }
    }
    let (left_column, right_column) = *pairs.get(rng.random_range(0..pairs.len().max(1)))?;

    let kind = match rng.random_range(0..4) {
        0 => JoinKind::Inner,
        1 => JoinKind::Left,
        2 => JoinKind::Right,
        _ => JoinKind::Full,
    };

    let op = if rng.random_range(0..100) < 80 {
        BinaryOp::Equal
    } else {
        BinaryOp::NotEqual
    };

    Some(Join {
        kind,
        table: right.name.clone(),
        on: Expr::Binary {
            op,
            left: Box::new(Expr::Column(reference(left, &left_column.name))),
            right: Box::new(Expr::Column(reference(right, &right_column.name))),
        },
    })
}

/// One aggregate over a column of this table.
///
/// `COUNT(*)` and `COUNT(x)` are both generated because they ask different questions —
/// `COUNT(x)` skips `NULL`s — and the generator puts `NULL`s in a quarter of all cells, so
/// the difference is exercised rather than theoretical.
///
/// `SUM` is restricted to 32-bit `INTEGER` columns: DuckDB widens a sum to `HUGEINT` while
/// SQLite keeps an integer until it overflows into `REAL`, so summing a `BIGINT` column of
/// extreme values would reproduce the documented overflow difference instead of testing
/// aggregation. With at most 8 rows of values below 2^31, a sum cannot leave `i64`.
fn generate_aggregate(rng: &mut SeededRng, table: &Table, bounds: Bounds) -> Expr {
    let column = &table.columns[rng.random_range(0..table.columns.len())];

    // **Conditional aggregation** — `SUM(CASE WHEN c > 0 THEN c ELSE NULL END)` — added at S9.12
    // because the corpus-shape check reported "CASE inside a grouped query" at **0%**. It was
    // not impossible, it was unreachable: `CASE` was only emitted by `generate_projection`,
    // which runs for row queries, while grouped queries build their projection from keys and
    // aggregates. A bare `CASE` in a grouped projection would be invalid SQL anyway — neither a
    // key nor an aggregate — so *inside* an aggregate is the form that is both valid and
    // interesting.
    //
    // It is the only place in this generator where two `NULL` mechanisms compose: the `CASE`'s
    // missing-`ELSE` route produces `NULL`s, and the aggregate then has to decide what to do
    // with them (`COUNT(x)` skips them, `SUM` of all-`NULL` is `NULL`, not 0).
    let argument = if bounds.case_expressions && rng.random_range(0..100) < 30 {
        generate_case(rng, table, column.sql_type, bounds)
    } else {
        Expr::Column(reference(table, &column.name))
    };
    let column_ref = argument;

    let summable = column.sql_type == SqlType::Integer;
    let choice = rng.random_range(0..100);

    let (func, arg) = match choice {
        0..=24 => (AggregateFunc::CountRows, None),
        25..=49 => (AggregateFunc::Count, Some(column_ref)),
        50..=69 => (AggregateFunc::Min, Some(column_ref)),
        70..=89 => (AggregateFunc::Max, Some(column_ref)),
        _ if summable => (AggregateFunc::Sum, Some(column_ref)),
        // Not summable: fall back to counting rows rather than skewing toward one column.
        _ => (AggregateFunc::CountRows, None),
    };

    Expr::Aggregate {
        func,
        arg: arg.map(Box::new),
    }
}

/// What the query returns: between one column and all of them, sometimes computed.
fn generate_projection(rng: &mut SeededRng, table: &Table, bounds: Bounds) -> Vec<Expr> {
    let count = rng.random_range(1..=table.columns.len());

    (0..count)
        .map(|_| {
            let column = &table.columns[rng.random_range(0..table.columns.len())];
            // Mostly bare columns: they keep minimized repros readable, and a divergence in
            // a projected column is easier to argue about than one inside an expression.
            if bounds.case_expressions && rng.random_range(0..100) < 30 {
                generate_case(rng, table, column.sql_type, bounds)
            } else if rng.random_range(0..100) < 70 {
                Expr::Column(reference(table, &column.name))
            } else {
                generate_scalar(rng, table, column.sql_type, bounds, 0)
            }
        })
        .collect()
}

/// `CASE WHEN c THEN v ... [ELSE v] END`, with every branch at `result_type`.
///
/// **The `ELSE` is omitted 45% of the time, and that is the axis's whole point.** A row matching
/// no branch then yields `NULL` — a value produced by the *absence* of a clause. The other 55%
/// are the control: if something fires with an `ELSE` present too, it is not the missing-`ELSE`
/// route.
///
/// Every branch value is generated at one type chosen by the caller. Mixing types would make the
/// result column's type differ between engines — the text-versus-integer hazard the `HAVING`
/// axis reintroduced at S9.5 and paid 825 spurious findings for. The type walk in the tests
/// asserts this independently rather than trusting the comment.
fn generate_case(rng: &mut SeededRng, table: &Table, result_type: SqlType, bounds: Bounds) -> Expr {
    let count = rng.random_range(1..=2);
    let branches: Vec<(Expr, Expr)> = (0..count)
        .map(|_| {
            // The condition is an ordinary predicate, so it can itself be UNKNOWN — the second
            // `NULL` route, independent of the missing `ELSE`.
            let when = generate_predicate(rng, table, bounds, bounds.max_expr_depth - 1);
            let then = generate_scalar(rng, table, result_type, bounds, bounds.max_expr_depth - 1);
            (when, then)
        })
        .collect();

    let otherwise = (rng.random_range(0..100) >= 45).then(|| {
        Box::new(generate_scalar(
            rng,
            table,
            result_type,
            bounds,
            bounds.max_expr_depth - 1,
        ))
    });

    Expr::Case {
        branches,
        otherwise,
    }
}

/// A `WHERE` clause: comparisons and `IS NULL` tests, combined with `AND`/`OR`/`NOT`.
///
/// `depth` counts down against `bounds.max_expr_depth`. Bounding depth matters more than it
/// looks: expression size grows exponentially in depth, so an unbounded generator produces a
/// few enormous cases rather than many small ones, and enormous cases are both slow to run
/// and miserable to minimize.
fn generate_predicate(rng: &mut SeededRng, table: &Table, bounds: Bounds, depth: usize) -> Expr {
    if depth >= bounds.max_expr_depth {
        return generate_comparison(rng, table, bounds, depth);
    }

    match rng.random_range(0..100) {
        0..=45 => generate_comparison(rng, table, bounds, depth),
        46..=60 => {
            // `IS NULL` / `IS NOT NULL` — the one way to ask about `NULL` that yields a
            // definite answer rather than an unknown, and therefore worth generating often.
            let column = &table.columns[rng.random_range(0..table.columns.len())];
            Expr::Unary {
                op: if rng.random_range(0..2) == 0 {
                    UnaryOp::IsNull
                } else {
                    UnaryOp::IsNotNull
                },
                operand: Box::new(Expr::Column(reference(table, &column.name))),
            }
        }
        61..=75 => Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(generate_predicate(rng, table, bounds, depth + 1)),
        },
        _ => Expr::Binary {
            op: if rng.random_range(0..2) == 0 {
                BinaryOp::And
            } else {
                BinaryOp::Or
            },
            left: Box::new(generate_predicate(rng, table, bounds, depth + 1)),
            right: Box::new(generate_predicate(rng, table, bounds, depth + 1)),
        },
    }
}

/// A comparison between two things of the *same* type.
///
/// The type is chosen first, from a column that exists, and both sides are then built at
/// that type. This is what makes `1 = 'x'` — where the engines' coercion rules differ —
/// impossible to produce rather than merely unlikely.
fn generate_comparison(rng: &mut SeededRng, table: &Table, bounds: Bounds, depth: usize) -> Expr {
    let column = &table.columns[rng.random_range(0..table.columns.len())];
    let sql_type = column.sql_type;

    let op = match rng.random_range(0..6) {
        0 => BinaryOp::Equal,
        1 => BinaryOp::NotEqual,
        2 => BinaryOp::Less,
        3 => BinaryOp::LessOrEqual,
        4 => BinaryOp::Greater,
        _ => BinaryOp::GreaterOrEqual,
    };

    Expr::Binary {
        op,
        left: Box::new(Expr::Column(reference(table, &column.name))),
        right: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
    }
}

/// A value expression of a given type: a column, a literal, or something computed.
fn generate_scalar(
    rng: &mut SeededRng,
    table: &Table,
    sql_type: SqlType,
    bounds: Bounds,
    depth: usize,
) -> Expr {
    let matching: Vec<&crate::schema::Column> = table
        .columns
        .iter()
        .filter(|candidate| sql_type.accepts(candidate.sql_type))
        .collect();

    let can_recurse = depth < bounds.max_expr_depth;
    let choice = rng.random_range(0..100);

    match sql_type {
        SqlType::Integer | SqlType::BigInt => {
            if choice < 40 || !can_recurse {
                Expr::Literal(crate::gen_schema::generate_literal(rng, sql_type))
            } else if choice < 70 && !matching.is_empty() {
                let column = matching[rng.random_range(0..matching.len())];
                Expr::Column(reference(table, &column.name))
            } else if choice < 85 {
                // Arithmetic, over **small literals only, and never nested**.
                //
                // Measured, not guessed: with column operands and the interesting-value
                // pool in play, `i64::MAX + 1` is reachable — and the engines then part
                // company. SQLite silently promotes the overflowed result to `REAL`
                // (observed: `Real(9.223372036854776e18)`), while DuckDB raises a
                // conversion error. Both behaviours are plausible and neither is obviously
                // wrong, so this is a legal-difference question, not a bug — and it
                // accounted for *every* unjudged case in a 10,000-case run.
                //
                // Bounding both operands to ±100 with no nesting caps any result at 10,000,
                // inside even a 32-bit column. That trades away overflow coverage
                // deliberately: it is a rich area (`PENDING` 2.6) and it comes back at S4 as
                // a *catalogued* experiment, once each engine's behaviour is cited rather
                // than observed. Keeping it now would mean an oracle whose noisiest signal
                // is a difference we cannot yet defend.
                let op = match rng.random_range(0..3) {
                    0 => BinaryOp::Add,
                    1 => BinaryOp::Subtract,
                    _ => BinaryOp::Multiply,
                };
                if bounds.wide_arithmetic {
                    // Overflow is reachable from here, deliberately: operands may be
                    // columns or pool values, so `i64::MAX + 1` occurs. Measured against
                    // the bounded setting at S5 to answer whether it finds anything.
                    Expr::Binary {
                        op,
                        left: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                        right: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                    }
                } else {
                    Expr::Binary {
                        op,
                        left: Box::new(Expr::Literal(crate::schema::Literal::Integer(
                            rng.random_range(-100..=100),
                        ))),
                        right: Box::new(Expr::Literal(crate::schema::Literal::Integer(
                            rng.random_range(-100..=100),
                        ))),
                    }
                }
            } else if choice < 92 {
                // Negation, over a small literal for the same reason: `-(i32::MIN)` has no
                // representation in 32 bits, and the two engines need not agree on what to
                // do about that.
                Expr::Unary {
                    op: UnaryOp::Negate,
                    operand: Box::new(Expr::Literal(crate::schema::Literal::Integer(
                        rng.random_range(-100..=100),
                    ))),
                }
            } else {
                // A cast, and only ever a **widening** one.
                //
                // The first version of this generated either integer width as the target
                // and called it "a widening that cannot fail". That was wrong, and running
                // it said so: `CAST(<bigint> AS INTEGER)` is a *narrowing* cast, which
                // DuckDB refuses when the value exceeds `INT32` while SQLite accepts it —
                // DuckDB's `INTEGER` is four bytes, SQLite's storage class is variable
                // width (`SPECS.md` §2.1, §3.4). Every remaining one-sided refusal in a
                // 2,000-case run was this.
                //
                // `CAST(text AS INTEGER)` is a separate documented difference
                // (`SPECS.md` §5.5, unretrieved) and is not generated at all.
                Expr::Cast {
                    expr: Box::new(generate_scalar(
                        rng,
                        table,
                        SqlType::Integer,
                        bounds,
                        depth + 1,
                    )),
                    to: SqlType::BigInt,
                }
            }
        }
        SqlType::Text => {
            if choice < 60 || matching.is_empty() {
                Expr::Literal(crate::gen_schema::generate_literal(rng, SqlType::Text))
            } else {
                let column = matching[rng.random_range(0..matching.len())];
                Expr::Column(reference(table, &column.name))
            }
        }
        SqlType::Decimal | SqlType::Boolean => {
            unreachable!("{sql_type:?} is not generated in v1; see SqlType::GENERATED")
        }
    }
}

/// `ORDER BY` over some prefix of the table's columns, in a random order.
///
/// Often generated, because an ordered query and an unordered one are compared by different
/// rules and both need to occur. Direction and `NULLS FIRST`/`LAST` are always stated
/// explicitly — never left to an engine default, since the defaults may differ and that
/// difference would be legal (`SPECS.md` §5.6).
fn generate_order_by(rng: &mut SeededRng, table: &Table) -> Vec<OrderKey> {
    if rng.random_range(0..100) < 40 {
        return Vec::new();
    }

    let count = rng.random_range(1..=table.columns.len());
    let mut available: Vec<&crate::schema::Column> = table.columns.iter().collect();
    let mut keys = Vec::with_capacity(count);

    for _ in 0..count {
        // Removed once chosen: ordering by the same column twice is legal and says nothing,
        // and it would make a "total" order look achievable when it is not.
        let column = available.remove(rng.random_range(0..available.len()));
        keys.push(OrderKey {
            column: reference(table, &column.name),
            direction: if rng.random_range(0..2) == 0 {
                Direction::Ascending
            } else {
                Direction::Descending
            },
            nulls_first: rng.random_range(0..2) == 0,
        });
    }

    keys
}

fn reference(table: &Table, column: &str) -> ColumnRef {
    ColumnRef {
        table: table.name.clone(),
        column: column.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_schema::{generate_data, generate_schema};

    fn generated(seed: u64) -> (Vec<Table>, Vec<InsertRows>, SelectStmt) {
        let mut rng = SeededRng::from_seed(seed);
        let tables = generate_schema(&mut rng, Bounds::V1);
        let data = generate_data(&mut rng, &tables, Bounds::V1);
        let query = generate_query(&mut rng, &tables, &data, Bounds::V1);
        (tables, data, query)
    }

    /// The `case` axis must fire, must omit the `ELSE` often enough to test the route it was
    /// added for, and must never mix branch types.
    #[test]
    fn the_case_axis_omits_the_else_often_enough_to_matter() {
        fn walk(expression: &Expr, total: &mut usize, without_else: &mut usize) {
            if let Expr::Case {
                branches,
                otherwise,
            } = expression
            {
                *total += 1;
                if otherwise.is_none() {
                    *without_else += 1;
                }
                assert!(
                    !branches.is_empty(),
                    "`CASE END` with no branches is invalid SQL"
                );
            }
            match expression {
                Expr::Case {
                    branches,
                    otherwise,
                } => {
                    for (when, then) in branches {
                        walk(when, total, without_else);
                        walk(then, total, without_else);
                    }
                    if let Some(otherwise) = otherwise {
                        walk(otherwise, total, without_else);
                    }
                }
                Expr::Unary { operand, .. } => walk(operand, total, without_else),
                Expr::Binary { left, right, .. } => {
                    walk(left, total, without_else);
                    walk(right, total, without_else);
                }
                _ => {}
            }
        }

        let (mut total, mut without_else) = (0usize, 0usize);
        for seed in 0..2_000 {
            let mut rng = SeededRng::from_seed(seed);
            let bounds = Bounds::V1_CASE;
            let tables = generate_schema(&mut rng, bounds);
            let data = generate_data(&mut rng, &tables, bounds);
            let query = generate_query(&mut rng, &tables, &data, bounds);
            for expression in &query.projection {
                walk(expression, &mut total, &mut without_else);
            }
        }

        assert!(total > 200, "only {total} CASE expressions in 2000 cases");
        // **The rate that decides whether the axis tests its own premise.** A `CASE` with an
        // `ELSE` cannot reach `NULL` by the missing-clause route at all.
        assert!(
            without_else * 4 > total,
            "only {without_else} of {total} CASE expressions omitted ELSE — the missing-clause \
             route to NULL is barely being generated"
        );
        // And the control form must exist too, or nothing distinguishes the two routes.
        assert!(
            without_else < total,
            "every CASE omitted its ELSE; no control form"
        );

        // Off by default.
        let mut rng = SeededRng::from_seed(0);
        let tables = generate_schema(&mut rng, Bounds::V1);
        let data = generate_data(&mut rng, &tables, Bounds::V1);
        let query = generate_query(&mut rng, &tables, &data, Bounds::V1);
        let (mut off, mut _e) = (0usize, 0usize);
        for expression in &query.projection {
            walk(expression, &mut off, &mut _e);
        }
        assert_eq!(off, 0, "V1 must not generate CASE expressions");
    }

    /// The correlated `NOT IN` axis must fire, must actually correlate, and must produce cases
    /// that pass the case-level validity check — which resolves names in **two scopes**.
    #[test]
    fn the_correlated_not_in_axis_correlates_and_stays_valid() {
        use crate::ast::SqlCase;

        fn correlated(expression: &Expr, outer_table: &str) -> Option<bool> {
            match expression {
                Expr::InSubquery { query, .. } => {
                    // The inner filter must reference the OUTER table, or the subquery is
                    // uncorrelated and this axis is silently generating the S8 form again.
                    Some(query.filter.as_ref().is_some_and(|filter| {
                        filter.columns().iter().any(|c| c.table == outer_table)
                    }))
                }
                Expr::Unary { operand, .. } => correlated(operand, outer_table),
                Expr::Binary { left, right, .. } => {
                    correlated(left, outer_table).or_else(|| correlated(right, outer_table))
                }
                _ => None,
            }
        }

        let (mut found, mut truly_correlated) = (0usize, 0usize);
        for seed in 0..2_000 {
            let mut rng = SeededRng::from_seed(seed);
            let bounds = Bounds::V1_NOT_IN_CORRELATED;
            let tables = generate_schema(&mut rng, bounds);
            let data = generate_data(&mut rng, &tables, bounds);
            let query = generate_query(&mut rng, &tables, &data, bounds);

            let case = SqlCase {
                schema: tables.clone(),
                data: data.clone(),
                query: query.clone(),
            };
            // **The check that matters for a correlated construct.** Two-scope resolution is
            // where an inner reference to an outer column either works or silently names
            // nothing, and `validate` is what would notice.
            assert!(
                case.validate().is_ok(),
                "seed {seed}: correlated case failed validation: {:?}",
                case.validate()
            );

            if let Some(filter) = &query.filter
                && let Some(is_correlated) = correlated(filter, &query.from)
            {
                found += 1;
                if is_correlated {
                    truly_correlated += 1;
                }
            }
        }

        assert!(
            found > 200,
            "only {found} membership subqueries in 2000 cases"
        );
        assert_eq!(
            found,
            truly_correlated,
            "every subquery on this axis must reference the outer row — {} did not",
            found - truly_correlated
        );
    }

    /// The `distinct` axis must fire, must stay on row queries, and must never attach an
    /// `ORDER BY` key that is not projected — which DuckDB refuses.
    #[test]
    fn the_distinct_axis_stays_valid_and_does_not_suppress_ordering() {
        let (mut distinct, mut distinct_and_ordered, mut ordered) = (0usize, 0usize, 0usize);

        for seed in 0..2_000 {
            let mut rng = SeededRng::from_seed(seed);
            let tables = generate_schema(&mut rng, Bounds::V1_DISTINCT);
            let data = generate_data(&mut rng, &tables, Bounds::V1_DISTINCT);
            let query = generate_query(&mut rng, &tables, &data, Bounds::V1_DISTINCT);

            if !query.order_by.is_empty() {
                ordered += 1;
            }
            if !query.distinct {
                continue;
            }
            distinct += 1;

            // Never on an aggregate or grouped query — a no-op there, and it costs the
            // metamorphic oracle coverage it would otherwise have.
            assert!(
                query.group_by.is_empty() && !query.projection.iter().any(Expr::contains_aggregate),
                "seed {seed}: DISTINCT on an aggregate/grouped query"
            );
            assert!(
                query.set_op.is_none(),
                "seed {seed}: DISTINCT with a set operation"
            );

            // **The validity rule DuckDB enforces**: every ordering key must be projected.
            for key in &query.order_by {
                assert!(
                    query
                        .projection
                        .iter()
                        .any(|e| matches!(e, Expr::Column(column) if *column == key.column)),
                    "seed {seed}: ORDER BY {:?} is not in the projection of a DISTINCT query",
                    key.column
                );
            }
            if !query.order_by.is_empty() {
                distinct_and_ordered += 1;
            }
        }

        assert!(distinct > 200, "only {distinct} DISTINCT queries in 2000");
        assert!(
            ordered > 200,
            "the axis must not suppress ordering: {ordered} ordered"
        );
        // The two must co-occur, or "DISTINCT is only applied where ordering permits it" has
        // silently become "DISTINCT is only applied to unordered queries".
        assert!(
            distinct_and_ordered > 20,
            "only {distinct_and_ordered} queries were both DISTINCT and ordered — the axis is \
             avoiding ordering rather than accommodating it"
        );
    }

    /// The `not_in_list` axis must fire, must favour the negated form, and must usually put a
    /// `NULL` in the list — without which the construct tests nothing it was added for.
    #[test]
    fn the_not_in_list_axis_generates_null_holding_negated_lists() {
        fn count(expression: &Expr, negated: &mut usize, plain: &mut usize, with_null: &mut usize) {
            match expression {
                Expr::InList { not, list, .. } => {
                    if *not {
                        *negated += 1;
                    } else {
                        *plain += 1;
                    }
                    if list.contains(&Literal::Null) {
                        *with_null += 1;
                    }
                    assert!(
                        list.len() >= 2,
                        "a one-element list is a disguised equality"
                    );
                }
                Expr::Unary { operand, .. } => count(operand, negated, plain, with_null),
                Expr::Binary { left, right, .. } => {
                    count(left, negated, plain, with_null);
                    count(right, negated, plain, with_null);
                }
                _ => {}
            }
        }

        let (mut negated, mut plain, mut with_null) = (0usize, 0usize, 0usize);
        for seed in 0..2_000 {
            let mut rng = SeededRng::from_seed(seed);
            let tables = generate_schema(&mut rng, Bounds::V1_NOT_IN_LIST);
            let data = generate_data(&mut rng, &tables, Bounds::V1_NOT_IN_LIST);
            let query = generate_query(&mut rng, &tables, &data, Bounds::V1_NOT_IN_LIST);
            if let Some(filter) = &query.filter {
                count(filter, &mut negated, &mut plain, &mut with_null);
            }
        }

        assert!(negated > 200, "only {negated} negated lists in 2000 cases");
        assert!(
            plain > 0,
            "the positive control form must also be generated"
        );
        assert!(negated > plain * 2, "{negated} negated vs {plain} plain");

        // **The rate that decides whether this axis tests anything.** A list with no `NULL`
        // exercises ordinary membership; only a `NULL` reaches the trap.
        let total = negated + plain;
        assert!(
            with_null * 2 > total,
            "only {with_null} of {total} lists held a NULL — most must, or the axis is testing \
             ordinary membership under the name of a three-valued-logic probe"
        );

        // Off by default, or every earlier measurement changes meaning.
        let mut rng = SeededRng::from_seed(0);
        let tables = generate_schema(&mut rng, Bounds::V1);
        let data = generate_data(&mut rng, &tables, Bounds::V1);
        let query = generate_query(&mut rng, &tables, &data, Bounds::V1);
        let (mut a, mut b, mut c) = (0, 0, 0);
        if let Some(filter) = &query.filter {
            count(filter, &mut a, &mut b, &mut c);
        }
        assert_eq!((a, b), (0, 0), "V1 must not generate membership lists");
    }

    /// The `not_in` axis must actually produce the construct, and must produce the **negated**
    /// form most of the time — the positive form cannot exhibit the trap at all.
    ///
    /// Pinned as a rate rather than "at least one", because an axis that fires once in ten
    /// thousand cases is indistinguishable from a disabled one over a short run.
    #[test]
    fn the_not_in_axis_generates_mostly_negated_membership_tests() {
        fn count(expression: &Expr, negated: &mut usize, plain: &mut usize) {
            match expression {
                Expr::InSubquery { not: true, .. } => *negated += 1,
                Expr::InSubquery { not: false, .. } => *plain += 1,
                Expr::Unary { operand, .. } => count(operand, negated, plain),
                Expr::Binary { left, right, .. } => {
                    count(left, negated, plain);
                    count(right, negated, plain);
                }
                _ => {}
            }
        }

        let (mut negated, mut plain) = (0usize, 0usize);
        for seed in 0..2_000 {
            let mut rng = SeededRng::from_seed(seed);
            let tables = generate_schema(&mut rng, Bounds::V1_NOT_IN);
            let data = generate_data(&mut rng, &tables, Bounds::V1_NOT_IN);
            let query = generate_query(&mut rng, &tables, &data, Bounds::V1_NOT_IN);
            if let Some(filter) = &query.filter {
                count(filter, &mut negated, &mut plain);
            }
        }

        assert!(
            negated > 200,
            "only {negated} negated membership tests in 2000 cases — the axis is barely firing"
        );
        assert!(
            plain > 0,
            "no plain `IN` was generated; the control form is needed to tell the trap from a \
             general membership bug"
        );
        assert!(
            negated > plain * 2,
            "expected NOT IN to dominate ({negated} negated vs {plain} plain)"
        );

        // And the axis must be **off** by default, or every earlier measurement changes meaning.
        let mut rng = SeededRng::from_seed(0);
        let tables = generate_schema(&mut rng, Bounds::V1);
        let data = generate_data(&mut rng, &tables, Bounds::V1);
        let query = generate_query(&mut rng, &tables, &data, Bounds::V1);
        let (mut off_negated, mut off_plain) = (0usize, 0usize);
        if let Some(filter) = &query.filter {
            count(filter, &mut off_negated, &mut off_plain);
        }
        assert_eq!(
            (off_negated, off_plain),
            (0, 0),
            "V1 must not generate membership tests"
        );
    }

    fn table_of<'a>(tables: &'a [Table], name: &str) -> &'a Table {
        tables
            .iter()
            .find(|table| table.name == name)
            .expect("the query reads a table that exists")
    }

    #[test]
    fn generation_is_deterministic() {
        let first = serde_json::to_string(&generated(4242).2).unwrap();
        let second = serde_json::to_string(&generated(4242).2).unwrap();
        assert_eq!(first, second);
    }

    /// Name resolution, checked over many cases: every column a query mentions must exist
    /// in the table it reads.
    #[test]
    fn every_referenced_column_exists() {
        for seed in 0..500 {
            let (tables, _, query) = generated(seed);
            let table = table_of(&tables, &query.from);

            let mut referenced: Vec<&ColumnRef> = Vec::new();
            for expression in &query.projection {
                referenced.extend(expression.columns());
            }
            if let Some(filter) = &query.filter {
                referenced.extend(filter.columns());
            }
            for key in &query.order_by {
                referenced.push(&key.column);
            }

            for reference in referenced {
                assert_eq!(reference.table, table.name, "seed {seed}");
                assert!(
                    table.column(&reference.column).is_some(),
                    "seed {seed}: {} is not a column of {}",
                    reference.column,
                    table.name
                );
            }
        }
    }

    /// The Lever-1 rule that needed the data to enforce.
    #[test]
    fn a_limit_only_appears_with_a_totally_ordered_query() {
        let mut with_limit = 0;

        for seed in 0..500 {
            let (tables, data, query) = generated(seed);
            if query.limit.is_none() {
                continue;
            }
            with_limit += 1;

            let table = table_of(&tables, &query.from);
            let rows = data
                .iter()
                .find(|insert| insert.table == table.name)
                .map(|insert| insert.rows.as_slice())
                .unwrap_or_default();

            assert!(
                orders_rows_totally(&query.order_by, table, rows),
                "seed {seed}: a LIMIT on a query whose order is not total lets the engines \
                 return different rows, both legally"
            );
        }

        // A rule nothing exercises is not a tested rule.
        assert!(with_limit > 0, "no LIMIT was generated in 500 seeds");
    }

    /// Both shapes must occur, since they are compared under different rules at S3.
    #[test]
    fn ordered_and_unordered_queries_both_occur() {
        let ordered = (0..200)
            .filter(|&seed| !generated(seed).2.order_by.is_empty())
            .count();
        assert!(ordered > 20, "too few ordered queries: {ordered}/200");
        assert!(ordered < 180, "too few unordered queries: {ordered}/200");
    }

    /// The type rule, checked structurally: no comparison ever mixes text with integers.
    ///
    /// This is what keeps `1 = '1'` — where the two engines differ — out of every case, and
    /// it is worth testing rather than trusting, because the generator's type discipline is
    /// spread across three functions.
    /// **Runs over every configuration, not just `V1` — and that is the point.**
    ///
    /// This test used to walk only `Bounds::V1`, which is the *narrowest* setting and contains
    /// none of the constructs added after it. Every axis since — subqueries, `not_in`,
    /// `not_in_list`, `distinct`, `having` — was outside its reach, so it could not have failed
    /// no matter what they generated.
    ///
    /// It did not fail, and a campaign found what it missed: a `HAVING MAX(text_column) > -2`,
    /// comparing `TEXT` against an integer, which SQLite coerces and DuckDB refuses. 825
    /// findings, all of them our own invalid SQL. **A test pinned to the narrowest
    /// configuration tests the code that was written before it and nothing since.**
    #[test]
    fn comparisons_never_mix_text_with_integers() {
        for (name, bounds) in [
            ("V1", Bounds::V1),
            ("V1_ALL", Bounds::V1_ALL),
            ("V1_HAVING", Bounds::V1_HAVING),
            ("V1_SUBQUERIES", Bounds::V1_SUBQUERIES),
            ("V1_NOT_IN_LIST", Bounds::V1_NOT_IN_LIST),
            ("V1_CASE", Bounds::V1_CASE),
        ] {
            for seed in 0..500 {
                let mut rng = SeededRng::from_seed(seed);
                let tables = generate_schema(&mut rng, bounds);
                let data = generate_data(&mut rng, &tables, bounds);
                let query = generate_query(&mut rng, &tables, &data, bounds);
                let table = table_of(&tables, &query.from);

                if let Some(filter) = &query.filter {
                    assert_types_agree(filter, table, seed);
                }
                for expression in &query.projection {
                    assert_types_agree(expression, table, seed);
                }
                // **The clause the campaign found missing.** A `HAVING` compares an aggregate
                // against a literal, and nothing was checking that the two had compatible
                // types.
                if let Some(having) = &query.having {
                    assert!(
                        !matches!(assert_types_agree(having, table, seed), Some(SqlType::Text)),
                        "{name} seed {seed}: HAVING is a text comparison"
                    );
                }
            }
        }
    }

    /// A `HAVING` must compare a **numeric** aggregate, never a text one.
    ///
    /// Stated directly as well as via the type walk, because this is the specific rule the
    /// campaign's 825 findings came from and it deserves a test that names it.
    #[test]
    fn having_never_compares_a_text_aggregate_against_a_number() {
        let mut seen = 0;
        for seed in 0..3_000 {
            let mut rng = SeededRng::from_seed(seed);
            let tables = generate_schema(&mut rng, Bounds::V1_HAVING);
            let data = generate_data(&mut rng, &tables, Bounds::V1_HAVING);
            let query = generate_query(&mut rng, &tables, &data, Bounds::V1_HAVING);
            let table = table_of(&tables, &query.from);

            let Some(Expr::Binary { left, .. }) = &query.having else {
                continue;
            };
            seen += 1;
            assert!(
                numeric_aggregate(left, table),
                "seed {seed}: HAVING compares a non-numeric aggregate {left:?}"
            );
        }
        assert!(
            seen > 100,
            "only {seen} HAVING clauses generated in 3000 cases"
        );
    }

    /// Walk an expression, checking that every binary operator sees compatible operand
    /// types. Returns the expression's own type where it has one.
    fn assert_types_agree(expression: &Expr, table: &Table, seed: u64) -> Option<SqlType> {
        match expression {
            Expr::Literal(literal) => literal.sql_type(),
            Expr::Column(reference) => table
                .column(&reference.column)
                .map(|(_, column)| column.sql_type),
            Expr::Unary { operand, op } => {
                let inner = assert_types_agree(operand, table, seed);
                match op {
                    // A truth value; v1 has no BOOLEAN type to report, so `None` stands for
                    // "not a stored value type".
                    UnaryOp::Not | UnaryOp::IsNull | UnaryOp::IsNotNull => None,
                    UnaryOp::Negate => inner,
                }
            }
            Expr::Cast { expr, to } => {
                assert_types_agree(expr, table, seed);
                Some(*to)
            }
            // A subquery has its own scope, so this walk — which checks types against one
            // table — cannot say anything about its insides. It returns the *shape* of the
            // subquery's result and stops, which is the honest boundary.
            Expr::Exists { .. } => None,
            // Both carry a left operand in *this* scope and a query in its own, so the walk
            // checks the operand and stops at the subquery boundary. `InSubquery` yields a
            // truth value, so `None` is right for it in the same way it is for `EXISTS`.
            Expr::ScalarSubquery { left, .. } | Expr::InSubquery { left, .. } => {
                assert_types_agree(left, table, seed);
                None
            }
            // The second arm that **asserts**: every branch of a `CASE` must yield the same
            // type, or the engines disagree about the result column's type — the same
            // text-versus-integer hazard the `HAVING` axis paid 825 spurious findings for.
            Expr::Case {
                branches,
                otherwise,
            } => {
                let mut result_type = None;
                for (when, then) in branches {
                    assert_types_agree(when, table, seed);
                    let branch = assert_types_agree(then, table, seed);
                    match (result_type, branch) {
                        (None, found) => result_type = found,
                        // `NULL` fits any branch, so a `None` here is not a mismatch.
                        (Some(_), None) => {}
                        (Some(expected), Some(found)) => assert!(
                            expected.accepts(found),
                            "seed {seed}: CASE mixes {expected:?} and {found:?}"
                        ),
                    }
                }
                if let Some(otherwise) = otherwise
                    && let (Some(expected), Some(found)) =
                        (result_type, assert_types_agree(otherwise, table, seed))
                {
                    assert!(
                        expected.accepts(found),
                        "seed {seed}: CASE ELSE is {found:?}, branches are {expected:?}"
                    );
                }
                result_type
            }
            // The one arm here that **asserts** rather than merely descends: a literal list is
            // the easiest place to accidentally compare text against an integer, which is a
            // documented engine difference this subset keeps unrepresentable.
            Expr::InList { left, list, .. } => {
                let operand = assert_types_agree(left, table, seed);
                assert!(!list.is_empty(), "seed {seed}: `IN ()` is a syntax error");
                if let Some(operand) = operand {
                    for literal in list {
                        // `NULL` has no type and fits anywhere, which is the whole point of it.
                        if let Some(literal_type) = literal.sql_type() {
                            assert!(
                                operand.accepts(literal_type),
                                "seed {seed}: {operand:?} list contains {literal_type:?}"
                            );
                        }
                    }
                }
                None
            }
            Expr::Aggregate { func, arg } => {
                if let Some(inner) = arg {
                    assert_types_agree(inner, table, seed);
                }
                match func {
                    // A count is a number whatever it counted.
                    AggregateFunc::CountRows | AggregateFunc::Count => Some(SqlType::BigInt),
                    // MIN/MAX/SUM carry their argument's type — and SUM is only ever
                    // generated over an integer column.
                    _ => arg
                        .as_ref()
                        .and_then(|inner| assert_types_agree(inner, table, seed)),
                }
            }
            Expr::Binary { op, left, right } => {
                let left_type = assert_types_agree(left, table, seed);
                let right_type = assert_types_agree(right, table, seed);

                if let (Some(left_type), Some(right_type)) = (left_type, right_type) {
                    assert!(
                        left_type.accepts(right_type),
                        "seed {seed}: {op:?} mixes {left_type:?} with {right_type:?}"
                    );
                }

                if op.is_predicate() { None } else { left_type }
            }
        }
    }

    #[test]
    fn a_projection_is_never_empty() {
        for seed in 0..300 {
            assert!(!generated(seed).2.projection.is_empty(), "seed {seed}");
        }
    }

    #[test]
    fn expressions_stay_within_the_depth_bound() {
        // Depth bounds size, and size bounds both runtime and how painful minimization is.
        // The bound is on depth, so the node count is bounded exponentially — stated here
        // as an explicit ceiling rather than left as a surprise.
        let ceiling = 2usize.pow(Bounds::V1.max_expr_depth as u32 + 2);

        for seed in 0..500 {
            let (_, _, query) = generated(seed);
            if let Some(filter) = &query.filter {
                assert!(
                    filter.node_count() <= ceiling,
                    "seed {seed}: {} nodes exceeds the ceiling of {ceiling}",
                    filter.node_count()
                );
            }
        }
    }
}
