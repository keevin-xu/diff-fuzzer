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
    AggregateFunc, BinaryOp, ColumnRef, Direction, Expr, InsertRows, OrderKey, SelectStmt,
    SetBranch, SetOp, SqlType, Table, UnaryOp,
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
            let projection = (0..count).map(|_| generate_aggregate(rng, table)).collect();
            (projection, Vec::new())
        }
        QueryShape::Grouped => {
            let key = &table.columns[rng.random_range(0..table.columns.len())];
            let key_ref = reference(table, &key.name);
            let mut projection = vec![Expr::Column(key_ref.clone())];
            for _ in 0..rng.random_range(1..=2) {
                projection.push(generate_aggregate(rng, table));
            }
            (projection, vec![key_ref])
        }
    };

    let filter = (rng.random_range(0..100) < 70).then(|| generate_predicate(rng, table, bounds, 0));

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
        QueryShape::Rows if wants_set_op => Vec::new(),
        QueryShape::Rows => generate_order_by(rng, table),
        QueryShape::WholeTableAggregate | QueryShape::Grouped => Vec::new(),
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
                    projection: projection.clone(),
                    from: table.name.clone(),
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
                projection: projection.clone(),
                from: table.name.clone(),
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

    SelectStmt {
        projection,
        from: table.name.clone(),
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
fn generate_aggregate(rng: &mut SeededRng, table: &Table) -> Expr {
    let column = &table.columns[rng.random_range(0..table.columns.len())];
    let column_ref = Expr::Column(reference(table, &column.name));

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
            if rng.random_range(0..100) < 70 {
                Expr::Column(reference(table, &column.name))
            } else {
                generate_scalar(rng, table, column.sql_type, bounds, 0)
            }
        })
        .collect()
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
    #[test]
    fn comparisons_never_mix_text_with_integers() {
        for seed in 0..500 {
            let (tables, _, query) = generated(seed);
            let table = table_of(&tables, &query.from);

            if let Some(filter) = &query.filter {
                assert_types_agree(filter, table, seed);
            }
            for expression in &query.projection {
                assert_types_agree(expression, table, seed);
            }
        }
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
