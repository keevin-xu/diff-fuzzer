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
    BinaryOp, ColumnRef, Direction, Expr, InsertRows, OrderKey, SelectStmt, SqlType, Table, UnaryOp,
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

    let projection = generate_projection(rng, table, bounds);
    let filter = (rng.random_range(0..100) < 70).then(|| generate_predicate(rng, table, bounds, 0));
    let order_by = generate_order_by(rng, table);

    // **The rule that needs the data.** A `LIMIT` on a query whose order is not total lets
    // two engines return different *rows*, both legally — a difference no normalization can
    // repair and no catalog entry could honestly excuse. So the limit is only offered when
    // the order has been shown to be total for this case's rows.
    let limit = if orders_rows_totally(&order_by, table, rows) && rng.random_range(0..100) < 30 {
        Some(rng.random_range(0..=rows.len() as u32))
    } else {
        None
    };

    SelectStmt {
        projection,
        from: table.name.clone(),
        filter,
        order_by,
        limit,
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
                // Arithmetic, which brings overflow with it — `i64::MAX + 1` is in reach
                // because the value pool contains `i64::MAX` deliberately. The engines may
                // well disagree about overflow; that is a *candidate finding*, not a
                // validity problem, and S4 decides whether it is legal.
                Expr::Binary {
                    op: match rng.random_range(0..3) {
                        0 => BinaryOp::Add,
                        1 => BinaryOp::Subtract,
                        _ => BinaryOp::Multiply,
                    },
                    left: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                    right: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                }
            } else if choice < 92 {
                Expr::Unary {
                    op: UnaryOp::Negate,
                    operand: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                }
            } else {
                // A cast, but only between the integer widths — a widening that cannot
                // fail. `CAST(text AS INTEGER)` is exactly where the engines are documented
                // to differ (`SPECS.md` §5.5, unretrieved), so it is not generated.
                Expr::Cast {
                    expr: Box::new(generate_scalar(rng, table, sql_type, bounds, depth + 1)),
                    to: if rng.random_range(0..2) == 0 {
                        SqlType::Integer
                    } else {
                        SqlType::BigInt
                    },
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
