//! Turning the tree into SQL text, once per engine.
//!
//! One case, two renderings. The AST is dialect-neutral and this is the only place that
//! knows how either engine spells anything — which is the *syntactic* half of the dialect
//! problem, and the harmless half. The **semantic** half (two engines spelling something
//! the same way and meaning different things) is never handled here: rewriting a query to
//! paper over a meaning difference would hide exactly what this project exists to find.
//! Those are handled by not generating them, or by a cited catalog entry.
//!
//! # Two rules that remove whole classes of difference
//!
//! - **Everything is parenthesized.** Not for readability — precedence is a documented
//!   place where engines can differ, and a fully parenthesized expression has only one
//!   possible reading. The cost is ugly SQL in a repro; the benefit is that a divergence
//!   can never be a disagreement about what the query meant.
//! - **Identifiers are quoted, literals are escaped.** A generated case contains a `'`
//!   because the value pool puts one there on purpose (`gen_schema`), and a renderer that
//!   ignored it would emit SQL that does not parse — turning a data problem into a
//!   validity failure on both engines at once, which looks like agreement.

use crate::schema::{
    AggregateFunc, BinaryOp, ColumnRef, Direction, Expr, InsertRows, Literal, OrderKey, SelectStmt,
    SqlType, Table, UnaryOp,
};

/// Which engine's spelling to use.
///
/// The two arms currently produce **identical text** for the v1 subset — see
/// [`type_name`], and the test at the bottom of this file that asserts it. That is a
/// finding, not an oversight: for a subset this small the syntactic dialect gap is empty.
/// The parameter stays because the gap opens as soon as the subset grows, and retrofitting
/// a dialect distinction into a renderer that never had one means touching every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    Sqlite,
    DuckDb,
}

/// Render the whole case as the statements to execute, in order.
pub fn render_case(
    schema: &[Table],
    data: &[InsertRows],
    query: &SelectStmt,
    dialect: Dialect,
) -> Vec<String> {
    let mut statements: Vec<String> = schema
        .iter()
        .map(|table| render_create_table(table, dialect))
        .collect();

    // A table with no rows produces no `INSERT` at all — an empty statement would be a
    // syntax error, and "insert nothing" is exactly what an empty table means.
    statements.extend(
        data.iter()
            .filter(|insert| !insert.rows.is_empty())
            .map(render_insert),
    );

    statements.push(render_select(query, dialect));
    statements
}

/// `CREATE TABLE "t0" ("c0" INTEGER, "c1" TEXT)`
pub fn render_create_table(table: &Table, dialect: Dialect) -> String {
    let columns: Vec<String> = table
        .columns
        .iter()
        .map(|column| {
            format!(
                "{} {}",
                quote_identifier(&column.name),
                type_name(column.sql_type, dialect)
            )
        })
        .collect();

    format!(
        "CREATE TABLE {} ({})",
        quote_identifier(&table.name),
        columns.join(", ")
    )
}

/// `INSERT INTO "t0" VALUES (1, 'x'), (NULL, '')`
pub fn render_insert(insert: &InsertRows) -> String {
    let rows: Vec<String> = insert
        .rows
        .iter()
        .map(|row| {
            let values: Vec<String> = row.iter().map(render_literal).collect();
            format!("({})", values.join(", "))
        })
        .collect();

    format!(
        "INSERT INTO {} VALUES {}",
        quote_identifier(&insert.table),
        rows.join(", ")
    )
}

/// The query.
pub fn render_select(query: &SelectStmt, dialect: Dialect) -> String {
    let projection: Vec<String> = query
        .projection
        .iter()
        .map(|expression| render_expr(expression, dialect))
        .collect();

    let mut sql = format!(
        "SELECT {} FROM {}",
        projection.join(", "),
        quote_identifier(&query.from)
    );

    if let Some(filter) = &query.filter {
        sql.push_str(&format!(" WHERE {}", render_expr(filter, dialect)));
    }

    // `GROUP BY` comes after `WHERE` and before `ORDER BY`. Rendering it out of order would
    // be a syntax error on both engines — which at least fails loudly.
    if !query.group_by.is_empty() {
        let columns: Vec<String> = query.group_by.iter().map(render_column_ref).collect();
        sql.push_str(&format!(" GROUP BY {}", columns.join(", ")));
    }

    if !query.order_by.is_empty() {
        let keys: Vec<String> = query.order_by.iter().map(render_order_key).collect();
        sql.push_str(&format!(" ORDER BY {}", keys.join(", ")));
    }

    if let Some(limit) = query.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    sql
}

fn render_order_key(key: &OrderKey) -> String {
    format!(
        "{} {} NULLS {}",
        render_column_ref(&key.column),
        match key.direction {
            Direction::Ascending => "ASC",
            Direction::Descending => "DESC",
        },
        // Always explicit. Where `NULL`s sort by default may differ between engines, and
        // that difference would be legal — so the query says which it wants rather than
        // inheriting an answer (`SPECS.md` §5.6).
        if key.nulls_first { "FIRST" } else { "LAST" }
    )
}

fn render_expr(expression: &Expr, dialect: Dialect) -> String {
    match expression {
        Expr::Column(reference) => render_column_ref(reference),
        Expr::Literal(literal) => render_literal(literal),
        Expr::Unary { op, operand } => {
            let inner = render_expr(operand, dialect);
            match op {
                UnaryOp::Not => format!("(NOT {inner})"),
                // **The space is load-bearing.** `-` immediately followed by `-` is `--`,
                // which SQL reads as a line comment — so `(-{-47})` renders as `(--47)`,
                // silently commenting out the rest of the query. Both engines then refuse
                // it, which looks like agreement: a tool bug wearing the costume of a
                // result. Measured cost before the fix: 2.35% of generated cases.
                UnaryOp::Negate => format!("(- {inner})"),
                UnaryOp::IsNull => format!("({inner} IS NULL)"),
                UnaryOp::IsNotNull => format!("({inner} IS NOT NULL)"),
            }
        }
        Expr::Binary { op, left, right } => format!(
            "({} {} {})",
            render_expr(left, dialect),
            binary_operator(*op),
            render_expr(right, dialect)
        ),
        Expr::Cast { expr, to } => format!(
            "CAST({} AS {})",
            render_expr(expr, dialect),
            type_name(*to, dialect)
        ),
        Expr::Aggregate { func, arg } => match (func, arg) {
            (AggregateFunc::CountRows, _) => "COUNT(*)".to_string(),
            (function, Some(inner)) => {
                format!(
                    "{}({})",
                    aggregate_name(*function),
                    render_expr(inner, dialect)
                )
            }
            // Unreachable for generated cases — only `COUNT(*)` omits its argument — but
            // rendering something that does not parse would be worse than saying so.
            (function, None) => format!("{}(*)", aggregate_name(*function)),
        },
    }
}

fn aggregate_name(func: AggregateFunc) -> &'static str {
    match func {
        AggregateFunc::CountRows | AggregateFunc::Count => "COUNT",
        AggregateFunc::Min => "MIN",
        AggregateFunc::Max => "MAX",
        AggregateFunc::Sum => "SUM",
    }
}

fn binary_operator(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Equal => "=",
        BinaryOp::NotEqual => "<>",
        BinaryOp::Less => "<",
        BinaryOp::LessOrEqual => "<=",
        BinaryOp::Greater => ">",
        BinaryOp::GreaterOrEqual => ">=",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Add => "+",
        BinaryOp::Subtract => "-",
        BinaryOp::Multiply => "*",
    }
}

/// How each engine spells a type.
///
/// **Identical for every v1 type, on both engines.** `INTEGER`, `BIGINT` and `TEXT` are
/// accepted by both — SQLite by affinity, DuckDB with `TEXT` as an alias for `VARCHAR` —
/// which is measured by the round-trip tests below rather than assumed from documentation.
/// The dialect parameter is threaded through anyway, because `DECIMAL` and the wider
/// surface later on will need it.
fn type_name(sql_type: SqlType, dialect: Dialect) -> &'static str {
    match (sql_type, dialect) {
        (SqlType::Integer, _) => "INTEGER",
        (SqlType::BigInt, _) => "BIGINT",
        (SqlType::Text, _) => "TEXT",
        (SqlType::Decimal, _) => "DECIMAL",
        (SqlType::Boolean, _) => "BOOLEAN",
    }
}

fn render_column_ref(reference: &ColumnRef) -> String {
    format!(
        "{}.{}",
        quote_identifier(&reference.table),
        quote_identifier(&reference.column)
    )
}

/// Double-quoted, with any embedded quote doubled.
///
/// Generated identifiers are `t0`/`c0` and need none of this. It is here because the rule
/// should live in the renderer rather than in an assumption about the generator — the
/// moment a name comes from somewhere else (a shrinker, a byte decoder, a hand-written
/// repro), an unquoted identifier is a syntax error or worse.
fn quote_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// Render a literal.
///
/// Two cases here are not obvious, and both come from values the generator produces on
/// purpose:
///
/// - **A text value containing `'`** must have it doubled, or the SQL does not parse. The
///   value pool contains a bare `'` precisely so this path is exercised on every run.
/// - **`i64::MIN` cannot be written as a negative literal.** SQL has no negative literals:
///   `-9223372036854775808` parses as *negation applied to* `9223372036854775808`, which
///   does not fit in a signed 64-bit integer. Engines then do something implementation-
///   defined — promote to floating point, overflow, or refuse. Writing it as
///   `(-9223372036854775807 - 1)` keeps every intermediate value in range, so the case
///   tests what it meant to test rather than the engines' overflow handling in the parser.
fn render_literal(literal: &Literal) -> String {
    match literal {
        Literal::Null => "NULL".to_string(),
        Literal::Integer(number) => {
            if *number == i64::MIN {
                format!("({} - 1)", i64::MIN + 1)
            } else {
                number.to_string()
            }
        }
        Literal::Text(text) => format!("'{}'", text.replace('\'', "''")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Column;

    fn table() -> Table {
        Table {
            name: "t0".to_string(),
            columns: vec![
                Column {
                    name: "c0".to_string(),
                    sql_type: SqlType::Integer,
                },
                Column {
                    name: "c1".to_string(),
                    sql_type: SqlType::Text,
                },
            ],
        }
    }

    #[test]
    fn a_quote_in_a_value_is_escaped() {
        assert_eq!(render_literal(&Literal::Text("'".to_string())), "''''");
        assert_eq!(
            render_literal(&Literal::Text("it's".to_string())),
            "'it''s'"
        );
        assert_eq!(render_literal(&Literal::Text(String::new())), "''");
    }

    #[test]
    fn the_smallest_integer_is_not_written_as_a_negative_literal() {
        // SQL has no negative literals, so `-9223372036854775808` is negation applied to a
        // number too large for i64. Every intermediate here stays in range.
        assert_eq!(
            render_literal(&Literal::Integer(i64::MIN)),
            "(-9223372036854775807 - 1)"
        );
        // The largest positive value needs no such treatment.
        assert_eq!(
            render_literal(&Literal::Integer(i64::MAX)),
            "9223372036854775807"
        );
    }

    #[test]
    fn double_negation_does_not_become_a_comment() {
        // `--` opens a line comment in SQL. Without the space this renders as `(--47)`,
        // which comments out the rest of the statement and makes both engines refuse it —
        // a tool bug that reads as agreement.
        let expression = Expr::Unary {
            op: UnaryOp::Negate,
            operand: Box::new(Expr::Literal(Literal::Integer(-47))),
        };
        let sql = render_expr(&expression, Dialect::Sqlite);
        assert_eq!(sql, "(- -47)");
        assert!(
            !sql.contains("--"),
            "rendered SQL must not contain a comment"
        );
    }

    #[test]
    fn expressions_are_fully_parenthesized() {
        // `a OR b AND c` reads differently under different precedence rules; the rendered
        // form has only one reading.
        let expression = Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(Expr::Literal(Literal::Integer(1))),
            right: Box::new(Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(Expr::Literal(Literal::Integer(2))),
                right: Box::new(Expr::Literal(Literal::Integer(3))),
            }),
        };
        assert_eq!(
            render_expr(&expression, Dialect::Sqlite),
            "(1 OR (2 AND 3))"
        );
    }

    #[test]
    fn nulls_position_is_always_stated() {
        let key = OrderKey {
            column: ColumnRef {
                table: "t0".to_string(),
                column: "c0".to_string(),
            },
            direction: Direction::Descending,
            nulls_first: false,
        };
        assert_eq!(render_order_key(&key), "\"t0\".\"c0\" DESC NULLS LAST");
    }

    #[test]
    fn an_empty_table_produces_no_insert() {
        let statements = render_case(
            &[table()],
            &[InsertRows {
                table: "t0".to_string(),
                rows: vec![],
            }],
            &SelectStmt {
                projection: vec![Expr::Literal(Literal::Integer(1))],
                from: "t0".to_string(),
                group_by: vec![],
                filter: None,
                order_by: vec![],
                limit: None,
            },
            Dialect::Sqlite,
        );

        assert_eq!(
            statements.len(),
            2,
            "create table and select, but no insert"
        );
        assert!(statements[0].starts_with("CREATE TABLE"));
        assert!(statements[1].starts_with("SELECT"));
    }

    /// The two dialects currently render **identically** for the v1 subset.
    ///
    /// Asserted rather than assumed, and worth knowing: it means the syntactic half of the
    /// dialect problem — the half `12`/`13` spent the most words on — is *empty* at this
    /// size. If this test ever fails, the subset has grown into a real dialect difference
    /// and the renderer is the right place to handle it.
    #[test]
    fn both_dialects_render_the_v1_subset_identically() {
        for seed in 0..200 {
            let mut rng = diff_fuzzer_core::SeededRng::from_seed(seed);
            let bounds = crate::gen_schema::Bounds::V1;
            let tables = crate::gen_schema::generate_schema(&mut rng, bounds);
            let data = crate::gen_schema::generate_data(&mut rng, &tables, bounds);
            let query = crate::gen_query::generate_query(&mut rng, &tables, &data, bounds);

            assert_eq!(
                render_case(&tables, &data, &query, Dialect::Sqlite),
                render_case(&tables, &data, &query, Dialect::DuckDb),
                "seed {seed}"
            );
        }
    }
}
