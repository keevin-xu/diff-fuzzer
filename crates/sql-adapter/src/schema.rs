//! The typed tree a case is made of: types, tables, values, expressions, and one query.
//!
//! These are declarations only. Building them is the generator's job (S2.2–S2.3), turning
//! them back into SQL text is the renderer's (S2.4), and [`crate::ast::SqlCase`] switches
//! over from strings to this tree at that point — not before, because until a renderer
//! exists there would be nothing for the engines to execute.
//!
//! # What the type system is doing here
//!
//! Several of this domain's rules are enforced by *what cannot be written down* rather than
//! by a check that runs later. That is the strongest form of correct-by-construction: a
//! generator cannot emit what the AST cannot represent, so the rule cannot be broken by an
//! oversight in generation, in shrinking, or in a future decoder.
//!
//! - **There is no floating-point type.** Not "we avoid it" — [`SqlType`] has no variant
//!   for it, so no column, literal, or cast can be one.
//! - **There is no division operator.** [`BinaryOp`] omits it, so the "never emit a bare
//!   `/`" rule is not a rule anything has to remember. (SQLite's `/` is integer division
//!   and DuckDB's is not — `SPECS.md` §5.3, still unretrieved, which is exactly why the
//!   safe move is to make it unrepresentable rather than to catalog it.)
//! - **A row's values are literals, never expressions**, so seed data cannot smuggle in
//!   computation that would have to be evaluated to know what was stored.

use serde::{Deserialize, Serialize};

/// A column type.
///
/// Five variants, of which the generator emits **three**. `Decimal` and `Boolean` are
/// defined so the AST stays stable when they are switched on, but they are not generated:
/// SQLite has no native form of either — booleans are `0`/`1` integers and a `DECIMAL`
/// column lands on IEEE 754 binary64 by affinity — while DuckDB has both natively, so every
/// such cell would differ for reasons of representation rather than correctness
/// (`SPECS.md` §4.1–4.2, `POLICY.md` §3).
///
/// **No floating-point variant exists at all**, and none should be added: the whole point
/// of the v1 subset is that comparison stays discrete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SqlType {
    /// 32-bit-declared integer. Both engines widen it on the way out; we read `i64`.
    Integer,
    /// 64-bit integer.
    BigInt,
    /// Text, compared with binary collation.
    Text,
    /// Defined, **not generated** — see the type-level note above.
    Decimal,
    /// Defined, **not generated**.
    Boolean,
}

impl SqlType {
    /// The types the v1 generator is allowed to produce.
    ///
    /// A function rather than a comment, so the generator and its tests read the same list
    /// and cannot drift apart.
    pub const GENERATED: [SqlType; 3] = [SqlType::Integer, SqlType::BigInt, SqlType::Text];

    /// Is this type one the generator may emit?
    pub fn is_generated(self) -> bool {
        Self::GENERATED.contains(&self)
    }
}

/// A literal value: what appears in seed rows and in expressions.
///
/// Mirrors [`crate::outcome::Cell`] deliberately — the values we put in are the kinds of
/// value we expect back — but stays a separate type, because one describes a case and the
/// other describes a result. Merging them would tie the shape of what we generate to the
/// shape of what engines return, and those change for different reasons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Literal {
    /// `NULL`, which is where a great many engine disagreements live.
    Null,
    Integer(i64),
    Text(String),
}

impl Literal {
    /// The type this literal can be stored in, or `None` for `NULL`, which fits anywhere.
    ///
    /// `Integer` reports [`SqlType::Integer`], but a `BigInt` column accepts it too — see
    /// [`SqlType::accepts`].
    pub fn sql_type(&self) -> Option<SqlType> {
        match self {
            Literal::Null => None,
            Literal::Integer(_) => Some(SqlType::Integer),
            Literal::Text(_) => Some(SqlType::Text),
        }
    }
}

impl SqlType {
    /// May a value of type `other` be stored in, or compared against, this type?
    ///
    /// The one place implicit conversion is allowed, and it is allowed because it is not
    /// implicit *conversion*: an integer literal is the same value whether the column is
    /// declared `INTEGER` or `BIGINT`. Text against integer is refused, which is what keeps
    /// `1 = '1'` — where the two engines differ — out of every generated case.
    pub fn accepts(self, other: SqlType) -> bool {
        matches!(
            (self, other),
            (
                SqlType::Integer | SqlType::BigInt,
                SqlType::Integer | SqlType::BigInt
            ) | (SqlType::Text, SqlType::Text)
                | (SqlType::Decimal, SqlType::Decimal)
                | (SqlType::Boolean, SqlType::Boolean)
        )
    }
}

/// One column of a table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub sql_type: SqlType,
}

/// One table: a name and its columns.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Table {
    pub name: String,
    pub columns: Vec<Column>,
}

impl Table {
    /// Find a column by name, with its position.
    ///
    /// The position matters because a row is a positional list of values: losing the
    /// correspondence between column order and value order is a silent way to store the
    /// right data in the wrong places.
    pub fn column(&self, name: &str) -> Option<(usize, &Column)> {
        self.columns
            .iter()
            .enumerate()
            .find(|(_, column)| column.name == name)
    }
}

/// Rows to insert into one table.
///
/// Values are positional and must line up with the table's columns — checked by
/// [`crate::ast::SqlCase::validate`] once the case type switches over, rather than trusted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InsertRows {
    pub table: String,
    pub rows: Vec<Vec<Literal>>,
}

/// A reference to a column of a table in scope.
///
/// Carries the table name as well as the column, even though v1 queries read from a single
/// table. Two tables arrive with joins, and a reference that only names the column would
/// become ambiguous exactly when ambiguity starts to matter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnRef {
    pub table: String,
    pub column: String,
}

/// Binary operators.
///
/// **No division.** SQLite's `/` performs integer division where DuckDB's does not, which
/// makes it a legal cross-engine difference we would then have to catalog — and the
/// evidence for that difference has not been retrieved (`SPECS.md` §5.3). Omitting the
/// operator entirely costs a little coverage and removes the question.
///
/// Also no modulo, for the same reason plus the sign-of-negatives question, and no string
/// concatenation, whose `NULL` handling differs between dialects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    And,
    Or,
    Add,
    Subtract,
    Multiply,
}

impl BinaryOp {
    /// Does this operator produce a truth value (rather than a number)?
    pub fn is_predicate(self) -> bool {
        matches!(
            self,
            BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::Less
                | BinaryOp::LessOrEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterOrEqual
                | BinaryOp::And
                | BinaryOp::Or
        )
    }
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Logical negation.
    Not,
    /// Arithmetic negation.
    Negate,
    /// `IS NULL` — spelled as an operator because that is what it is, and because it is one
    /// of the few ways to ask about `NULL` without falling into three-valued logic.
    IsNull,
    /// `IS NOT NULL`.
    IsNotNull,
}

/// An aggregate function.
///
/// Deliberately excludes `AVG`: it returns `REAL` on SQLite and `DOUBLE` on DuckDB
/// (measured), and floating point is the one thing this subset exists to keep out — an
/// average would reintroduce it under a name that does not look numeric-fragile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregateFunc {
    /// `COUNT(*)` — counts rows, including those that are entirely `NULL`.
    CountRows,
    /// `COUNT(expr)` — counts non-`NULL` values, which is a different question and a
    /// classic place for engines to differ.
    Count,
    Min,
    Max,
    /// `SUM(expr)`. Restricted at generation to 32-bit columns: DuckDB widens a sum to
    /// `HUGEINT` while SQLite keeps it in an integer until it overflows into `REAL`, so an
    /// unrestricted sum reproduces the overflow difference rather than testing aggregation.
    Sum,
}

impl AggregateFunc {
    /// Does this aggregate take an argument? `COUNT(*)` does not.
    pub fn takes_argument(self) -> bool {
        self != AggregateFunc::CountRows
    }
}

/// An expression.
///
/// Recursive, and therefore boxed: a `Box<Expr>` is a pointer to another expression, which
/// is what gives the type a finite size. Without it the compiler cannot lay out a value
/// that may contain another value of the same type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    Column(ColumnRef),
    Literal(Literal),
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// An explicit `CAST`. Explicit because the alternative — letting the engines coerce
    /// implicitly — is a documented place where they differ.
    Cast {
        expr: Box<Expr>,
        to: SqlType,
    },
    /// An aggregate over the rows of a group (or of the whole table, with no `GROUP BY`).
    Aggregate {
        func: AggregateFunc,
        /// `None` only for `COUNT(*)`.
        arg: Option<Box<Expr>>,
    },
}

impl Expr {
    /// How many nodes this expression contains, itself included.
    ///
    /// Used by minimization to know that a reduction actually reduced something, and by the
    /// generator to bound how deep it may build.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Column(_) | Expr::Literal(_) => 1,
            Expr::Unary { operand, .. } => 1 + operand.node_count(),
            Expr::Binary { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Expr::Cast { expr, .. } => 1 + expr.node_count(),
            Expr::Aggregate { arg, .. } => 1 + arg.as_ref().map_or(0, |inner| inner.node_count()),
        }
    }

    /// Every column this expression reads.
    ///
    /// Name resolution checks this against what is in scope: a query referring to a column
    /// that does not exist is invalid, and an invalid case tests the parser rather than the
    /// engine.
    pub fn columns(&self) -> Vec<&ColumnRef> {
        let mut found = Vec::new();
        self.collect_columns(&mut found);
        found
    }

    fn collect_columns<'a>(&'a self, found: &mut Vec<&'a ColumnRef>) {
        match self {
            Expr::Column(reference) => found.push(reference),
            Expr::Literal(_) => {}
            Expr::Unary { operand, .. } => operand.collect_columns(found),
            Expr::Binary { left, right, .. } => {
                left.collect_columns(found);
                right.collect_columns(found);
            }
            Expr::Cast { expr, .. } => expr.collect_columns(found),
            Expr::Aggregate { arg, .. } => {
                if let Some(inner) = arg {
                    inner.collect_columns(found);
                }
            }
        }
    }

    /// Is this expression an aggregate, or does it contain one?
    ///
    /// A projection that aggregates behaves completely differently from one that does not:
    /// it collapses rows. The generator and the validity rules both need to ask.
    pub fn contains_aggregate(&self) -> bool {
        match self {
            Expr::Aggregate { .. } => true,
            Expr::Column(_) | Expr::Literal(_) => false,
            Expr::Unary { operand, .. } => operand.contains_aggregate(),
            Expr::Binary { left, right, .. } => {
                left.contains_aggregate() || right.contains_aggregate()
            }
            Expr::Cast { expr, .. } => expr.contains_aggregate(),
        }
    }

    /// Does this expression mention a `NULL` literal anywhere?
    ///
    /// A small thing now; the predicate vocabulary of S7 is built from questions like this,
    /// and they are cheap on a tree and awkward on text.
    pub fn contains_null_literal(&self) -> bool {
        match self {
            Expr::Literal(Literal::Null) => true,
            Expr::Literal(_) | Expr::Column(_) => false,
            Expr::Unary { operand, .. } => operand.contains_null_literal(),
            Expr::Binary { left, right, .. } => {
                left.contains_null_literal() || right.contains_null_literal()
            }
            Expr::Cast { expr, .. } => expr.contains_null_literal(),
            Expr::Aggregate { arg, .. } => arg
                .as_ref()
                .is_some_and(|inner| inner.contains_null_literal()),
        }
    }
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Ascending,
    Descending,
}

/// One `ORDER BY` key.
///
/// A column rather than an arbitrary expression, deliberately. Ordering by an expression is
/// legal SQL, but whether an order is *total* has to be decidable from the case, and that
/// is far easier to establish for a column whose values are sitting in the seed data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OrderKey {
    pub column: ColumnRef,
    pub direction: Direction,
    /// Where `NULL`s sort. Always stated explicitly, never left to the engine's default,
    /// because the defaults may differ and the difference would be legal
    /// (`SPECS.md` §5.6 — unretrieved, so the safe move is to not depend on it).
    pub nulls_first: bool,
}

/// The query under test.
///
/// One `SELECT` over one table in v1. Joins, grouping and subqueries are where DuckDB's
/// bugs are most likely to be, and they come after the pipeline is trustworthy — the fields
/// are not present yet because an unused field is an invitation to fill it in early.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectStmt {
    /// What to return. Never empty — `SELECT` with nothing to select is not a case.
    pub projection: Vec<Expr>,
    /// The single table read from.
    pub from: String,
    /// `WHERE`, if any.
    pub filter: Option<Expr>,
    /// `GROUP BY` columns. Empty means no grouping.
    ///
    /// When non-empty, every projected expression must be either one of these columns or an
    /// aggregate — SQLite permits looser forms and DuckDB refuses them, so the strict rule is
    /// the one that both accept.
    pub group_by: Vec<ColumnRef>,
    /// `ORDER BY` keys, in order. Empty means the engine may return rows in any order.
    pub order_by: Vec<OrderKey>,
    /// `LIMIT`, if any.
    ///
    /// **Only meaningful alongside an `ORDER BY` that totally orders the rows.** Without
    /// one, `LIMIT n` lets two engines legally return *different rows* rather than the same
    /// rows in a different order — which sorting cannot repair and no catalog entry could
    /// honestly excuse. The generator enforces the pairing; `SqlCase::validate` will check
    /// it, so a hand-built or shrunk case cannot slip past.
    pub limit: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_ref(column: &str) -> ColumnRef {
        ColumnRef {
            table: "t".to_string(),
            column: column.to_string(),
        }
    }

    #[test]
    fn only_three_types_are_generated() {
        assert!(SqlType::Integer.is_generated());
        assert!(SqlType::BigInt.is_generated());
        assert!(SqlType::Text.is_generated());
        // Defined so the AST stays stable, but not produced — the retrieved SQLite/DuckDB
        // difference in `SPECS.md` §4.1–4.2 is what keeps them out.
        assert!(!SqlType::Decimal.is_generated());
        assert!(!SqlType::Boolean.is_generated());
    }

    #[test]
    fn integer_widths_are_compatible_but_text_and_integer_are_not() {
        assert!(SqlType::Integer.accepts(SqlType::BigInt));
        assert!(SqlType::BigInt.accepts(SqlType::Integer));
        assert!(SqlType::Text.accepts(SqlType::Text));
        // The rule that keeps `1 = '1'` — where the engines differ — unrepresentable.
        assert!(!SqlType::Integer.accepts(SqlType::Text));
        assert!(!SqlType::Text.accepts(SqlType::Integer));
    }

    #[test]
    fn a_null_literal_has_no_type_and_fits_anywhere() {
        assert_eq!(Literal::Null.sql_type(), None);
        assert_eq!(Literal::Integer(1).sql_type(), Some(SqlType::Integer));
        assert_eq!(
            Literal::Text("x".to_string()).sql_type(),
            Some(SqlType::Text)
        );
    }

    #[test]
    fn node_count_counts_every_node() {
        let leaf = Expr::Literal(Literal::Integer(1));
        assert_eq!(leaf.node_count(), 1);

        // (a > 1) AND (b IS NULL) — 1 for the AND, 3 for the comparison, 2 for IS NULL.
        let expression = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Greater,
                left: Box::new(Expr::Column(column_ref("a"))),
                right: Box::new(Expr::Literal(Literal::Integer(1))),
            }),
            right: Box::new(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(Expr::Column(column_ref("b"))),
            }),
        };
        assert_eq!(expression.node_count(), 6);
    }

    #[test]
    fn columns_are_found_at_every_depth() {
        let expression = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(Expr::Cast {
                expr: Box::new(Expr::Column(column_ref("a"))),
                to: SqlType::BigInt,
            }),
            right: Box::new(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(Expr::Column(column_ref("b"))),
            }),
        };

        let names: Vec<&str> = expression
            .columns()
            .iter()
            .map(|reference| reference.column.as_str())
            .collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn a_null_literal_is_found_wherever_it_hides() {
        let buried = Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(Expr::Binary {
                op: BinaryOp::Equal,
                left: Box::new(Expr::Column(column_ref("a"))),
                right: Box::new(Expr::Literal(Literal::Null)),
            }),
        };
        assert!(buried.contains_null_literal());

        let without = Expr::Column(column_ref("a"));
        assert!(!without.contains_null_literal());
    }

    #[test]
    fn predicates_are_distinguished_from_arithmetic() {
        assert!(BinaryOp::Equal.is_predicate());
        assert!(BinaryOp::And.is_predicate());
        assert!(!BinaryOp::Add.is_predicate());
        assert!(!BinaryOp::Multiply.is_predicate());
    }

    #[test]
    fn a_column_is_found_with_its_position() {
        let table = Table {
            name: "t".to_string(),
            columns: vec![
                Column {
                    name: "a".to_string(),
                    sql_type: SqlType::Integer,
                },
                Column {
                    name: "b".to_string(),
                    sql_type: SqlType::Text,
                },
            ],
        };

        let (index, column) = table.column("b").expect("b exists");
        assert_eq!(index, 1);
        assert_eq!(column.sql_type, SqlType::Text);
        assert!(table.column("nope").is_none());
    }

    #[test]
    fn the_tree_survives_a_round_trip_through_json() {
        let statement = SelectStmt {
            projection: vec![Expr::Column(column_ref("a"))],
            from: "t".to_string(),
            group_by: vec![],
            filter: Some(Expr::Binary {
                op: BinaryOp::Greater,
                left: Box::new(Expr::Column(column_ref("a"))),
                right: Box::new(Expr::Literal(Literal::Integer(0))),
            }),
            order_by: vec![OrderKey {
                column: column_ref("a"),
                direction: Direction::Ascending,
                nulls_first: true,
            }],
            limit: None,
        };

        let json = serde_json::to_string(&statement).expect("serializes");
        let back: SelectStmt = serde_json::from_str(&json).expect("deserializes");
        assert_eq!(statement, back);
    }
}
