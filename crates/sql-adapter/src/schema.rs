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

/// A secondary index on one or more columns of a table.
///
/// # Why this exists, and it is not "another construct"
///
/// Every axis before this one tested **evaluation** — what a query computes. An index tests
/// **plan choice**: whether the engine reaches the answer by scanning or by seeking, and that is
/// the one subsystem where SQLite and DuckDB are architecturally unalike.
///
/// **The asymmetry is expected even on eight rows.** SQLite without `ANALYZE` assumes a table
/// holds roughly a million rows whatever its real size (`SPECS.md` §5.10 — the exact constant
/// was not retrievable, but that the fallback is a constant independent of our data is what
/// matters), so it will use an index it can use. DuckDB gathers real statistics and on eight
/// rows will likely decline. **Two engines taking different plans for the same query is exactly
/// the condition a differential oracle exists for**, and nothing in this project has produced it
/// before.
///
/// It also supplies the strongest oracle here: **adding an index must not change a query's
/// results.** If it does, that is a bug with no interpretation required — no tolerance, no legal
/// difference, no argument about which engine is right.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Index {
    pub name: String,
    pub table: String,
    /// Column names, in index order. Order matters to the planner: an index on `(a, b)` can
    /// serve a lookup on `a` and cannot serve one on `b` alone.
    pub columns: Vec<String>,
    /// **Not generated in v1.** A `UNIQUE` index fails outright when the data has duplicates,
    /// which would surface as a setup failure and a skipped case rather than a finding. Its
    /// `NULL` rule is also a genuine divergence candidate (SQLite permits several `NULL`s in a
    /// unique index) and deserves its own axis with a duplicate check, not a flag here.
    pub unique: bool,
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
    /// `EXISTS (SELECT ...)` — a truth value.
    ///
    /// The inner query may reference the **outer** row, which is what makes it *correlated*:
    /// it is re-evaluated per outer row rather than once. That is where the interesting bugs
    /// live, and it is also what makes name resolution two-scoped and shrinking dangerous —
    /// dropping an outer column can orphan a reference buried inside the subquery.
    Exists {
        not: bool,
        query: Box<SelectStmt>,
    },
    /// `expr <op> (SELECT ...)` — a comparison against a **scalar** subquery.
    ///
    /// The inner query must return one column. It may return **no rows**, in which case SQL
    /// says the scalar is `NULL` and the comparison is unknown — a case worth generating
    /// deliberately, since it is where "no rows" and "a `NULL` value" become the same thing.
    ScalarSubquery {
        op: BinaryOp,
        left: Box<Expr>,
        query: Box<SelectStmt>,
    },
    /// `expr IN (SELECT ...)` / `expr NOT IN (SELECT ...)`.
    ///
    /// **This variant exists for one specific bug, and it is worth stating exactly.** `NOT IN`
    /// against a subquery whose column contains a `NULL` returns **UNKNOWN for every row**, so
    /// `WHERE x NOT IN (SELECT y FROM t)` correctly returns **no rows at all** when any `y` is
    /// `NULL` — even for an `x` that plainly does not appear in `y`.
    ///
    /// The reason: `x NOT IN (a, b, NULL)` means `x <> a AND x <> b AND x <> NULL`, and the last
    /// conjunct is UNKNOWN. `false AND unknown` is false, but `true AND unknown` is **unknown**,
    /// so a row that differs from every non-`NULL` value cannot reach TRUE. `IN` is not
    /// symmetric here: `x IN (a, b, NULL)` still returns TRUE when `x = a`, because
    /// `true OR unknown` is true. The asymmetry is the whole trap.
    ///
    /// It is the most famous three-valued-logic trap in SQL, engines get it wrong *in the same
    /// direction* (treating the UNKNOWN as FALSE and returning rows that should be excluded),
    /// and a shared wrong answer is invisible to a differential oracle — which is precisely
    /// what the metamorphic oracle exists to reach.
    InSubquery {
        not: bool,
        left: Box<Expr>,
        query: Box<SelectStmt>,
    },
    /// `expr IN (1, 2, NULL)` / `expr NOT IN (1, 2, NULL)` — a **literal** list.
    ///
    /// The same three-valued-logic trap as [`Expr::InSubquery`], reached by a different route
    /// **and that is the entire reason it exists as a separate variant.** A subquery must be
    /// executed; a literal list can be **constant-folded** at plan time, so an engine may
    /// evaluate this through completely different code — rewriting it to a chain of `OR`s, to a
    /// hash probe, or to a precomputed set. Each rewrite has to preserve `NULL` semantics
    /// independently, and each is somewhere the same mistake can be made again.
    ///
    /// The list holds [`Literal`]s rather than [`Expr`]s deliberately: an expression list would
    /// defeat constant folding and so would test the path this variant was added to reach.
    /// `CASE WHEN c1 THEN v1 WHEN c2 THEN v2 ELSE v3 END`.
    ///
    /// **Two `NULL` traps in one construct, and they are different from each other.**
    ///
    /// 1. **An omitted `ELSE` yields `NULL`.** A row matching no branch does not error and does
    ///    not return the last value — it returns `NULL`. The absence of a clause produces a
    ///    value, which is unusual enough that an engine can get it wrong by simply not
    ///    implementing the case.
    /// 2. **An UNKNOWN condition is not taken.** `WHEN NULL THEN x` behaves as `WHEN FALSE`, so
    ///    a branch whose condition involves a `NULL` falls through. Combined with (1), a row can
    ///    silently reach `NULL` through two independent routes.
    ///
    /// All branch results carry the **same type**, chosen once at generation. Mixing them would
    /// reintroduce the text-versus-integer comparison the subset keeps unrepresentable — the
    /// mistake the `HAVING` axis made at S9.5 and paid 825 spurious findings for.
    Case {
        /// `(condition, value)` pairs, in order. Never empty — `CASE END` is not valid SQL.
        branches: Vec<(Expr, Expr)>,
        /// The `ELSE`. `None` is the interesting case: it means `NULL` for unmatched rows.
        otherwise: Option<Box<Expr>>,
    },
    InList {
        not: bool,
        left: Box<Expr>,
        /// Two or more literals, at the left operand's type. May contain `NULL` — and usually
        /// does, since without one the negated form is perfectly well-behaved.
        list: Vec<Literal>,
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
            // A subquery counts as its own size plus its query's, so minimization sees
            // removing one as the large reduction it is.
            Expr::Exists { query, .. } => 1 + query.node_count(),
            Expr::ScalarSubquery { left, query, .. } | Expr::InSubquery { left, query, .. } => {
                1 + left.node_count() + query.node_count()
            }
            // Each literal counts, so minimization can see that shortening the list is a
            // genuine reduction — which it is, and often the one that isolates the `NULL`.
            Expr::InList { left, list, .. } => 1 + left.node_count() + list.len(),
            // Every condition and value counts, so removing a branch reads as a real reduction
            // to minimization — which it is, and often the one that isolates the `NULL` route.
            Expr::Case {
                branches,
                otherwise,
            } => {
                1 + branches
                    .iter()
                    .map(|(when, then)| when.node_count() + then.node_count())
                    .sum::<usize>()
                    + otherwise.as_ref().map_or(0, |e| e.node_count())
            }
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
            // Deliberately **does** descend into subqueries: the whole point of a correlated
            // one is that it references the outer scope, and a caller checking which columns a
            // query needs must see those references or it will happily delete them.
            Expr::Exists { query, .. } => query.collect_columns(found),
            Expr::ScalarSubquery { left, query, .. } | Expr::InSubquery { left, query, .. } => {
                left.collect_columns(found);
                query.collect_columns(found);
            }
            // The list is literals, so only the left operand can reference a column.
            Expr::InList { left, .. } => left.collect_columns(found),
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (when, then) in branches {
                    when.collect_columns(found);
                    then.collect_columns(found);
                }
                if let Some(otherwise) = otherwise {
                    otherwise.collect_columns(found);
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
            // An aggregate *inside* a subquery belongs to the subquery, not to this query —
            // `SELECT c0 FROM t WHERE c0 = (SELECT MAX(c1) FROM u)` is not an aggregate query.
            // Reporting otherwise would make the grouping rules reject a valid case.
            Expr::Exists { .. } | Expr::ScalarSubquery { .. } | Expr::InSubquery { .. } => false,
            // Unlike the subquery forms, there is no inner scope here — an aggregate in the
            // left operand would belong to *this* query, so it is reported rather than hidden.
            Expr::InList { left, .. } => left.contains_aggregate(),
            Expr::Case {
                branches,
                otherwise,
            } => {
                branches
                    .iter()
                    .any(|(when, then)| when.contains_aggregate() || then.contains_aggregate())
                    || otherwise.as_ref().is_some_and(|e| e.contains_aggregate())
            }
        }
    }

    /// Column references at **this level only**, not descending into subqueries.
    ///
    /// The counterpart to [`Expr::columns`], and the distinction is what makes correlated
    /// subqueries checkable: a reference inside a subquery must be resolved against the
    /// *subquery's* scope plus the outer one, not against the outer scope alone. A single walk
    /// that flattened both would reject every valid inner reference.
    pub fn columns_here(&self) -> Vec<&ColumnRef> {
        let mut found = Vec::new();
        self.collect_columns_here(&mut found);
        found
    }

    fn collect_columns_here<'a>(&'a self, found: &mut Vec<&'a ColumnRef>) {
        match self {
            Expr::Column(reference) => found.push(reference),
            Expr::Literal(_) => {}
            Expr::Unary { operand, .. } => operand.collect_columns_here(found),
            Expr::Binary { left, right, .. } => {
                left.collect_columns_here(found);
                right.collect_columns_here(found);
            }
            Expr::Cast { expr, .. } => expr.collect_columns_here(found),
            Expr::Aggregate { arg, .. } => {
                if let Some(inner) = arg {
                    inner.collect_columns_here(found);
                }
            }
            // Stops here, deliberately.
            Expr::Exists { .. } => {}
            Expr::ScalarSubquery { left, .. } | Expr::InSubquery { left, .. } => {
                left.collect_columns_here(found)
            }
            Expr::InList { left, .. } => left.collect_columns_here(found),
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (when, then) in branches {
                    when.collect_columns_here(found);
                    then.collect_columns_here(found);
                }
                if let Some(otherwise) = otherwise {
                    otherwise.collect_columns_here(found);
                }
            }
        }
    }

    /// The statements of any subqueries directly inside this expression.
    pub fn subqueries(&self) -> Vec<&SelectStmt> {
        let mut found = Vec::new();
        self.collect_subqueries(&mut found);
        found
    }

    fn collect_subqueries<'a>(&'a self, found: &mut Vec<&'a SelectStmt>) {
        match self {
            Expr::Exists { query, .. } => found.push(query),
            Expr::ScalarSubquery { left, query, .. } | Expr::InSubquery { left, query, .. } => {
                left.collect_subqueries(found);
                found.push(query);
            }
            Expr::InList { left, .. } => left.collect_subqueries(found),
            Expr::Case {
                branches,
                otherwise,
            } => {
                for (when, then) in branches {
                    when.collect_subqueries(found);
                    then.collect_subqueries(found);
                }
                if let Some(otherwise) = otherwise {
                    otherwise.collect_subqueries(found);
                }
            }
            Expr::Column(_) | Expr::Literal(_) => {}
            Expr::Unary { operand, .. } => operand.collect_subqueries(found),
            Expr::Binary { left, right, .. } => {
                left.collect_subqueries(found);
                right.collect_subqueries(found);
            }
            Expr::Cast { expr, .. } => expr.collect_subqueries(found),
            Expr::Aggregate { arg, .. } => {
                if let Some(inner) = arg {
                    inner.collect_subqueries(found);
                }
            }
        }
    }

    /// Does this expression contain a subquery?
    pub fn contains_subquery(&self) -> bool {
        match self {
            Expr::Exists { .. } | Expr::ScalarSubquery { .. } | Expr::InSubquery { .. } => true,
            // A literal list is not a subquery, whatever it superficially resembles.
            Expr::InList { left, .. } => left.contains_subquery(),
            Expr::Case {
                branches,
                otherwise,
            } => {
                branches
                    .iter()
                    .any(|(when, then)| when.contains_subquery() || then.contains_subquery())
                    || otherwise.as_ref().is_some_and(|e| e.contains_subquery())
            }
            Expr::Column(_) | Expr::Literal(_) => false,
            Expr::Unary { operand, .. } => operand.contains_subquery(),
            Expr::Binary { left, right, .. } => {
                left.contains_subquery() || right.contains_subquery()
            }
            Expr::Cast { expr, .. } => expr.contains_subquery(),
            Expr::Aggregate { arg, .. } => {
                arg.as_ref().is_some_and(|inner| inner.contains_subquery())
            }
        }
    }

    /// Does this expression produce a truth value rather than a stored value?
    pub fn is_predicate_shaped(&self) -> bool {
        match self {
            // `InSubquery` is listed explicitly rather than left to the `_` arm below. The
            // wildcard is why the compiler said nothing when this variant was added, and
            // defaulting to `false` would have quietly declared `x NOT IN (...)` not a
            // predicate — the one classification this variant exists to get right. `InList`
            // is here for the same reason: the wildcard stayed silent a second time.
            Expr::Exists { .. }
            | Expr::ScalarSubquery { .. }
            | Expr::InSubquery { .. }
            | Expr::InList { .. } => true,
            // **`Case` is NOT predicate-shaped** — it yields a value, not a truth value — and it
            // is listed explicitly anyway. The `_ => false` arm below would give the right
            // answer by accident; naming it means the next variant's author has to decide
            // rather than inherit. Third time this wildcard has stayed silent on a new variant.
            Expr::Case { .. } => false,
            Expr::Unary { op, .. } => {
                matches!(op, UnaryOp::Not | UnaryOp::IsNull | UnaryOp::IsNotNull)
            }
            Expr::Binary { op, .. } => op.is_predicate(),
            _ => false,
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
            Expr::Exists { query, .. } => query.contains_null_literal(),
            Expr::ScalarSubquery { left, query, .. } | Expr::InSubquery { left, query, .. } => {
                left.contains_null_literal() || query.contains_null_literal()
            }
            // **The important one for this variant.** A `NULL` in the list is exactly what makes
            // the negated form return nothing, so a vocabulary that missed it would be blind to
            // the feature this construct exists to exercise.
            Expr::InList { left, list, .. } => {
                left.contains_null_literal() || list.contains(&Literal::Null)
            }
            Expr::Case {
                branches,
                otherwise,
            } => {
                branches.iter().any(|(when, then)| {
                    when.contains_null_literal() || then.contains_null_literal()
                }) || otherwise
                    .as_ref()
                    .is_some_and(|e| e.contains_null_literal())
            }
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

/// How two tables are joined.
///
/// The outer kinds are the interesting ones, and `RIGHT`/`FULL` especially so: **SQLite gained
/// them only in 3.39.0 (2022)**, having refused them for its entire prior history, while DuckDB
/// has had them throughout. One side's implementation is therefore much younger than the other's,
/// which is exactly the asymmetry a differential test wants.
///
/// They are also where `NULL` arrives *from the join itself* rather than from the data — an
/// unmatched row is padded with `NULL`s — so a predicate that behaved one way on stored `NULL`s
/// meets a second source of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinKind {
    Inner,
    Left,
    Right,
    Full,
}

impl JoinKind {
    pub fn as_sql(self) -> &'static str {
        match self {
            JoinKind::Inner => "INNER JOIN",
            JoinKind::Left => "LEFT OUTER JOIN",
            JoinKind::Right => "RIGHT OUTER JOIN",
            JoinKind::Full => "FULL OUTER JOIN",
        }
    }

    /// Can this join introduce `NULL`s that were never in the data?
    pub fn pads_with_nulls(self) -> bool {
        self != JoinKind::Inner
    }
}

/// A join of the query's table with one other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Join {
    pub kind: JoinKind,
    /// The table being joined in. Must differ from the query's own table.
    pub table: String,
    /// The `ON` predicate. Never absent: a join without one is a cross product, which is a
    /// different construct and is not what this axis is testing.
    pub on: Expr,
}

/// A set operation combining two queries.
///
/// **`UNION`, `INTERSECT` and `EXCEPT` deduplicate; `UNION ALL` does not** — which makes the
/// pair a natural probe: the same two branches under `UNION` and `UNION ALL` must differ
/// exactly by the duplicates.
///
/// They are also the place `NULL` stops behaving as it does elsewhere. SQL's `=` says
/// `NULL = NULL` is unknown, but set operations treat two `NULL`s as *the same value* for
/// deduplication and matching. An engine that implemented one rule where the other applies
/// would diverge here and nowhere else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetOp {
    Union,
    UnionAll,
    Intersect,
    Except,
}

impl SetOp {
    pub fn as_sql(self) -> &'static str {
        match self {
            SetOp::Union => "UNION",
            SetOp::UnionAll => "UNION ALL",
            SetOp::Intersect => "INTERSECT",
            SetOp::Except => "EXCEPT",
        }
    }

    /// Does this operation remove duplicate rows?
    pub fn deduplicates(self) -> bool {
        self != SetOp::UnionAll
    }
}

/// The right-hand side of a set operation.
///
/// Boxed because it contains a [`SelectStmt`], which contains this — the same recursion
/// problem `Expr` has, and the same answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetBranch {
    pub op: SetOp,
    pub right: Box<SelectStmt>,
}

/// The query under test.
///
/// One `SELECT` over one table in v1. Joins, grouping and subqueries are where DuckDB's
/// bugs are most likely to be, and they come after the pipeline is trustworthy — the fields
/// are not present yet because an unused field is an invitation to fill it in early.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectStmt {
    /// `SELECT DISTINCT` rather than plain `SELECT`.
    ///
    /// **Worth an axis because deduplication changes `NULL`'s rules.** Everywhere else in SQL,
    /// `NULL = NULL` is UNKNOWN — two `NULL`s are not equal. `DISTINCT` contradicts that: it
    /// treats two `NULL`s as *the same value* and collapses them to one. It is the same
    /// exception the set operations make, applied within a single query, and an engine has to
    /// implement it as a deliberate special case rather than by reusing its equality.
    ///
    /// **It also breaks TLP's multiset relation**, which is why the metamorphic side handles it
    /// separately — see `metamorphic::check_distinct`.
    pub distinct: bool,
    /// What to return. Never empty — `SELECT` with nothing to select is not a case.
    pub projection: Vec<Expr>,
    /// The tables in the `FROM` clause, comma-joined.
    ///
    /// **Never empty**, and the first entry is the *primary* table — the one predicates and
    /// projections are built against. Additional entries are comma-joins, i.e. cross products.
    ///
    /// # Why this is a list rather than a single table
    ///
    /// It was a single `String` until S10. The restriction was deliberate and its stated reason
    /// was *"so a case never depends on the associativity of chained joins"* — the same reason
    /// set operations are limited to one. **Both restrictions turned out to hide a divergence**,
    /// and both for the same underlying cause: precedence.
    ///
    /// SQLite documents its own parser as getting comma-join precedence wrong (`SPECS.md`
    /// §2.11), and measurement confirms it — `FROM a, b RIGHT JOIN c` yields `(NULL, NULL, 3)` on
    /// SQLite against `(1, NULL, 3)` on DuckDB. That case **could not be built at all** while
    /// this field was a `String`, so the one real finding this project has could not be minimized,
    /// signatured or triaged. Widening it is the prerequisite for using the finding, not merely
    /// another axis.
    #[serde(deserialize_with = "one_or_many_tables")]
    pub from: Vec<String>,
    /// `WHERE`, if any.
    pub filter: Option<Expr>,
    /// A join with one other table, if any.
    ///
    /// At most one, so a case never depends on the associativity of chained joins — the same
    /// discipline the set operations follow, and for the same reason.
    pub join: Option<Join>,
    /// A set operation with another query, if any.
    ///
    /// Deliberately **not chained**: at most one operation, so no case can depend on the
    /// precedence of `INTERSECT` against `UNION`/`EXCEPT` — which is a documented divergence
    /// (SQLite gives them equal precedence left-to-right; SQL92 binds `INTERSECT` tighter),
    /// and one whose DuckDB side has not been retrieved. Chaining is a separate axis to add
    /// deliberately, not something to inherit by accident.
    pub set_op: Option<SetBranch>,
    /// `HAVING`, if any — a filter on **groups**, applied after aggregation.
    ///
    /// Only meaningful when the query aggregates, and only generated then. Worth an axis
    /// because it applies three-valued logic to *aggregate results* rather than to rows: a
    /// `HAVING SUM(x) > 0` on a group whose `SUM` is `NULL` is UNKNOWN, and the group vanishes.
    ///
    /// **It breaks the `WHERE`-partitioned TLP forms**, which is why they refuse it: the
    /// partitions have different aggregate values than the whole, so a group passing
    /// `HAVING SUM > 5` with `SUM = 6` fails in both partitions when split into 2 and 4.
    /// Partitioning on *this* predicate instead is sound — see `metamorphic::partition_having`.
    pub having: Option<Expr>,
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

/// Accept a `FROM` clause written either as a single table name or as a list.
///
/// # Why this exists, and why it is not merely convenience
///
/// Findings on disk store the **whole `SqlCase`**, not the seed. [`crate::ast::SqlCase`] explains
/// why: a seed reproduces a case only for the exact generator that produced it, and the tensor
/// domain lost 810 of 814 recorded findings to that. Storing the case makes a finding durable
/// across generator changes.
///
/// **Widening `from` from `String` to `Vec<String>` broke that guarantee** — every finding
/// recorded before S10 became unloadable, so the durability held against *generator* changes and
/// not against *AST* changes, which this project makes constantly. Accepting both spellings
/// restores it for six lines.
///
/// The reverse direction needs nothing: new findings serialise as a list, and nothing older reads
/// them.
fn one_or_many_tables<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }

    Ok(match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(name) => vec![name],
        OneOrMany::Many(names) => names,
    })
}

impl SelectStmt {
    /// The **primary** table: the first entry of `FROM`, which predicates and projections are
    /// built against. Additional entries are comma-joins.
    ///
    /// Returns `""` rather than panicking when `FROM` is empty. An empty `FROM` is not valid SQL
    /// and [`crate::ast::SqlCase::validate`] rejects it — but this is a fuzzer, and a case that
    /// slipped through should make a lookup fail cleanly rather than abort a campaign hours in.
    pub fn primary(&self) -> &str {
        self.from.first().map(String::as_str).unwrap_or_default()
    }

    /// Every table in scope for this query: the `FROM` list plus any joined table.
    ///
    /// Name resolution needs all of them, and before S10 "all of them" was at most two and could
    /// be written out by hand at each site. With a comma-join list it cannot.
    pub fn tables_in_scope(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.from.iter().map(String::as_str).collect();
        if let Some(join) = &self.join {
            names.push(&join.table);
        }
        names
    }

    /// Every expression this statement contains, at its own level.
    fn expressions(&self) -> impl Iterator<Item = &Expr> {
        self.projection
            .iter()
            .chain(self.filter.iter())
            .chain(self.join.iter().map(|join| &join.on))
    }

    /// How many expression nodes this statement contains, following subqueries and set
    /// operations. Used by minimization to know a reduction reduced something.
    pub fn node_count(&self) -> usize {
        self.expressions().map(Expr::node_count).sum::<usize>()
            + self.order_by.len()
            + usize::from(self.limit.is_some())
            + self.group_by.len()
            + self
                .set_op
                .as_ref()
                .map_or(0, |branch| branch.right.node_count())
    }

    /// Every column reference anywhere in this statement, **including inside subqueries**.
    ///
    /// The inclusion is the point: a correlated subquery's references to the outer scope are
    /// exactly what a caller must not delete, and a walk that stopped at the subquery boundary
    /// would report them as absent.
    pub fn collect_columns<'a>(&'a self, found: &mut Vec<&'a ColumnRef>) {
        for expression in self.expressions() {
            found.extend(expression.columns());
        }
        for key in &self.order_by {
            found.push(&key.column);
        }
        found.extend(self.group_by.iter());
        if let Some(branch) = &self.set_op {
            branch.right.collect_columns(found);
        }
    }

    /// Does anything in this statement mention a `NULL` literal?
    pub fn contains_null_literal(&self) -> bool {
        self.expressions().any(Expr::contains_null_literal)
            || self
                .set_op
                .as_ref()
                .is_some_and(|branch| branch.right.contains_null_literal())
    }

    /// Does this statement contain a subquery anywhere?
    pub fn contains_subquery(&self) -> bool {
        self.expressions().any(Expr::contains_subquery)
            || self
                .set_op
                .as_ref()
                .is_some_and(|branch| branch.right.contains_subquery())
    }
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
            having: None,
            distinct: false,
            projection: vec![Expr::Column(column_ref("a"))],
            from: vec!["t".to_string()],
            join: None,
            set_op: None,
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
