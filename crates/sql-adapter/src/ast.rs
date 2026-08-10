//! What a test case *is* in this domain.
//!
//! Not one call — a whole small database program: a schema to create, rows to insert, and
//! one query to run. All three travel together in a [`SqlCase`], and that is the property
//! the entire design rests on. Because the case carries its own world, running it means
//! opening a database, using it, and dropping it *inside a single call*; nothing persists
//! between cases; so the shared engine's `Implementation::run(&self, ..)` never needs to
//! become `&mut self`. The statefulness that looked like this domain's hard problem
//! dissolves into the shape of the case.
//!
//! # Why the case is a tree and not SQL text
//!
//! It was text through S1, which was enough to prove one case could flow through every
//! seam. It is a tree now, for three reasons that all arrive later and would each be
//! painful to retrofit:
//!
//! - **Shrinking needs structure.** Minimizing a finding means dropping a predicate, a
//!   column, a row. On text those are edits that can produce SQL which no longer parses; on
//!   a tree they are node removals that cannot.
//! - **Two dialects need one meaning.** The same case is rendered as each engine spells it
//!   ([`crate::render`]). From a tree that is two printers; from text it is search and
//!   replace, which is how a "translation" quietly changes what the query asks.
//! - **Later phases read properties off the case** — "does this put a `NULL` in a
//!   comparison?" — which is a walk on a tree and string-matching on text.

use crate::render::{Dialect, render_case};
use crate::schema::{Index, InsertRows, Literal, SelectStmt, Table};
use diff_fuzzer_core::traits::Input;
use serde::{Deserialize, Serialize};

/// One self-contained SQL test case: schema, seed data, and the query under test.
///
/// `Clone` because minimization repeatedly produces modified copies; `Debug` because a case
/// that cannot be printed cannot be reported; `Serialize`/`Deserialize` because the *whole
/// case* is what gets written to a finding. Not the seed — a seed only reproduces a case for
/// the exact generator that produced it, and generators change. The tensor domain learned
/// that when 810 of 814 recorded findings stopped reproducing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlCase {
    /// The tables to create, in order.
    pub schema: Vec<Table>,
    /// Secondary indexes, created after the tables and before the rows.
    ///
    /// Semantically inert by definition — an index changes *how* an answer is reached, never
    /// *what* it is — which is what makes them worth generating: any observable effect is a bug.
    pub indexes: Vec<Index>,
    /// The rows to insert, per table.
    pub data: Vec<InsertRows>,
    /// The single `SELECT` under test. Exactly one, so a divergence names one query.
    pub query: SelectStmt,
}

/// Marks `SqlCase` as a test case for the engine.
///
/// The trait has no methods — it exists so the other seams can say `type In: Input` and be
/// sure whatever flows through them can be cloned, printed, and therefore reported.
impl Input for SqlCase {}

impl SqlCase {
    /// The statements to execute, in order, spelled for `dialect`.
    ///
    /// One definition of execution order, used by both engines, so the two can never
    /// disagree because one of them applied the case differently. A divergence manufactured
    /// by this crate is the worst kind: it looks exactly like a finding.
    pub fn statements(&self, dialect: Dialect) -> Vec<String> {
        render_case(
            &self.schema,
            &self.indexes,
            &self.data,
            &self.query,
            dialect,
        )
    }

    /// The table the query reads, if the schema contains it.
    pub fn queried_table(&self) -> Option<&Table> {
        self.schema
            .iter()
            .find(|table| table.name == self.query.primary())
    }

    /// The rows of the table the query reads.
    ///
    /// An empty slice when the table has no `INSERT` — which is what an empty table *is*,
    /// not a missing value.
    pub fn queried_rows(&self) -> &[Vec<Literal>] {
        self.data
            .iter()
            .find(|insert| insert.table == self.query.primary())
            .map(|insert| insert.rows.as_slice())
            .unwrap_or_default()
    }

    /// Does the query's `ORDER BY` put *these* rows into exactly one legal order?
    ///
    /// The property that decides whether row order is part of the answer. It belongs to the
    /// case rather than to the query, because ties in the data are what break a total order
    /// — see [`crate::ordering`].
    pub fn is_totally_ordered(&self) -> bool {
        // **A grouped query ordered by its grouping columns IS totally ordered** — provably,
        // not by inspection of the data. `GROUP BY c` emits exactly one row per distinct value
        // of `c` (and exactly one row for `NULL`, since grouping treats `NULL`s as equal), so
        // ordering by `c` puts those rows in one legal order however the data ties.
        //
        // This is the only case where ordering can be established for a query whose output
        // rows are not the seeded rows, and it matters: grouped queries were 30% of the
        // combined corpus and every one of them was being sorted away before comparison.
        if !self.query.group_by.is_empty()
            && !self.query.order_by.is_empty()
            && self
                .query
                .group_by
                .iter()
                .all(|key| self.query.order_by.iter().any(|by| &by.column == key))
        {
            return true;
        }

        // **Otherwise a grouped or aggregated query is not treated as totally ordered.**
        //
        // `orders_rows_totally` asks whether the *seeded rows* tie on the ordering columns —
        // but a grouped query does not return seeded rows, it returns one row per group, and
        // an aggregate with no `GROUP BY` returns exactly one row from all of them. The
        // question the function answers is simply not the question being asked.
        //
        // Rather than teach it to compute groups, take the safe answer: not ordered, so both
        // sides are sorted before comparing and no `LIMIT` is ever attached. Being wrong this
        // way costs the ability to catch an ordering bug in a grouped query; being wrong the
        // other way would invent divergences on every grouped query with a tie. The first is
        // a gap, the second is noise that would swamp the oracle.
        if !self.query.group_by.is_empty() || self.aggregates() {
            return false;
        }

        // A join changes which rows exist: an outer join manufactures rows that are in no
        // table, padded with `NULL`s. The ordering check inspects the *queried table's* seeded
        // rows, which are no longer what comes back, so — as with grouping and set operations —
        // take the safe answer and sort before comparing.
        // A comma-join is a cross product, so the rows returned are not the seeded rows of any
        // one table — the same reason an explicit join disqualifies the check.
        if self.query.join.is_some() || self.query.from.len() > 1 {
            return false;
        }

        // A set operation's output rows are neither branch's rows: `UNION` may drop
        // duplicates, `EXCEPT` removes matches. The seeded rows the ordering check inspects
        // are not what comes back, so — as with grouping — take the safe answer.
        if self.query.set_op.is_some() {
            return false;
        }

        match self.queried_table() {
            Some(table) => crate::ordering::orders_rows_totally(
                &self.query.order_by,
                table,
                self.queried_rows(),
            ),
            None => false,
        }
    }

    /// Does the query aggregate — collapsing many rows into one?
    pub fn aggregates(&self) -> bool {
        self.query
            .projection
            .iter()
            .any(crate::schema::Expr::contains_aggregate)
    }

    /// A small fixed case, for tests that need a known answer rather than a generated one.
    ///
    /// Carries the values that break result comparison first: a `NULL`, an empty string
    /// beside it (the two must never render alike), a row the `WHERE` excludes, and a
    /// total `ORDER BY` so row order is genuinely part of the answer.
    pub fn fixed_example() -> Self {
        use crate::schema::{Column, ColumnRef, Direction, Expr, OrderKey, SelectStmt, SqlType};

        let reference = |column: &str| ColumnRef {
            table: "t0".to_string(),
            column: column.to_string(),
        };

        Self {
            indexes: Vec::new(),
            schema: vec![Table {
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
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![
                    vec![Literal::Integer(1), Literal::Text("one".to_string())],
                    vec![Literal::Integer(2), Literal::Text(String::new())],
                    vec![Literal::Integer(3), Literal::Null],
                    vec![Literal::Integer(-1), Literal::Text("neg".to_string())],
                ],
            }],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(reference("c0")), Expr::Column(reference("c1"))],
                from: vec!["t0".to_string()],
                join: None,
                set_op: None,
                group_by: vec![],
                filter: Some(Expr::Binary {
                    op: crate::schema::BinaryOp::Greater,
                    left: Box::new(Expr::Column(reference("c0"))),
                    right: Box::new(Expr::Literal(Literal::Integer(0))),
                }),
                order_by: vec![OrderKey {
                    column: reference("c0"),
                    direction: Direction::Ascending,
                    nulls_first: true,
                }],
                limit: None,
            },
        }
    }

    /// Check the invariants a valid case must hold.
    ///
    /// Everything here is guaranteed by construction *for generated cases*. It is checked
    /// anyway, because generation is not the only source of cases: minimization edits them,
    /// a byte decoder will build them, and a saved repro is loaded from a file someone may
    /// have touched. A case that violates these is a defect in whichever of those produced
    /// it, and finding out at the boundary beats finding out from an engine's parser.
    pub fn validate(&self) -> Result<(), String> {
        let _table = self.queried_table().ok_or_else(|| {
            format!(
                "query reads table {} which is not in the schema",
                self.query.primary()
            )
        })?;

        if self.query.projection.is_empty() {
            return Err("a query must project at least one expression".to_string());
        }

        for insert in &self.data {
            let target = self
                .schema
                .iter()
                .find(|candidate| candidate.name == insert.table)
                .ok_or_else(|| {
                    format!("data for table {} which is not in the schema", insert.table)
                })?;

            for row in &insert.rows {
                if row.len() != target.columns.len() {
                    return Err(format!(
                        "a row of {} has {} values for {} columns",
                        target.name,
                        row.len(),
                        target.columns.len()
                    ));
                }
                for (value, column) in row.iter().zip(target.columns.iter()) {
                    if let Some(value_type) = value.sql_type()
                        && !column.sql_type.accepts(value_type)
                    {
                        return Err(format!(
                            "{value:?} does not fit the {:?} column {}.{}",
                            column.sql_type, target.name, column.name
                        ));
                    }
                }
            }
        }

        // **Name resolution, scope by scope.**
        //
        // A correlated subquery may reference the outer row, so a reference is legal if it
        // resolves in *its own* scope or in any enclosing one. Resolving everything against a
        // single flattened scope would accept an outer query referencing an inner table, which
        // is not legal SQL and which the engines would refuse — a case that tests the parser
        // rather than the engine.
        resolve_scopes(&self.query, &self.schema, &[])?;

        // Grouping rules. SQLite permits a bare column alongside an aggregate without grouping
        // by it (picking an arbitrary row); DuckDB refuses. Generating only the strict form is
        // what both accept — so this is a validity rule here, not a legal difference to catalog.
        if !self.query.group_by.is_empty() {
            for expression in &self.query.projection {
                let is_grouping_column = matches!(
                    expression,
                    crate::schema::Expr::Column(reference)
                        if self.query.group_by.iter().any(|key| key == reference)
                );
                if !is_grouping_column && !expression.contains_aggregate() {
                    return Err(
                        "a grouped query may project only its grouping columns and aggregates"
                            .to_string(),
                    );
                }
            }
        } else if self.aggregates() {
            for expression in &self.query.projection {
                if !expression.contains_aggregate() {
                    return Err(
                        "an aggregate query without GROUP BY may not project bare columns"
                            .to_string(),
                    );
                }
            }
        }

        // Join rules. A self join needs aliases, which the v1 AST cannot express; the joined
        // table's existence and its `ON` predicate's references are covered by `resolve_scopes`.
        if let Some(join) = &self.query.join
            && join.table == self.query.primary()
        {
            return Err("a self join needs aliases, which v1 cannot express".to_string());
        }

        // Set-operation rules. Both branches must project the same number of columns, and
        // neither may carry its own `ORDER BY` or `LIMIT` — those would bind to one branch
        // rather than to the combined result, which is not a question worth asking the two
        // engines to agree about.
        if let Some(branch) = &self.query.set_op {
            if branch.right.projection.len() != self.query.projection.len() {
                return Err(format!(
                    "a set operation joins {} columns to {}",
                    self.query.projection.len(),
                    branch.right.projection.len()
                ));
            }
            if !branch.right.order_by.is_empty() || branch.right.limit.is_some() {
                return Err("a set operation's right branch may not order or limit".to_string());
            }
            if branch.right.primary() != self.query.primary() {
                return Err("both branches must read the same table in v1".to_string());
            }
            // Chaining is permitted to one further level, and no more: two operations is
            // enough for precedence to be observable, and each extra level multiplies the
            // shapes a minimized case has to be read through.
            if let Some(inner) = &branch.right.set_op {
                if inner.right.set_op.is_some() {
                    return Err("set operations chain at most twice".to_string());
                }
                if inner.right.projection.len() != self.query.projection.len() {
                    return Err("every branch of a chain must project the same columns".to_string());
                }
                if !inner.right.order_by.is_empty() || inner.right.limit.is_some() {
                    return Err("no branch of a chain may order or limit".to_string());
                }
                if inner.right.primary() != self.query.primary() {
                    return Err("every branch must read the same table in v1".to_string());
                }
            }
        }

        // The rule that cannot be expressed in the type system, and that a shrinker could
        // otherwise break by deleting a row: a `LIMIT` without a total order lets two
        // engines return different rows, both legally.
        if self.query.limit.is_some() && !self.is_totally_ordered() {
            return Err(
                "a LIMIT requires an ORDER BY that totally orders this case's rows".to_string(),
            );
        }

        Ok(())
    }
}

/// Check every column reference in `statement`, given the tables enclosing scopes provide.
///
/// Walks outward: a reference resolves against this statement's own tables first, then against
/// anything an enclosing query had in scope. That outward search *is* correlation — remove it
/// and a correlated subquery becomes unrepresentable; widen it to a flat set and an outer query
/// could illegally reference an inner table.
fn resolve_scopes(
    statement: &SelectStmt,
    schema: &[Table],
    outer: &[&Table],
) -> Result<(), String> {
    // **Every table in `FROM` is in scope**, not just the first. Before S10 this was a single
    // table and the list could not exist; a comma-join puts several in scope at once, and a
    // resolver that saw only the first would reject valid references to the others.
    let mut scope: Vec<&Table> = Vec::new();
    for name in &statement.from {
        let table = schema
            .iter()
            .find(|table| table.name == *name)
            .ok_or_else(|| format!("query reads table {name}, which is not in the schema"))?;
        scope.push(table);
    }
    if scope.is_empty() {
        return Err("query has an empty FROM clause".to_string());
    }
    if let Some(join) = &statement.join {
        let joined = schema
            .iter()
            .find(|table| table.name == join.table)
            .ok_or_else(|| {
                format!(
                    "join names table {}, which is not in the schema",
                    join.table
                )
            })?;
        scope.push(joined);
    }
    scope.extend_from_slice(outer);

    let resolves = |reference: &crate::schema::ColumnRef| {
        scope
            .iter()
            .any(|table| table.name == reference.table && table.column(&reference.column).is_some())
    };

    let mut here: Vec<&crate::schema::ColumnRef> = Vec::new();
    for expression in statement
        .projection
        .iter()
        .chain(statement.filter.iter())
        .chain(statement.join.iter().map(|join| &join.on))
    {
        here.extend(expression.columns_here());
    }
    for key in &statement.order_by {
        here.push(&key.column);
    }
    here.extend(statement.group_by.iter());

    for reference in here {
        if !resolves(reference) {
            return Err(format!(
                "{}.{} is not in scope for the query over {}",
                reference.table,
                reference.column,
                statement.from.join(", ")
            ));
        }
    }

    // Recurse, handing this statement's scope down as the enclosing one.
    for expression in statement
        .projection
        .iter()
        .chain(statement.filter.iter())
        .chain(statement.join.iter().map(|join| &join.on))
    {
        for subquery in expression.subqueries() {
            resolve_scopes(subquery, schema, &scope)?;
        }
    }
    if let Some(branch) = &statement.set_op {
        resolve_scopes(&branch.right, schema, outer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_schema::Bounds;
    use crate::schema::{Column, ColumnRef, Direction, Expr, OrderKey, SqlType};
    use diff_fuzzer_core::SeededRng;

    fn simple_case() -> SqlCase {
        SqlCase {
            indexes: Vec::new(),
            schema: vec![Table {
                name: "t0".to_string(),
                columns: vec![Column {
                    name: "c0".to_string(),
                    sql_type: SqlType::Integer,
                }],
            }],
            data: vec![InsertRows {
                table: "t0".to_string(),
                rows: vec![vec![Literal::Integer(1)], vec![Literal::Integer(2)]],
            }],
            query: SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(ColumnRef {
                    table: "t0".to_string(),
                    column: "c0".to_string(),
                })],
                from: vec!["t0".to_string()],
                join: None,
                set_op: None,
                group_by: vec![],
                filter: None,
                order_by: vec![],
                limit: None,
            },
        }
    }

    fn generated(seed: u64) -> SqlCase {
        let mut rng = SeededRng::from_seed(seed);
        let schema = crate::gen_schema::generate_schema(&mut rng, Bounds::V1);
        let data = crate::gen_schema::generate_data(&mut rng, &schema, Bounds::V1);
        let query = crate::gen_query::generate_query(&mut rng, &schema, &data, Bounds::V1);
        SqlCase {
            indexes: Vec::new(),
            schema,
            data,
            query,
        }
    }

    #[test]
    fn statements_run_schema_then_data_then_query() {
        let statements = simple_case().statements(Dialect::Sqlite);
        assert_eq!(statements.len(), 3);
        assert!(statements[0].starts_with("CREATE TABLE"));
        assert!(statements[1].starts_with("INSERT INTO"));
        assert!(statements[2].starts_with("SELECT"));
    }

    #[test]
    fn a_case_survives_a_round_trip_through_json() {
        // The property a finding depends on: if a case cannot be written and read back
        // unchanged, a saved divergence is a story rather than a reproduction.
        for seed in 0..50 {
            let case = generated(seed);
            let json = serde_json::to_string(&case).expect("a case serializes");
            let back: SqlCase = serde_json::from_str(&json).expect("and deserializes");
            assert_eq!(case, back, "seed {seed}");
        }
    }

    #[test]
    fn every_generated_case_is_valid() {
        for seed in 0..1000 {
            if let Err(problem) = generated(seed).validate() {
                panic!("seed {seed} generated an invalid case: {problem}");
            }
        }
    }

    #[test]
    fn validate_rejects_a_column_that_does_not_exist() {
        let mut case = simple_case();
        case.query.projection = vec![Expr::Column(ColumnRef {
            table: "t0".to_string(),
            column: "nope".to_string(),
        })];
        assert!(case.validate().is_err());
    }

    #[test]
    fn validate_rejects_a_row_of_the_wrong_width() {
        let mut case = simple_case();
        case.data[0]
            .rows
            .push(vec![Literal::Integer(1), Literal::Integer(2)]);
        assert!(case.validate().is_err());
    }

    #[test]
    fn validate_rejects_a_value_of_the_wrong_type() {
        let mut case = simple_case();
        case.data[0].rows.push(vec![Literal::Text("x".to_string())]);
        assert!(case.validate().is_err());
    }

    /// The grouping rules, as negative tests.
    ///
    /// These exist because the rules were once **deleted by accident** during a refactor and
    /// nothing failed: every generated case still validated, because the generator does not
    /// produce the shapes the rules reject. A rule with only positive coverage can be removed
    /// silently, which makes it no rule at all.
    #[test]
    fn validate_rejects_a_bare_column_beside_an_aggregate() {
        use crate::schema::{AggregateFunc, ColumnRef, Expr};

        let mut case = SqlCase::fixed_example();
        let reference = ColumnRef {
            table: "t0".to_string(),
            column: "c0".to_string(),
        };
        // SQLite would accept this and pick an arbitrary row; DuckDB refuses it. Generating
        // only the strict form is what both accept.
        case.query.projection = vec![
            Expr::Column(reference.clone()),
            Expr::Aggregate {
                func: AggregateFunc::CountRows,
                arg: None,
            },
        ];
        assert!(case.validate().is_err(), "aggregate beside a bare column");

        // Grouping by that column makes it legal again.
        case.query.group_by = vec![reference];
        assert!(case.validate().is_ok(), "grouped, so the column is legal");
    }

    #[test]
    fn validate_rejects_a_projection_that_is_not_grouped_or_aggregated() {
        use crate::schema::{ColumnRef, Expr};

        let mut case = SqlCase::fixed_example();
        case.query.group_by = vec![ColumnRef {
            table: "t0".to_string(),
            column: "c0".to_string(),
        }];
        // Projects c1, which is neither the grouping column nor an aggregate.
        case.query.projection = vec![Expr::Column(ColumnRef {
            table: "t0".to_string(),
            column: "c1".to_string(),
        })];
        assert!(case.validate().is_err());
    }

    #[test]
    fn validate_rejects_a_self_join() {
        use crate::schema::{BinaryOp, ColumnRef, Expr, Join, JoinKind};

        let mut case = SqlCase::fixed_example();
        let reference = ColumnRef {
            table: "t0".to_string(),
            column: "c0".to_string(),
        };
        case.query.join = Some(Join {
            kind: JoinKind::Inner,
            table: "t0".to_string(),
            on: Expr::Binary {
                op: BinaryOp::Equal,
                left: Box::new(Expr::Column(reference.clone())),
                right: Box::new(Expr::Column(reference)),
            },
        });
        assert!(
            case.validate().is_err(),
            "a self join needs aliases the v1 AST cannot express"
        );
    }

    /// Correlation is the whole point of a subquery here: an inner query referencing the
    /// **outer** row must validate, and an outer query referencing an inner table must not.
    #[test]
    fn scopes_resolve_outward_but_never_inward() {
        use crate::schema::{BinaryOp, Column, ColumnRef, Expr, SelectStmt, SqlType, Table};

        let mut case = SqlCase::fixed_example();
        case.schema.push(Table {
            name: "t1".to_string(),
            columns: vec![Column {
                name: "d0".to_string(),
                sql_type: SqlType::Integer,
            }],
        });
        case.data.push(crate::schema::InsertRows {
            table: "t1".to_string(),
            rows: vec![vec![Literal::Integer(1)]],
        });

        let outer_column = ColumnRef {
            table: "t0".to_string(),
            column: "c0".to_string(),
        };
        let inner_column = ColumnRef {
            table: "t1".to_string(),
            column: "d0".to_string(),
        };

        // An inner query comparing the inner table to the OUTER row — correlated, and legal.
        case.query.filter = Some(Expr::Exists {
            not: false,
            query: Box::new(SelectStmt {
                having: None,
                distinct: false,
                projection: vec![Expr::Column(inner_column.clone())],
                from: vec!["t1".to_string()],
                join: None,
                set_op: None,
                group_by: vec![],
                filter: Some(Expr::Binary {
                    op: BinaryOp::Equal,
                    left: Box::new(Expr::Column(inner_column)),
                    right: Box::new(Expr::Column(outer_column)),
                }),
                order_by: vec![],
                limit: None,
            }),
        });
        assert!(
            case.validate().is_ok(),
            "an inner query may reference the outer row: {:?}",
            case.validate()
        );

        // The reverse must fail: the outer query cannot see inside the subquery.
        let mut inverted = SqlCase::fixed_example();
        inverted.query.projection = vec![Expr::Column(ColumnRef {
            table: "t1".to_string(),
            column: "d0".to_string(),
        })];
        assert!(
            inverted.validate().is_err(),
            "an outer query must not reference a table it does not read"
        );
    }

    /// The invariant a shrinker is most likely to break, since it is about the *data* and
    /// the query at once.
    #[test]
    fn validate_rejects_a_limit_whose_order_is_not_total() {
        let mut case = simple_case();
        case.query.order_by = vec![OrderKey {
            column: ColumnRef {
                table: "t0".to_string(),
                column: "c0".to_string(),
            },
            direction: Direction::Ascending,
            nulls_first: true,
        }];
        case.query.limit = Some(1);
        // Distinct values: the order is total, so the limit is fine.
        assert!(case.validate().is_ok());

        // Now make the two rows tie. Nothing about the *query* changed, and it is now
        // invalid — which is the whole point of totality being a property of the case.
        case.data[0].rows = vec![vec![Literal::Integer(1)], vec![Literal::Integer(1)]];
        assert!(case.validate().is_err());
    }
}
