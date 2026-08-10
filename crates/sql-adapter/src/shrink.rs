//! Making a failing case small enough to look at.
//!
//! A generated divergence is a whole database program — several tables, a dozen rows, an
//! expression tree. Nobody can tell from that which part matters, and a maintainer handed
//! one would be entitled to ignore it. Minimization repeatedly proposes simpler cases and
//! keeps any that still diverges, until nothing simpler does.
//!
//! # The two obligations, and why SQL makes the first one hard
//!
//! `Shrink` requires that every candidate be **valid** and **strictly simpler**.
//!
//! *Strictly simpler* is easy here: [`complexity`] counts nodes and cells, and every move
//! removes or shrinks something. It is checked anyway (see [`SqlCase::candidates`]),
//! because relying on a contract for termination is not the same as enforcing it — and the
//! tensor domain's shrinker had exactly this bug, two moves that proposed each other
//! forever.
//!
//! *Valid* is where SQL differs from tensors, and it is the interesting part: **reductions
//! here are not local.** Dropping a column can orphan a reference in the `WHERE` clause.
//! Dropping a *row* can turn a totally-ordered query into an ambiguous one, invalidating a
//! `LIMIT` that was fine a moment ago — a change to the **data** breaking a rule about the
//! **query**. `09` §3 flagged exactly this worry for grammar domains.
//!
//! The answer is not to reason about which moves are safe. It is to propose freely and let
//! [`SqlCase::validate`] reject: one predicate, already written, already tested, and
//! impossible to forget to apply.

use crate::ast::SqlCase;
use crate::schema::{Expr, Literal};
use diff_fuzzer_core::minimize::Shrink;

/// How big a case is, for the purpose of deciding whether a reduction reduced anything.
///
/// A pair, compared lexicographically: the query's node count first, then the data. Query
/// structure is what a reader has to understand, so simplifying it is worth more than
/// deleting a row — but a case with fewer rows is genuinely simpler at equal structure, and
/// without the second element the search would stop while the data was still noisy.
pub fn complexity(case: &SqlCase) -> (usize, usize) {
    // **Delegates to `SelectStmt::node_count` rather than counting by hand.** It used to
    // reimplement a narrower count — projection, filter, `ORDER BY`, `LIMIT` — and drifted as
    // the query grew: the join's `ON`, `GROUP BY`, `HAVING` and set operations were all invisible
    // to it, and so were indexes. A reduction the shrinker offered but the guard could not see
    // produced a candidate of *equal* complexity, which the strictly-simpler rule then rejected.
    //
    // The effect was silent and only visible on a real finding: minimizing the comma-join case
    // left a `CREATE INDEX` in the repro because dropping it changed no counted quantity.
    // **A monotonicity guard that cannot see part of the case silently forbids simplifying it.**
    let query_nodes = case.query.node_count()
        // Schema-level structure the shrinker can remove. Counted with the query rather than
        // with the data because, like the query, it is something a reader must understand:
        // an index in a repro invites the question "does the index matter?".
        + case.indexes.len()
        + case.query.from.len();

    let cells: usize = case
        .data
        .iter()
        .map(|insert| insert.rows.iter().map(Vec::len).sum::<usize>())
        .sum();

    (query_nodes, cells)
}

impl Shrink for SqlCase {
    /// Simpler versions of this case, most aggressive first.
    ///
    /// Every candidate is filtered through two gates before being offered: it must be
    /// **strictly simpler** by [`complexity`], and it must **validate**. Neither gate is
    /// decoration — the first prevents a cycle, and the second is what makes non-local
    /// reductions safe to propose without reasoning about them one by one.
    fn candidates(&self) -> Vec<Self> {
        let mut proposals = Vec::new();

        // Most aggressive first: whole clauses, then structure, then data, then values.
        // Order is about speed, not correctness — a greedy search takes the first candidate
        // that still fails, so leading with big cuts reaches a small case in fewer rounds.

        // Drop an index. **Added at S10, and its absence until then was a real gap**: the
        // shrinker was written at S5 and indexes arrived five phases later, so every minimized
        // case carried whatever indexes it started with — including ones irrelevant to the
        // failure. A repro with an unnecessary `CREATE INDEX` is a repro that invites the
        // maintainer to wonder whether the index matters.
        //
        // Cheap and high-value: an index is one statement, and removing it is the difference
        // between "this is about indexing" and "this is not".
        for index in 0..self.indexes.len() {
            let mut candidate = self.clone();
            candidate.indexes.remove(index);
            proposals.push(candidate);
        }

        // Drop the WHERE clause entirely.
        if self.query.filter.is_some() {
            let mut candidate = self.clone();
            candidate.query.filter = None;
            proposals.push(candidate);
        }

        // Drop the LIMIT. Worth trying early: it is the clause most likely to be incidental,
        // and removing it also frees the ordering constraint that keeps it valid.
        if self.query.limit.is_some() {
            let mut candidate = self.clone();
            candidate.query.limit = None;
            proposals.push(candidate);
        }

        // Drop ORDER BY keys, last first.
        if !self.query.order_by.is_empty() {
            let mut candidate = self.clone();
            candidate.query.order_by.pop();
            proposals.push(candidate);
        }

        // Split a conjunction: keep one side of an AND/OR. Halves the predicate rather than
        // peeling one node, which is what makes deep trees collapse quickly.
        if let Some(filter) = &self.query.filter {
            for half in split_binary(filter) {
                let mut candidate = self.clone();
                candidate.query.filter = Some(half);
                proposals.push(candidate);
            }
        }

        // Drop a projected expression.
        if self.query.projection.len() > 1 {
            for index in 0..self.query.projection.len() {
                let mut candidate = self.clone();
                candidate.query.projection.remove(index);
                proposals.push(candidate);
            }
        }

        // Replace a subquery predicate with a constant. The single largest reduction
        // available on a correlated case: it removes a whole nested query, its correlation,
        // and the second table's relevance in one move.
        if let Some(filter) = &self.query.filter
            && filter.contains_subquery()
        {
            let mut candidate = self.clone();
            candidate.query.filter = None;
            proposals.push(candidate);
        }

        // Simplify a projected expression to a leaf, keeping the column count.
        for (index, expression) in self.query.projection.iter().enumerate() {
            if expression.node_count() > 1 {
                let mut candidate = self.clone();
                candidate.query.projection[index] = Expr::Literal(Literal::Integer(0));
                proposals.push(candidate);
            }
        }

        // Drop a table the query does not read. Two-table schemas are common and the second
        // table is usually irrelevant to the failure.
        for (index, table) in self.schema.iter().enumerate() {
            if !self.query.from.contains(&table.name) {
                let mut candidate = self.clone();
                candidate.schema.remove(index);
                candidate.data.retain(|insert| insert.table != table.name);
                proposals.push(candidate);
            }
        }

        // Halve the rows of a table, then drop them one at a time. Halving first is the
        // difference between a handful of rounds and one per row.
        for (index, insert) in self.data.iter().enumerate() {
            if insert.rows.len() > 1 {
                let mut candidate = self.clone();
                candidate.data[index].rows.truncate(insert.rows.len() / 2);
                proposals.push(candidate);
            }
            for row in 0..insert.rows.len() {
                let mut candidate = self.clone();
                candidate.data[index].rows.remove(row);
                proposals.push(candidate);
            }
        }

        // Simplify one value at a time, toward NULL and toward zero/empty. A minimized case
        // full of `i64::MIN` and `'  '` invites questions about values that turn out to be
        // irrelevant.
        for (insert_index, insert) in self.data.iter().enumerate() {
            for (row_index, row) in insert.rows.iter().enumerate() {
                for (cell_index, value) in row.iter().enumerate() {
                    for simpler in simplify_literal(value) {
                        let mut candidate = self.clone();
                        candidate.data[insert_index].rows[row_index][cell_index] = simpler;
                        proposals.push(candidate);
                    }
                }
            }
        }

        // Drop an unreferenced column.
        for (table_index, table) in self.schema.iter().enumerate() {
            if table.columns.len() > 1 {
                for column_index in 0..table.columns.len() {
                    let mut candidate = self.clone();
                    candidate.schema[table_index].columns.remove(column_index);
                    for insert in &mut candidate.data {
                        if insert.table == table.name {
                            for row in &mut insert.rows {
                                row.remove(column_index);
                            }
                        }
                    }
                    proposals.push(candidate);
                }
            }
        }

        let before = complexity(self);
        proposals.retain(|candidate| {
            // **Strictly simpler**, enforced rather than trusted. A candidate equal to its
            // parent would let the search cycle forever, and an infinite loop inside a
            // reporting path loses the finding it was called to describe.
            complexity(candidate) < before
                // **Still a valid case.** This is what makes non-local reductions safe to
                // propose: dropping a row can invalidate a `LIMIT`, dropping a column can
                // orphan a reference, and neither has to be reasoned about here.
                && candidate.validate().is_ok()
        });

        proposals
    }
}

/// If this is an `AND`/`OR`, its two sides; otherwise nothing.
fn split_binary(expression: &Expr) -> Vec<Expr> {
    match expression {
        Expr::Binary { op, left, right } if op.is_predicate() => {
            vec![(**left).clone(), (**right).clone()]
        }
        Expr::Unary { operand, .. } => vec![(**operand).clone()],
        _ => Vec::new(),
    }
}

/// Simpler values than this one: `NULL` first, then toward zero or the empty string.
fn simplify_literal(value: &Literal) -> Vec<Literal> {
    match value {
        // `NULL` is already the simplest value to *write*, but it is the least simple to
        // *reason about* — so it is never proposed as a simplification of something else.
        // A minimized case keeps its `NULL`s only if they matter, because the moves below
        // will replace them with `0` when they do not.
        Literal::Null => vec![Literal::Integer(0)],
        Literal::Integer(0) => Vec::new(),
        Literal::Integer(_) => vec![Literal::Integer(0)],
        Literal::Text(text) if text.is_empty() => Vec::new(),
        Literal::Text(_) => vec![Literal::Text(String::new())],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Needed for `Bounds::description` — a trait method since the engine's `GenerationAxes`
    // was adopted, so the trait must be in scope at the call site.
    use crate::gen_schema::Bounds;
    use crate::generator::SqlGenerator;
    use diff_fuzzer_core::GenerationAxes;
    use diff_fuzzer_core::SeededRng;
    use diff_fuzzer_core::minimize::minimize;
    use diff_fuzzer_core::traits::Generator;

    fn generated(seed: u64) -> SqlCase {
        SqlGenerator::default().generate(&mut SeededRng::from_seed(seed))
    }

    fn generated_with(seed: u64, bounds: Bounds) -> SqlCase {
        SqlGenerator::new(bounds).generate(&mut SeededRng::from_seed(seed))
    }

    /// The contract, checked on **every axis**, not just the default one.
    ///
    /// Reductions in this domain are not local: dropping a row can invalidate a `LIMIT`,
    /// dropping a table can orphan a correlated subquery's reference. Those interactions are
    /// only reachable when the axis that creates them is on, so a contract test that only
    /// ever saw default cases would be checking the easy half.
    #[test]
    fn the_contract_holds_on_every_axis() {
        for (name, bounds) in [
            ("default", Bounds::V1),
            ("aggregates", Bounds::V1_AGGREGATES),
            ("set ops", Bounds::V1_SET_OPS),
            ("chained set ops", Bounds::V1_CHAINED_SET_OPS),
            ("joins", Bounds::V1_JOINS),
            ("subqueries", Bounds::V1_SUBQUERIES),
            ("all", Bounds::V1_ALL),
        ] {
            for seed in 0..120 {
                let case = generated_with(seed, bounds);
                let before = complexity(&case);
                for candidate in case.candidates() {
                    assert!(
                        complexity(&candidate) < before,
                        "{name}, seed {seed}: a candidate was not strictly simpler"
                    );
                    candidate.validate().unwrap_or_else(|problem| {
                        panic!("{name}, seed {seed}: invalid candidate: {problem}")
                    });
                }
            }
        }
    }

    /// A correlated case must shrink without orphaning the reference that correlates it.
    #[test]
    fn shrinking_a_correlated_case_never_orphans_its_reference() {
        for seed in 0..200 {
            let case = generated_with(seed, Bounds::V1_SUBQUERIES);
            if !case
                .query
                .filter
                .as_ref()
                .is_some_and(crate::schema::Expr::contains_subquery)
            {
                continue;
            }

            // Keep failing while the subquery survives: the search is then forced to try
            // every reduction *around* it, including dropping the table it reads.
            let result = minimize(case, |candidate| {
                candidate
                    .query
                    .filter
                    .as_ref()
                    .is_some_and(crate::schema::Expr::contains_subquery)
            });
            result
                .input
                .validate()
                .unwrap_or_else(|problem| panic!("seed {seed}: {problem}"));
        }
    }

    #[test]
    fn every_candidate_is_strictly_simpler_and_still_valid() {
        // The `Shrink` contract, checked over real generated cases rather than argued.
        for seed in 0..300 {
            let case = generated(seed);
            let before = complexity(&case);

            for candidate in case.candidates() {
                assert!(
                    complexity(&candidate) < before,
                    "seed {seed}: a candidate was not strictly simpler"
                );
                candidate.validate().unwrap_or_else(|problem| {
                    panic!("seed {seed}: a candidate was invalid: {problem}")
                });
            }
        }
    }

    #[test]
    fn shrinking_terminates_and_reaches_something_small() {
        // "Still fails" here is a stand-in for the oracle: pretend any case with data
        // diverges. The search should strip everything it can and stop.
        for seed in 0..50 {
            let case = generated(seed);
            let result = minimize(case, |candidate| !candidate.queried_rows().is_empty());

            assert!(result.input.validate().is_ok(), "seed {seed}");
            // It stopped because nothing simpler failed, not because it ran out of budget.
            assert!(
                result.is_minimal(),
                "seed {seed}: stopped at {:?}",
                result.stopped
            );
        }
    }

    #[test]
    fn a_case_that_does_not_fail_is_returned_untouched() {
        let case = generated(1);
        let result = minimize(case.clone(), |_| false);
        assert_eq!(result.input, case);
        assert_eq!(result.steps, 0);
    }

    /// The non-local reduction `09` §3 warned about, as an executable case.
    ///
    /// Removing a **row** can break a rule about the **query**: with two distinct rows the
    /// `ORDER BY` is total and the `LIMIT` is legal, but make the rows tie and it is not.
    /// The validity gate catches it without anyone having enumerated the interaction.
    #[test]
    fn a_reduction_that_would_invalidate_a_limit_is_never_offered() {
        use crate::schema::{ColumnRef, Direction, OrderKey};

        let mut case = SqlCase::fixed_example();
        case.query.order_by = vec![OrderKey {
            column: ColumnRef {
                table: "t0".to_string(),
                column: "c0".to_string(),
            },
            direction: Direction::Ascending,
            nulls_first: true,
        }];
        case.query.limit = Some(2);
        assert!(case.validate().is_ok());

        for candidate in case.candidates() {
            // Whatever was dropped, the case still makes sense — in particular no candidate
            // keeps a `LIMIT` while destroying the ordering that justified it.
            assert!(
                candidate.validate().is_ok(),
                "an invalid candidate was offered: {candidate:?}"
            );
        }
    }

    #[test]
    fn complexity_falls_when_anything_is_removed() {
        let case = SqlCase::fixed_example();
        let full = complexity(&case);

        let mut fewer_rows = case.clone();
        fewer_rows.data[0].rows.pop();
        assert!(complexity(&fewer_rows) < full);

        let mut no_filter = case.clone();
        no_filter.query.filter = None;
        assert!(complexity(&no_filter) < full);
    }

    #[test]
    fn bounds_are_respected_by_the_starting_case() {
        // Guards against a shrinker test that silently exercises a different distribution
        // from the one the campaign uses.
        assert_eq!(
            SqlGenerator::default().description(),
            Bounds::V1.description()
        );
    }
}
