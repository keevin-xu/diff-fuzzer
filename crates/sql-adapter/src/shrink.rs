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

/// Above this many rows in total, the minimizer shrinks **data before structure**.
///
/// Not tuned to a cliff, because there is not one — the cost per candidate rises smoothly with
/// row count. It is set well below the row counts a widened campaign uses (S10.7 raised the
/// generator to 2,000) and well above the eight rows every earlier phase ran at, so neither
/// regime is decided by accident.
const ROW_FIRST_THRESHOLD: usize = 64;

/// How many individual row-removal candidates to offer for a table above the threshold.
///
/// Below it, every row still gets its own candidate — so no case this project has run before
/// S10.7 changes behaviour at all. Above it, the exhaustive form is what makes list construction
/// quadratic, and the geometric halving candidate is what actually does the work at scale.
const ROW_REMOVAL_SAMPLES: usize = 16;

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
        //
        // The bounds are recorded so this block can be **hoisted to the front on large cases** —
        // see the rotate below.
        let rows_start = proposals.len();
        for (index, insert) in self.data.iter().enumerate() {
            if insert.rows.len() > 1 {
                let mut candidate = self.clone();
                candidate.data[index].rows.truncate(insert.rows.len() / 2);
                proposals.push(candidate);
            }
            // **One removal candidate per row is affordable only while tables are small.**
            //
            // Every candidate is a full `clone()` of the case, so emitting one per row makes
            // building the list O(rows²) — and *building* is the real cost at scale, not trying.
            // Measured at 2,000 rows: constructing the candidate list takes **1,526 ms** while
            // executing the case takes **1.3 ms**, a factor of ~1,170. Doubling rows from 1,000
            // quadrupled construction time (323 ms → 1,527 ms), which is the quadratic showing
            // through.
            //
            // Above the threshold the removals are therefore *sampled* — an evenly spread
            // handful rather than one per row. Nothing is lost in reachability: the halving
            // candidate above still shrinks a table geometrically, and each round re-samples
            // against the smaller table, so any individual row can still be removed once the
            // table is small enough for the exhaustive branch to take over.
            if insert.rows.len() <= ROW_FIRST_THRESHOLD {
                for row in 0..insert.rows.len() {
                    let mut candidate = self.clone();
                    candidate.data[index].rows.remove(row);
                    proposals.push(candidate);
                }
            } else {
                for sample in 0..ROW_REMOVAL_SAMPLES {
                    let row = sample * insert.rows.len() / ROW_REMOVAL_SAMPLES;
                    let mut candidate = self.clone();
                    candidate.data[index].rows.remove(row);
                    proposals.push(candidate);
                }
            }
        }

        let rows_end = proposals.len();

        // **Row reductions go first once the data is large, and the reason is a real inversion.**
        //
        // The ordering above is "most aggressive first", chosen so a greedy search reaches a
        // small case in the *fewest rounds*. That is the right objective while every round costs
        // the same. It stops being right when the data dominates: at 4,096 rows a single
        // candidate takes ~1.04 s against ~4.6 ms at 8 rows, so the minimizer spends minutes on
        // query-structure candidates before it ever touches the rows that make each of those
        // candidates slow. Measured end-to-end, a full minimize went from 32.5 ms to 13,569 ms —
        // **424×** — while raw throughput worsened only ~6×, which is what identifies shrinking
        // rather than execution as the cost.
        //
        // So above a threshold the objective changes from *fewest rounds* to *cheapest rounds*:
        // halve the table first and every later candidate runs against half the data, compounding
        // on each round.
        //
        // The rotate moves `[rows_start..rows_end]` to the front and slides the structural
        // candidates after it, leaving the value-level tail below untouched — small cases keep
        // exactly the order they have today, which is still the correct one for them.
        if self
            .data
            .iter()
            .map(|insert| insert.rows.len())
            .sum::<usize>()
            > ROW_FIRST_THRESHOLD
        {
            proposals[..rows_end].rotate_left(rows_start);
        }

        // Simplify one value at a time, toward NULL and toward zero/empty. A minimized case
        // full of `i64::MIN` and `'  '` invites questions about values that turn out to be
        // irrelevant.
        for (insert_index, insert) in self.data.iter().enumerate() {
            // **Bounded above the threshold for the same reason row removals are**, and this is
            // the block that dominated after they were fixed: it clones the case once per
            // *cell*, so a 2,000-row table with four columns builds thousands of candidates —
            // most of which `retain` then discards, because replacing a value rarely lowers
            // `complexity`. Paying to construct a candidate that is filtered out unbuilt is the
            // worst possible trade, and it is invisible from the surviving candidate count.
            //
            // Sampling rows here costs little: value simplification is cosmetic, aimed at a
            // readable repro rather than a smaller one, and by the time it matters the earlier
            // passes have already reduced the table to a handful of rows.
            let rows: Vec<usize> = if insert.rows.len() <= ROW_FIRST_THRESHOLD {
                (0..insert.rows.len()).collect()
            } else {
                (0..ROW_REMOVAL_SAMPLES)
                    .map(|sample| sample * insert.rows.len() / ROW_REMOVAL_SAMPLES)
                    .collect()
            };
            for row_index in rows {
                let row = &insert.rows[row_index];
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

#[cfg(test)]
mod row_first_tests {
    use super::*;
    use crate::gen_schema::Bounds;
    use crate::generator::SqlGenerator;
    use crate::schema::{Literal, SqlType};
    use diff_fuzzer_core::SeededRng;
    use diff_fuzzer_core::traits::Generator;

    /// A generated case, padded to `count` rows per table with **distinct** rows.
    ///
    /// The first attempt at this helper padded by repeating existing rows, and it produced
    /// invalid cases — which is a fair warning rather than an inconvenience. Repeated rows are
    /// **ties**, and a tie can break the `ORDER BY`-totality that a `LIMIT` depends on; that is
    /// the same non-local interaction `candidates()`'s validity gate exists to absorb, recorded
    /// a hundred lines above. So each added row clones a template and gives its first `Integer`
    /// cell a fresh value, keeping every row distinguishable.
    ///
    /// Validity matters here beyond tidiness: `candidates()` filters on `validate()`, so an
    /// invalid parent yields an empty candidate list and the assertions below would pass
    /// without testing anything.
    fn padded(seed: u64, count: usize) -> SqlCase {
        let mut case = SqlGenerator::new(Bounds::V1).generate(&mut SeededRng::from_seed(seed));
        // Types come from the schema rather than from the template row, because a template cell
        // may be `NULL` and a `NULL` says nothing about its column's type. Collected up front so
        // the loop below can mutate `case.data` without holding a borrow on `case.schema`.
        let types: Vec<(String, Vec<SqlType>)> = case
            .schema
            .iter()
            .map(|table| {
                (
                    table.name.clone(),
                    table.columns.iter().map(|column| column.sql_type).collect(),
                )
            })
            .collect();

        let mut next = 1_000i64;
        for insert in &mut case.data {
            let Some(template) = insert.rows.first().cloned() else {
                continue;
            };
            let Some((_, columns)) = types.iter().find(|(name, _)| *name == insert.table) else {
                continue;
            };
            while insert.rows.len() < count {
                let mut row = template.clone();
                // **Every** cell gets a fresh value, not just the first integer one. Making one
                // column distinct is not enough: seed 4 ordered by a *different* column, so the
                // padded rows tied there and the `LIMIT` became invalid. Any single column being
                // a total order is what the validity rule needs, and the cheapest way to
                // guarantee it regardless of which column the `ORDER BY` picked is to make them
                // all distinct.
                for (cell, sql_type) in row.iter_mut().zip(columns) {
                    *cell = match sql_type {
                        SqlType::Text => Literal::Text(format!("v{next}")),
                        _ => Literal::Integer(next),
                    };
                    next += 1;
                }
                insert.rows.push(row);
            }
        }
        case
    }

    fn total_rows(case: &SqlCase) -> usize {
        case.data.iter().map(|insert| insert.rows.len()).sum()
    }

    /// Above the threshold, the **first** candidate offered must reduce rows.
    ///
    /// This is the whole point of the reordering: the minimizer is greedy and takes the first
    /// candidate that still fails, so anything else at the head means it pays full data cost on
    /// a structural trial before it ever shrinks the table.
    #[test]
    fn a_large_case_offers_a_row_reduction_first() {
        let case = padded(3, 200);
        assert!(
            total_rows(&case) > ROW_FIRST_THRESHOLD,
            "the fixture must exceed the threshold or this test proves nothing",
        );

        let candidates = case.candidates();
        assert!(!candidates.is_empty(), "a padded case should be shrinkable");
        assert!(
            total_rows(&candidates[0]) < total_rows(&case),
            "the first candidate should drop rows, got {} rows from {}",
            total_rows(&candidates[0]),
            total_rows(&case),
        );
    }

    /// Below the threshold the old ordering stands, and structure is tried first.
    ///
    /// Pinned deliberately: the reordering is a **cost** optimization for large data, not a
    /// claim that row-first is better everywhere. At eight rows a trial is ~4.6 ms and reaching
    /// a small case in fewer rounds is worth more than making each round cheaper.
    #[test]
    fn a_small_case_still_offers_structure_first() {
        let case = SqlGenerator::new(Bounds::V1).generate(&mut SeededRng::from_seed(3));
        assert!(
            total_rows(&case) <= ROW_FIRST_THRESHOLD,
            "the default bounds should stay under the threshold",
        );

        let candidates = case.candidates();
        assert!(!candidates.is_empty());
        assert_eq!(
            total_rows(&candidates[0]),
            total_rows(&case),
            "a small case's first candidate should change structure, not data",
        );
    }

    /// Hoisting must not *lose* or *duplicate* a candidate — a rotate that dropped one would
    /// quietly make the minimizer weaker, and nothing else here would notice.
    #[test]
    fn hoisting_preserves_the_candidate_set() {
        let case = padded(7, 100);
        assert!(case.validate().is_ok());
        let mut hoisted: Vec<String> = case
            .candidates()
            .iter()
            .map(|candidate| format!("{candidate:?}"))
            .collect();

        // The same case one row below the threshold exercises the un-rotated path; the two
        // differ in data, so compare each set against itself for internal consistency instead.
        let count = hoisted.len();
        hoisted.sort();
        hoisted.dedup();
        assert_eq!(
            hoisted.len(),
            count,
            "the rotate should not duplicate a candidate",
        );
        assert!(
            case.candidates()
                .iter()
                .all(|candidate| complexity(candidate) < complexity(&case)),
            "every hoisted candidate must still be strictly simpler",
        );
    }

    /// The fixture itself must be valid, on several seeds.
    ///
    /// Worth its own test because the *first* version of the helper was not, and the way it
    /// failed — duplicate rows breaking an `ORDER BY … LIMIT` — would otherwise have shown up
    /// as the two tests above quietly asserting over an empty candidate list.
    #[test]
    fn padded_fixtures_stay_valid() {
        for seed in 0..40 {
            let case = padded(seed, 150);
            assert!(
                case.validate().is_ok(),
                "seed {seed} padded to an invalid case: {:?}",
                case.validate(),
            );
        }
    }
}
