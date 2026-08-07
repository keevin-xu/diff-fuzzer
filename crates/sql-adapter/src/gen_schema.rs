//! Generating the *state* a query runs against: tables, and the rows in them.
//!
//! SQLancer's central idea, and the one this domain borrows: **generate the state first,
//! then generate a query against that known state.** A query built with the schema in hand
//! can only reference columns that exist, with types that match, so validity is a property
//! of how the case was built rather than something checked afterwards. Generate-and-reject
//! would instead spend the budget proving that the engines' parsers work.
//!
//! # What uniform sampling would never produce
//!
//! Sampling integers uniformly gives `0` with probability nil, never gives `i64::MIN`, and
//! never repeats a value. Every one of those absences matters here:
//!
//! - **`NULL`** — three-valued logic is where engines disagree most.
//! - **The empty string** — distinct from `NULL`, and easy for either engine to conflate.
//! - **The text `NULL`** — the value that would collide with a real `NULL` under a careless
//!   canonical form. Generating it means our own normalizer is under test too.
//! - **`0`, `1`, `-1`, `i64::MIN`, `i64::MAX`** — boundaries, and the overflow edge.
//! - **Repeated values** — the reason a query can have an `ORDER BY` and still not be
//!   totally ordered. Ties have to occur, or the sort-mode logic is never exercised.
//! - **An empty table** — a query over no rows at all.
//!
//! So values come from a deliberate pool far more often than from uniform sampling. Bugs
//! cluster exactly where random sampling does not go.

use crate::schema::{Column, InsertRows, Literal, SqlType, Table};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// How large a generated case may be.
///
/// **One definition, named in the generator's description**, because a negative case
/// recorded under one set of bounds cannot be scored against findings produced under
/// another — the distributions differ even though both say "generated". The tensor domain
/// learned this when widened bounds silently produced an incomparable pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub max_tables: usize,
    pub max_columns: usize,
    pub max_rows: usize,
    /// Whether the generator may emit a join, including the outer kinds.
    ///
    /// Forces a two-table schema when enabled, since a join needs somewhere to join to.
    pub joins: bool,
    /// Whether set operations may be **chained** — `A UNION B INTERSECT C`, unparenthesized.
    ///
    /// This is the probe for the one difference documented on one side only: SQLite states
    /// that it groups `UNION`/`INTERSECT`/`EXCEPT` left-to-right and that SQL92 disagrees
    /// (`SPECS.md` §1.4, §4.11); DuckDB documents nothing (§5.9, two failed retrievals).
    /// Generating the construct is the only remaining way to learn what DuckDB does.
    pub chained_set_ops: bool,
    /// Whether the generator may emit a set operation (`UNION`, `UNION ALL`, `INTERSECT`,
    /// `EXCEPT`).
    ///
    /// The second widening, aimed where the two engines share least code and where `NULL`
    /// changes its own rules: set operations treat two `NULL`s as *the same value* for
    /// deduplication, unlike `=` everywhere else in SQL.
    pub set_ops: bool,
    /// Whether the generator may emit aggregates and `GROUP BY`.
    ///
    /// The first widening past the v1 subset, and pointed where DuckDB plausibly differs
    /// from SQLite: it is an analytics engine, and grouping is its home ground. `NULL`
    /// handling inside aggregates is the classic place engines part company — `COUNT(x)`
    /// skips `NULL`s while `COUNT(*)` does not, and an aggregate over *no* rows has to
    /// decide what to return.
    pub aggregates: bool,
    /// Whether arithmetic may overflow.
    ///
    /// `false` (v1) bounds arithmetic to small literals so a result cannot leave any
    /// integer range. `true` restores column and pool operands, which reaches
    /// `i64::MAX + 1` — where the two engines make **incompatible documented promises**:
    /// SQLite falls back to floating-point arithmetic, DuckDB raises (`SPECS.md` §4.9).
    ///
    /// It is a knob rather than a constant because the question "does widening this find
    /// anything?" is answered by measuring both settings, not by choosing one. Note it is
    /// part of [`Bounds::description`], so cases drawn under the two settings can never be
    /// mistaken for one distribution.
    pub wide_arithmetic: bool,
    /// How deeply expressions may nest.
    ///
    /// Bounds *size*, which bounds runtime and how painful minimization is. Note that
    /// bounding each knob does not bound the case: depth is exponential in node count, so
    /// this number is small on purpose.
    pub max_expr_depth: usize,
}

impl Bounds {
    /// The v1 bounds. Small on purpose: the pipeline has to be trustworthy before breadth
    /// is worth anything, and every bound this project has chosen by intuition was wrong —
    /// so these are a starting point to be *measured* at S2.8 and S5, not a considered
    /// answer.
    pub const V1: Bounds = Bounds {
        max_tables: 2,
        max_columns: 4,
        max_rows: 8,
        max_expr_depth: 3,
        wide_arithmetic: false,
        aggregates: false,
        set_ops: false,
        chained_set_ops: false,
        joins: false,
    };

    /// V1 plus joins. One axis, as always.
    pub const V1_JOINS: Bounds = Bounds {
        joins: true,
        ..Bounds::V1
    };

    /// **Every axis at once** — the configuration a campaign runs.
    ///
    /// Each axis was measured alone, which is right for attributing a difference in yield to
    /// one change. But a per-axis sweep tests no **interaction**, and interactions are where
    /// engines classically part company: an aggregate over an outer-joined table, where the
    /// `NULL`s being counted were manufactured by the join rather than stored; a set operation
    /// over grouped queries. Nothing so far has generated any of those.
    ///
    /// **Two axes stay off, and for the same reason: their divergences are already understood.**
    ///
    /// - `wide_arithmetic` — integer overflow, catalogued in `known.rs` with citations.
    /// - `chained_set_ops` — set-operation precedence (`SPECS.md` §4.11). *Not* catalogued,
    ///   because DuckDB's side is measured rather than documented, so nothing filters it.
    ///
    /// Leaving either on floods a campaign with a difference nobody needs to see again.
    /// Measured, not assumed: a first attempt at this configuration with `chained_set_ops`
    /// enabled produced **494 findings in under two minutes, every one of them the precedence
    /// mechanism** — while each one paid the cost of minimization. An unknown finding would
    /// have been one line among hundreds of identical ones.
    ///
    /// The general rule, which cost two attempts to learn: **a campaign's configuration
    /// excludes the axes whose answers are known.** Widening exists to find new mechanisms;
    /// once a mechanism is understood, generating more of it buys nothing and hides the rest.
    /// Both axes remain available for deliberate use.
    pub const V1_ALL: Bounds = Bounds {
        aggregates: true,
        set_ops: true,
        chained_set_ops: false,
        joins: true,
        wide_arithmetic: false,
        ..Bounds::V1
    };

    /// V1 plus set operations. One axis, as always.
    pub const V1_SET_OPS: Bounds = Bounds {
        set_ops: true,
        ..Bounds::V1
    };

    /// V1 plus **chained** set operations, where precedence becomes observable.
    pub const V1_CHAINED_SET_OPS: Bounds = Bounds {
        set_ops: true,
        chained_set_ops: true,
        ..Bounds::V1
    };

    /// V1 plus aggregates and `GROUP BY`. One axis, as always, so a difference in yield is
    /// attributable to this change and nothing else.
    pub const V1_AGGREGATES: Bounds = Bounds {
        aggregates: true,
        ..Bounds::V1
    };

    /// V1 with overflow reachable. Everything else identical, so a difference in yield
    /// between the two is attributable to this one axis — the tensor domain's rule that a
    /// sweep must vary one thing at a time.
    pub const V1_WIDE_ARITHMETIC: Bounds = Bounds {
        wide_arithmetic: true,
        ..Bounds::V1
    };

    /// A description that names the parameters, for recording alongside a case.
    ///
    /// Compared as an exact string when deciding whether two pools are comparable, so it
    /// must change whenever the numbers do. The test below fails if they drift apart.
    pub fn description(&self) -> String {
        format!(
            "sql-v1(tables<={}, columns<={}, rows<={}, depth<={}, wide-arith={}, aggregates={}, set-ops={}, chained={}, joins={})",
            self.max_tables,
            self.max_columns,
            self.max_rows,
            self.max_expr_depth,
            self.wide_arithmetic,
            self.aggregates,
            self.set_ops,
            self.chained_set_ops,
            self.joins
        )
    }
}

/// Integer values worth trying, **for a 64-bit `BIGINT` column**.
const INTERESTING_BIGINTS: [i64; 7] = [0, 1, -1, 2, -2, i64::MAX, i64::MIN];

/// Integer values worth trying, **for a 32-bit `INTEGER` column**.
///
/// # The engines do not agree on what `INTEGER` means
///
/// DuckDB's `INTEGER` is `INT4` — four bytes, range −2³¹ to 2³¹−1. SQLite's `INTEGER` is a
/// storage class of variable width, "stored in 0, 1, 2, 3, 4, 6, or 8 bytes depending on the
/// magnitude of the value" — so it swallows a 64-bit value happily.
///
/// This was found by *running*: a generated case put `i64::MIN` in an `INTEGER` column,
/// SQLite accepted it, and DuckDB refused the `INSERT` with a conversion error. The case
/// was then skipped rather than judged, which is the correct outcome and a useless one —
/// a case neither engine judged teaches nothing.
///
/// The fix is correct-by-construction rather than a catalog entry: a literal is drawn from
/// **its declared column's range**, so an `INTEGER` column never sees a value that only one
/// engine can store. Cited in `SPECS.md` §2.1, §3.4 and §4.4.
const INTERESTING_INTS: [i64; 7] = [0, 1, -1, 2, -2, i32::MAX as i64, i32::MIN as i64];

/// Text values worth trying.
///
/// `"NULL"` is here on purpose: it is the string that would be indistinguishable from a
/// real `NULL` under a canonical form that renders both bare. Generating it keeps our own
/// normalizer honest. `"'"` is here because a value containing a quote is what a careless
/// renderer turns into broken SQL.
const INTERESTING_TEXT: [&str; 6] = ["", "NULL", "'", "a", "A", "  "];

/// How often a nullable cell is actually `NULL`, in percent.
///
/// High — far higher than uniform sampling would give — because `NULL` propagation through
/// comparisons is the single richest source of engine disagreement in SQL.
const NULL_PERCENT: u32 = 25;

/// Build a schema: one or two tables, each with a few typed columns.
///
/// Names are positional (`t0`, `c0`) rather than random. A random identifier would add
/// noise to every minimized repro without testing anything — engines do not care what a
/// column is called, and a human reading a finding does.
pub fn generate_schema(rng: &mut SeededRng, bounds: Bounds) -> Vec<Table> {
    // A join needs somewhere to join to, so the schema is forced to two tables when the axis
    // is on. Without this the generator would produce single-table schemas most of the time
    // and quietly test joins far less than the run appears to — the same shape of confound
    // that made the first set-op sweep meaningless.
    let table_count = if bounds.joins {
        // Exactly two. A join needs somewhere to join to, and v1 caps the schema at two
        // anyway — so this is not a clamp, it is the only value that makes the axis mean
        // anything.
        debug_assert!(
            bounds.max_tables >= 2,
            "the joins axis needs a schema of at least two tables"
        );
        2
    } else {
        rng.random_range(1..=bounds.max_tables)
    };

    (0..table_count)
        .map(|table_index| {
            let column_count = rng.random_range(1..=bounds.max_columns);
            let columns = (0..column_count)
                .map(|column_index| Column {
                    name: format!("c{column_index}"),
                    // Only the three generated types — `DECIMAL` and `BOOLEAN` are defined
                    // in the AST but kept out of cases until their cross-engine behaviour
                    // is retrieved rather than recalled (`SPECS.md` §4.1–4.2).
                    sql_type: SqlType::GENERATED[rng.random_range(0..SqlType::GENERATED.len())],
                })
                .collect();

            Table {
                name: format!("t{table_index}"),
                columns,
            }
        })
        .collect()
}

/// Fill each table with rows, honouring its column types.
///
/// Row counts start at **zero**: an empty table is a real case, and one whose absence from
/// a corpus would go unnoticed. Aggregates over nothing, joins against nothing, and
/// `WHERE` clauses that match nothing all begin here.
pub fn generate_data(rng: &mut SeededRng, tables: &[Table], bounds: Bounds) -> Vec<InsertRows> {
    tables
        .iter()
        .map(|table| {
            let row_count = rng.random_range(0..=bounds.max_rows);
            let rows = (0..row_count)
                .map(|_| {
                    table
                        .columns
                        .iter()
                        .map(|column| generate_literal(rng, column.sql_type))
                        .collect()
                })
                .collect();

            InsertRows {
                table: table.name.clone(),
                rows,
            }
        })
        .collect()
}

/// One value of the given type — `NULL` a quarter of the time, otherwise usually a value
/// from the interesting pool.
///
/// Drawing from a small pool has a second effect worth naming: **values repeat**, which is
/// what creates ties. A generator whose values were all distinct would never produce a case
/// where `ORDER BY c0` fails to order the rows totally, and the sort-mode decision would go
/// permanently untested.
pub fn generate_literal(rng: &mut SeededRng, sql_type: SqlType) -> Literal {
    if rng.random_range(0..100) < NULL_PERCENT {
        return Literal::Null;
    }

    match sql_type {
        // The pool differs by declared width, because the engines do not agree on what
        // `INTEGER` holds — see [`INTERESTING_INTS`]. Drawing from the column's own range is
        // what keeps a case judgeable by both engines instead of skipped by one.
        SqlType::Integer | SqlType::BigInt => {
            let pool: &[i64] = if sql_type == SqlType::Integer {
                &INTERESTING_INTS
            } else {
                &INTERESTING_BIGINTS
            };

            // Mostly from the pool, sometimes an arbitrary value — the pool finds edges,
            // arbitrary values find everything the pool's author did not think of.
            if rng.random_range(0..100) < 75 {
                Literal::Integer(pool[rng.random_range(0..pool.len())])
            } else {
                Literal::Integer(rng.random_range(-1000..=1000))
            }
        }
        SqlType::Text => {
            Literal::Text(INTERESTING_TEXT[rng.random_range(0..INTERESTING_TEXT.len())].to_string())
        }
        // Unreachable by construction: `generate_schema` only emits generated types, and
        // this arm exists so that adding a type to `SqlType::GENERATED` without teaching
        // this function about it fails loudly here rather than silently producing nothing.
        SqlType::Decimal | SqlType::Boolean => {
            unreachable!("{sql_type:?} is not generated in v1; see SqlType::GENERATED")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_and_data(seed: u64) -> (Vec<Table>, Vec<InsertRows>) {
        let mut rng = SeededRng::from_seed(seed);
        let tables = generate_schema(&mut rng, Bounds::V1);
        let data = generate_data(&mut rng, &tables, Bounds::V1);
        (tables, data)
    }

    #[test]
    fn the_description_names_the_actual_bounds() {
        // The guard against the tensor domain's silent-incomparability bug: if the numbers
        // change and the description does not, two differently-drawn pools would both
        // claim to be the same distribution.
        let description = Bounds::V1.description();
        assert!(description.contains(&Bounds::V1.max_tables.to_string()));
        assert!(description.contains(&Bounds::V1.max_columns.to_string()));
        assert!(description.contains(&Bounds::V1.max_rows.to_string()));
        assert!(description.contains(&Bounds::V1.max_expr_depth.to_string()));
        // The overflow axis must be visible in the description too, or two pools drawn
        // under different settings would claim to be the same distribution.
        assert_ne!(
            Bounds::V1.description(),
            Bounds::V1_WIDE_ARITHMETIC.description()
        );
        assert_ne!(
            Bounds::V1.description(),
            Bounds::V1_AGGREGATES.description()
        );

        let wider = Bounds {
            max_rows: Bounds::V1.max_rows + 1,
            ..Bounds::V1
        };
        assert_ne!(wider.description(), Bounds::V1.description());
    }

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(schema_and_data(12345), schema_and_data(12345));
    }

    #[test]
    fn different_seeds_give_different_cases() {
        // Weak by design — two seeds *could* coincide — but across this many it would take
        // a broken generator. This is the test S1 could not have.
        // Compared as JSON rather than by hashing, so the AST types are not forced to
        // derive `Hash` for the sake of one test.
        let distinct = (0..20)
            .map(|seed| serde_json::to_string(&schema_and_data(seed)).expect("serializes"))
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(
            distinct > 10,
            "only {distinct} distinct cases from 20 seeds"
        );
    }

    #[test]
    fn everything_stays_inside_the_bounds() {
        for seed in 0..200 {
            let (tables, data) = schema_and_data(seed);

            assert!(!tables.is_empty() && tables.len() <= Bounds::V1.max_tables);
            for table in &tables {
                assert!(!table.columns.is_empty());
                assert!(table.columns.len() <= Bounds::V1.max_columns);
            }
            for insert in &data {
                assert!(insert.rows.len() <= Bounds::V1.max_rows);
            }
        }
    }

    #[test]
    fn only_generated_types_appear() {
        for seed in 0..200 {
            let (tables, _) = schema_and_data(seed);
            for table in &tables {
                for column in &table.columns {
                    assert!(
                        column.sql_type.is_generated(),
                        "{:?} must not be generated in v1",
                        column.sql_type
                    );
                }
            }
        }
    }

    #[test]
    fn every_row_matches_its_tables_column_types() {
        // Name resolution's first half: the data must fit the schema positionally and by
        // type, or the case is invalid before any query is even written.
        for seed in 0..200 {
            let (tables, data) = schema_and_data(seed);

            for insert in &data {
                let table = tables
                    .iter()
                    .find(|table| table.name == insert.table)
                    .expect("data references a table that exists");

                for row in &insert.rows {
                    assert_eq!(
                        row.len(),
                        table.columns.len(),
                        "row width matches the table"
                    );
                    for (value, column) in row.iter().zip(table.columns.iter()) {
                        match value.sql_type() {
                            // NULL fits any column.
                            None => {}
                            Some(value_type) => assert!(
                                column.sql_type.accepts(value_type),
                                "{value:?} does not fit a {:?} column",
                                column.sql_type
                            ),
                        }
                    }
                }
            }
        }
    }

    /// The awkward values have to actually occur, or generating them is a claim rather than
    /// a fact. Each of these is a case the pipeline is specifically supposed to handle.
    #[test]
    fn the_values_that_uniform_sampling_would_miss_do_occur() {
        let mut saw_null = false;
        let mut saw_empty_text = false;
        let mut saw_the_text_null = false;
        let mut saw_quote = false;
        let mut saw_extreme_integer = false;
        let mut saw_empty_table = false;
        let mut saw_a_tie = false;

        for seed in 0..300 {
            let (_, data) = schema_and_data(seed);
            for insert in &data {
                if insert.rows.is_empty() {
                    saw_empty_table = true;
                }

                // A tie is two rows sharing a value in the same column — the reason an
                // `ORDER BY` can exist and still not order the rows totally.
                for column_index in 0..insert.rows.first().map_or(0, Vec::len) {
                    let column: Vec<_> = insert
                        .rows
                        .iter()
                        .filter_map(|row| row.get(column_index))
                        .collect();
                    for (position, value) in column.iter().enumerate() {
                        if column[position + 1..].contains(value) {
                            saw_a_tie = true;
                        }
                    }
                }

                for value in insert.rows.iter().flatten() {
                    match value {
                        Literal::Null => saw_null = true,
                        Literal::Text(text) if text.is_empty() => saw_empty_text = true,
                        Literal::Text(text) if text == "NULL" => saw_the_text_null = true,
                        Literal::Text(text) if text == "'" => saw_quote = true,
                        // Both widths' boundaries count: a 32-bit column's extremes are as
                        // much an edge as a 64-bit column's, and after the INTEGER/BIGINT
                        // discovery they are what an `INTEGER` column actually receives.
                        Literal::Integer(number)
                            if [i64::MAX, i64::MIN, i32::MAX as i64, i32::MIN as i64]
                                .contains(number) =>
                        {
                            saw_extreme_integer = true;
                        }
                        _ => {}
                    }
                }
            }
        }

        assert!(saw_null, "NULL must occur");
        assert!(saw_empty_text, "the empty string must occur");
        assert!(
            saw_the_text_null,
            "the text 'NULL' must occur — it is what tests our own canonical form"
        );
        assert!(
            saw_quote,
            "a quote must occur — it is what tests the renderer"
        );
        assert!(saw_extreme_integer, "integer boundaries must occur");
        assert!(saw_empty_table, "an empty table must occur");
        assert!(
            saw_a_tie,
            "repeated values must occur, or sort modes go untested"
        );
    }
}
