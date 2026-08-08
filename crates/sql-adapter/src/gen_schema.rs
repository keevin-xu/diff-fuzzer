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
use diff_fuzzer_core::GenerationAxes;
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// FNV-1a over a byte slice, continuing from `hash`.
///
/// A **`const fn`**: it runs at *compile* time, so the value below costs nothing at runtime.
/// `const fn` may use loops (since Rust 1.46) but not iterators, which is why this is a `while`
/// over an index rather than a `for` over `.iter()`.
///
/// FNV-1a is chosen for being short enough to read in one sitting. It is **not** cryptographic
/// and does not need to be: this detects *accidental* drift, not an adversary.
const fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

/// A fingerprint of the **generation logic itself**, computed at compile time from the source.
///
/// # The problem this solves
///
/// [`Bounds::description`] names every *bound*, which is what a reader would think identifies a
/// corpus. It does not. A change to the generation *logic* that leaves every bound alone — the
/// set-op ordering fix, the grouped-ordering change, making joins probabilistic — produces a
/// completely different corpus under a **byte-identical description**. `Pool::matched` compares
/// that string exactly, so it would accept two incomparable pools as the same one and quietly
/// mix them. Three such logic changes have already happened in this project.
///
/// # Why this is derived rather than a number someone bumps
///
/// A hand-maintained `GENERATOR_VERSION` would work only if it were always remembered — and
/// "someone will remember" is precisely the discipline that failed and created the problem.
/// `include_bytes!` embeds the source of the generation modules at compile time, so the
/// fingerprint changes **whenever they change**, with nothing to forget.
///
/// # The cost, stated honestly
///
/// It is **over-sensitive**: reformatting or editing a comment in these files changes the
/// fingerprint and so invalidates pools that are in fact still comparable. That is the direction
/// to err in. A spurious mismatch costs a re-run and is visible; a missed one silently corrupts
/// a measurement and is not.
pub const GENERATOR_FINGERPRINT: u32 = {
    // Every module that decides what a case looks like. A module added here later must be
    // added to this list too — the one thing this scheme still asks a human to remember.
    let hash = fnv1a(include_bytes!("gen_schema.rs"), 0xcbf2_9ce4_8422_2325);
    let hash = fnv1a(include_bytes!("gen_query.rs"), hash);
    let hash = fnv1a(include_bytes!("generator.rs"), hash);
    // Fold the 64-bit hash into 32 bits so the description stays short; collision risk is
    // irrelevant for detecting accidental drift.
    (hash ^ (hash >> 32)) as u32
};

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
    /// Whether the generator may emit **correlated subqueries** — `EXISTS (SELECT ... WHERE
    /// inner.c = outer.c)` and comparisons against a scalar subquery.
    ///
    /// Historically the highest-yield SQL construct, and the reason is structural: a
    /// correlated subquery is re-evaluated per outer row, so it exercises the optimizer's
    /// decisions about *when* to evaluate what. Two engines that agree on every value can
    /// still disagree here if one rewrites the subquery into a join and the other does not.
    ///
    /// Needs two tables, like joins, so it forces the same two-table schema.
    pub subqueries: bool,
    /// Whether the generator may emit `x IN (SELECT ...)` / `x NOT IN (SELECT ...)`.
    ///
    /// **Aimed at one specific bug**, unlike the other axes, which widen the surface
    /// generally. `NOT IN` against a subquery column containing a `NULL` is UNKNOWN for every
    /// row, so the query correctly returns nothing — and engines get this wrong *in the same
    /// direction*, by treating the UNKNOWN as FALSE and returning rows that should be
    /// excluded. A shared wrong answer is invisible to the differential oracle, which is why
    /// this axis exists to be pointed at the **metamorphic** one.
    ///
    /// Needs two tables, like joins and correlated subqueries.
    pub not_in: bool,
    /// Whether the generator may emit `x IN (1, 2, NULL)` / `x NOT IN (…)` over a **literal
    /// list**.
    ///
    /// Separate from [`Bounds::not_in`] because it reaches different engine code for the same
    /// logic. A subquery must be executed; a literal list can be **constant-folded** at plan
    /// time and rewritten — to a chain of `OR`s, a hash probe, a precomputed set — and each
    /// rewrite has to preserve `NULL` semantics on its own. The subquery form came back clean
    /// over 30,000 cases on both oracles, which is precisely why the folded path is worth its
    /// own axis rather than being assumed to behave the same.
    ///
    /// **Needs no second table**, unlike `not_in` — a list is self-contained. So this axis is
    /// reachable in single-table schemas where the subquery form is not.
    pub not_in_list: bool,
    /// Whether the generator may emit `SELECT DISTINCT`.
    ///
    /// **Deduplication is where `NULL` stops behaving like `NULL`.** Everywhere else in SQL
    /// `NULL = NULL` is UNKNOWN, so two `NULL`s are not equal — but `DISTINCT` collapses them
    /// to one row, treating them as the same value. An engine cannot get this right by reusing
    /// its equality; it has to special-case it, and a special case is somewhere to be wrong.
    /// The set operations make the same exception, and this applies it within a single query.
    ///
    /// **Constrained at generation:** with `DISTINCT`, an `ORDER BY` key must appear in the
    /// projection or DuckDB refuses the query. So `DISTINCT` is only applied to queries whose
    /// ordering keys are already projected — which keeps the axis *additive*, rather than
    /// suppressing ordering the way three earlier widenings did.
    pub distinct: bool,
    /// Whether the generator may emit `HAVING` — a filter on **groups**, after aggregation.
    ///
    /// Three-valued logic moved one level up: `HAVING SUM(x) > 0` on a group whose `SUM` is
    /// `NULL` is UNKNOWN, and the group disappears. The same trap as a `WHERE` on a `NULL`, but
    /// on a value the engine *computed* rather than one that was stored — so it exercises the
    /// aggregation path's `NULL` handling rather than the comparison's.
    ///
    /// Requires aggregation, so it only attaches to grouped or whole-table-aggregate queries.
    pub having: bool,
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
        subqueries: false,
        not_in: false,
        not_in_list: false,
        distinct: false,
        having: false,
    };

    /// V1 plus correlated subqueries. One axis, as always.
    pub const V1_SUBQUERIES: Bounds = Bounds {
        subqueries: true,
        ..Bounds::V1
    };

    /// V1 plus `IN`/`NOT IN` subqueries. One axis, as always — so its yield can be measured
    /// against the others rather than confounded with them.
    pub const V1_NOT_IN: Bounds = Bounds {
        not_in: true,
        ..Bounds::V1
    };

    /// V1 plus `IN`/`NOT IN` over a literal list. One axis, as always.
    pub const V1_NOT_IN_LIST: Bounds = Bounds {
        not_in_list: true,
        ..Bounds::V1
    };

    /// V1 plus `SELECT DISTINCT`. One axis, as always.
    pub const V1_DISTINCT: Bounds = Bounds {
        distinct: true,
        ..Bounds::V1
    };

    /// V1 plus `HAVING`. Needs aggregates to attach to, so this preset enables both — the one
    /// axis in this crate that cannot be varied entirely alone, and it is noted rather than
    /// hidden: yield here is `having` **given** aggregates, measured against `V1_AGGREGATES`.
    pub const V1_HAVING: Bounds = Bounds {
        aggregates: true,
        having: true,
        ..Bounds::V1
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
        subqueries: true,
        not_in: true,
        not_in_list: true,
        distinct: true,
        having: true,
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
}

/// The engine's cross-domain view of a generator configuration.
///
/// Adopted 2026-08-07 at the tensor domain's request, after both domains built the same
/// mechanism independently. **The derived `description()` replaces the hand-written format
/// string, and that is the point:** the old one had to be edited by hand for every new axis,
/// and adding `not_in` required remembering to add `not-in={}` to it. Forgetting would have
/// produced two configurations sharing one identity — exactly what the trait prevents.
///
/// # The logic fingerprint goes through `logic_version()`
///
/// The trait derives identity from **declared** axes and scalars. That catches a configuration
/// changing without its description changing — but only when the *configuration* changed. It
/// does not catch the generation **logic** changing while every axis stays put, and this domain
/// has had three of those (see [`GENERATOR_FINGERPRINT`]).
///
/// The sharpest example is the trait's own: `axes.rs` cites this adapter's joins-versus-ordering
/// finding, and the **fix** for it was making joins probabilistic at 60% rather than
/// unconditional. That changed no axis and no scalar, so a description derived from declared
/// configuration alone gives the corpus before that fix and the corpus after it byte-identical
/// identities.
///
/// This was first worked around by reporting the fingerprint as a scalar, which put a source
/// hash in the slot meant for bounds. The engine added `logic_version()` in response
/// (2026-08-07), and the fingerprint now goes there. **The rendered description is unchanged** —
/// `logic=<hex>` was already last — so this is a change of provenance, not of identity, and no
/// corpus is invalidated by it.
impl GenerationAxes for Bounds {
    /// Every axis, **including the disabled ones**, in a fixed order.
    fn axes(&self) -> Vec<(&'static str, bool)> {
        vec![
            ("wide-arith", self.wide_arithmetic),
            ("aggregates", self.aggregates),
            ("set-ops", self.set_ops),
            ("chained", self.chained_set_ops),
            ("joins", self.joins),
            ("subqueries", self.subqueries),
            ("not-in", self.not_in),
            ("not-in-list", self.not_in_list),
            ("distinct", self.distinct),
            ("having", self.having),
        ]
    }

    /// The size bounds. The logic fingerprint is **not** here — see `logic_version` below.
    fn scalars(&self) -> Vec<(&'static str, String)> {
        vec![
            ("tables", self.max_tables.to_string()),
            ("columns", self.max_columns.to_string()),
            ("rows", self.max_rows.to_string()),
            ("depth", self.max_expr_depth.to_string()),
        ]
    }

    /// The generation-logic fingerprint — the half of drift the axes cannot see.
    ///
    /// Answering `Some` rather than taking the `None` default is deliberate: `None` claims that
    /// generation logic never changes in ways that matter, and this domain has falsified that
    /// three times.
    fn logic_version(&self) -> Option<String> {
        Some(format!("{GENERATOR_FINGERPRINT:08x}"))
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
    let table_count = if bounds.joins || bounds.subqueries {
        // Exactly two. A join needs somewhere to join to, and v1 caps the schema at two
        // anyway — so this is not a clamp, it is the only value that makes the axis mean
        // anything.
        debug_assert!(
            bounds.max_tables >= 2,
            "the joins and subqueries axes need a schema of at least two tables"
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

    /// The fingerprint must actually distinguish different logic. Tested on `fnv1a` directly,
    /// because the constant itself cannot vary within one compilation — which is precisely why
    /// the *function* is what needs pinning.
    #[test]
    fn the_fingerprint_function_separates_different_sources() {
        let seed = 0xcbf2_9ce4_8422_2325;
        let original = fnv1a(b"if rng.random_range(0..100) < 60 {", seed);
        // The set-op ordering fix was a change of exactly this shape: one condition, no bound
        // touched, an entirely different corpus.
        let changed = fnv1a(b"if rng.random_range(0..100) < 70 {", seed);
        assert_ne!(
            original, changed,
            "a one-character logic change must change the fingerprint"
        );

        // Order matters: hashing the same modules in a different order is a different value,
        // so the chain cannot be silently reordered.
        let forward = fnv1a(b"second", fnv1a(b"first", seed));
        let backward = fnv1a(b"first", fnv1a(b"second", seed));
        assert_ne!(forward, backward);

        // And it is stable — the same bytes always give the same value, or a description would
        // change between runs and nothing could ever be compared.
        assert_eq!(fnv1a(b"stable", seed), fnv1a(b"stable", seed));
    }

    /// Every description must carry the fingerprint, and two *different* bound sets must still
    /// be distinguishable from each other as well.
    #[test]
    fn the_description_carries_the_logic_fingerprint_and_every_axis() {
        let description = Bounds::V1_ALL.description();

        assert!(
            description.contains(&format!("logic={GENERATOR_FINGERPRINT:08x}")),
            "description must name the generation logic, got {description}"
        );

        // Every axis still named — the fingerprint supplements the bounds, it does not replace
        // them. A reader must be able to see *what was configured* without recompiling.
        // Spellings follow the engine's derived format (`name=on`/`name=off`/`name=value`),
        // adopted 2026-08-07. The old hand-written form read `tables<=2`; the change is
        // cosmetic, and every axis is still required to appear.
        for axis in [
            "tables=",
            "columns=",
            "rows=",
            "depth=",
            "wide-arith=",
            "aggregates=",
            "set-ops=",
            "chained=",
            "joins=",
            "subqueries=",
            "not-in=",
        ] {
            assert!(
                description.contains(axis),
                "{axis} missing from {description}"
            );
        }

        // A **disabled** axis must still be named, or "off" is indistinguishable from "this
        // axis did not exist yet" — which is what makes an old corpus silently incomparable.
        assert!(
            Bounds::V1.description().contains("joins=off"),
            "a disabled axis must appear: {}",
            Bounds::V1.description()
        );

        // Two configurations differing only in one bound remain distinguishable.
        assert_ne!(Bounds::V1.description(), Bounds::V1_ALL.description());
        assert_ne!(
            Bounds::V1_NOT_IN.description(),
            Bounds::V1_JOINS.description()
        );

        // The fingerprint is shared by all of them: it describes the *code*, not the config.
        assert!(
            Bounds::V1
                .description()
                .contains(&format!("logic={GENERATOR_FINGERPRINT:08x}"))
        );
    }

    /// The fingerprint reaches the description through the engine's `logic_version()` hook,
    /// **not** through `scalars()`.
    ///
    /// Worth pinning rather than trusting, because the two routes render identically — the move
    /// from one to the other changed no output at all. A test that only checked the rendered
    /// string would pass under either, and would not notice the fingerprint silently dropping
    /// back into the bounds slot.
    #[test]
    fn the_logic_fingerprint_comes_from_the_hook_and_not_from_the_bounds() {
        let bounds = Bounds::V1_ALL;

        assert_eq!(
            bounds.logic_version(),
            Some(format!("{GENERATOR_FINGERPRINT:08x}")),
            "the hook must report the fingerprint"
        );

        // `scalars()` is for bounds. A source hash there reads as a bound to whoever adopts
        // this trait next, which is why the engine grew a separate slot for it.
        assert!(
            !bounds.scalars().iter().any(|(name, _)| *name == "logic"),
            "the fingerprint must not also be a scalar: {:?}",
            bounds.scalars()
        );

        // And it still lands in the description, which is what actually scopes a corpus.
        assert!(
            bounds
                .description()
                .contains(&format!("logic={GENERATOR_FINGERPRINT:08x}"))
        );
    }

    /// A non-zero, non-trivial fingerprint. A zero here would mean `include_bytes!` resolved to
    /// nothing and every corpus would silently share one identity again.
    #[test]
    fn the_fingerprint_is_not_degenerate() {
        assert_ne!(GENERATOR_FINGERPRINT, 0);
        assert_ne!(GENERATOR_FINGERPRINT, u32::MAX);
    }

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
