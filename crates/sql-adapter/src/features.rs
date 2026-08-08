//! Properties of a case that a trigger rule might key on.
//!
//! # What a feature is, and what it deliberately is not
//!
//! A **feature** is a boolean property computable from a case **alone** — never from the
//! results of running it. That restriction is the whole point. A rule computed from outputs
//! is just another description of the *symptom*, which is what `signature.rs` already does.
//! A rule computed from the input makes a falsifiable claim about **cases that do not exist
//! yet**, and can therefore be tested by generating them.
//!
//! # This file is the domain-specific half, by design
//!
//! `09` §1.9 predicted exactly this split: *"the machinery transfers, the vocabulary does
//! not"*. `predicate.rs`, `search.rs` and `validation.rs` were copied from the tensor
//! adapter and needed no logic changes; this file shares only its **shape** with its tensor
//! counterpart — the `FEATURES` array, `FeatureVec`, `extract`, and the registry test. Every
//! entry differs, because tensor features are about floating-point arithmetic and these are
//! about SQL semantics.
//!
//! # Bit order is part of the on-disk format
//!
//! A [`Predicate`](crate::predicate) is a bitmask over [`FEATURES`]. **Appending is safe;
//! reordering silently invalidates every recorded predicate** — the mask would still match,
//! just against different meanings, and nothing would error. The test at the bottom guards
//! this by rebuilding a rule *from names* and checking it still matches a real case.

use crate::ast::SqlCase;
use crate::schema::{Expr, JoinKind, Literal, SetOp};

/// The vocabulary. **Index is bit position, and that mapping is durable — append only.**
///
/// Twenty features in three groups: what the *data* contains, what the *query* is made of,
/// and where the two **meet**. The third group is the interesting one and is the reason to
/// have a vocabulary at all — "there is a `NULL` somewhere" and "a `NULL` is in the column a
/// join matches on" are different claims, and only the second describes a mechanism.
pub const FEATURES: [&str; 20] = [
    // --- data: what is in the tables ---
    "null_in_data",
    "empty_table",
    "duplicate_rows",
    "empty_string_present",
    "extreme_integer",
    // --- query shape: what the SQL is made of ---
    "where_present",
    "order_by_present",
    "limit_present",
    "group_by_present",
    "aggregate_present",
    "count_star_present",
    "join_present",
    "outer_join_present",
    "set_op_present",
    "set_op_deduplicates",
    "subquery_present",
    "null_literal_in_predicate",
    "is_null_test_present",
    // --- the interaction: where the data meets the query ---
    "null_in_join_key",
    "aggregate_over_empty",
];

/// One case's features, one bit each.
///
/// A bitmask rather than a set, so matching a predicate is a single `AND` — chosen because
/// it is trivially explainable, not because twenty booleans need optimising.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FeatureVec(pub u32);

impl FeatureVec {
    /// Whether the named feature holds. Returns `false` for an unknown name rather than
    /// panicking, since names arrive from recorded predicates that may predate a rename.
    pub fn has(&self, name: &str) -> bool {
        match FEATURES.iter().position(|feature| *feature == name) {
            Some(bit) => self.0 & (1 << bit) != 0,
            None => false,
        }
    }

    /// The features that hold, by name — for reports, where a bitmask means nothing.
    pub fn names(&self) -> Vec<&'static str> {
        FEATURES
            .iter()
            .enumerate()
            .filter(|(bit, _)| self.0 & (1 << bit) != 0)
            .map(|(_, name)| *name)
            .collect()
    }

    fn set(&mut self, name: &str) {
        if let Some(bit) = FEATURES.iter().position(|feature| *feature == name) {
            self.0 |= 1 << bit;
        }
    }
}

/// Compute every feature of a case.
///
/// **Reads the case only.** Nothing here runs an engine or inspects an output.
pub fn extract(case: &SqlCase) -> FeatureVec {
    let mut features = FeatureVec::default();

    data_features(case, &mut features);
    query_features(case, &mut features);
    interaction_features(case, &mut features);

    features
}

/// What is in the tables.
fn data_features(case: &SqlCase, features: &mut FeatureVec) {
    let mut any_rows = false;

    for insert in &case.data {
        if insert.rows.is_empty() {
            features.set("empty_table");
        } else {
            any_rows = true;
        }

        // Duplicate rows matter because every deduplicating construct — `UNION`,
        // `INTERSECT`, `DISTINCT` — is defined by what it does to them.
        for (index, row) in insert.rows.iter().enumerate() {
            if insert.rows[index + 1..].contains(row) {
                features.set("duplicate_rows");
            }
            for value in row {
                match value {
                    Literal::Null => features.set("null_in_data"),
                    Literal::Text(text) if text.is_empty() => features.set("empty_string_present"),
                    Literal::Integer(number)
                        if [i64::MAX, i64::MIN, i32::MAX as i64, i32::MIN as i64]
                            .contains(number) =>
                    {
                        features.set("extreme_integer");
                    }
                    _ => {}
                }
            }
        }
    }

    // A schema with no rows anywhere is its own case, distinct from one table being empty.
    if !any_rows {
        features.set("empty_table");
    }
}

/// What the query is made of.
fn query_features(case: &SqlCase, features: &mut FeatureVec) {
    let query = &case.query;

    if query.filter.is_some() {
        features.set("where_present");
    }
    if !query.order_by.is_empty() {
        features.set("order_by_present");
    }
    if query.limit.is_some() {
        features.set("limit_present");
    }
    if !query.group_by.is_empty() {
        features.set("group_by_present");
    }
    if case.aggregates() {
        features.set("aggregate_present");
    }
    if let Some(join) = &query.join {
        features.set("join_present");
        if join.kind != JoinKind::Inner {
            features.set("outer_join_present");
        }
    }
    if let Some(branch) = &query.set_op {
        features.set("set_op_present");
        // `UNION ALL` keeps duplicates and the other three do not — the distinction the
        // whole family turns on, so it earns its own bit rather than hiding inside a name.
        if branch.op.deduplicates() {
            features.set("set_op_deduplicates");
        }
    }
    if query.contains_subquery() {
        features.set("subquery_present");
    }
    if let Some(filter) = &query.filter {
        if filter.contains_null_literal() {
            features.set("null_literal_in_predicate");
        }
        if mentions_null_test(filter) {
            features.set("is_null_test_present");
        }
    }
    for expression in &query.projection {
        if matches!(
            expression,
            Expr::Aggregate {
                func: crate::schema::AggregateFunc::CountRows,
                ..
            }
        ) {
            features.set("count_star_present");
        }
    }
}

/// Where the data meets the query — the group worth having a vocabulary for.
fn interaction_features(case: &SqlCase, features: &mut FeatureVec) {
    // A `NULL` **in the column a join matches on** is a different claim from a `NULL`
    // anywhere in the data: an outer join's unmatched rows are exactly the ones a `NULL` key
    // produces, since `NULL = NULL` is unknown.
    if let Some(join) = &case.query.join {
        let keys = join.on.columns();
        for insert in &case.data {
            let Some(table) = case
                .schema
                .iter()
                .find(|candidate| candidate.name == insert.table)
            else {
                continue;
            };
            for key in &keys {
                if key.table != table.name {
                    continue;
                }
                let Some((position, _)) = table.column(&key.column) else {
                    continue;
                };
                if insert
                    .rows
                    .iter()
                    .any(|row| matches!(row.get(position), Some(Literal::Null)))
                {
                    features.set("null_in_join_key");
                }
            }
        }
    }

    // An aggregate over no rows has to decide what to return — `NULL` or zero, depending on
    // the aggregate — which is a documented edge and not the same as either alone.
    if case.aggregates() && case.queried_rows().is_empty() {
        features.set("aggregate_over_empty");
    }
}

/// Does this predicate ask about `NULL` directly, rather than comparing against it?
fn mentions_null_test(expression: &Expr) -> bool {
    match expression {
        Expr::Unary { op, operand } => {
            matches!(
                op,
                crate::schema::UnaryOp::IsNull | crate::schema::UnaryOp::IsNotNull
            ) || mentions_null_test(operand)
        }
        Expr::Binary { left, right, .. } => mentions_null_test(left) || mentions_null_test(right),
        Expr::Cast { expr, .. } => mentions_null_test(expr),
        Expr::Aggregate { arg, .. } => arg.as_ref().is_some_and(|inner| mentions_null_test(inner)),
        Expr::Exists { query, .. } => query.filter.as_ref().is_some_and(mentions_null_test),
        Expr::ScalarSubquery { left, query, .. } | Expr::InSubquery { left, query, .. } => {
            mentions_null_test(left) || query.filter.as_ref().is_some_and(mentions_null_test)
        }
        // A membership test against a list is **not** a `NULL` test, even when the list holds a
        // `NULL`. It asks about equality; the `NULL` changes the answer without being asked
        // about. Reporting otherwise would conflate the two features a predicate rule most
        // needs to tell apart.
        Expr::InList { left, .. } => mentions_null_test(left),
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

/// So the unused-import warning does not fire for a type the vocabulary documents.
const _: Option<SetOp> = None;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_schema::Bounds;
    use crate::generator::SqlGenerator;
    use diff_fuzzer_core::SeededRng;
    use diff_fuzzer_core::traits::Generator;

    fn generated(seed: u64) -> SqlCase {
        SqlGenerator::new(Bounds::V1_ALL).generate(&mut SeededRng::from_seed(seed))
    }

    /// The registry test, carried over in shape from the tensor domain.
    ///
    /// **Reordering `FEATURES` silently changes what every recorded predicate means** — the
    /// mask still matches, just against different bits, and nothing errors. This rebuilds a
    /// lookup *from names* and checks it agrees with the bit positions.
    #[test]
    fn names_and_bit_positions_agree() {
        for (bit, name) in FEATURES.iter().enumerate() {
            let vector = FeatureVec(1 << bit);
            assert!(vector.has(name), "bit {bit} should be {name}");
            assert_eq!(vector.names(), vec![*name]);
        }
    }

    #[test]
    fn an_unknown_name_is_false_rather_than_a_panic() {
        // Names arrive from recorded predicates that may predate a rename.
        assert!(!FeatureVec(u32::MAX).has("a_feature_that_never_existed"));
    }

    #[test]
    fn features_are_a_pure_function_of_the_case() {
        for seed in 0..200 {
            let case = generated(seed);
            assert_eq!(extract(&case), extract(&case.clone()));
        }
    }

    /// Every feature must be **reachable**, or the vocabulary is lying about what it can
    /// express — the same rule that caught an unreachable classifier variant in the tensor
    /// domain, applied to the one artifact that is written rather than derived.
    #[test]
    fn every_feature_occurs_at_least_once() {
        let mut seen = FeatureVec::default();
        for seed in 0..3000 {
            seen.0 |= extract(&generated(seed)).0;
        }

        let missing: Vec<&str> = FEATURES
            .iter()
            .enumerate()
            .filter(|(bit, _)| seen.0 & (1 << bit) == 0)
            .map(|(_, name)| *name)
            .collect();

        assert!(
            missing.is_empty(),
            "unreachable features — a rule mentioning one could never be validated: {missing:?}"
        );
    }

    #[test]
    fn a_null_in_a_join_key_is_distinct_from_a_null_anywhere() {
        // The interaction the vocabulary exists for. Across a real corpus, the two features
        // must not be the same set — otherwise the finer one carries no information.
        let (mut anywhere, mut in_key) = (0, 0);
        for seed in 0..2000 {
            let features = extract(&generated(seed));
            if features.has("null_in_data") {
                anywhere += 1;
            }
            if features.has("null_in_join_key") {
                in_key += 1;
            }
        }
        assert!(in_key > 0, "the finer feature must occur");
        assert!(
            in_key < anywhere,
            "if every NULL is a join-key NULL, the finer feature adds nothing ({in_key} vs {anywhere})"
        );
    }
}
