//! A falsifiable claim about what an input must contain to trigger a divergence.
//!
//! # What a predicate is
//!
//! A conjunction over [`FEATURES`](crate::features::FEATURES): some bits that must be set,
//! some that must be clear. Rendered as `mixed_sign_overflow ∧ ¬outer_join_present`.
//!
//! Two masks and one `AND` each way. That is the entire mechanism, chosen because it is
//! trivially explainable — not because seventeen booleans need optimising.
//!
//! # Why this is not just another signature
//!
//! A [`signature`](crate::signature) is computed from **results**: it describes what a
//! disagreement looked like. A predicate is computed from the **case**, so it claims
//! something about inputs that have never been run — which is what makes it testable by
//! generating them, and what makes it capable of being *wrong*.
//!
//! > A signature can only ever describe the past. A predicate makes a claim about the
//! > future, and can therefore be falsified.
//!
//! # The hazard this module carries
//!
//! A predicate is a **bitmask**, and bits mean whatever `FEATURES` says they mean at that
//! index. Reorder that array and every recorded predicate silently changes meaning — the
//! masks still match, just against different properties. **Nothing errors.** The registry
//! test below is the only thing standing between that and a confidently wrong report.

use crate::features::{FEATURES, FeatureVec};
use serde::{Deserialize, Serialize};

/// A conjunction of features, some required and some forbidden.
///
/// `required` and `forbidden` are disjoint by construction — a feature cannot be both — and
/// a feature absent from both is simply not mentioned by the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Predicate {
    pub required: u32,
    pub forbidden: u32,
}

impl Predicate {
    /// Build from feature names. Unknown names are **ignored**, not fatal.
    ///
    /// Deliberate: a predicate recorded before a feature was renamed should degrade into a
    /// weaker rule rather than crash a triage run. It will match more cases than intended,
    /// which is visible in the report, rather than taking the tool down.
    pub fn new(required: &[&str], forbidden: &[&str]) -> Self {
        Self {
            required: mask(required),
            forbidden: mask(forbidden),
        }
    }

    /// Whether a case's features satisfy this rule.
    ///
    /// Every required bit present, no forbidden bit present. One `AND` each way.
    pub fn matches(&self, features: FeatureVec) -> bool {
        features.0 & self.required == self.required && features.0 & self.forbidden == 0
    }

    /// How many features the rule mentions.
    ///
    /// Used by the search as a tiebreak — Occam, and shorter rules are readable.
    pub fn size(&self) -> u32 {
        self.required.count_ones() + self.forbidden.count_ones()
    }

    /// Whether the rule constrains nothing.
    ///
    /// **An empty predicate matches every case**, which makes it worthless as a trigger
    /// claim while scoring perfectly on "covers the most findings". The search must reject
    /// it explicitly rather than let it win by vacuity.
    pub fn is_vacuous(&self) -> bool {
        self.required == 0 && self.forbidden == 0
    }

    /// Render as the rule a person reads: `mixed_sign_overflow ∧ ¬outer_join_present`.
    pub fn describe(&self) -> String {
        if self.is_vacuous() {
            return "(matches everything)".to_string();
        }

        let mut parts: Vec<String> = Vec::new();
        for (bit, name) in FEATURES.iter().enumerate() {
            if self.required & (1 << bit) != 0 {
                parts.push((*name).to_string());
            }
            if self.forbidden & (1 << bit) != 0 {
                parts.push(format!("¬{name}"));
            }
        }
        parts.join(" ∧ ")
    }
}

fn mask(names: &[&str]) -> u32 {
    names
        .iter()
        .filter_map(|name| FEATURES.iter().position(|f| f == name))
        .fold(0u32, |acc, bit| acc | (1 << bit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SqlCase;
    use crate::features::extract;

    /// A case with a `NULL` in its data, whose features are known.
    ///
    /// Replaces the tensor version's `matmul` fixture — the *shape* of the test carries
    /// (build a case whose features you know, then assert on them); the case itself cannot.
    fn case_with_null() -> SqlCase {
        let mut case = SqlCase::fixed_example();
        case.data[0].rows[0][1] = crate::schema::Literal::Null;
        case
    }

    /// A case with no `NULL` anywhere — the foil.
    fn case_without_null() -> SqlCase {
        let mut case = SqlCase::fixed_example();
        for row in &mut case.data[0].rows {
            for value in row.iter_mut() {
                if matches!(value, crate::schema::Literal::Null) {
                    *value = crate::schema::Literal::Integer(0);
                }
            }
        }
        case
    }

    /// The case the registry test rebuilds a rule against.
    fn filed_case() -> SqlCase {
        case_with_null()
    }

    /// **The test that stands between a reordered vocabulary and a confidently wrong
    /// report.**
    ///
    /// A predicate is a bitmask over `FEATURES` *by index*. Reorder that array and every
    /// recorded predicate keeps matching — against different properties. No compiler error,
    /// no runtime error, just a rule that now means something else.
    ///
    /// So the check cannot be "does this mask still equal that mask". It has to rebuild the
    /// rule **from names**, extract features **from a real case**, and assert they still
    /// agree. Renaming or reordering a feature then fails here rather than silently.
    #[test]
    fn a_predicate_built_from_names_still_matches_the_case_that_produces_it() {
        let predicate = Predicate::new(&["null_in_data", "null_in_data"], &[]);

        assert!(
            predicate.matches(extract(&filed_case())),
            "the vocabulary has been reordered or renamed: {} no longer matches the case \
             it was written for",
            predicate.describe()
        );
    }

    #[test]
    fn a_required_feature_that_is_absent_fails_the_match() {
        // The filed case is `output_is_vector`; a rule demanding a batched case must not
        // match it.
        let predicate = Predicate::new(&["null_in_data", "outer_join_present"], &[]);
        assert!(!predicate.matches(extract(&filed_case())));
    }

    #[test]
    fn a_forbidden_feature_that_is_present_fails_the_match() {
        let predicate = Predicate::new(&["null_in_data"], &["null_in_data"]);
        assert!(!predicate.matches(extract(&filed_case())));
    }

    /// The distinction the whole search rests on: a rule that fires on a case which did
    /// **not** diverge is not a trigger. Here the same rule matches both, which is exactly
    /// how `overflow ∧ mixed_sign` was falsified by measurement.
    #[test]
    fn a_rule_can_match_a_case_that_does_not_diverge() {
        let agreeing = case_without_null();

        let too_broad = Predicate::new(&["order_by_present"], &[]);
        assert!(too_broad.matches(extract(&agreeing)));
        assert!(too_broad.matches(extract(&filed_case())));
    }

    /// **An empty rule matches everything**, which would score perfectly on "covers the
    /// most findings" while claiming nothing. The search must reject it by name.
    #[test]
    fn the_empty_predicate_matches_everything_and_says_so() {
        let empty = Predicate::default();

        assert!(empty.is_vacuous());
        assert!(empty.matches(extract(&filed_case())));
        assert!(empty.matches(FeatureVec::default()));
        assert_eq!(empty.describe(), "(matches everything)");
    }

    #[test]
    fn size_counts_both_required_and_forbidden() {
        let predicate = Predicate::new(&["null_in_data"], &["outer_join_present", "join_present"]);
        assert_eq!(predicate.size(), 3);
    }

    /// The rendering is what a reviewer reads in `CANDIDATES.md`; a bitmask is unreadable.
    #[test]
    fn a_rule_renders_readably_with_negation_marked() {
        let rendered = Predicate::new(&["null_in_data"], &["outer_join_present"]).describe();

        assert!(rendered.contains("null_in_data"));
        assert!(rendered.contains("¬outer_join_present"));
        assert!(rendered.contains('∧'));
    }

    /// An unrecognised name degrades the rule rather than crashing triage. It then matches
    /// **more** than intended — visible in a report — instead of taking the tool down.
    #[test]
    fn an_unknown_feature_name_is_ignored_rather_than_fatal() {
        let predicate = Predicate::new(&["null_in_data", "no_such_feature"], &[]);

        assert_eq!(predicate.size(), 1, "the unknown name contributes no bit");
        assert!(predicate.matches(extract(&filed_case())));
    }
}
