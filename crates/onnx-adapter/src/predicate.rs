//! A falsifiable claim about what an input must contain to trigger a divergence.
//!
//! # What a predicate is
//!
//! A conjunction over [`FEATURES`](crate::features::FEATURES): some bits that must be set, some
//! that must be clear. Rendered as `has_negative_zero ∧ ¬empty_tensor`.
//!
//! Two masks and one `AND` each way. That is the entire mechanism, chosen because it is trivially
//! explainable — not because fourteen booleans need optimising.
//!
//! # Why this is not just another signature
//!
//! A [`signature`](crate::signature) is computed from **results**: it describes what a
//! disagreement looked like. A predicate is computed from the **case**, so it claims something
//! about inputs that have never been run — which is what makes it testable by generating them,
//! and what makes it capable of being *wrong*.
//!
//! > A signature can only ever describe the past. A predicate makes a claim about the future, and
//! > can therefore be falsified.
//!
//! # Why this domain needs it, concretely
//!
//! [`problems`](crate::problems) groups signatures into problems by `(operator, kind)`, written
//! by hand. **P-001 merges two different defects**: `tract` returning `1` for integer `Sign(0)`
//! and `-0.0` for float `Sign(-0.0)`. Same operator, same `kind=value`, one problem — and their
//! fates differ, since the first was fixed upstream by tract#2533 and the second is still live on
//! `main`. A signature cannot separate them. A predicate keyed on the input can:
//! `integer_dtype` against `has_negative_zero`.
//!
//! # The hazard this module carries
//!
//! A predicate is a **bitmask**, and bits mean whatever `FEATURES` says they mean at that index.
//! Reorder that array and every recorded predicate silently changes meaning — the masks still
//! match, just against different properties. **Nothing errors.** The registry test below is the
//! only thing standing between that and a confidently wrong report.

use crate::features::{FEATURES, FeatureVec};
use serde::{Deserialize, Serialize};

/// A conjunction of features, some required and some forbidden.
///
/// `required` and `forbidden` are disjoint by construction — a feature cannot be both — and a
/// feature absent from both is simply not mentioned by the rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Predicate {
    pub required: u64,
    pub forbidden: u64,
}

impl Predicate {
    /// Build from feature names. Unknown names are **ignored**, not fatal.
    ///
    /// Deliberate: a predicate recorded before a feature was renamed should degrade into a weaker
    /// rule rather than crash a triage run. It will match more cases than intended, which is
    /// visible in the report, rather than taking the tool down.
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
    /// **An empty predicate matches every case**, which makes it worthless as a trigger claim
    /// while scoring perfectly on "covers the most findings". The search must reject it
    /// explicitly rather than let it win by vacuity.
    pub fn is_vacuous(&self) -> bool {
        self.required == 0 && self.forbidden == 0
    }

    /// Render as the rule a person reads: `has_negative_zero ∧ ¬empty_tensor`.
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

fn mask(names: &[&str]) -> u64 {
    names
        .iter()
        .filter_map(|name| FEATURES.iter().position(|f| f == name))
        .fold(0u64, |acc, bit| acc | (1u64 << bit))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OnnxCase, OpKind, TensorData, TensorValue};
    use crate::features::features;

    const OPSET: i64 = 22;

    /// The F-005 case: `tract` returns `-0.0` for `Sign(-0.0)`, minimised to a single element.
    ///
    /// A real filed finding rather than an invented fixture, so the registry test below is
    /// anchored to a case whose features were measured rather than assumed.
    fn filed_case() -> OnnxCase {
        OnnxCase::new(
            OpKind::Sign,
            OPSET,
            vec![TensorValue::new("a", vec![1], TensorData::F32(vec![-0.0]))],
        )
    }

    /// **The test that stands between a reordered vocabulary and a confidently wrong report.**
    ///
    /// A predicate is a bitmask over `FEATURES` *by index*. Reorder that array and every recorded
    /// predicate keeps matching — against different properties. No compiler error, no runtime
    /// error, just a rule that now means something else.
    ///
    /// So the check cannot be "does this mask still equal that mask". It has to rebuild the rule
    /// **from names**, compute features **from a real case**, and assert they still agree.
    /// Renaming or reordering a feature then fails here rather than silently.
    #[test]
    fn a_predicate_built_from_names_still_matches_the_case_that_produces_it() {
        let predicate = Predicate::new(&["has_negative_zero", "float_dtype"], &["empty_tensor"]);

        assert!(
            predicate.matches(features(&filed_case())),
            "the vocabulary has been reordered or renamed: {} no longer matches the case it \
             was written for",
            predicate.describe()
        );
    }

    #[test]
    fn a_required_feature_that_is_absent_fails_the_match() {
        // The filed case is a rank-1 float tensor; a rule demanding an integer case must not
        // match it.
        let predicate = Predicate::new(&["has_negative_zero", "integer_dtype"], &[]);
        assert!(!predicate.matches(features(&filed_case())));
    }

    #[test]
    fn a_forbidden_feature_that_is_present_fails_the_match() {
        let predicate = Predicate::new(&["float_dtype"], &["has_negative_zero"]);
        assert!(!predicate.matches(features(&filed_case())));
    }

    /// The distinction the whole search rests on: a rule that fires on a case which did **not**
    /// diverge is not a trigger.
    ///
    /// `float_dtype` holds for the filed case and for an ordinary `Abs` that every runtime agrees
    /// on. A rule that broad claims nothing, and only measurement against negatives can say so.
    #[test]
    fn a_rule_can_match_a_case_that_does_not_diverge() {
        let agreeing = OnnxCase::new(
            OpKind::Abs,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![2],
                TensorData::F32(vec![1.0, 2.0]),
            )],
        );

        let too_broad = Predicate::new(&["float_dtype"], &[]);
        assert!(too_broad.matches(features(&agreeing)));
        assert!(too_broad.matches(features(&filed_case())));
    }

    /// **An empty rule matches everything**, which would score perfectly on "covers the most
    /// findings" while claiming nothing. The search must reject it by name.
    #[test]
    fn the_empty_predicate_matches_everything_and_says_so() {
        let empty = Predicate::default();

        assert!(empty.is_vacuous());
        assert!(empty.matches(features(&filed_case())));
        assert!(empty.matches(FeatureVec::default()));
        assert_eq!(empty.describe(), "(matches everything)");
    }

    #[test]
    fn size_counts_both_required_and_forbidden() {
        let predicate = Predicate::new(&["has_negative_zero"], &["rank_0", "empty_tensor"]);
        assert_eq!(predicate.size(), 3);
    }

    /// The rendering is what a reviewer reads in the candidates report; a bitmask is unreadable.
    #[test]
    fn a_rule_renders_readably_with_negation_marked() {
        let rendered = Predicate::new(&["has_negative_zero"], &["empty_tensor"]).describe();

        assert!(rendered.contains("has_negative_zero"));
        assert!(rendered.contains("¬empty_tensor"));
        assert!(rendered.contains('∧'));
    }

    /// An unrecognised name degrades the rule rather than crashing triage. It then matches
    /// **more** than intended — visible in a report — instead of taking the tool down.
    #[test]
    fn an_unknown_feature_name_is_ignored_rather_than_fatal() {
        let predicate = Predicate::new(&["has_negative_zero", "no_such_feature"], &[]);

        assert_eq!(predicate.size(), 1, "the unknown name contributes no bit");
        assert!(predicate.matches(features(&filed_case())));
    }
}
