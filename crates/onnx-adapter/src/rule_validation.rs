//! Testing a predicate against cases it was never fitted to.
//!
//! # The problem this exists to solve
//!
//! **A predicate that fits *n* findings can always be found.** With a few thousand candidate
//! rules and a few dozen findings, the search is guaranteed to return something — and "something"
//! is exactly what overfitting looks like. The rule may describe the findings perfectly and
//! predict nothing.
//!
//! The only honest test of a trigger claim is the one the claim itself invites: **generate cases
//! nobody has seen, keep the ones the rule matches, and see whether they diverge.** A real trigger
//! predicts divergence. A coincidence does not.
//!
//! # Rejection sampling
//!
//! Draw from the generator, discard anything the predicate does not match, run what remains.
//! Simple, and it samples from *the generator's distribution restricted to the rule* — which is
//! what makes the resulting rate comparable to the pool the findings came from.
//!
//! Its cost is the reason for the third outcome below: a rule describing something the generator
//! effectively never produces will reject nearly everything drawn.
//!
//! # The four outcomes
//!
//! - [`Outcome::Trigger`] — matched cases diverge often. The claim survived.
//! - [`Outcome::Coincidence`] — matched cases mostly *don't* diverge. The rule described the
//!   findings, not the bug. Discard it.
//! - [`Outcome::NeverSampled`] — **nothing matched at all.** Not a failure of the rule: a
//!   statement that the generator cannot reach what the rule describes, so this method cannot
//!   judge it either way. Silently scoring it as failure would hide that.
//! - [`Outcome::Inconclusive`] — a few matched, too few to distinguish a real rate from noise.
//!   Reported rather than rounded to one of the verdicts above.
//!
//! # Why this file is not called `validation.rs`
//!
//! Because this adapter already has one, and it answers a different question: whether a *model*
//! is well formed before any runtime sees it. Two modules named `validation` in one crate would
//! be a genuine ambiguity rather than a cosmetic one — "is this case legal ONNX" and "is this
//! rule a real trigger" are unrelated. The rename is recorded in `PENDING` §4 as one of the two
//! changes the copied machinery required.

use crate::case::OnnxCase;
use crate::features::features;
use crate::predicate::Predicate;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Generator;

/// The share of matched cases that must diverge for a rule to count as a trigger.
///
/// **A judgment, and deliberately not tuned.** Half is chosen because it is the point where a
/// rule carries information at all: below it, knowing the rule matches makes divergence *less*
/// likely than not. Tightening it to 0.9 would have been fitting the threshold to the findings,
/// which is the error this whole module exists to catch.
pub const TRIGGER_RATE: f64 = 0.5;

/// Matched cases needed before a rate is reported as a verdict rather than as inconclusive.
///
/// With fewer than this, one lucky draw moves the rate across the threshold. Thirty is the
/// conventional small-sample floor; the exact number matters less than refusing to call a verdict
/// on three cases.
pub const MIN_MATCHED: usize = 30;

/// What sampling concluded about a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Matched cases diverged at or above [`TRIGGER_RATE`].
    Trigger,
    /// Matched cases diverged below [`TRIGGER_RATE`]. The rule fitted the findings only.
    Coincidence,
    /// Enough cases were drawn, but too few matched to judge.
    Inconclusive,
    /// **Nothing matched.** The rule describes a region the generator does not reach.
    NeverSampled,
}

/// The evidence behind an [`Outcome`], so a reader can check the verdict rather than take it.
#[derive(Debug, Clone, PartialEq)]
pub struct Validation {
    pub predicate: Predicate,
    /// The seed the sampling ran from. **Every claim here is replayable from this number.**
    pub seed: u64,
    /// Cases drawn from the generator.
    pub sampled: usize,
    /// Of those, how many the rule matched.
    pub matched: usize,
    /// Of the matched, how many diverged.
    pub diverged: usize,
    pub outcome: Outcome,
}

impl Validation {
    /// `diverged / matched`, or `None` when nothing matched.
    ///
    /// Returns `Option` rather than 0.0 on purpose: a rate of zero means "matched cases reliably
    /// agreed", and no matches at all means something entirely different. This is the same
    /// distinction the metamorphic campaign had to learn the hard way — a relation that reported
    /// `0` violations because it *declined to apply* looked identical to one that held.
    pub fn rate(&self) -> Option<f64> {
        (self.matched > 0).then(|| self.diverged as f64 / self.matched as f64)
    }

    /// One line of evidence, leading with the numbers rather than the verdict.
    pub fn describe(&self) -> String {
        let head = match self.rate() {
            Some(rate) => format!(
                "{}/{} matched cases diverged ({:.0}%)",
                self.diverged,
                self.matched,
                rate * 100.0
            ),
            None => "no case matched".to_string(),
        };
        format!(
            "{head}, from {} sampled, seed {} — {}",
            self.sampled,
            self.seed,
            match self.outcome {
                Outcome::Trigger => "TRIGGER",
                Outcome::Coincidence => "COINCIDENCE, discard",
                Outcome::Inconclusive => "INCONCLUSIVE, too few matched",
                Outcome::NeverSampled => "NOT REACHABLE by this generator",
            }
        )
    }
}

/// Sample `budget` cases and measure how often the matched ones diverge.
///
/// `diverges` runs a case through the differential and says whether it diverged. It is a closure
/// rather than a set of runtimes so the accounting can be tested without ONNX Runtime in the
/// process — and so the caller decides what "diverged" means, which is the oracle's job, not this
/// module's.
///
/// Rust note: `impl FnMut(&OnnxCase) -> bool` takes any closure that may hold mutable state (a
/// runtime handle, a counter). `impl Generator<In = OnnxCase>` is the same idea for the source of
/// cases: this function names the capability it needs, not a concrete type.
pub fn validate(
    predicate: Predicate,
    generator: &impl Generator<In = OnnxCase>,
    seed: u64,
    budget: usize,
    mut diverges: impl FnMut(&OnnxCase) -> bool,
) -> Validation {
    let mut rng = SeededRng::from_seed(seed);
    let mut matched = 0usize;
    let mut diverged = 0usize;

    for _ in 0..budget {
        let case = generator.generate(&mut rng);
        // Rejection: draw, test, discard. Cases the rule does not match cost a generation and
        // nothing else — no runtime runs for them, which is what keeps this affordable.
        if !predicate.matches(features(&case)) {
            continue;
        }
        matched += 1;
        if diverges(&case) {
            diverged += 1;
        }
    }

    let outcome = if matched == 0 {
        Outcome::NeverSampled
    } else if matched < MIN_MATCHED {
        Outcome::Inconclusive
    } else if (diverged as f64 / matched as f64) >= TRIGGER_RATE {
        Outcome::Trigger
    } else {
        Outcome::Coincidence
    };

    Validation {
        predicate,
        seed,
        sampled: budget,
        matched,
        diverged,
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureVec;
    use crate::gen_shape::Bounds;
    use crate::generator::OnnxGenerator;

    fn generator() -> OnnxGenerator {
        OnnxGenerator::new(Bounds::default().with_special_values())
    }

    /// A rule nothing matches is **not reachable**, not false.
    ///
    /// The distinction is the whole reason the outcome exists: scoring it as a failure would say
    /// "this rule is wrong" when the honest statement is "this generator cannot test it". In the
    /// tensor domain 763 of 814 findings landed in the equivalent bucket, and reading that as
    /// 763 refutations would have been badly wrong.
    #[test]
    fn a_rule_the_generator_cannot_reach_is_reported_as_such() {
        // `quantized_op` requires the quantized axis, which this generator does not enable.
        let unreachable = Predicate::new(&["quantized_op"], &[]);
        let result = validate(unreachable, &generator(), 1, 500, |_| true);

        assert_eq!(result.outcome, Outcome::NeverSampled);
        assert_eq!(result.matched, 0);
        assert_eq!(result.rate(), None, "no matches is not a rate of zero");
        assert!(result.describe().contains("NOT REACHABLE"));
    }

    /// A rule whose matched cases reliably diverge is a trigger.
    ///
    /// The oracle is stubbed to always diverge, so this tests the *accounting*, not any runtime.
    #[test]
    fn a_rule_whose_matches_diverge_is_a_trigger() {
        let broad = Predicate::new(&["float_dtype"], &[]);
        let result = validate(broad, &generator(), 2, 2_000, |_| true);

        assert!(result.matched >= MIN_MATCHED, "{}", result.describe());
        assert_eq!(result.outcome, Outcome::Trigger);
        assert_eq!(result.rate(), Some(1.0));
    }

    /// The same rule, with an oracle that never diverges, must be discarded as a coincidence.
    ///
    /// **This is the pair that matters.** A validator that returned `Trigger` for everything
    /// would pass the test above and fail here; only running both proves it reads the oracle.
    #[test]
    fn a_rule_whose_matches_agree_is_a_coincidence() {
        let broad = Predicate::new(&["float_dtype"], &[]);
        let result = validate(broad, &generator(), 2, 2_000, |_| false);

        assert!(result.matched >= MIN_MATCHED);
        assert_eq!(result.outcome, Outcome::Coincidence);
        assert_eq!(result.rate(), Some(0.0));
        assert!(result.describe().contains("discard"));
    }

    /// Too few matches is reported as inconclusive rather than rounded to a verdict.
    #[test]
    fn a_handful_of_matches_is_inconclusive() {
        let broad = Predicate::new(&["float_dtype"], &[]);
        // A budget small enough that fewer than MIN_MATCHED can match.
        let result = validate(broad, &generator(), 3, 10, |_| true);

        assert!(result.matched < MIN_MATCHED);
        assert!(matches!(
            result.outcome,
            Outcome::Inconclusive | Outcome::NeverSampled
        ));
    }

    /// **The vacuous rule matches everything**, so it would be scored a perfect trigger by any
    /// oracle that diverges at all. The search must exclude it; this pins that it is the search's
    /// job, because validation alone cannot tell the difference.
    #[test]
    fn validation_cannot_save_us_from_the_vacuous_rule() {
        let empty = Predicate::default();
        assert!(empty.matches(FeatureVec::default()));

        let result = validate(empty, &generator(), 4, 500, |_| true);
        assert_eq!(
            result.outcome,
            Outcome::Trigger,
            "a rule claiming nothing validates perfectly — the search must reject it by name"
        );
    }

    /// Same seed, same numbers. A validation that cannot be replayed is not evidence.
    #[test]
    fn validation_is_reproducible_from_its_seed() {
        let rule = Predicate::new(&["float_dtype"], &[]);
        let a = validate(rule, &generator(), 7, 300, |_| true);
        let b = validate(rule, &generator(), 7, 300, |_| true);
        assert_eq!(a, b);
    }
}
