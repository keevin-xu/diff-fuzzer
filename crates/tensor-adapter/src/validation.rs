//! Testing a predicate against cases it was never fitted to.
//!
//! # The problem this exists to solve
//!
//! **A predicate that fits *n* findings can always be found.** With 6,018 candidate rules
//! and a few dozen findings, the search is guaranteed to return something — and "something"
//! is exactly what overfitting looks like. The rule may describe the findings perfectly and
//! predict nothing.
//!
//! The only honest test of a trigger claim is the one the claim itself invites: **generate
//! cases nobody has seen, keep the ones the rule matches, and see whether they diverge.** A
//! real trigger predicts divergence. A coincidence does not.
//!
//! # Rejection sampling
//!
//! Draw from the generator, discard anything the predicate does not match, run what remains.
//! Simple, and it samples from *the generator's distribution restricted to the rule* — which
//! is what makes the resulting rate comparable to the pool the findings came from.
//!
//! Its cost is the reason for the third outcome below: a rule describing something the
//! generator effectively never produces will reject nearly everything drawn.
//!
//! # The four outcomes
//!
//! - [`Outcome::Trigger`] — matched cases diverge often. The claim survived.
//! - [`Outcome::Coincidence`] — matched cases mostly *don't* diverge. The rule described the
//!   findings, not the bug. Discard it.
//! - [`Outcome::NeverSampled`] — **nothing matched at all.** Not a failure of the rule: a
//!   statement that the generator cannot reach what the rule describes, so this method
//!   cannot judge it either way. Silently scoring it as failure would hide that.
//! - [`Outcome::Inconclusive`] — a few matched, too few to distinguish a real rate from
//!   noise. Reported rather than rounded to one of the verdicts above.

use crate::features::extract;
use crate::input::TensorOp;
use crate::predicate::Predicate;
use diff_fuzzer_core::{Generator, SeededRng};

/// The share of matched cases that must diverge for a rule to count as a trigger.
///
/// **A judgment, and deliberately not tuned.** Half is chosen because it is the point where
/// a rule carries information at all: below it, knowing the rule matches makes divergence
/// *less* likely than not. Tightening it to 0.9 would have been fitting the threshold to the
/// findings, which is the error this whole module exists to catch.
pub const TRIGGER_RATE: f64 = 0.5;

/// Matched cases needed before a rate is reported as a verdict rather than as inconclusive.
///
/// With fewer than this, one lucky draw moves the rate across the threshold. Thirty is the
/// conventional small-sample floor; the exact number matters less than refusing to call a
/// verdict on three cases.
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
    /// Returns `Option` rather than 0.0 on purpose: a rate of zero means "matched cases
    /// reliably agreed", and no matches at all means something entirely different.
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
/// `diverges` runs a case through the differential and says whether it diverged. It is a
/// closure rather than a backend so the accounting can be tested without libtorch — and so
/// the caller decides what "diverged" means, which is the oracle's job, not this module's.
///
/// Rust note: `impl FnMut(&TensorOp) -> bool` takes any closure that may hold mutable state
/// (a backend handle, a counter). `impl Generator<In = TensorOp>` is the same idea for the
/// source of cases: this function names the capability it needs, not a concrete type.
pub fn validate(
    predicate: Predicate,
    generator: &impl Generator<In = TensorOp>,
    seed: u64,
    budget: usize,
    mut diverges: impl FnMut(&TensorOp) -> bool,
) -> Validation {
    let mut rng = SeededRng::from_seed(seed);
    let mut matched = 0usize;
    let mut diverged = 0usize;

    for _ in 0..budget {
        let case = generator.generate(&mut rng);
        // Rejection: draw, test, discard. Cases the rule does not match cost a generation
        // and nothing else — no backend runs for them, which is what keeps this affordable.
        if !predicate.matches(extract(&case)) {
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
    use crate::generator::TensorOpGenerator;
    use crate::ops::Bounds;

    fn generator() -> TensorOpGenerator {
        TensorOpGenerator::new(Bounds::default())
    }

    /// A rule the generator reaches, on a differential that always diverges: the rate is 1
    /// and the verdict is `Trigger`.
    #[test]
    fn a_rule_whose_matched_cases_always_diverge_is_a_trigger() {
        let everything_diverges = |_: &TensorOp| true;
        // `¬rank_ge_3` is common enough that the default generator produces it constantly.
        let predicate = Predicate::new(&[], &["rank_ge_3"]);

        let result = validate(predicate, &generator(), 7, 400, everything_diverges);

        assert_eq!(result.outcome, Outcome::Trigger);
        assert_eq!(result.rate(), Some(1.0));
        assert!(result.matched >= MIN_MATCHED);
    }

    /// **The overfitting guard doing its job.** The same commonly-matched rule, on a
    /// differential where nothing diverges, is a coincidence — the rule described the
    /// findings it was fitted to and predicts nothing.
    #[test]
    fn a_rule_whose_matched_cases_never_diverge_is_a_coincidence() {
        let nothing_diverges = |_: &TensorOp| false;
        let predicate = Predicate::new(&[], &["rank_ge_3"]);

        let result = validate(predicate, &generator(), 7, 400, nothing_diverges);

        assert_eq!(result.outcome, Outcome::Coincidence);
        assert_eq!(result.rate(), Some(0.0));
    }

    /// **The third outcome, and the reason it is not folded into failure.** A rule the
    /// generator cannot reach has not been shown wrong — it has not been tested at all, and
    /// reporting that is what points at the generator rather than at the rule.
    #[test]
    fn a_rule_the_generator_never_produces_is_reported_as_unreachable_not_as_failed() {
        let mut ran = 0usize;
        // `input_special` means "an operand is NaN or infinite", and the generator *cannot*
        // produce one: `SPECIAL_VALUES` holds ten finite entries, and the uniform path draws
        // from `-magnitude..magnitude`. So this is a real, well-formed rule that rejection
        // sampling can never test — not a contrived one.
        let unreachable = Predicate::new(&["input_special"], &[]);

        let result = validate(unreachable, &generator(), 7, 2000, |_| {
            ran += 1;
            true
        });

        assert_eq!(result.outcome, Outcome::NeverSampled);
        assert_eq!(result.matched, 0);
        assert_eq!(result.rate(), None, "a rate of zero would be a lie here");
        assert_eq!(ran, 0, "no backend should run for a rule nothing matches");
        assert!(result.describe().contains("NOT REACHABLE"));
    }

    /// A handful of matches is not a verdict. One lucky draw would move a 3-case rate across
    /// the threshold, so the sample size is reported instead of a conclusion.
    #[test]
    fn too_few_matched_cases_yield_no_verdict() {
        let predicate = Predicate::new(&[], &["rank_ge_3"]);
        // A budget small enough that fewer than MIN_MATCHED can match.
        let result = validate(predicate, &generator(), 7, 10, |_| true);

        assert!(result.matched < MIN_MATCHED);
        assert_eq!(result.outcome, Outcome::Inconclusive);
    }

    /// **Determinism is sacred** (`CLAUDE.md` §3): the same seed must give the same verdict,
    /// or a validation result is not evidence of anything.
    #[test]
    fn the_same_seed_gives_the_same_result() {
        let predicate = Predicate::new(&[], &["rank_ge_3"]);
        let first = validate(predicate, &generator(), 99, 200, |c| c.rank() == 2);
        let second = validate(predicate, &generator(), 99, 200, |c| c.rank() == 2);

        assert_eq!(first, second);
    }

    /// Different seeds must actually explore differently, or the first test proves nothing
    /// about reproducibility — it would hold for a generator that ignored its seed.
    #[test]
    fn a_different_seed_explores_different_cases() {
        let predicate = Predicate::new(&[], &["rank_ge_3"]);
        let first = validate(predicate, &generator(), 1, 200, |c| c.rank() == 2);
        let second = validate(predicate, &generator(), 2, 200, |c| c.rank() == 2);

        assert_ne!(
            (first.matched, first.diverged),
            (second.matched, second.diverged)
        );
    }

    /// The description leads with the counts, not the verdict — the file it lands in must
    /// support review rather than assent.
    #[test]
    fn the_description_leads_with_evidence() {
        let predicate = Predicate::new(&[], &["rank_ge_3"]);
        let result = validate(predicate, &generator(), 7, 400, |_| true);
        let described = result.describe();

        assert!(described.starts_with(&format!("{}/{}", result.diverged, result.matched)));
        assert!(described.contains("seed 7"));
    }
}
