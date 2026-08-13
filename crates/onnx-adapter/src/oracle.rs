//! What counts as a divergence.
//!
//! # The order of the checks is load-bearing
//!
//! 1. **Self-evident defects first.** A crash or a hang accuses its runtime on its own,
//!    before any comparison. It must never reach the skip path — that is the entire thesis
//!    of this domain, and the engine's default (`SkipReason::CouldNotRun`) does exactly the
//!    opposite.
//! 2. **Then remove legitimate abstainers.** Only `Unsupported` excuses a participant. A
//!    runtime that never claimed to implement an operator cannot be wrong about it.
//! 3. **Then check there is still something to compare.** Fewer than two results is a skip
//!    with a stated reason, never a silent pass.
//! 4. **Then compare all pairs.**
//!
//! Getting 1 before 2 is what stops a crash being written off as an abstention. Getting 3
//! before 4 is what stops a single surviving result reading as unanimous agreement.
//!
//! # All pairs, and naming the outlier
//!
//! With two participants, "compare against the first" and "compare all pairs" are the same
//! operation — so the choice between them is untested, and this project found the resulting
//! **two-implementation blindness in four separate places** across its earlier domains.
//! This domain has three runtimes plus a specification oracle, so the distinction is real
//! and is exercised.
//!
//! When exactly one participant disagrees with a consensus, the oracle **names it**. That
//! turns "two implementations differ" into "this one is wrong", which is the difference
//! between a report a maintainer can act on and one they can dismiss.

use std::collections::BTreeMap;

use diff_fuzzer_core::report::Divergence;
use diff_fuzzer_core::traits::{NamedOutput, Oracle, SkipReason, Verdict};

use crate::case::OnnxCase;
use crate::normalize::{Agreement, Canonical, compare, equivalent, licensed_elements};

/// The differential oracle for ONNX runtimes.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnnxOracle;

impl Oracle for OnnxOracle {
    type In = OnnxCase;
    type Canon = Canonical;

    fn check(&self, input: &OnnxCase, outputs: &[NamedOutput<Canonical>]) -> Verdict {
        // ── 1. Self-evident defects ────────────────────────────────────────────────
        // Checked before anything else so a crash cannot be filtered out as an
        // abstention further down.
        let broken: Vec<&NamedOutput<Canonical>> = outputs
            .iter()
            .filter(|o| o.output.is_self_evident_defect())
            .collect();

        if !broken.is_empty() {
            let names: Vec<&str> = broken.iter().map(|o| o.implementation.as_str()).collect();
            let all_broke = broken.len() == outputs.len();
            let summary = if all_broke {
                // Not a *differential* finding — nobody disagreed. Still reported, and the
                // classification is stated explicitly rather than left implicit, because
                // "every runtime falls over on this input" is worth knowing and would
                // otherwise vanish.
                format!(
                    "every participant failed on {} ({})",
                    input.op.onnx_name(),
                    names.join(", ")
                )
            } else {
                format!(
                    "{} on {} while others answered: {}",
                    if names.len() == 1 {
                        format!("{} failed", names[0])
                    } else {
                        format!("{} failed", names.join(", "))
                    },
                    input.op.onnx_name(),
                    describe(outputs)
                )
            };
            return Verdict::Diverged(divergence(input, outputs, summary));
        }

        // ── 2. Remove legitimate abstainers ────────────────────────────────────────
        let comparable: Vec<&NamedOutput<Canonical>> = outputs
            .iter()
            .filter(|o| !o.output.is_unsupported())
            .collect();

        // ── 3. Is there anything left to compare? ──────────────────────────────────
        if comparable.len() < 2 {
            return Verdict::Skipped(SkipReason::TooFewResults {
                available: comparable.len(),
            });
        }

        // ── 4. Compare every pair ──────────────────────────────────────────────────
        // Grouping is the all-pairs comparison expressed once: two participants land in
        // the same group exactly when they agree, so one group means unanimity and the
        // group sizes are what identify an outlier.
        let groups = group_by_result(&comparable);

        if groups.len() == 1 {
            // Unanimous. But agreement on a result that could not have differed is empty
            // evidence, and counting it as a pass would inflate what a campaign appears to
            // prove — hence a stated skip rather than a silent `Agree`.
            if comparable.iter().all(|o| o.output.is_degenerate()) {
                let elements = comparable
                    .first()
                    .map(|o| o.output.tensors.iter().map(|t| t.bits.len()).sum())
                    .unwrap_or(0);
                return Verdict::Skipped(SkipReason::NothingComparable { elements });
            }

            // ── 5. A licensed difference is not a pass ─────────────────────────────
            // The participants landed in one group, but some pair got there only because a
            // rule in the special-value table forgave a difference in their bits. Nothing was
            // verified about that case, so it is recorded as unjudged.
            //
            // **This check sits after the defect check on purpose.** A crash alongside a
            // licensed difference must still be reported — the licence excuses the difference
            // it names, never the complaint standing next to it. Step 1 has already returned
            // by the time control reaches here, which is what makes that ordering structural
            // rather than a matter of remembering it.
            let first = &comparable[0].output;
            let licensed: usize = comparable
                .iter()
                .filter(|o| compare(first, &o.output) == Agreement::ByLicense)
                .map(|o| licensed_elements(first, &o.output))
                .max()
                .unwrap_or(0);
            if licensed > 0 {
                return Verdict::Skipped(SkipReason::NothingComparable { elements: licensed });
            }
            return Verdict::Agree;
        }

        Verdict::Diverged(divergence(
            input,
            outputs,
            describe_disagreement(input, &groups),
        ))
    }
}

/// Participants grouped by the result they produced, keyed by group order of first
/// appearance so the output is deterministic.
fn group_by_result<'a>(
    outputs: &[&'a NamedOutput<Canonical>],
) -> Vec<Vec<&'a NamedOutput<Canonical>>> {
    let mut groups: Vec<Vec<&NamedOutput<Canonical>>> = Vec::new();
    for output in outputs {
        match groups
            .iter_mut()
            .find(|group| equivalent(&group[0].output, &output.output))
        {
            Some(group) => group.push(output),
            None => groups.push(vec![output]),
        }
    }
    groups
}

/// State the disagreement, naming the lone dissenter when there is exactly one.
fn describe_disagreement(input: &OnnxCase, groups: &[Vec<&NamedOutput<Canonical>>]) -> String {
    let op = input.op.onnx_name();

    // Exactly two groups, one of which holds a single participant: that participant is the
    // outlier and the rest are a consensus. This is the case worth naming.
    if groups.len() == 2 {
        let (small, large) = if groups[0].len() < groups[1].len() {
            (&groups[0], &groups[1])
        } else {
            (&groups[1], &groups[0])
        };
        if small.len() == 1 && large.len() > 1 {
            // Sorted, not left in iteration order. A summary whose text depends on the
            // order participants happened to run in cannot be de-duplicated by signature
            // and cannot be diffed between runs — and determinism is not negotiable here.
            let mut consensus: Vec<&str> =
                large.iter().map(|o| o.implementation.as_str()).collect();
            consensus.sort_unstable();
            return format!(
                "{op}: {} disagrees with {} others ({})",
                small[0].implementation,
                consensus.len(),
                consensus.join(", ")
            );
        }
    }

    // Same reasoning: sorted within each faction, and the factions themselves sorted, so
    // the summary is a function of *who agreed with whom* and not of run order.
    let mut factions: Vec<String> = groups
        .iter()
        .map(|group| {
            let mut names: Vec<&str> = group.iter().map(|o| o.implementation.as_str()).collect();
            names.sort_unstable();
            names.join("+")
        })
        .collect();
    factions.sort();
    format!(
        "{op}: {} distinct results — {}",
        groups.len(),
        factions.join(" | ")
    )
}

/// A compact rendering of who produced what, for a summary line.
fn describe(outputs: &[NamedOutput<Canonical>]) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for output in outputs {
        *counts.entry(output.output.kind).or_default() += 1;
    }
    counts
        .iter()
        .map(|(kind, count)| format!("{count}×{kind}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Package a divergence for the engine.
///
/// The input is recorded by its `Debug` form here, which is what the engine's `Divergence`
/// accepts. **That is a triage aid, not the artifact** — the self-contained record that
/// must survive a generator change stores the serialized `OnnxCase`, and is built at N7.
fn divergence(input: &OnnxCase, outputs: &[NamedOutput<Canonical>], summary: String) -> Divergence {
    Divergence {
        input: format!("{input:?}"),
        outputs: outputs
            .iter()
            .map(|o| (o.implementation.clone(), render(&o.output)))
            .collect(),
        summary,
    }
}

/// Render a canonical result for a report.
///
/// Bit patterns are shown next to values, because a report that prints `0` for both `+0.0`
/// and `-0.0` hides the disagreement it exists to document.
fn render(canonical: &Canonical) -> String {
    if !canonical.is_ok() {
        return format!("{}: {}", canonical.kind, canonical.detail);
    }
    let tensors: Vec<String> = canonical
        .tensors
        .iter()
        .map(|t| {
            let values: Vec<String> = t
                .bits
                .iter()
                .take(16)
                .map(|b| format!("{b:#018x}"))
                .collect();
            let elided = if t.bits.len() > 16 {
                format!(" ...[{} more]", t.bits.len() - 16)
            } else {
                String::new()
            };
            format!("{:?}[{}{elided}]", t.dims, values.join(" "))
        })
        .collect();
    tensors.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::normalize::OnnxNormalizer;
    use crate::outcome::OnnxOutcome;
    use crate::validation::well_formed;
    use diff_fuzzer_core::Normalizer;

    fn case() -> OnnxCase {
        well_formed(OpKind::Add, &[2], 22)
    }

    fn named(name: &str, outcome: OnnxOutcome) -> NamedOutput<Canonical> {
        NamedOutput {
            implementation: name.to_string(),
            output: OnnxNormalizer.normalize(outcome),
        }
    }

    fn answered(name: &str, values: Vec<f32>) -> NamedOutput<Canonical> {
        named(
            name,
            OnnxOutcome::Ok(vec![TensorValue::f32(
                "out",
                vec![values.len() as i64],
                values,
            )]),
        )
    }

    #[test]
    fn unanimous_agreement_is_agreement() {
        let outputs = vec![
            answered("ort", vec![1.0, 2.0]),
            answered("tract", vec![1.0, 2.0]),
            answered("candle", vec![1.0, 2.0]),
        ];
        assert_eq!(OnnxOracle.check(&case(), &outputs), Verdict::Agree);
    }

    /// The property two participants cannot test. With three, "compare against the first"
    /// and "compare all pairs" give different answers, and the outlier has a name.
    #[test]
    fn a_lone_dissenter_is_named() {
        let outputs = vec![
            answered("ort", vec![1.0, 2.0]),
            answered("tract", vec![9.0, 9.0]),
            answered("reference", vec![1.0, 2.0]),
        ];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("a disagreement must diverge");
        };
        assert!(
            divergence.summary.contains("tract"),
            "the outlier must be named: {}",
            divergence.summary
        );
        assert!(divergence.summary.contains("disagrees with 2"));
    }

    /// An even split has no outlier, and the oracle must not invent one.
    #[test]
    fn an_even_split_names_the_factions_not_an_outlier() {
        let outputs = vec![answered("ort", vec![1.0]), answered("tract", vec![2.0])];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("must diverge");
        };
        assert!(
            !divergence.summary.contains("disagrees with"),
            "no outlier exists here: {}",
            divergence.summary
        );
        assert!(divergence.summary.contains("2 distinct results"));
    }

    /// **The thesis.** A crash must be a divergence, never a skip.
    #[test]
    fn a_crash_is_a_divergence_not_a_skip() {
        let outputs = vec![
            answered("ort", vec![1.0]),
            named(
                "tract",
                OnnxOutcome::Crashed {
                    detail: "index out of bounds".into(),
                },
            ),
            answered("reference", vec![1.0]),
        ];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("a crash must be reported, not skipped");
        };
        assert!(divergence.summary.contains("tract"));
    }

    #[test]
    fn a_timeout_is_a_divergence() {
        let outputs = vec![
            answered("ort", vec![1.0]),
            named("tract", OnnxOutcome::TimedOut { after_ms: 5000 }),
        ];
        assert!(matches!(
            OnnxOracle.check(&case(), &outputs),
            Verdict::Diverged(_)
        ));
    }

    /// A crash must not be rescued by the abstention filter. This is the ordering test:
    /// if the `Unsupported` filter ran first and a crashing runtime were also the only
    /// other participant, the case could fall out as `TooFewResults`.
    #[test]
    fn a_crash_beside_an_abstention_still_diverges() {
        let outputs = vec![
            named(
                "tract",
                OnnxOutcome::Crashed {
                    detail: "boom".into(),
                },
            ),
            named(
                "candle",
                OnnxOutcome::Unsupported {
                    reason: "operator not implemented".into(),
                },
            ),
        ];
        assert!(
            matches!(OnnxOracle.check(&case(), &outputs), Verdict::Diverged(_)),
            "a crash must survive the abstention filter"
        );
    }

    /// An unimplemented operator is never a bug.
    #[test]
    fn unsupported_participants_are_excused() {
        let outputs = vec![
            answered("ort", vec![1.0]),
            answered("tract", vec![1.0]),
            named(
                "candle",
                OnnxOutcome::Unsupported {
                    reason: "no Sign".into(),
                },
            ),
        ];
        assert_eq!(OnnxOracle.check(&case(), &outputs), Verdict::Agree);
    }

    #[test]
    fn one_survivor_is_a_stated_skip_not_a_pass() {
        let outputs = vec![
            answered("ort", vec![1.0]),
            named("tract", OnnxOutcome::Unsupported { reason: "x".into() }),
            named("candle", OnnxOutcome::Unsupported { reason: "y".into() }),
        ];
        assert_eq!(
            OnnxOracle.check(&case(), &outputs),
            Verdict::Skipped(SkipReason::TooFewResults { available: 1 })
        );
    }

    /// Agreement on a result that could not have differed is empty evidence, and must be
    /// reported as unjudged rather than counted as a pass.
    #[test]
    fn unanimous_but_degenerate_is_not_counted_as_a_pass() {
        let outputs = vec![
            answered("ort", vec![f32::NAN, f32::INFINITY]),
            answered("tract", vec![f32::NAN, f32::INFINITY]),
        ];
        assert!(
            matches!(
                OnnxOracle.check(&case(), &outputs),
                Verdict::Skipped(SkipReason::NothingComparable { .. })
            ),
            "an all-undefined agreement compares nothing"
        );
    }

    /// Two runtimes rejecting a case agree. Comparing message text across implementations
    /// would report a divergence on every rejected case.
    #[test]
    fn mutual_rejection_is_agreement() {
        let outputs = vec![
            named(
                "ort",
                OnnxOutcome::Rejected {
                    detail: "bad".into(),
                },
            ),
            named(
                "tract",
                OnnxOutcome::Rejected {
                    detail: "completely different wording".into(),
                },
            ),
        ];
        assert_eq!(OnnxOracle.check(&case(), &outputs), Verdict::Agree);
    }

    /// But one answering while another refuses is a real disagreement.
    #[test]
    fn answered_versus_rejected_diverges() {
        let outputs = vec![
            answered("ort", vec![1.0]),
            named(
                "tract",
                OnnxOutcome::Rejected {
                    detail: "no".into(),
                },
            ),
        ];
        assert!(matches!(
            OnnxOracle.check(&case(), &outputs),
            Verdict::Diverged(_)
        ));
    }

    /// Every participant crashing is not a *differential* finding, but it is still
    /// reported, and the summary says which situation it is.
    #[test]
    fn everyone_crashing_is_reported_and_labelled() {
        let outputs = vec![
            named("ort", OnnxOutcome::Crashed { detail: "a".into() }),
            named("tract", OnnxOutcome::Crashed { detail: "b".into() }),
        ];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("must be reported");
        };
        assert!(divergence.summary.contains("every participant failed"));
    }

    /// A divergence report must show the disagreement. A signed-zero difference rendered
    /// as `0` versus `0` is a report nobody can act on.
    #[test]
    fn a_report_shows_signed_zero() {
        let outputs = vec![answered("ort", vec![0.0]), answered("tract", vec![-0.0])];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("signed zero must diverge");
        };
        let rendered: String = divergence
            .outputs
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(
            rendered.contains("80000000"),
            "the negative-zero bit pattern must be visible: {rendered}"
        );
    }

    /// **The licensed-difference ordering.** Two runtimes producing `NaN`s with different
    /// payloads agree under the special-value table — but the table *forgave* that difference
    /// rather than verifying anything, so the case must be recorded as unjudged.
    ///
    /// Reporting it as `Agree` would make an excused difference look like evidence of
    /// correctness, which is the failure `SkipReason::Unjudgeable` exists to document.
    #[test]
    fn a_licensed_difference_is_skipped_not_agreed() {
        // Same value in element 0, two different `NaN` payloads in element 1.
        let quiet = f32::from_bits(0x7fc0_0000);
        let other = f32::from_bits(0x7fc0_1234);
        let outputs = vec![
            answered("ort", vec![1.0, quiet]),
            answered("tract", vec![1.0, other]),
        ];
        assert_eq!(
            OnnxOracle.check(&case(), &outputs),
            Verdict::Skipped(SkipReason::NothingComparable { elements: 1 }),
            "an excused difference is not a pass"
        );
    }

    /// The other half of the ordering: a licence excuses the difference it names, never a
    /// complaint standing beside it. A crash alongside a licensed difference still diverges.
    #[test]
    fn a_licence_does_not_excuse_a_crash_beside_it() {
        let quiet = f32::from_bits(0x7fc0_0000);
        let other = f32::from_bits(0x7fc0_1234);
        let outputs = vec![
            answered("ort", vec![1.0, quiet]),
            answered("tract", vec![1.0, other]),
            named(
                "candle",
                OnnxOutcome::Crashed {
                    detail: "boom".into(),
                },
            ),
        ];
        let Verdict::Diverged(divergence) = OnnxOracle.check(&case(), &outputs) else {
            panic!("the crash must survive the licence");
        };
        assert!(divergence.summary.contains("candle"));
    }

    /// And a licence must not rescue a real disagreement: agreeing about a `NaN` in one
    /// element says nothing about a numeric difference in another.
    #[test]
    fn a_licence_does_not_absorb_a_real_disagreement() {
        let quiet = f32::from_bits(0x7fc0_0000);
        let other = f32::from_bits(0x7fc0_1234);
        let outputs = vec![
            answered("ort", vec![1.0, quiet]),
            answered("tract", vec![2.0, other]),
        ];
        assert!(
            matches!(OnnxOracle.check(&case(), &outputs), Verdict::Diverged(_)),
            "a forgiven NaN must not forgive the number next to it"
        );
    }

    /// Order of participants must not change the verdict **or its wording**.
    ///
    /// The wording matters as much as the verdict: findings are de-duplicated by signature
    /// at N7, and a summary that depends on the order runtimes happened to execute in would
    /// make the same problem look like several. Caught by this test, which originally
    /// failed on nothing but the consensus listing order.
    #[test]
    fn the_verdict_does_not_depend_on_participant_order() {
        let a = answered("ort", vec![1.0]);
        let b = answered("tract", vec![2.0]);
        let c = answered("candle", vec![1.0]);

        let forward = OnnxOracle.check(&case(), &[a.clone(), b.clone(), c.clone()]);
        let reversed = OnnxOracle.check(&case(), &[c, b, a]);

        let (Verdict::Diverged(f), Verdict::Diverged(r)) = (forward, reversed) else {
            panic!("both orders must diverge");
        };
        assert_eq!(
            f.summary, r.summary,
            "the outlier must be identified regardless of ordering"
        );
    }
}
