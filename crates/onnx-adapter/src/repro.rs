//! Replaying a stored finding.
//!
//! # The question a replay is actually asking
//!
//! "Does this finding still reproduce?" looks like one question and is really two:
//!
//! 1. do the runtimes still behave as they did, and
//! 2. do we still judge that behaviour the same way?
//!
//! A naive replay conflates them. It re-runs the case under **today's** comparison rules and
//! reports agree-or-diverge, which silently answers question 1 using an answer to question 2 that
//! nobody checked. If the policy loosened in between, a real defect quietly becomes "no longer
//! reproduces" — the most expensive possible false negative, because it closes a finding.
//!
//! So replay checks the policy fingerprint first and **refuses to render a verdict** when it has
//! moved. That is not a limitation being worked around; it is the honest answer. The rules are
//! code, and code from six months ago is not available to a record written six months ago.
//!
//! > **Detecting that the question changed is worth more than confidently answering the wrong
//! > one.**

use diff_fuzzer_core::Normalizer;
use diff_fuzzer_core::traits::{Implementation, NamedOutput, Oracle, Verdict};

use crate::case::OnnxCase;
use crate::findings::StoredFinding;
use crate::normalize::{Canonical, OnnxNormalizer};
use crate::oracle::OnnxOracle;
use crate::outcome::OnnxOutcome;

/// What happened when a stored finding was re-run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Replay {
    /// It diverges again, with the same signature. The finding stands.
    Reproduced { signature: String },

    /// It diverges, but **differently**. Not the same finding.
    ///
    /// Distinguished from [`Self::Reproduced`] because a report that says "still reproduces" while
    /// showing a different disagreement is worse than one that says nothing: it attaches evidence
    /// to the wrong claim.
    DivergedDifferently { was: String, now: String },

    /// The runtimes now agree. Either they were fixed, or the finding was never real.
    NoLongerDiverges,

    /// The comparison rules changed since this was recorded, so no verdict is claimed.
    ///
    /// Carries both descriptions so a human can see *what* changed rather than only that
    /// something did.
    PolicyDrift { recorded: String, current: String },

    /// The environment changed — a different runtime version, a different platform.
    ///
    /// Reported rather than ignored: "reproduces on a different version of `tract`" and
    /// "reproduces on the version it was found on" are different claims, and only one of them is
    /// what a maintainer asked for.
    EnvironmentDrift { differences: Vec<String> },
}

impl Replay {
    /// Did this confirm the original finding?
    pub fn confirms(&self) -> bool {
        matches!(self, Replay::Reproduced { .. })
    }
}

/// Re-run a stored finding against the given participants.
///
/// **Checks drift before running anything.** A replay under changed rules should not consume the
/// runtimes' time to produce a verdict that will be discarded, and more importantly should not
/// produce a number that might be quoted.
pub fn replay(
    finding: &StoredFinding,
    participants: &[(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)],
) -> Replay {
    // ── 1. Are we still asking the same question? ───────────────────────────────────
    if !finding.policy_is_current() {
        return Replay::PolicyDrift {
            recorded: if finding.policy.is_empty() {
                "(none recorded)".to_string()
            } else {
                finding.policy.clone()
            },
            current: crate::policy::describe(),
        };
    }

    // ── 2. Is it still the same environment? ────────────────────────────────────────
    let current = crate::environment::environment();
    let differences: Vec<String> = finding
        .environment
        .components
        .iter()
        .filter_map(|(name, recorded)| {
            let now = current
                .components
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_str());
            match now {
                Some(now) if now != recorded => {
                    Some(format!("{name}: recorded {recorded}, now {now}"))
                }
                None => Some(format!("{name}: recorded {recorded}, now absent")),
                _ => None,
            }
        })
        .collect();
    if !differences.is_empty() {
        return Replay::EnvironmentDrift { differences };
    }

    // ── 3. Re-run the stored case — not a regenerated one ───────────────────────────
    // The case comes out of the record. Regenerating it from the seed would be re-deriving the
    // very thing the record exists to preserve.
    let outputs: Vec<NamedOutput<Canonical>> = participants
        .iter()
        .map(|(name, implementation)| NamedOutput {
            implementation: (*name).to_string(),
            output: OnnxNormalizer.normalize(implementation.run(&finding.case).unwrap_or(
                OnnxOutcome::Crashed {
                    detail: "the adapter returned an error".to_string(),
                },
            )),
        })
        .collect();

    match OnnxOracle.check(&finding.case, &outputs) {
        Verdict::Diverged(_) => match crate::signature::of(&finding.case, &outputs) {
            Some(signature) if signature.key() == finding.signature => Replay::Reproduced {
                signature: signature.key(),
            },
            Some(signature) => Replay::DivergedDifferently {
                was: finding.signature.clone(),
                now: signature.key(),
            },
            // Diverged but no signature could be derived — the oracle and the signature
            // disagree about what counts. Reported as a difference rather than as a
            // reproduction, because claiming the latter would rest on the disagreement.
            None => Replay::DivergedDifferently {
                was: finding.signature.clone(),
                now: "(no signature)".to_string(),
            },
        },
        _ => Replay::NoLongerDiverges,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::findings::StoredFinding;
    use crate::gen_shape::Bounds;
    use crate::validation::well_formed;
    use diff_fuzzer_core::axes::GenerationAxes;
    use diff_fuzzer_core::traits::RunError;

    /// A runtime that returns whatever it is told to.
    #[derive(Clone)]
    struct Fixed(OnnxOutcome);

    impl Implementation for Fixed {
        type In = OnnxCase;
        type Out = OnnxOutcome;
        fn name(&self) -> &str {
            "fixed"
        }
        fn run(&self, _input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
            Ok(self.0.clone())
        }
    }

    fn answered(values: Vec<f32>) -> Fixed {
        Fixed(OnnxOutcome::Ok(vec![TensorValue::f32(
            "out",
            vec![values.len() as i64],
            values,
        )]))
    }

    fn stored(signature: &str) -> StoredFinding {
        StoredFinding::new(
            signature,
            "a disagreement",
            7,
            Bounds::default().description(),
            well_formed(OpKind::Add, &[2], 22),
            vec![],
        )
    }

    #[test]
    fn a_finding_that_still_diverges_the_same_way_is_reproduced() {
        let finding = stored("Add/22/F32/rank1/value");
        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 9.0]);
        let result = replay(&finding, &[("ort", &a), ("tract", &b)]);
        assert_eq!(
            result,
            Replay::Reproduced {
                signature: "Add/22/F32/rank1/value".to_string()
            }
        );
        assert!(result.confirms());
    }

    #[test]
    fn agreement_means_it_no_longer_diverges() {
        let finding = stored("Add/22/F32/rank1/value");
        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 2.0]);
        assert_eq!(
            replay(&finding, &[("ort", &a), ("tract", &b)]),
            Replay::NoLongerDiverges
        );
    }

    /// **Bug hijacking, at replay time.** A case that diverges *differently* must not be reported
    /// as a reproduction — that would attach evidence to the wrong claim.
    #[test]
    fn a_different_disagreement_is_not_a_reproduction() {
        let finding = stored("Add/22/F32/rank1/value");
        let a = answered(vec![1.0, 2.0]);
        let crashed = Fixed(OnnxOutcome::Crashed {
            detail: "boom".to_string(),
        });
        let result = replay(&finding, &[("ort", &a), ("tract", &crashed)]);
        assert!(
            matches!(result, Replay::DivergedDifferently { .. }),
            "expected a different signature, got {result:?}"
        );
        assert!(!result.confirms());
    }

    /// **The property the module exists for.** A finding recorded under different rules gets no
    /// verdict at all — not "reproduced", and crucially not "no longer diverges", which would
    /// close a real finding on the strength of a tool change.
    #[test]
    fn a_policy_change_suspends_the_verdict_rather_than_answering() {
        let mut finding = stored("Add/22/F32/rank1/value");
        finding.policy = "comparison=bit-exact fingerprint=deadbeef".to_string();

        // These two agree, so a naive replay would confidently report NoLongerDiverges.
        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 2.0]);
        let result = replay(&finding, &[("ort", &a), ("tract", &b)]);

        match result {
            Replay::PolicyDrift { recorded, current } => {
                assert!(recorded.contains("deadbeef"));
                assert!(current.contains("fingerprint="));
            }
            other => panic!("a policy change must suspend the verdict, got {other:?}"),
        }
    }

    /// A record with no policy at all is unverifiable, and must be treated as drifted rather than
    /// as matching. The convenient default is how a stale record starts being trusted.
    #[test]
    fn a_finding_with_no_recorded_policy_is_not_silently_judged() {
        let mut finding = stored("Add/22/F32/rank1/value");
        finding.policy = String::new();
        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 9.0]);
        assert!(matches!(
            replay(&finding, &[("ort", &a), ("tract", &b)]),
            Replay::PolicyDrift { .. }
        ));
    }

    /// A version change is reported rather than absorbed: "reproduces on a different version" is
    /// a different claim from the one a maintainer asked about.
    #[test]
    fn a_version_change_is_reported() {
        let mut finding = stored("Add/22/F32/rank1/value");
        finding.environment.components.push((
            "tract-onnx".to_string(),
            "0.0.1-not-the-real-one".to_string(),
        ));

        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 9.0]);
        match replay(&finding, &[("ort", &a), ("tract", &b)]) {
            Replay::EnvironmentDrift { differences } => {
                assert!(
                    differences.iter().any(|d| d.contains("tract-onnx")),
                    "{differences:?}"
                );
            }
            other => panic!("expected environment drift, got {other:?}"),
        }
    }

    /// Round trip: a finding written to disk, read back, and replayed must behave identically to
    /// the in-memory one. The stored case is the artifact, so the path through serialization is
    /// the one that matters.
    #[test]
    fn a_finding_round_trips_through_disk_and_still_replays() {
        use crate::findings::FindingsLog;
        let path = std::env::temp_dir().join(format!("dfrepro-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let finding = stored("Add/22/F32/rank1/value");
        let mut log = FindingsLog::open(&path).unwrap();
        log.record(&finding).unwrap();

        let loaded = FindingsLog::load(&path).unwrap();
        let a = answered(vec![1.0, 2.0]);
        let b = answered(vec![1.0, 9.0]);
        assert_eq!(
            replay(&loaded[0], &[("ort", &a), ("tract", &b)]),
            replay(&finding, &[("ort", &a), ("tract", &b)]),
            "a finding must replay the same way after a round trip"
        );
        let _ = std::fs::remove_file(&path);
    }
}
