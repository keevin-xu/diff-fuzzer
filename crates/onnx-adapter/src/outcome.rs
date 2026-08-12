//! What came back from a runtime — **including its failures, as values**.
//!
//! This module is where the domain's methodological thesis lives, so it is worth being
//! precise about what it claims and why the shape is what it is.
//!
//! # Errors are values, not failures to run
//!
//! The engine's [`diff_fuzzer_core::Implementation::run`] returns
//! `Result<Out, RunError>`, and the driver treats an `Err` as *"this participant could not
//! be compared"* — it collects the reasons and, if fewer than two results survive, records
//! `SkipReason::CouldNotRun`. That is the right default for a general engine.
//!
//! It is the wrong default here. **Roughly three quarters of published bugs in this space
//! are crashes rather than wrong answers** (NNSmith 55 of 72; ORTHRUS 13 of 21; OATest a
//! majority), and routing a crash through `Err` would file that entire class under "not
//! evidence of being wrong".
//!
//! So this domain's `Implementation::Out` is [`OnnxOutcome`], and `run` returns
//! `Ok(outcome)` in every case — including when the runtime blew up. The oracle then judges
//! outcomes against each other and decides what a crash *means*. `RunError` is never
//! constructed by this adapter at all.
//!
//! **Stated plainly, because it flatters the seams otherwise:** this domain adds itself
//! with zero changes to `diff-fuzzer-core`, but it does so by *routing around* the core's
//! error model rather than by that model fitting a third domain. That is a real result
//! about the `Out` associated type being properly unconstrained, and it is not the same
//! claim as "the error model generalised". Recorded in `PENDING` §4.
//!
//! # The four ways a runtime can fail to give an answer
//!
//! Telling these apart is the whole method:
//!
//! | situation | variant | finding? |
//! |---|---|---|
//! | declares it does not implement this | [`OnnxOutcome::Unsupported`] | **no** — it made no claim |
//! | clean typed error about the *input* | [`OnnxOutcome::Rejected`] | a comparable **value** |
//! | claims the operator, then panics or aborts | [`OnnxOutcome::Crashed`] | **yes** |
//! | never returns | [`OnnxOutcome::TimedOut`] | **yes** |
//!
//! The variants exist from N1 so the plumbing carries them, but **classifying correctly
//! between rows one and three needs the capability model from N2**: without knowing what a
//! runtime claims to support, "it errored" cannot be sorted into "legitimately doesn't do
//! this" versus "does this, and broke". Until then everything non-`Ok` that is not clearly
//! a panic is conservatively [`OnnxOutcome::Rejected`], which is the variant that makes no
//! accusation. Over-reporting crashes before the model exists would manufacture findings.

use serde::{Deserialize, Serialize};

use crate::case::TensorValue;

/// What one runtime produced for one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OnnxOutcome {
    /// It ran and produced these tensors.
    Ok(Vec<TensorValue>),

    /// It returned a clean, typed error about the *input*.
    ///
    /// A comparable value, not a non-answer. The SQL domain found rows-versus-error one of
    /// its most productive signals: one engine accepting what another rejects is a real
    /// disagreement about what the specification permits.
    ///
    /// Most often, early on, it means **our own model was invalid** — which is information
    /// worth having quickly.
    Rejected { detail: String },

    /// It declares it does not implement this operator, opset, or dtype.
    ///
    /// **Never a bug.** A runtime that never claimed to do something cannot be wrong about
    /// it. This is the variant that must not be confused with [`Self::Crashed`], and
    /// telling them apart is what the N2 capability census is for.
    Unsupported { reason: String },

    /// It claims the operator and then panicked, aborted, or segfaulted.
    ///
    /// **A finding.** Only sound when the model is valid: our own malformed model crashing
    /// a runtime is our bug. Two gates stand in front of this — our `validate`, and the
    /// reference implementation accepting the model.
    Crashed { detail: String },

    /// It did not return within the bound.
    ///
    /// **A finding**, and one nothing in this project modelled before. The bound is
    /// derived from the measured runtime distribution at N5, never from intuition, so a
    /// slow-but-correct runtime is not reported as hung.
    TimedOut { after_ms: u64 },
}

impl OnnxOutcome {
    /// The tensors, if it produced any.
    pub fn tensors(&self) -> Option<&[TensorValue]> {
        match self {
            OnnxOutcome::Ok(tensors) => Some(tensors),
            _ => None,
        }
    }

    /// Whether this outcome is one the oracle should treat as evidence of a defect *on its
    /// own*, before any comparison.
    ///
    /// A crash or a hang accuses the runtime by itself. `Rejected` does not — it only
    /// becomes interesting next to a peer that answered.
    pub fn is_self_evident_defect(&self) -> bool {
        matches!(
            self,
            OnnxOutcome::Crashed { .. } | OnnxOutcome::TimedOut { .. }
        )
    }

    /// Whether this outcome legitimately removes a participant from the comparison.
    ///
    /// Only `Unsupported` does. Note what is **not** here: a crash never excuses a
    /// participant from the comparison, because being excused is exactly how the abundant
    /// bug class got discarded in the first place.
    pub fn is_legitimate_skip(&self) -> bool {
        matches!(self, OnnxOutcome::Unsupported { .. })
    }

    /// A short, stable label for grouping and for reading a report at a glance.
    pub fn kind(&self) -> &'static str {
        match self {
            OnnxOutcome::Ok(_) => "ok",
            OnnxOutcome::Rejected { .. } => "rejected",
            OnnxOutcome::Unsupported { .. } => "unsupported",
            OnnxOutcome::Crashed { .. } => "crashed",
            OnnxOutcome::TimedOut { .. } => "timed-out",
        }
    }
}

impl std::fmt::Display for OnnxOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnnxOutcome::Ok(tensors) => {
                write!(f, "ok(")?;
                for (index, tensor) in tensors.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}=", tensor.dims)?;
                    // Printed as bits alongside the value, because a report that shows
                    // `0` for both `+0.0` and `-0.0` hides the disagreement it exists to
                    // document.
                    // Rendered as bit patterns, not values. A report that prints `0` for
                    // both `+0.0` and `-0.0` hides the disagreement it exists to document.
                    write!(f, "{:?}[", tensor.elem_type())?;
                    for (i, bits) in tensor.data.to_bit_keys().iter().enumerate() {
                        if i > 0 {
                            write!(f, " ")?;
                        }
                        write!(f, "{bits:#x}")?;
                    }
                    write!(f, "]")?;
                }
                write!(f, ")")
            }
            OnnxOutcome::Rejected { detail } => write!(f, "rejected: {}", first_line(detail)),
            OnnxOutcome::Unsupported { reason } => write!(f, "unsupported: {}", first_line(reason)),
            OnnxOutcome::Crashed { detail } => write!(f, "CRASHED: {}", first_line(detail)),
            OnnxOutcome::TimedOut { after_ms } => write!(f, "TIMED OUT after {after_ms}ms"),
        }
    }
}

/// A runtime's error can be a whole Python traceback. Reports stay readable if only the
/// first line is shown; the full text is still in the stored finding.
fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("").trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_variant() -> Vec<OnnxOutcome> {
        vec![
            OnnxOutcome::Ok(vec![TensorValue::f32("out", vec![1], vec![1.0])]),
            OnnxOutcome::Rejected {
                detail: "bad input".into(),
            },
            OnnxOutcome::Unsupported {
                reason: "no such operator".into(),
            },
            OnnxOutcome::Crashed {
                detail: "index out of bounds".into(),
            },
            OnnxOutcome::TimedOut { after_ms: 5000 },
        ]
    }

    /// The classification that the entire method rests on. Written as one test over every
    /// variant so a new variant cannot be added without deciding which side it falls on.
    #[test]
    fn only_unsupported_excuses_a_participant() {
        for outcome in every_variant() {
            let excused = outcome.is_legitimate_skip();
            match outcome {
                OnnxOutcome::Unsupported { .. } => {
                    assert!(excused, "an unimplemented operator is a legitimate skip")
                }
                _ => assert!(
                    !excused,
                    "{} must not excuse a participant from comparison",
                    outcome.kind()
                ),
            }
        }
    }

    #[test]
    fn crashes_and_hangs_accuse_on_their_own() {
        for outcome in every_variant() {
            let accuses = outcome.is_self_evident_defect();
            match outcome {
                OnnxOutcome::Crashed { .. } | OnnxOutcome::TimedOut { .. } => {
                    assert!(accuses, "{} is a finding by itself", outcome.kind())
                }
                _ => assert!(!accuses, "{} needs a peer to mean anything", outcome.kind()),
            }
        }
    }

    /// An outcome must round-trip, because it is stored inside a finding.
    #[test]
    fn every_variant_survives_serialization() {
        for outcome in every_variant() {
            let json = serde_json::to_string(&outcome).expect("serialize");
            let restored: OnnxOutcome = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(outcome, restored, "{} did not round-trip", outcome.kind());
        }
    }

    /// A `NaN` inside a stored outcome must survive too — the outcome carries
    /// `TensorValue`, so it inherits the bit-pattern encoding, but that inheritance should
    /// be tested rather than assumed.
    #[test]
    fn a_stored_outcome_keeps_special_values() {
        let outcome = OnnxOutcome::Ok(vec![TensorValue::f32("out", vec![2], vec![f32::NAN, -0.0])]);
        let restored: OnnxOutcome =
            serde_json::from_str(&serde_json::to_string(&outcome).unwrap()).unwrap();

        let values = restored.tensors().unwrap()[0].as_f32().expect("f32 tensor");
        assert!(values[0].is_nan());
        assert_eq!(values[1].to_bits(), (-0.0f32).to_bits());
    }

    /// Display must distinguish `+0.0` from `-0.0`. A report that renders both as `0`
    /// hides the very disagreement it was written to document.
    #[test]
    fn display_distinguishes_the_two_zeros() {
        let positive = OnnxOutcome::Ok(vec![TensorValue::f32("out", vec![1], vec![0.0])]);
        let negative = OnnxOutcome::Ok(vec![TensorValue::f32("out", vec![1], vec![-0.0])]);

        assert_ne!(
            positive.to_string(),
            negative.to_string(),
            "signed zero must be visible in a report"
        );
    }

    #[test]
    fn kinds_are_distinct() {
        let kinds: Vec<&str> = every_variant().iter().map(OnnxOutcome::kind).collect();
        let mut unique = kinds.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(kinds.len(), unique.len(), "two variants share a label");
    }
}
