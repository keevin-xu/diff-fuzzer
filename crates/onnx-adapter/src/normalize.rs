//! Turning what a runtime produced into a form two runtimes can be compared in.
//!
//! # What "canonical" means here, and what it deliberately does not mean yet
//!
//! Canonicalizing is the cheap half of comparison and tolerance is the expensive half:
//! every unit of tolerance is sensitivity given away. So this module canonicalizes what it
//! can — shape, element type, and the exact bits of every value — and tolerates nothing.
//!
//! **At N1 that means bit-exact equality, including `NaN` versus `NaN`.** That is
//! *stricter* than the policy the domain intends: `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §4
//! says two `NaN`s should **agree**, because the payload and sign bit are unspecified. That
//! rule is a **loosening**, and the project's inherited asymmetry is explicit about the
//! direction of travel:
//!
//! > Holding a bound needs only evidence it is achievable; loosening one needs a
//! > specification.
//!
//! The specification has not been retrieved (it sits in `SPECS.md` §5), so the loosening
//! waits for N4 and N6. Being too strict now costs false positives, which are noisy,
//! visible and self-correcting. Being too loose would hide defects, silently and
//! permanently. Starting strict is the choice that can be safely revised.
//!
//! # Why bits rather than values
//!
//! Comparing `f32` with `==` gets two things wrong that matter enormously here: `NaN != NaN`
//! is *always* true, so two identical results would be reported as a divergence; and
//! `-0.0 == 0.0` is *also* true, so a genuine signed-zero disagreement would be reported as
//! agreement. Comparing bit patterns gets both right, and it is what makes `Eq` derivable.

use serde::{Deserialize, Serialize};

use crate::case::{ElemType, TensorValue};
use crate::outcome::OnnxOutcome;

/// One tensor in canonical form.
///
/// Values are `u32` bit patterns, never `f32` — see the module note. That also lets this
/// derive `Eq` and `Hash`, which `f32` cannot, and grouping participants by their result
/// is exactly what naming an outlier requires.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonTensor {
    pub dims: Vec<i64>,
    pub elem_type: ElemType,
    /// One entry per element, holding its exact bits.
    ///
    /// `u64` for every type, not just the 64-bit ones, so the canonical form is a single
    /// shape regardless of what the tensor held. The element type is carried alongside, so
    /// widening loses nothing: an `I32` and an `F32` with the same bit pattern are still
    /// distinguishable because their `elem_type` differs.
    pub bits: Vec<u64>,
    /// Whether every element was `NaN` or infinite, computed **before** widening.
    ///
    /// Stored rather than recomputed because `u64` bits alone cannot answer it — the same
    /// pattern means different things for `F32` and `I64`, and reinterpreting after the
    /// fact is how a reasonable-looking check quietly starts measuring the wrong thing.
    pub entirely_undefined: bool,
}

impl CanonTensor {
    fn from(tensor: &TensorValue) -> Self {
        Self {
            dims: tensor.dims.clone(),
            elem_type: tensor.elem_type(),
            bits: tensor.data.to_bit_keys(),
            entirely_undefined: tensor.data.is_entirely_undefined(),
        }
    }

    /// Whether every element is `NaN` or infinite.
    ///
    /// Used to recognise a result that cannot discriminate between implementations. A case
    /// where both sides produce only undefined values *agrees*, but the agreement is empty:
    /// no arithmetic was actually compared. Counting that as a pass inflates the evidence a
    /// campaign appears to provide, which is why the engine has a `NothingComparable` skip
    /// reason at all.
    pub fn is_entirely_undefined(&self) -> bool {
        self.entirely_undefined
    }
}

/// A runtime's result, in the form comparison happens on.
///
/// The `kind` is carried alongside the tensors so that "produced a tensor" and "returned an
/// error" are directly comparable — an implementation's error is a legitimate outcome, and
/// answered-versus-rejected is one of the signals this domain most wants to see.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Canonical {
    /// `"ok"`, `"rejected"`, `"unsupported"`, `"crashed"`, `"timed-out"`.
    pub kind: &'static str,
    /// Present only for `ok`.
    pub tensors: Vec<CanonTensor>,
    /// For non-`ok` outcomes: the reason, **normalized**. See [`normalize_detail`].
    pub detail: String,
}

impl Canonical {
    pub fn is_ok(&self) -> bool {
        self.kind == "ok"
    }

    pub fn is_unsupported(&self) -> bool {
        self.kind == "unsupported"
    }

    pub fn is_self_evident_defect(&self) -> bool {
        self.kind == "crashed" || self.kind == "timed-out"
    }

    /// Whether this result could not discriminate between implementations.
    ///
    /// An empty tensor, or one made entirely of `NaN`/infinity. Reported rather than
    /// counted as a pass: `05-MEASUREMENT-AND-CAMPAIGNS.md` requires the degenerate-output
    /// rate to be measured, because it is what turns a nominal bound into an effective one.
    /// In the SQL domain 44.4% of results were empty, making the honest bound ~1.8× worse
    /// than the nominal one.
    pub fn is_degenerate(&self) -> bool {
        self.is_ok()
            && (self.tensors.is_empty()
                || self
                    .tensors
                    .iter()
                    .all(|t| t.bits.is_empty() || t.is_entirely_undefined()))
    }
}

/// Reduce a runtime's error text to something two runtimes could plausibly be compared on.
///
/// Error messages are **not** comparable across implementations and this makes no attempt
/// to pretend otherwise — ORT, `tract` and Python phrase the same complaint completely
/// differently. What matters is that the *kind* of outcome is comparable, so the detail is
/// kept only as a single trimmed line for a human reading a report.
///
/// A deliberate consequence: two runtimes that both reject a case **agree**, whatever they
/// said about it. Comparing message text would report a divergence on every rejected case,
/// which is noise, not signal.
fn normalize_detail(text: &str) -> String {
    // A Python traceback's last line is the exception; a Rust error is usually one line.
    let line = text
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    // Bounded so a report stays readable. A megabyte of Debug output once produced a
    // 224 MB log for 235 findings, which is a file nobody opens.
    line.chars().take(200).collect()
}

/// The [`diff_fuzzer_core::Normalizer`] for this domain.
#[derive(Debug, Clone, Copy, Default)]
pub struct OnnxNormalizer;

impl diff_fuzzer_core::Normalizer for OnnxNormalizer {
    type Out = OnnxOutcome;
    type Canon = Canonical;

    /// Takes ownership of the outcome rather than borrowing it: extracting data from a
    /// runtime's representation usually consumes it, and consuming avoids copying what may
    /// be a large buffer.
    fn normalize(&self, out: OnnxOutcome) -> Canonical {
        let kind = out.kind();
        match out {
            OnnxOutcome::Ok(tensors) => Canonical {
                kind,
                tensors: tensors.iter().map(CanonTensor::from).collect(),
                detail: String::new(),
            },
            OnnxOutcome::Rejected { detail } | OnnxOutcome::Crashed { detail } => Canonical {
                kind,
                tensors: Vec::new(),
                detail: normalize_detail(&detail),
            },
            OnnxOutcome::Unsupported { reason } => Canonical {
                kind,
                tensors: Vec::new(),
                detail: normalize_detail(&reason),
            },
            OnnxOutcome::TimedOut { after_ms } => Canonical {
                kind,
                tensors: Vec::new(),
                detail: format!("{after_ms}ms"),
            },
        }
    }
}

/// Compare two canonical results for the purposes of the oracle.
///
/// A free function rather than relying on `==` alone, because **what counts as equal is a
/// policy decision that will change at N4**, and it should change in one place. Today it is
/// exact, except that two rejections are equal regardless of wording.
pub fn equivalent(left: &Canonical, right: &Canonical) -> bool {
    if left.kind != right.kind {
        return false;
    }
    match left.kind {
        // Message text is not comparable across implementations; the *kind* is.
        "rejected" | "unsupported" | "crashed" | "timed-out" => true,
        // Bit-exact. See the module note on why this is strict on purpose at N1.
        _ => left.tensors == right.tensors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use diff_fuzzer_core::Normalizer;

    fn canon(outcome: OnnxOutcome) -> Canonical {
        OnnxNormalizer.normalize(outcome)
    }

    fn ok(values: Vec<f32>) -> Canonical {
        canon(OnnxOutcome::Ok(vec![TensorValue::f32(
            "out",
            vec![values.len() as i64],
            values,
        )]))
    }

    /// The two failures that comparing `f32` with `==` would cause, both checked.
    #[test]
    fn bit_comparison_gets_nan_and_signed_zero_right() {
        // `NaN != NaN` under `==`, so a value-based comparison would report two identical
        // results as a divergence.
        assert!(
            equivalent(&ok(vec![f32::NAN]), &ok(vec![f32::NAN])),
            "identical NaN bit patterns must compare equal"
        );

        // `-0.0 == 0.0` under `==`, so a value-based comparison would report a genuine
        // signed-zero disagreement as agreement.
        assert!(
            !equivalent(&ok(vec![0.0]), &ok(vec![-0.0])),
            "signed zero must be visible to the oracle"
        );
    }

    #[test]
    fn shape_is_part_of_the_answer() {
        let flat = canon(OnnxOutcome::Ok(vec![TensorValue::f32(
            "out",
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
        )]));
        let square = canon(OnnxOutcome::Ok(vec![TensorValue::f32(
            "out",
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )]));
        assert!(
            !equivalent(&flat, &square),
            "same values in a different shape is a divergence, not a match"
        );
    }

    /// Two runtimes that both reject a case agree, whatever they said. Comparing message
    /// text across implementations would report a divergence on every rejected case.
    #[test]
    fn two_rejections_agree_regardless_of_wording() {
        let ort = canon(OnnxOutcome::Rejected {
            detail: "Invalid input shape for operator Add".into(),
        });
        let tract = canon(OnnxOutcome::Rejected {
            detail: "translating node #0 \"add_0\" Add ToTypedTranslator".into(),
        });
        assert!(equivalent(&ort, &tract));
    }

    /// But answered-versus-rejected is a real disagreement, and one of the signals this
    /// domain most wants. The SQL domain found rows-versus-error highly productive.
    #[test]
    fn answered_versus_rejected_is_a_disagreement() {
        let answered = ok(vec![1.0]);
        let refused = canon(OnnxOutcome::Rejected {
            detail: "nope".into(),
        });
        assert!(!equivalent(&answered, &refused));
    }

    /// Every non-`Ok` kind is distinct from every other. A crash must never be equivalent
    /// to an unsupported operator — collapsing those is the exact mistake the domain exists
    /// to correct.
    #[test]
    fn the_outcome_kinds_are_mutually_distinct() {
        let kinds = [
            canon(OnnxOutcome::Rejected { detail: "x".into() }),
            canon(OnnxOutcome::Unsupported { reason: "x".into() }),
            canon(OnnxOutcome::Crashed { detail: "x".into() }),
            canon(OnnxOutcome::TimedOut { after_ms: 1 }),
            ok(vec![1.0]),
        ];
        for (i, left) in kinds.iter().enumerate() {
            for (j, right) in kinds.iter().enumerate() {
                assert_eq!(
                    equivalent(left, right),
                    i == j,
                    "{} vs {} compared wrongly",
                    left.kind,
                    right.kind
                );
            }
        }
    }

    #[test]
    fn degenerate_results_are_recognised() {
        assert!(
            ok(vec![]).is_degenerate(),
            "an empty tensor cannot disagree"
        );
        assert!(
            ok(vec![f32::NAN, f32::INFINITY]).is_degenerate(),
            "an all-undefined result compares nothing"
        );
        assert!(
            !ok(vec![1.0, f32::NAN]).is_degenerate(),
            "one finite value is comparable"
        );
        assert!(!ok(vec![0.0]).is_degenerate(), "zero is a real answer");
    }

    #[test]
    fn a_traceback_is_reduced_to_its_last_line() {
        let traceback = "Traceback (most recent call last):\n  File \"x.py\", line 1\n\
                         ValueError: something specific\n";
        let reduced = canon(OnnxOutcome::Rejected {
            detail: traceback.into(),
        });
        assert_eq!(reduced.detail, "ValueError: something specific");
    }

    #[test]
    fn detail_is_bounded_so_reports_stay_readable() {
        let huge = "x".repeat(10_000);
        let reduced = canon(OnnxOutcome::Crashed { detail: huge });
        assert!(reduced.detail.chars().count() <= 200);
    }
}
