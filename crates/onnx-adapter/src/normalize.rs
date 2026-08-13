//! Turning what a runtime produced into a form two runtimes can be compared in.
//!
//! # What "canonical" means here, and what it deliberately does not mean yet
//!
//! Canonicalizing is the cheap half of comparison and tolerance is the expensive half:
//! every unit of tolerance is sensitivity given away. So this module canonicalizes what it
//! can — shape, element type, and the exact bits of every value — and tolerates nothing.
//!
//! **The special-value table is the one place any looseness lives**, and it is decided before
//! any numeric comparison — see [`values_agree`]. Four of its five rows tighten or preserve;
//! exactly one loosens, and the project's inherited asymmetry says only that one needed a
//! citation:
//!
//! > Holding a bound needs only evidence it is achievable; loosening one needs a
//! > specification.
//!
//! The loosening is `NaN` vs `NaN` agreeing. `+0.0` vs `−0.0` **disagrees**, which is the tight
//! direction and therefore free. Tolerance — the expensive option, where every unit is
//! sensitivity given away — is still zero: Tier A and Tier B admit no rounding argument.
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

/// How two results agreed — because *why* they agreed decides whether the case counts.
///
/// # Why a bool was not enough
///
/// `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §6 forbids reporting a **licensed** difference as
/// `Agree`: a licensed difference is `Skipped`. The reasoning is the same one behind
/// `SkipReason::NothingComparable` and `Unjudgeable` in the engine — a rule that *excuses* a
/// difference has not verified anything, and counting it as a pass inflates what a campaign
/// appears to prove. The tensor domain measured that exact failure: 96% of its `matmul` cases
/// carried a bound nothing could fail, and six hours of fuzzing reported agreement rather than
/// "could not judge".
///
/// So the comparison reports three states, not two. `ByLicense` means the two results differed
/// in bits and a rule in the special-value table forgave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agreement {
    /// Identical bit patterns, or the same non-tensor outcome kind. Real evidence.
    Exactly,
    /// Agreed **only** because a rule licensed the difference away. Not evidence.
    ByLicense,
    /// Did not agree.
    No,
}

/// How two values of the same element type compare, before any tolerance is considered.
///
/// # The table, and the one row that is a *loosening*
///
/// From `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §4. Every row is decided here, before any numeric
/// comparison, and the inherited rule is absolute: **no tolerance, however large, may absorb a
/// disagreement in this table, and none, however strict, may break an agreement in it.**
///
/// | comparison | verdict | why |
/// |---|---|---|
/// | `NaN` vs `NaN` | **agree** | both produced "not a number"; *which* `NaN` is not specified |
/// | `NaN` vs a number | **disagree** | a disagreement about whether an answer exists |
/// | same infinity | **agree** | |
/// | opposite infinities, or infinity vs finite | **disagree** | not a difference of degree |
/// | `+0.0` vs `−0.0` | **disagree** | the sign of a zero result is determined for these operators |
///
/// **Four of these five rows tighten or preserve; one loosens, and only that one needed a
/// citation.** `NaN` vs `NaN` agreeing is the loosening: it accepts as equal two bit patterns
/// that differ. `SPECS.md` §5.3 records what supports it — multiple consistent secondary
/// sources stating that which `NaN` payload propagates is implementation-defined — and records
/// equally plainly that **the primary text was not retrieved**.
///
/// `+0.0` vs `−0.0` disagreeing is the *tight* direction, so it needs only evidence that it is
/// achievable, which every measurement so far provides: all four participants agree on the sign
/// of zero everywhere it has been observed. Loosening it later would need a citation; adopting
/// it now does not. `PENDING` 1.6.
fn values_agree(elem: ElemType, left: u64, right: u64) -> Agreement {
    if left == right {
        return Agreement::Exactly;
    }
    // Only the floating types have values that differ in bits while agreeing in meaning.
    if !elem.is_floating() {
        return Agreement::No;
    }
    let (a, b) = match elem {
        ElemType::F32 => (
            f64::from(f32::from_bits(left as u32)),
            f64::from(f32::from_bits(right as u32)),
        ),
        _ => (f64::from_bits(left), f64::from_bits(right)),
    };
    // The single loosening: two `NaN`s agree whatever their payload or sign bit.
    //
    // Everything else falls through to the bit comparison already made above — which is what
    // keeps `+0.0` vs `−0.0` a disagreement. Comparing with `==` here instead would silently
    // make them agree, since `-0.0 == 0.0` is true, and that is precisely the blind spot the
    // tensor domain documented.
    if a.is_nan() && b.is_nan() {
        Agreement::ByLicense
    } else {
        Agreement::No
    }
}

/// Whether two canonical tensors agree under the special-value table, **and how**.
fn tensors_agree(left: &[CanonTensor], right: &[CanonTensor]) -> Agreement {
    if left.len() != right.len() {
        return Agreement::No;
    }
    let mut verdict = Agreement::Exactly;
    for (a, b) in left.iter().zip(right.iter()) {
        // Shape and element type are structural: no rule in the table can absorb a difference
        // in either, and neither can any tolerance.
        if a.dims != b.dims || a.elem_type != b.elem_type || a.bits.len() != b.bits.len() {
            return Agreement::No;
        }
        for (x, y) in a.bits.iter().zip(b.bits.iter()) {
            match values_agree(a.elem_type, *x, *y) {
                Agreement::No => return Agreement::No,
                // One licensed element is enough to taint the whole comparison: the case can
                // no longer be said to have been judged, even if every other element matched
                // bit for bit.
                Agreement::ByLicense => verdict = Agreement::ByLicense,
                Agreement::Exactly => {}
            }
        }
    }
    verdict
}

/// Compare two canonical results for the purposes of the oracle.
///
/// A free function rather than relying on `==` alone, because **what counts as equal is a
/// policy decision** and it should live in one place.
pub fn compare(left: &Canonical, right: &Canonical) -> Agreement {
    if left.kind != right.kind {
        return Agreement::No;
    }
    match left.kind {
        // Message text is not comparable across implementations; the *kind* is.
        "rejected" | "unsupported" | "crashed" | "timed-out" => Agreement::Exactly,
        _ => tensors_agree(&left.tensors, &right.tensors),
    }
}

/// How many elements agreed **only because a rule forgave the difference**.
///
/// Reported into the skip reason so the record says how much of the case went unjudged, rather
/// than merely that some of it did.
pub fn licensed_elements(left: &Canonical, right: &Canonical) -> usize {
    left.tensors
        .iter()
        .zip(right.tensors.iter())
        .map(|(a, b)| {
            if a.elem_type != b.elem_type || a.bits.len() != b.bits.len() {
                return 0;
            }
            a.bits
                .iter()
                .zip(b.bits.iter())
                .filter(|(x, y)| values_agree(a.elem_type, **x, **y) == Agreement::ByLicense)
                .count()
        })
        .sum()
}

/// Whether two canonical results agree at all, discarding *how*.
///
/// Grouping participants only needs the yes/no; the oracle asks `compare` separately for the
/// distinction that decides `Agree` versus `Skipped`.
pub fn equivalent(left: &Canonical, right: &Canonical) -> bool {
    compare(left, right) != Agreement::No
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

    /// **The special-value table, every row, in both directions.**
    ///
    /// N4.2 asks for both directions specifically, and the reason is that a table tested only
    /// where it says "agree" would pass with a comparison that agrees with everything.
    #[test]
    fn the_special_value_table_holds_in_both_directions() {
        let nan_alt = f32::from_bits(f32::NAN.to_bits() ^ 0x0000_0001); // a different payload
        let nan_neg = -f32::NAN; // a different sign bit

        // ── rows that AGREE ───────────────────────────────────────────────────────
        assert!(
            equivalent(&ok(vec![f32::NAN]), &ok(vec![nan_alt])),
            "two NaNs agree whatever their payload — which one propagates is not specified"
        );
        assert!(
            equivalent(&ok(vec![f32::NAN]), &ok(vec![nan_neg])),
            "two NaNs agree whatever their sign bit"
        );
        assert!(
            equivalent(&ok(vec![f32::INFINITY]), &ok(vec![f32::INFINITY])),
            "the same infinity agrees"
        );
        assert!(
            equivalent(&ok(vec![1.5]), &ok(vec![1.5])),
            "equal numbers agree"
        );

        // ── rows that DISAGREE ────────────────────────────────────────────────────
        assert!(
            !equivalent(&ok(vec![f32::NAN]), &ok(vec![1.0])),
            "NaN vs a number is a disagreement about whether an answer exists"
        );
        assert!(
            !equivalent(&ok(vec![f32::INFINITY]), &ok(vec![f32::NEG_INFINITY])),
            "opposite infinities are not a difference of degree"
        );
        assert!(
            !equivalent(&ok(vec![f32::INFINITY]), &ok(vec![f32::MAX])),
            "infinity vs finite is not a difference of degree"
        );
        assert!(
            !equivalent(&ok(vec![0.0]), &ok(vec![-0.0])),
            "signed zero is a disagreement: the sign of a zero result is determined here"
        );
        assert!(
            !equivalent(&ok(vec![1.5]), &ok(vec![1.5000001])),
            "no tolerance exists at Tier A or Tier B — the nearest distinct float disagrees"
        );
    }

    /// The `NaN` loosening must apply **only** to floats. An integer bit pattern that happens
    /// to look like a `NaN` when reinterpreted must not be quietly accepted as equal to a
    /// different one — that would be the widening escaping its type.
    #[test]
    fn the_nan_rule_does_not_leak_into_integer_types() {
        use crate::case::TensorData;
        let int_tensor = |bits: i64| {
            canon(OnnxOutcome::Ok(vec![TensorValue::new(
                "out",
                vec![1],
                TensorData::I64(vec![bits]),
            )]))
        };
        // Two i64 values whose bits are NaN patterns as f64, and which differ.
        let a = f64::NAN.to_bits() as i64;
        let b = (f64::NAN.to_bits() ^ 1) as i64;
        assert!(
            !equivalent(&int_tensor(a), &int_tensor(b)),
            "integers compare by value; the NaN rule must not reach them"
        );
    }

    /// The table is decided before any tolerance, and **no tolerance may absorb a
    /// disagreement in it**. There is no tolerance here to test against, so the check is that
    /// the largest possible numeric gap and the smallest both behave per the table.
    #[test]
    fn the_table_is_decided_before_magnitude_matters() {
        // A vast finite difference and a one-ulp difference are both simply disagreements.
        assert!(!equivalent(&ok(vec![0.0]), &ok(vec![f32::MAX])));
        assert!(!equivalent(
            &ok(vec![1.0]),
            &ok(vec![f32::from_bits(1.0f32.to_bits() + 1)])
        ));
        // ...and NaN-ness beats magnitude entirely: NaN agrees with NaN, disagrees with the
        // largest and smallest finite values alike.
        assert!(equivalent(&ok(vec![f32::NAN]), &ok(vec![f32::NAN])));
        assert!(!equivalent(&ok(vec![f32::NAN]), &ok(vec![f32::MAX])));
        assert!(!equivalent(&ok(vec![f32::NAN]), &ok(vec![0.0])));
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
