//! Relations that check a **single** implementation against itself.
//!
//! # Why this lives beside the seams rather than in them
//!
//! `Oracle::check(input, outputs)` is handed one input and judges outputs someone else produced. A
//! metamorphic relation **derives its own inputs and must run them**, so it does not fit that
//! shape — and this was known in advance (`03-CONCEPTS.md` §9) rather than discovered here. The SQL
//! adapter's `metamorphic.rs` sits beside its seams for the same reason.
//!
//! # What it catches that a differential oracle structurally cannot
//!
//! A differential oracle reports disagreement. If every implementation is wrong **in the same
//! way**, they agree, and agreement is what it reports — the defect is not merely missed, it is
//! rendered as a pass. `04-ARCHITECTURE.md` §5 names this as the known blind spot of using
//! `onnx.reference` as a confirmer.
//!
//! A metamorphic relation compares an implementation against **arithmetic that must hold**, so a
//! shared defect has nothing to hide behind.
//!
//! It also covers a case the differential oracle cannot reach at all: N9 measured that only ONNX
//! Runtime implements `QuantizeLinear` and `DequantizeLinear`, and *an oracle over one participant
//! is not an oracle*. The round-trip relation judged 2,598,050 values there.
//!
//! # A wrong transform is a wrong oracle
//!
//! `PHASE-N10` names this first among its risks, and it is the reason every relation here has a
//! **precondition** and a test that it fires on a deliberately broken answer. A metamorphic
//! "finding" that is really a bug in the transform is still our bug, and it arrives wearing the
//! costume of a discovery.

use crate::attrs::Attrs;
use crate::case::{ElemType, OnnxCase, OpKind, TensorValue};

/// What a relation concluded about one case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The relation applied and held.
    Held,
    /// The relation applied and was **violated**. A finding.
    Violated,
    /// The relation did not apply — its precondition failed, or the runtime declined a step.
    ///
    /// Counted separately and never as a pass, for the reason `SkipReason::Unjudgeable` records:
    /// a check that could not run looks identical to one that succeeded, in a total.
    NotApplicable,
}

/// **N10.1 — the shape the specification infers must be the shape the runtime produces.**
///
/// Discrete, value-free and very cheap: no arithmetic is involved, so there is no tolerance
/// argument and no special-value policy. A mismatch means one of the two is wrong.
///
/// The inferred shape comes from [`crate::ops::output_spec`], and the licence for that is a
/// measurement rather than an assumption: over 400 generated models, ONNX's own
/// `shape_inference.infer_shapes(strict_mode=True)` produced a shape for 323 outputs and agreed
/// with `output_spec` on **all 323**, with zero strict-mode failures. Using our inference is
/// therefore using the specification's, at no protocol cost.
///
/// **This is single-implementation.** Every runtime producing the same wrong shape would satisfy
/// the differential oracle and fail here.
pub fn shape_matches_inference(case: &OnnxCase, produced: &[TensorValue]) -> Verdict {
    let Some(first) = produced.first() else {
        return Verdict::NotApplicable;
    };
    let (inferred_elem, inferred_dims) = crate::ops::output_spec(case);

    // An operator whose output shape depends on input *values* — `Reshape`'s target, `Slice`'s
    // bounds — is inferable only when those values are constants we put there. They are, since
    // shape inputs are emitted as initializers, so this stays applicable.
    if inferred_dims.iter().any(|d| *d < 0) {
        return Verdict::NotApplicable;
    }
    if first.dims == inferred_dims && first.elem_type() == inferred_elem {
        Verdict::Held
    } else {
        Verdict::Violated
    }
}

/// **N10.2 — the same model at two opsets where the operator did not change.**
///
/// # Where the version diff comes from, and why it is not a reading
///
/// `PHASE-N10` calls N10.2.1 the phase's real work: establish which opset changes are semantic and
/// which are cosmetic, so a genuine semantic change is never reported as a bug. The obvious way is
/// to read each operator's published version history — and reading thirty-seven of them is thirty-
/// seven chances to misread one, each producing a flood of violations that are entirely ours.
///
/// There is an exact source instead. The schema registry's **`since_version`** at a given opset is
/// *the last version at which the operator changed*. So for any `N` with
/// `since_version <= N <= 22`, the operator's definition is **identical by construction** — not by
/// interpretation.
///
/// Measured from the registry (`SPECS.md` §2.12): 36 of the 37 operators in scope have a
/// non-trivial span. `Round` is the exception, introduced at opset 22, and the relation correctly
/// declines it because there is no earlier opset to compare against.
///
/// **This is single-implementation**: the same runtime, the same values, two opsets it should not
/// be able to tell apart.
pub fn opset_invariant(case: &OnnxCase) -> Option<OnnxCase> {
    let since = crate::ops::spec(case.op).since;
    if since >= case.opset {
        // No earlier opset at which this operator is defined identically.
        return None;
    }
    let mut earlier = case.clone();
    earlier.opset = since;
    // The model must still be one we would have built — an operator whose *inputs* differ across
    // the span would be a semantic change, and `since_version` says there is none.
    if !crate::validation::is_valid(&earlier) {
        return None;
    }
    Some(earlier)
}

/// **N10.3 — `Transpose(Transpose(x, p), p⁻¹) == x`**, exactly.
///
/// Returns the second model given the first's output, or `None` when the relation does not apply.
///
/// The inverse permutation is computed rather than assumed: `p⁻¹[p[i]] = i`. Getting that backwards
/// would make the relation false for every non-involutive permutation and produce a flood of
/// "findings" that were entirely ours — which is exactly the risk `PHASE-N10` lists first.
pub fn transpose_inverse(case: &OnnxCase, produced: &TensorValue) -> Option<OnnxCase> {
    if case.op != OpKind::Transpose {
        return None;
    }
    let perm = match case.attrs.get("perm") {
        Some(crate::attrs::AttrValue::Ints(p)) => p.clone(),
        _ => return None,
    };
    let mut inverse = vec![0i64; perm.len()];
    for (position, axis) in perm.iter().enumerate() {
        let axis = usize::try_from(*axis).ok()?;
        if axis >= inverse.len() {
            return None;
        }
        inverse[axis] = position as i64;
    }
    Some(
        OnnxCase::new(
            OpKind::Transpose,
            case.opset,
            vec![TensorValue::new(
                "a",
                produced.dims.clone(),
                produced.data.clone(),
            )],
        )
        .with_attrs(Attrs::new().ints("perm", inverse)),
    )
}

/// **N10.3 — a widening `Cast` round-trip is lossless**, so `Cast(Cast(x, wide), narrow) == x`.
///
/// # The precondition is the whole safety argument
///
/// Only widenings within a family round-trip: `int32 → int64 → int32` and `float32 → float64 →
/// float32` are exact, because every value of the narrow type is representable in the wide one.
///
/// **Everything else is excluded**, and the exclusions are not stylistic. `float → int` truncates.
/// `int64 → float32` loses precision above 2^24. `float → int` out of range is explicitly
/// undefined (`SPECS.md` §2.5). Any of them would make the relation false and every violation ours.
pub fn cast_round_trip(case: &OnnxCase) -> Option<(OnnxCase, ElemType)> {
    if case.op != OpKind::Cast {
        return None;
    }
    let source = case.inputs.first()?.elem_type();
    let wide = match source {
        ElemType::I32 => ElemType::I64,
        ElemType::F32 => ElemType::F64,
        _ => return None,
    };
    let input = case.inputs.first()?;
    Some((
        OnnxCase::new(OpKind::Cast, case.opset, vec![input.clone()])
            .with_attrs(Attrs::new().int("to", i64::from(wide.wire()))),
        source,
    ))
}

/// Build the narrowing half of a cast round trip.
pub fn cast_back(produced: &TensorValue, back_to: ElemType, opset: i64) -> OnnxCase {
    OnnxCase::new(
        OpKind::Cast,
        opset,
        vec![TensorValue::new(
            "a",
            produced.dims.clone(),
            produced.data.clone(),
        )],
    )
    .with_attrs(Attrs::new().int("to", i64::from(back_to.wire())))
}

/// Do two tensors match exactly, bit for bit?
///
/// **Bit patterns, not values**, for the reason this project has now recorded twice in findings:
/// `-0.0 == 0.0` is true, so a value comparison would report a genuine signed-zero round-trip
/// failure as success, and `NaN != NaN` would report a preserved `NaN` as a violation.
pub fn identical(left: &TensorValue, right: &TensorValue) -> bool {
    left.dims == right.dims
        && left.elem_type() == right.elem_type()
        && left.data.to_bit_keys() == right.data.to_bit_keys()
}

/// A tally over one relation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub held: usize,
    pub violated: usize,
    pub not_applicable: usize,
}

impl Tally {
    pub fn record(&mut self, verdict: Verdict) {
        match verdict {
            Verdict::Held => self.held += 1,
            Verdict::Violated => self.violated += 1,
            Verdict::NotApplicable => self.not_applicable += 1,
        }
    }

    /// Checks the relation actually judged. The denominator for any rate.
    pub fn judged(&self) -> usize {
        self.held + self.violated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::TensorData;
    use crate::validation::well_formed;

    fn tensor(dims: Vec<i64>, values: Vec<f32>) -> TensorValue {
        TensorValue::f32("out", dims, values)
    }

    /// The shape relation must accept a correct shape and reject a wrong one — the second half
    /// being what stops it from being an oracle that passes everything.
    #[test]
    fn the_shape_relation_discriminates() {
        let case = well_formed(OpKind::Add, &[2, 3], 22);
        let right = tensor(vec![2, 3], vec![0.0; 6]);
        let wrong_shape = tensor(vec![3, 2], vec![0.0; 6]);
        let wrong_rank = tensor(vec![6], vec![0.0; 6]);

        assert_eq!(shape_matches_inference(&case, &[right]), Verdict::Held);
        assert_eq!(
            shape_matches_inference(&case, &[wrong_shape]),
            Verdict::Violated
        );
        assert_eq!(
            shape_matches_inference(&case, &[wrong_rank]),
            Verdict::Violated
        );
    }

    /// A wrong element type is a violation too — the relation covers both halves of a shape.
    #[test]
    fn the_shape_relation_checks_the_element_type() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let wrong_type = TensorValue::new("out", vec![2], TensorData::I32(vec![0, 0]));
        assert_eq!(
            shape_matches_inference(&case, &[wrong_type]),
            Verdict::Violated
        );
    }

    /// No output at all is not a pass.
    #[test]
    fn nothing_produced_is_not_applicable_rather_than_held() {
        let case = well_formed(OpKind::Add, &[2], 22);
        assert_eq!(shape_matches_inference(&case, &[]), Verdict::NotApplicable);
    }

    /// **The inverse permutation must actually be the inverse.** Getting this backwards is the
    /// "wrong transform is a wrong oracle" failure, and it is invisible for symmetric
    /// permutations — so the test uses an asymmetric one, where `p⁻¹ != p`.
    #[test]
    fn the_transpose_inverse_is_the_real_inverse() {
        // A 3-cycle: [1, 2, 0]. Its inverse is [2, 0, 1], not itself.
        let case = OnnxCase::new(
            OpKind::Transpose,
            22,
            vec![TensorValue::f32("a", vec![2, 3, 4], vec![0.0; 24])],
        )
        .with_attrs(Attrs::new().ints("perm", vec![1, 2, 0]));

        let produced = tensor(vec![3, 4, 2], vec![0.0; 24]);
        let second = transpose_inverse(&case, &produced).expect("applies");
        assert_eq!(
            second.attrs.get("perm"),
            Some(&crate::attrs::AttrValue::Ints(vec![2, 0, 1])),
            "the inverse of [1,2,0] is [2,0,1]"
        );
    }

    /// And composing a permutation with the computed inverse must be the identity, checked over
    /// every permutation of rank 3 rather than one example.
    #[test]
    fn every_permutation_composes_to_the_identity() {
        for perm in [
            vec![0i64, 1, 2],
            vec![0, 2, 1],
            vec![1, 0, 2],
            vec![1, 2, 0],
            vec![2, 0, 1],
            vec![2, 1, 0],
        ] {
            let case = OnnxCase::new(
                OpKind::Transpose,
                22,
                vec![TensorValue::f32("a", vec![2, 3, 4], vec![0.0; 24])],
            )
            .with_attrs(Attrs::new().ints("perm", perm.clone()));
            let produced = tensor(vec![2, 3, 4], vec![0.0; 24]);
            let second = transpose_inverse(&case, &produced).expect("applies");
            let inverse = match second.attrs.get("perm") {
                Some(crate::attrs::AttrValue::Ints(p)) => p.clone(),
                _ => panic!("no perm"),
            };
            // Applying perm then inverse must map every axis back to itself.
            for axis in 0..3usize {
                assert_eq!(
                    inverse[perm[axis] as usize], axis as i64,
                    "perm {perm:?} inverse {inverse:?} is not an inverse at axis {axis}"
                );
            }
        }
    }

    /// **The cast round trip must only apply where it is lossless.** A `float → int` "round trip"
    /// truncates, and admitting it would make every fractional value a violation.
    #[test]
    fn the_cast_round_trip_refuses_lossy_directions() {
        let widening = OnnxCase::new(
            OpKind::Cast,
            22,
            vec![TensorValue::f32("a", vec![2], vec![1.5, -2.5])],
        )
        .with_attrs(Attrs::new().int("to", i64::from(ElemType::I64.wire())));
        // f32 widens to f64 — applicable, whatever the case's own `to` said.
        let (_, back) = cast_round_trip(&widening).expect("f32 has a widening");
        assert_eq!(back, ElemType::F32);

        // i64 and Bool have no wider type in this adapter, so the relation declines.
        let from_i64 = OnnxCase::new(
            OpKind::Cast,
            22,
            vec![TensorValue::new("a", vec![1], TensorData::I64(vec![1]))],
        )
        .with_attrs(Attrs::new().int("to", i64::from(ElemType::F32.wire())));
        assert!(cast_round_trip(&from_i64).is_none());
    }

    /// Comparison is on bit patterns. Two of this project's findings are about signed zero, and a
    /// value comparison would report a `-0.0` that came back as `+0.0` as a successful round trip.
    #[test]
    fn identity_is_decided_on_bits_not_values() {
        let negative = tensor(vec![1], vec![-0.0]);
        let positive = tensor(vec![1], vec![0.0]);
        assert!(
            !identical(&negative, &positive),
            "-0.0 and +0.0 are equal as values and must not be equal here"
        );

        let nan_a = tensor(vec![1], vec![f32::NAN]);
        let nan_b = tensor(vec![1], vec![f32::NAN]);
        assert!(
            identical(&nan_a, &nan_b),
            "an identically preserved NaN is a successful round trip"
        );
    }

    /// The opset relation must produce an *earlier* opset, and must decline where there is none.
    #[test]
    fn opset_invariance_declines_when_there_is_no_earlier_version() {
        // `Add` is unchanged from opset 14 to 22, so the relation applies and goes back to 14.
        let add = well_formed(OpKind::Add, &[2], 22);
        let earlier = opset_invariant(&add).expect("Add is unchanged since 14");
        assert_eq!(earlier.opset, 14);
        assert_eq!(earlier.inputs, add.inputs, "only the opset may differ");

        // `Round` was introduced at 22 — there is no earlier opset that defines it.
        let round = crate::validation::well_formed(OpKind::Round, &[2], 22);
        assert!(
            opset_invariant(&round).is_none(),
            "Round has no earlier opset and the relation must decline rather than invent one"
        );
    }

    /// Only the opset may change. If the relation altered anything else it would be comparing two
    /// different computations and every difference would be ours.
    #[test]
    fn opset_invariance_changes_nothing_but_the_opset() {
        for op in [OpKind::Mul, OpKind::Where, OpKind::Concat, OpKind::Gather] {
            let case = well_formed(op, &[2, 2], 22);
            if let Some(earlier) = opset_invariant(&case) {
                assert_eq!(earlier.op, case.op);
                assert_eq!(earlier.inputs, case.inputs);
                assert_eq!(earlier.attrs, case.attrs);
                assert!(earlier.opset < case.opset);
            }
        }
    }

    /// A tally must keep "did not apply" apart from "held".
    #[test]
    fn the_tally_separates_inapplicable_from_held() {
        let mut tally = Tally::default();
        tally.record(Verdict::Held);
        tally.record(Verdict::Violated);
        tally.record(Verdict::NotApplicable);
        assert_eq!(tally.judged(), 2);
        assert_eq!(tally.not_applicable, 1);
    }
}
