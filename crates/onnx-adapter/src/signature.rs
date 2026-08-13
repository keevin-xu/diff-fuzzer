//! What makes two divergences "the same problem".
//!
//! # Err finer, not coarser
//!
//! A signature groups occurrences so a campaign reports *problems* rather than *hits*. Getting it
//! wrong has asymmetric costs, and the asymmetry runs the opposite way to intuition:
//!
//! - **Too fine** splits one problem into several. The cost is a longer list, and a human notices
//!   immediately that the entries look alike.
//! - **Too coarse** merges two problems into one. The second problem is then *invisible* — and
//!   what it looks like is a shorter, cleaner list. **Merging looks exactly like success.**
//!
//! So the key is deliberately narrow, and where there was a choice this module took the finer one.
//!
//! # No runtime names in the key
//!
//! `06-ORACLES-AND-LEGAL-DIFFERENCES.md` is explicit, and the reason is worth stating: the same
//! defect produces different *summaries* depending on who else ran. F-001 appears as a two-faction
//! split when candle abstains and as a named lone dissenter when candle runs and agrees — same
//! bug, same operator, same wrong answer, two different sentences. A key built from the summary
//! would report it twice.
//!
//! The participants are recorded [alongside](Signature::participants) rather than inside the key,
//! so nothing is lost — the report can still say who disagreed with whom.

use std::fmt;

use crate::case::{ElemType, OnnxCase};
use crate::normalize::Canonical;
use diff_fuzzer_core::traits::NamedOutput;

/// What kind of disagreement it is.
///
/// Ordered from most to least self-evidently a defect, which is also the order they are detected
/// in — a crash is a crash whatever the values say.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum Kind {
    /// A participant panicked, aborted or segfaulted.
    Crash,
    /// A participant did not return within the bound.
    Timeout,
    /// One answered while another refused. The SQL domain's most productive signal.
    RejectedVersusOk,
    /// The results have different shapes.
    Shape,
    /// The results have different element types.
    Dtype,
    /// Same shape, same type, different values. The classic wrong answer.
    Value,
}

impl Kind {
    /// The token used in the key.
    pub fn token(self) -> &'static str {
        match self {
            Kind::Crash => "crash",
            Kind::Timeout => "timeout",
            Kind::RejectedVersusOk => "rejected-vs-ok",
            Kind::Shape => "shape",
            Kind::Dtype => "dtype",
            Kind::Value => "value",
        }
    }
}

/// The identity of a problem.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Signature {
    pub operator: String,
    pub opset: i64,
    /// The element type of the data input.
    ///
    /// **In the key deliberately.** F-001 is `tract` returning `Sign(0) = 1` at *integer* types
    /// while its float path is correct — the same operator, and only the type separates a defect
    /// from correct behaviour. A key without it would merge them and report the pair as one
    /// intermittent problem.
    pub elem_type: ElemType,
    /// The rank of the data input.
    ///
    /// Also in the key, for the reason `PENDING` 1.14 established twice over: behaviour differs
    /// by rank, and a key blind to rank merges a rank-0 problem into a rank-3 one.
    pub rank: usize,
    pub kind: Kind,
    /// Who disagreed with whom — recorded, **not** part of the key. See the module comment.
    pub participants: Vec<(String, String)>,
}

impl Signature {
    /// The de-duplication key. Stable, and free of runtime names.
    pub fn key(&self) -> String {
        format!(
            "{}/{}/{:?}/rank{}/{}",
            self.operator,
            self.opset,
            self.elem_type,
            self.rank,
            self.kind.token()
        )
    }
}

impl fmt::Display for Signature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.key())
    }
}

/// Derive the signature of a divergence.
///
/// Returns `None` when the outputs do not actually disagree — deriving a signature from agreement
/// would manufacture a finding.
pub fn of(case: &OnnxCase, outputs: &[NamedOutput<Canonical>]) -> Option<Signature> {
    let kind = kind_of(outputs)?;

    // Recorded sorted, so the same disagreement produces the same record regardless of the order
    // participants happened to execute in. The same determinism requirement the oracle's summary
    // has, for the same reason: a record that varies by run order cannot be de-duplicated.
    let mut participants: Vec<(String, String)> = outputs
        .iter()
        .map(|o| (o.implementation.clone(), o.output.kind.to_string()))
        .collect();
    participants.sort();

    Some(Signature {
        operator: case.op.onnx_name().to_string(),
        opset: case.opset,
        elem_type: crate::ops::data_elem_type(case),
        rank: crate::ops::data_rank(case),
        kind,
        participants,
    })
}

/// Classify the disagreement, most self-evident first.
fn kind_of(outputs: &[NamedOutput<Canonical>]) -> Option<Kind> {
    if outputs.iter().any(|o| o.output.kind == "crashed") {
        return Some(Kind::Crash);
    }
    if outputs.iter().any(|o| o.output.kind == "timed-out") {
        return Some(Kind::Timeout);
    }

    // Only participants that could have disagreed. An abstention is not a disagreement.
    let comparable: Vec<&NamedOutput<Canonical>> = outputs
        .iter()
        .filter(|o| !o.output.is_unsupported())
        .collect();
    if comparable.len() < 2 {
        return None;
    }

    let answered = comparable.iter().filter(|o| o.output.is_ok()).count();
    if answered > 0 && answered < comparable.len() {
        return Some(Kind::RejectedVersusOk);
    }
    if answered == 0 {
        // Everybody rejected. They agree, whatever they said about it.
        return None;
    }

    // All answered. Structure before values — a shape difference is a shape problem even if the
    // overlapping values also differ, and reporting it as a value problem would send a maintainer
    // looking in the wrong place.
    let first = &comparable[0].output;
    if comparable
        .iter()
        .any(|o| shapes_of(&o.output) != shapes_of(first))
    {
        return Some(Kind::Shape);
    }
    if comparable
        .iter()
        .any(|o| types_of(&o.output) != types_of(first))
    {
        return Some(Kind::Dtype);
    }
    if comparable
        .iter()
        .any(|o| !crate::normalize::equivalent(&o.output, first))
    {
        return Some(Kind::Value);
    }
    None
}

fn shapes_of(canonical: &Canonical) -> Vec<Vec<i64>> {
    canonical.tensors.iter().map(|t| t.dims.clone()).collect()
}

fn types_of(canonical: &Canonical) -> Vec<ElemType> {
    canonical.tensors.iter().map(|t| t.elem_type).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::normalize::OnnxNormalizer;
    use crate::outcome::OnnxOutcome;
    use crate::validation::{well_formed, well_formed_typed};
    use diff_fuzzer_core::Normalizer;

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
    fn a_value_difference_is_a_value_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![1.0, 2.0]),
            answered("tract", vec![1.0, 3.0]),
        ];
        let signature = of(&case, &outputs).expect("must have a signature");
        assert_eq!(signature.kind, Kind::Value);
        assert_eq!(signature.key(), "Add/22/F32/rank1/value");
    }

    /// **No runtime names in the key.** The same defect produces different summaries depending on
    /// who else took part; a key that varied with them would report one problem as several.
    #[test]
    fn the_key_does_not_depend_on_who_participated() {
        let case = well_formed(OpKind::Add, &[2], 22);

        let two = vec![answered("ort", vec![1.0]), answered("tract", vec![2.0])];
        let three = vec![
            answered("ort", vec![1.0]),
            answered("tract", vec![2.0]),
            answered("candle", vec![1.0]),
        ];
        assert_eq!(
            of(&case, &two).unwrap().key(),
            of(&case, &three).unwrap().key(),
            "the key changed because a third participant joined"
        );
        // But the participants are still recorded.
        assert_eq!(of(&case, &three).unwrap().participants.len(), 3);
    }

    /// And it must not depend on the order they ran in either.
    #[test]
    fn the_record_does_not_depend_on_execution_order() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let a = answered("ort", vec![1.0]);
        let b = answered("tract", vec![2.0]);
        let forward = of(&case, &[a.clone(), b.clone()]).unwrap();
        let reversed = of(&case, &[b, a]).unwrap();
        assert_eq!(forward, reversed);
    }

    /// **Element type is in the key**, because F-001 is a defect at integer types on an operator
    /// whose float path is correct. Merging them would report one intermittent problem instead of
    /// one real one.
    #[test]
    fn element_type_separates_two_problems() {
        let floats = well_formed_typed(OpKind::Sign, &[2], 22, ElemType::F32);
        let ints = well_formed_typed(OpKind::Sign, &[2], 22, ElemType::I64);
        let outputs = vec![answered("ort", vec![1.0]), answered("tract", vec![2.0])];
        assert_ne!(
            of(&floats, &outputs).unwrap().key(),
            of(&ints, &outputs).unwrap().key(),
            "a defect at one element type must not merge with the other"
        );
    }

    /// Rank likewise — the census had to learn this twice.
    #[test]
    fn rank_separates_two_problems() {
        let flat = well_formed(OpKind::Add, &[4], 22);
        let square = well_formed(OpKind::Add, &[2, 2], 22);
        let outputs = vec![answered("ort", vec![1.0]), answered("tract", vec![2.0])];
        assert_ne!(
            of(&flat, &outputs).unwrap().key(),
            of(&square, &outputs).unwrap().key()
        );
    }

    /// A crash outranks everything: it is a defect on its own, before any comparison.
    #[test]
    fn a_crash_outranks_a_value_difference() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![1.0]),
            named(
                "tract",
                OnnxOutcome::Crashed {
                    detail: "boom".into(),
                },
            ),
        ];
        assert_eq!(of(&case, &outputs).unwrap().kind, Kind::Crash);
    }

    /// Structure before values: a shape difference must not be reported as a value problem, or a
    /// maintainer goes looking in the wrong place.
    #[test]
    fn a_shape_difference_outranks_the_values_inside_it() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![1.0, 2.0]),
            answered("tract", vec![9.0, 9.0, 9.0]),
        ];
        assert_eq!(of(&case, &outputs).unwrap().kind, Kind::Shape);
    }

    #[test]
    fn answered_versus_rejected_is_its_own_kind() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![1.0]),
            named(
                "tract",
                OnnxOutcome::Rejected {
                    detail: "no".into(),
                },
            ),
        ];
        assert_eq!(of(&case, &outputs).unwrap().kind, Kind::RejectedVersusOk);
    }

    /// Agreement has no signature. Deriving one would manufacture a finding.
    #[test]
    fn agreement_has_no_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![answered("ort", vec![1.0]), answered("tract", vec![1.0])];
        assert!(of(&case, &outputs).is_none());
    }

    /// Two runtimes both rejecting agree, whatever they said about it.
    #[test]
    fn mutual_rejection_has_no_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            named("ort", OnnxOutcome::Rejected { detail: "a".into() }),
            named("tract", OnnxOutcome::Rejected { detail: "b".into() }),
        ];
        assert!(of(&case, &outputs).is_none());
    }

    /// An abstention is not a disagreement, and one survivor cannot disagree with anybody.
    #[test]
    fn a_lone_survivor_has_no_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![1.0]),
            named(
                "tract",
                OnnxOutcome::Unsupported {
                    reason: "no".into(),
                },
            ),
        ];
        assert!(of(&case, &outputs).is_none());
    }

    /// The licensed `NaN` rule must not produce a signature: forgiven differences are not
    /// findings, and a key derived without consulting the comparison would invent one.
    #[test]
    fn a_licensed_difference_produces_no_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let outputs = vec![
            answered("ort", vec![f32::from_bits(0x7fc0_0000)]),
            answered("tract", vec![f32::from_bits(0x7fc0_1234)]),
        ];
        assert!(
            of(&case, &outputs).is_none(),
            "a forgiven NaN difference is not a problem to report"
        );
    }
}
