//! What a test case *is* in this domain.
//!
//! An [`OnnxCase`] is one operator, the tensors going into it, and the opset revision that
//! says what the operator means. It is this domain's [`diff_fuzzer_core::Input`].
//!
//! # The case is the artifact; the seed is only context
//!
//! An earlier domain in this project stored seeds instead of cases, and a backend swap
//! made **810 of 814 findings stop reproducing** — a seed identifies a case only for the
//! exact generator that produced it, and generators change. So `OnnxCase` serializes
//! whole, and a finding carries the case rather than a recipe for regenerating it.
//!
//! # Why values are stored as bit patterns
//!
//! This is the subtle part, and getting it wrong would repeat that failure in a new form.
//!
//! Findings are written as JSON Lines by `diff-fuzzer-core`. **JSON cannot represent this
//! domain's subject matter**: `serde_json` writes `f32::NAN` and the infinities as `null`,
//! and while `-0.0` survives a round trip today, nothing in the format guarantees the sign
//! of zero. `NaN`, `±inf` and `±0.0` are not edge cases here — they are the *thesis*. Both
//! of this project's prior real findings were special-value bugs.
//!
//! So the wire form of a float tensor is its **`u32` bit patterns**, which are ordinary
//! integers to JSON and round-trip exactly, NaN payload included. In memory the values stay
//! `f32`, because that is what every runtime wants; only the serialized form differs. The
//! conversion lives in [`f32_bits`] and is covered by a round-trip test over hostile values.

use serde::{Deserialize, Serialize};

/// A tensor's element type, as ONNX numbers them on the wire.
///
/// Only the types this domain currently generates. The schema defines 27; adding one here
/// without also generating it would create a type nothing tests, and adding one *without*
/// extending the checks is the "check that silently narrows" failure this project has hit
/// ten times. **Adding a variant is a commit that also touches `validate`, the runtimes,
/// and the reference runner's dtype table.**
// `Hash` so the canonical form can derive it: grouping participants by the result they
// produced is how the oracle identifies an outlier, and that needs a hashable key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElemType {
    /// 32-bit IEEE-754 binary floating point. ONNX `TensorProto.DataType.FLOAT` = 1.
    F32,
}

impl ElemType {
    /// The integer ONNX uses for this type on the wire.
    pub fn wire(self) -> i32 {
        match self {
            // Cast to `i32` because the protobuf field is a plain `int32`: proto2 enums
            // are open, so the field's type cannot be the enum itself.
            ElemType::F32 => crate::pb::tensor_proto::DataType::Float as i32,
        }
    }
}

/// One tensor: its name in the graph, its shape, and its contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorValue {
    pub name: String,
    pub dims: Vec<i64>,
    pub elem_type: ElemType,
    /// The values, held as `f32` in memory and serialized as bit patterns. See the module
    /// note — this attribute is the whole reason a stored case survives a `NaN`.
    #[serde(with = "f32_bits")]
    pub values: Vec<f32>,
}

impl TensorValue {
    pub fn f32(name: &str, dims: Vec<i64>, values: Vec<f32>) -> Self {
        Self {
            name: name.to_owned(),
            dims,
            elem_type: ElemType::F32,
            values,
        }
    }

    /// How many elements the shape implies.
    ///
    /// Saturating at zero rather than wrapping: a negative dimension is invalid, and
    /// `validate` rejects it — but this must not produce a nonsense count in the meantime.
    pub fn element_count(&self) -> usize {
        if self.dims.iter().any(|d| *d < 0) {
            return 0;
        }
        self.dims.iter().product::<i64>().max(0) as usize
    }

    /// The rank (number of dimensions). Rank 0 is a scalar, which is legal in ONNX.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

/// Serializing `Vec<f32>` as `Vec<u32>` bit patterns, so JSON cannot destroy a value.
///
/// A `mod` used with `#[serde(with = "...")]`: serde looks for `serialize` and
/// `deserialize` functions inside it and calls them instead of the default ones. This is
/// the standard way to change how one field is represented without changing its type.
pub mod f32_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(values: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
        serializer.collect_seq(bits)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        let bits = Vec::<u32>::deserialize(deserializer)?;
        Ok(bits.into_iter().map(f32::from_bits).collect())
    }
}

/// The operators this domain can currently build.
///
/// Deliberately four. `08-RISKS.md` §4 is about checks that enumerate what existed when
/// they were written, and the countermeasure is a review rule: **adding a variant here
/// must be the same commit that extends `validate`, the arity table, and the tests.** A
/// small set keeps that rule cheap to honour while the skeleton is being built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpKind {
    /// Elementwise addition. Tier B — IEEE-754 governs it.
    Add,
    /// Elementwise subtraction. Tier B.
    Sub,
    /// Elementwise multiplication. Tier B.
    Mul,
    /// Passes its input through unchanged. Tier A — no arithmetic at all, which makes it
    /// the operator that tests the *plumbing* rather than any kernel.
    Identity,
}

impl OpKind {
    /// The `op_type` string ONNX knows this operator by.
    pub fn onnx_name(self) -> &'static str {
        match self {
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Identity => "Identity",
        }
    }

    /// How many inputs this operator takes.
    ///
    /// A table rather than a scatter of `if op == ...` checks, because arity is a property
    /// of the operator and every place that needs it should read the same answer. The SQL
    /// domain fixed one property at four separate sites because each site knew it locally.
    pub fn arity(self) -> usize {
        match self {
            OpKind::Add | OpKind::Sub | OpKind::Mul => 2,
            OpKind::Identity => 1,
        }
    }

    /// Whether this operator's output depends on the *values* of its inputs.
    ///
    /// Not decoration. The N2 go/no-go minimum requires at least 8 qualifying operators to
    /// be value-dependent, precisely so the bar cannot be met by operators that cannot
    /// exercise the adversarial-value thesis. That check needs this to be a property of
    /// the operator, recorded once.
    pub fn is_value_dependent(self) -> bool {
        match self {
            OpKind::Add | OpKind::Sub | OpKind::Mul => true,
            // Identity copies whatever it is given: the output depends on the input's
            // *bits*, but no arithmetic is performed, so no kernel is exercised.
            OpKind::Identity => false,
        }
    }

    /// Every operator, for tests that must cover all of them.
    ///
    /// Exists so a test cannot silently cover only the variants that existed when it was
    /// written — the failure mode `08-RISKS.md` §4 describes. A new variant joins this
    /// list and every exhaustive test picks it up automatically.
    pub const ALL: [OpKind; 4] = [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Identity];
}

/// One test case: an operator, its inputs, and the opset that defines its meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxCase {
    /// Which revision of the operator's semantics applies. Recorded per case because it is
    /// part of what a finding claims: "this is wrong" is only meaningful at a stated opset.
    pub opset: i64,
    pub op: OpKind,
    pub inputs: Vec<TensorValue>,
}

impl OnnxCase {
    pub fn new(op: OpKind, opset: i64, inputs: Vec<TensorValue>) -> Self {
        Self { opset, op, inputs }
    }

    /// The name the single output is given in the graph.
    pub const OUTPUT_NAME: &'static str = "out";

    /// The shape this case's output should have.
    ///
    /// Every operator here is shape-preserving over equally-shaped inputs, which is a
    /// consequence of `validate` refusing to build anything else — broadcasting is a
    /// deliberate N3 decision, not an N1 omission.
    pub fn output_dims(&self) -> Vec<i64> {
        self.inputs
            .first()
            .map(|t| t.dims.clone())
            .unwrap_or_default()
    }

    /// Total elements across all inputs. Used as a cheap size measure by shrinking and by
    /// corpus-shape reporting.
    pub fn total_elements(&self) -> usize {
        self.inputs.iter().map(TensorValue::element_count).sum()
    }
}

/// The marker that tells the engine this type is a test case.
///
/// `Input` has no methods; it exists so the other seams can say `type In: Input`, which is
/// what stops a generator from producing something unreportable. It requires `Clone`
/// (minimisation makes modified copies) and `Debug` (a case that cannot be printed cannot
/// be reported).
impl diff_fuzzer_core::Input for OnnxCase {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test that justifies the whole bit-pattern design.
    ///
    /// If this ever fails, stored findings containing special values are silently corrupt —
    /// which is exactly how an earlier domain lost 810 of 814 findings, in a different way.
    #[test]
    fn a_case_survives_json_with_its_special_values_intact() {
        let hostile = vec![
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
        ];
        let case = OnnxCase::new(
            OpKind::Add,
            22,
            vec![
                TensorValue::f32("a", vec![8], hostile.clone()),
                TensorValue::f32("b", vec![8], vec![0.0; 8]),
            ],
        );

        let json = serde_json::to_string(&case).expect("a case must serialize");
        let restored: OnnxCase = serde_json::from_str(&json).expect("and deserialize");

        // Compared as **bits**, not as values. `NaN != NaN` and `-0.0 == 0.0`, so an
        // equality check on the values would both fail spuriously and pass wrongly.
        for (original, round_tripped) in hostile.iter().zip(restored.inputs[0].values.iter()) {
            assert_eq!(
                original.to_bits(),
                round_tripped.to_bits(),
                "bit pattern changed crossing JSON"
            );
        }
        assert_eq!(case.op, restored.op);
        assert_eq!(case.opset, restored.opset);
    }

    /// Prove the check above could fail: storing these as plain JSON floats really does
    /// destroy them. Without this, the test above passes and nobody knows what it bought.
    #[test]
    fn plain_json_floats_would_have_destroyed_them() {
        let encoded = serde_json::to_string(&vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY])
            .expect("serde_json encodes non-finite floats as null rather than failing");

        assert_eq!(
            encoded, "[null,null,null]",
            "if this changed, re-examine whether the bit-pattern encoding is still needed"
        );
    }

    #[test]
    fn arity_is_defined_for_every_operator() {
        // Iterating `ALL` rather than listing operators, so a new variant is covered the
        // moment it is added rather than whenever someone remembers to extend this test.
        for op in OpKind::ALL {
            assert!(op.arity() >= 1, "{op:?} must take at least one input");
            assert!(!op.onnx_name().is_empty());
        }
    }

    #[test]
    fn element_count_handles_scalars_and_degenerate_shapes() {
        // Rank 0 is a scalar: the empty product is 1, which is correct, not a bug.
        assert_eq!(TensorValue::f32("s", vec![], vec![1.0]).element_count(), 1);
        // A zero dimension makes the tensor empty. Legal in ONNX and worth generating.
        assert_eq!(TensorValue::f32("e", vec![0, 3], vec![]).element_count(), 0);
        assert_eq!(
            TensorValue::f32("t", vec![2, 3], vec![0.0; 6]).element_count(),
            6
        );
        // A negative dimension is invalid; the count must not wrap into something huge.
        assert_eq!(
            TensorValue::f32("n", vec![-1, 3], vec![]).element_count(),
            0
        );
    }

    #[test]
    fn value_dependence_is_recorded_per_operator() {
        assert!(OpKind::Add.is_value_dependent());
        assert!(!OpKind::Identity.is_value_dependent());
        // The N2 minimum counts these, so at least some must qualify or the bar is
        // unmeetable by construction.
        assert!(
            OpKind::ALL
                .iter()
                .filter(|o| o.is_value_dependent())
                .count()
                >= 3
        );
    }
}
