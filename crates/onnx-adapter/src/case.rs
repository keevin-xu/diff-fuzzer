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
/// Only the types this domain generates. The schema defines 27; these five are what the
/// Tier A and Tier B operator surface actually needs — the logical operators require
/// `Bool`, the comparisons *produce* it, `Reshape` takes an `I64` shape input and `Gather`
/// takes `I64` indices, and `Cast` needs at least two integer widths to be worth testing.
///
/// **Adding a variant is a commit that also touches** `TensorData`, `validate`, every
/// runtime's extraction path, and the reference runner's dtype table. `08-RISKS.md` §4 is
/// about checks that enumerate what existed when they were written; the exhaustive `match`
/// on `TensorData` is what makes the compiler enforce that rule rather than a reviewer.
// `Hash` so the canonical form can derive it: grouping participants by the result they
// produced is how the oracle identifies an outlier, and that needs a hashable key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElemType {
    /// 32-bit IEEE-754 binary float. ONNX `FLOAT` = 1.
    F32,
    /// 64-bit IEEE-754 binary float. ONNX `DOUBLE` = 11.
    F64,
    /// ONNX `INT32` = 6.
    I32,
    /// ONNX `INT64` = 7. Also the type of shape and index inputs.
    I64,
    /// ONNX `BOOL` = 9.
    Bool,
}

impl ElemType {
    /// The integer ONNX uses for this type on the wire.
    pub fn wire(self) -> i32 {
        use crate::pb::tensor_proto::DataType;
        // Cast to `i32` because the protobuf field is a plain `int32`: proto2 enums are
        // open, so the field's type cannot be the enum itself.
        match self {
            ElemType::F32 => DataType::Float as i32,
            ElemType::F64 => DataType::Double as i32,
            ElemType::I32 => DataType::Int32 as i32,
            ElemType::I64 => DataType::Int64 as i32,
            ElemType::Bool => DataType::Bool as i32,
        }
    }

    /// The inverse of [`Self::wire`]. `None` for a type this adapter does not represent.
    ///
    /// Returning `Option` rather than defaulting: a type we cannot decode must be reported
    /// as such, never quietly read as something else. Decoding `INT64` bytes as `FLOAT`
    /// would produce a fabricated divergence that looks entirely real.
    pub fn from_wire(wire: i32) -> Option<Self> {
        ElemType::ALL.into_iter().find(|t| t.wire() == wire)
    }

    /// Whether this type has special values (`NaN`, `±inf`, `-0.0`) worth injecting.
    ///
    /// The integer and boolean types do not, which matters for the N2 go/no-go: an
    /// operator that only accepts them cannot exercise the adversarial-value thesis, and
    /// counting it toward a value-surface bar would be the kind of tidy number
    /// `02-METHODOLOGY.md` warns about.
    pub fn is_floating(self) -> bool {
        matches!(self, ElemType::F32 | ElemType::F64)
    }

    /// Every element type, so an exhaustive test cannot silently cover only the ones that
    /// existed when it was written.
    pub const ALL: [ElemType; 5] = [
        ElemType::F32,
        ElemType::F64,
        ElemType::I32,
        ElemType::I64,
        ElemType::Bool,
    ];
}

/// A tensor's contents, typed.
///
/// One enum rather than a `Vec<f32>` plus a separate `elem_type` field, because those two
/// can disagree and this cannot. The element type is *derived* from the data, so a tensor
/// claiming to be `Bool` while carrying floats is unrepresentable rather than merely
/// invalid — the compiler enforces what `validate` would otherwise have to check.
///
/// Floats serialize as bit patterns; see the module note. The integer and boolean variants
/// need no such treatment, because JSON represents them exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorData {
    F32(#[serde(with = "f32_bits")] Vec<f32>),
    F64(#[serde(with = "f64_bits")] Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
}

impl TensorData {
    /// The element type this data is.
    pub fn elem_type(&self) -> ElemType {
        match self {
            TensorData::F32(_) => ElemType::F32,
            TensorData::F64(_) => ElemType::F64,
            TensorData::I32(_) => ElemType::I32,
            TensorData::I64(_) => ElemType::I64,
            TensorData::Bool(_) => ElemType::Bool,
        }
    }

    /// How many values are actually stored.
    pub fn len(&self) -> usize {
        match self {
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
            TensorData::I32(v) => v.len(),
            TensorData::I64(v) => v.len(),
            TensorData::Bool(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The raw little-endian bytes, for the reference implementation's wire format and for
    /// building `TensorProto` payloads.
    ///
    /// Bytes rather than a numeric conversion, because a bit pattern is the only encoding
    /// that survives `NaN` payloads and the sign of zero intact. `Bool` is one byte per
    /// value, which is what ONNX and numpy both use.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        match self {
            TensorData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorData::Bool(v) => v.iter().map(|x| u8::from(*x)).collect(),
        }
    }

    /// Rebuild from raw little-endian bytes of a known type. The inverse of
    /// [`Self::to_le_bytes`].
    pub fn from_le_bytes(elem_type: ElemType, bytes: &[u8]) -> Self {
        fn chunks<const N: usize>(bytes: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
            bytes
                .chunks_exact(N)
                .map(|c| c.try_into().expect("chunks_exact yields exactly N bytes"))
        }
        match elem_type {
            ElemType::F32 => TensorData::F32(chunks::<4>(bytes).map(f32::from_le_bytes).collect()),
            ElemType::F64 => TensorData::F64(chunks::<8>(bytes).map(f64::from_le_bytes).collect()),
            ElemType::I32 => TensorData::I32(chunks::<4>(bytes).map(i32::from_le_bytes).collect()),
            ElemType::I64 => TensorData::I64(chunks::<8>(bytes).map(i64::from_le_bytes).collect()),
            ElemType::Bool => TensorData::Bool(bytes.iter().map(|b| *b != 0).collect()),
        }
    }

    /// The values as comparison keys — one `u64` per element, holding the exact bits.
    ///
    /// This is what the oracle compares. Bits rather than values because `NaN != NaN` would
    /// report identical results as a divergence, and `-0.0 == 0.0` would report a genuine
    /// signed-zero disagreement as agreement. Widened to `u64` so every type shares one
    /// representation and the canonical form stays a single shape.
    pub fn to_bit_keys(&self) -> Vec<u64> {
        match self {
            TensorData::F32(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
            TensorData::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
            TensorData::I32(v) => v.iter().map(|x| *x as u32 as u64).collect(),
            TensorData::I64(v) => v.iter().map(|x| *x as u64).collect(),
            TensorData::Bool(v) => v.iter().map(|x| u64::from(*x)).collect(),
        }
    }

    /// The values as `f32`, if that is what this holds.
    ///
    /// `Option` rather than a panicking accessor: asking a `Bool` tensor for its floats is
    /// a caller error, and returning `None` makes the caller say what it wants to happen.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            TensorData::F32(v) => Some(v),
            _ => None,
        }
    }

    /// Whether every element is `NaN` or infinite — a result that cannot discriminate
    /// between implementations. Never true for the non-floating types, which have no such
    /// values.
    pub fn is_entirely_undefined(&self) -> bool {
        match self {
            TensorData::F32(v) => !v.is_empty() && v.iter().all(|x| !x.is_finite()),
            TensorData::F64(v) => !v.is_empty() && v.iter().all(|x| !x.is_finite()),
            _ => false,
        }
    }
}

/// One tensor: its name in the graph, its shape, and its typed contents.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorValue {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: TensorData,
}

impl TensorValue {
    pub fn new(name: &str, dims: Vec<i64>, data: TensorData) -> Self {
        Self {
            name: name.to_owned(),
            dims,
            data,
        }
    }

    /// Shorthand for the common case. The other types get `new` with an explicit
    /// [`TensorData`], which keeps the type visible at the call site where it matters.
    pub fn f32(name: &str, dims: Vec<i64>, values: Vec<f32>) -> Self {
        Self::new(name, dims, TensorData::F32(values))
    }

    /// The element type, derived from the data rather than stored beside it.
    pub fn elem_type(&self) -> ElemType {
        self.data.elem_type()
    }

    /// How many elements the *shape* implies — which may disagree with how many are
    /// stored. `validate` is what catches that; this must not paper over it.
    ///
    /// Saturating at zero rather than wrapping: a negative dimension is invalid, and this
    /// must not produce a nonsense count before `validate` rejects it.
    pub fn element_count(&self) -> usize {
        if self.dims.iter().any(|d| *d < 0) {
            return 0;
        }
        self.dims.iter().product::<i64>().max(0) as usize
    }

    /// The rank. Rank 0 is a scalar, which is legal in ONNX and worth generating.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// The values as `f32`, if that is what this tensor holds. See [`TensorData::as_f32`].
    pub fn as_f32(&self) -> Option<&[f32]> {
        self.data.as_f32()
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

/// The same treatment for `f64`. See [`f32_bits`].
pub mod f64_bits {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(values: &[f64], serializer: S) -> Result<S::Ok, S::Error> {
        let bits: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
        serializer.collect_seq(bits)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f64>, D::Error> {
        let bits = Vec::<u64>::deserialize(deserializer)?;
        Ok(bits.into_iter().map(f64::from_bits).collect())
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
        for (original, round_tripped) in hostile
            .iter()
            .zip(restored.inputs[0].as_f32().expect("f32 tensor").iter())
        {
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
