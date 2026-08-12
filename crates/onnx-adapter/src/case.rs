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

use crate::attrs::Attrs;

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
// `Ord` so the capability model can key an ordered set on it — an ordered set rather than a
// hash set because a capability matrix is read by humans and iterated in reports, and a
// stable order is worth more than a marginally faster lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
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

/// What an operator input *is* — data flowing through the graph, or configuration.
///
/// # Why the distinction exists
///
/// ONNX makes no type-level difference between "the tensor to reshape" and "the shape to
/// reshape it to": both are node inputs. But they are not the same kind of thing, and treating
/// them alike broke two things at once.
///
/// N0.3 decided that values are **graph inputs** rather than baked-in `initializer` constants,
/// because an all-initializer graph can be constant-folded at load time — which would test the
/// optimizer while appearing to test the operator, and would do so silently. That reasoning is
/// correct for **data**.
///
/// It is wrong for a **shape vector**. The N2 census measured `Reshape`, `Squeeze`, `Unsqueeze`,
/// `Slice` and `Pad` failing **0/5 on `tract`** — 25 of its 29 rejections — because `tract`
/// types the graph statically at load and cannot infer an output shape whose shape input only
/// arrives at run time. A shape vector is not data; it is operator **configuration**, no
/// different in kind from the `perm` attribute that `Transpose` takes.
///
/// So the rule splits by **role**, not by operator:
///
/// | role | emitted as | why |
/// |---|---|---|
/// | [`InputRole::Data`] | a graph input, fed at execution | keeps the kernel in the path |
/// | [`InputRole::Initializer`] | a constant in the model | it is configuration, and static shape inference needs it |
///
/// The data input of every operator stays a graph input, so nothing can be folded away to a
/// constant — which is the property N0.3 was protecting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum InputRole {
    /// Fed at execution time through each runtime's own API. The default.
    #[default]
    Data,
    /// Baked into the model as a constant `TensorProto`.
    ///
    /// Not listed among the graph's inputs: since IR version 4 an initializer need not be, and
    /// listing it would make every runtime expect it to be fed as well.
    Initializer,
}

/// One tensor: its name in the graph, its shape, its typed contents, and its role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorValue {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: TensorData,
    /// Whether this is data or configuration. See [`InputRole`].
    ///
    /// `#[serde(default)]` so a finding stored before roles existed still loads, as `Data` —
    /// the backward-compatible-deserializer rule from `08-RISKS.md` §11.
    #[serde(default)]
    pub role: InputRole,
}

impl TensorValue {
    pub fn new(name: &str, dims: Vec<i64>, data: TensorData) -> Self {
        Self {
            name: name.to_owned(),
            dims,
            data,
            role: InputRole::Data,
        }
    }

    /// The same tensor, emitted as a constant in the model rather than fed at execution.
    ///
    /// For shape vectors, axis lists and pad amounts — operator configuration that a runtime
    /// doing static shape inference must be able to see at load time.
    #[must_use]
    pub fn as_initializer(mut self) -> Self {
        self.role = InputRole::Initializer;
        self
    }

    /// Whether this input is baked into the model rather than fed at execution.
    pub fn is_initializer(&self) -> bool {
        self.role == InputRole::Initializer
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

/// The same treatment for a single `f32` — used by float *attributes*, which can also
/// legitimately be an infinity. See [`f32_bits`].
pub mod f32_bits_scalar {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(value.to_bits())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        Ok(f32::from_bits(u32::deserialize(deserializer)?))
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

/// The operators this domain builds.
///
/// **Tier A and Tier B**, chosen by how tightly their *value* semantics are specified rather
/// than by popularity — legal-difference noise is what consumed the SQL domain, and these
/// admit no rounding argument. Per-operator facts (arity, accepted types, output type and
/// shape) live in [`crate::ops`], retrieved from the schema registry rather than recalled.
///
/// **Adding a variant is a commit that also touches** `ops::spec`, `ops::output_spec`,
/// `ops::probe`, and the census. The exhaustive `match` in `ops::spec` is what makes the
/// compiler enforce that instead of a reviewer having to remember it — `08-RISKS.md` §4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OpKind {
    // ── Tier A — structural ───────────────────────────────────────────────────────
    Identity,
    Reshape,
    Transpose,
    Concat,
    Squeeze,
    Unsqueeze,
    Shape,
    Size,
    Slice,
    Pad,
    // ── Tier A — discrete, value-reading ──────────────────────────────────────────
    Gather,
    Where,
    Cast,
    Equal,
    Greater,
    Less,
    And,
    Or,
    Xor,
    Not,
    // ── Tier B — IEEE-754 elementwise ─────────────────────────────────────────────
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Abs,
    Neg,
    Sign,
    Sqrt,
    Floor,
    Ceil,
    Round,
}

impl OpKind {
    /// The `op_type` string ONNX knows this operator by.
    ///
    /// Identical to the Rust variant name for every operator here, which is why this is a
    /// mechanical mapping rather than a table with room to disagree with itself.
    pub fn onnx_name(self) -> &'static str {
        match self {
            OpKind::Identity => "Identity",
            OpKind::Reshape => "Reshape",
            OpKind::Transpose => "Transpose",
            OpKind::Concat => "Concat",
            OpKind::Squeeze => "Squeeze",
            OpKind::Unsqueeze => "Unsqueeze",
            OpKind::Shape => "Shape",
            OpKind::Size => "Size",
            OpKind::Slice => "Slice",
            OpKind::Pad => "Pad",
            OpKind::Gather => "Gather",
            OpKind::Where => "Where",
            OpKind::Cast => "Cast",
            OpKind::Equal => "Equal",
            OpKind::Greater => "Greater",
            OpKind::Less => "Less",
            OpKind::And => "And",
            OpKind::Or => "Or",
            OpKind::Xor => "Xor",
            OpKind::Not => "Not",
            OpKind::Add => "Add",
            OpKind::Sub => "Sub",
            OpKind::Mul => "Mul",
            OpKind::Div => "Div",
            OpKind::Min => "Min",
            OpKind::Max => "Max",
            OpKind::Abs => "Abs",
            OpKind::Neg => "Neg",
            OpKind::Sign => "Sign",
            OpKind::Sqrt => "Sqrt",
            OpKind::Floor => "Floor",
            OpKind::Ceil => "Ceil",
            OpKind::Round => "Round",
        }
    }

    /// How many inputs this operator accepts, inclusive `(min, max)`.
    ///
    /// Delegates to the catalog so there is one answer rather than two that can drift.
    pub fn arity_range(self) -> (usize, usize) {
        crate::ops::arity_range(self)
    }

    /// Whether this operator's output depends on the *values* of its inputs.
    ///
    /// Delegates to the catalog so there is one answer, not two that can drift. Load-bearing
    /// for the N2 go/no-go bar, which requires a stated number of value-dependent operators
    /// precisely so it cannot be cleared by operators that read nothing.
    pub fn is_value_dependent(self) -> bool {
        crate::ops::spec(self).value_dependent
    }

    /// Every operator, so an exhaustive test cannot silently cover only the ones that
    /// existed when it was written.
    pub const ALL: [OpKind; 33] = [
        OpKind::Identity,
        OpKind::Reshape,
        OpKind::Transpose,
        OpKind::Concat,
        OpKind::Squeeze,
        OpKind::Unsqueeze,
        OpKind::Shape,
        OpKind::Size,
        OpKind::Slice,
        OpKind::Pad,
        OpKind::Gather,
        OpKind::Where,
        OpKind::Cast,
        OpKind::Equal,
        OpKind::Greater,
        OpKind::Less,
        OpKind::And,
        OpKind::Or,
        OpKind::Xor,
        OpKind::Not,
        OpKind::Add,
        OpKind::Sub,
        OpKind::Mul,
        OpKind::Div,
        OpKind::Min,
        OpKind::Max,
        OpKind::Abs,
        OpKind::Neg,
        OpKind::Sign,
        OpKind::Sqrt,
        OpKind::Floor,
        OpKind::Ceil,
        OpKind::Round,
    ];

    /// The elementwise subset the N1 skeleton generator produces.
    ///
    /// Separate from [`Self::ALL`] on purpose: the generator builds identically-shaped
    /// inputs, which is right for these and wrong for `Reshape`, `Gather` and friends. Those
    /// need per-operator construction, which arrives with the real generator at N3. Naming
    /// the subset keeps that limitation visible instead of implicit.
    pub const ELEMENTWISE: [OpKind; 4] = [OpKind::Add, OpKind::Sub, OpKind::Mul, OpKind::Identity];
}

/// One test case: an operator, its inputs, and the opset that defines its meaning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnnxCase {
    /// Which revision of the operator's semantics applies. Recorded per case because it is
    /// part of what a finding claims: "this is wrong" is only meaningful at a stated opset.
    pub opset: i64,
    pub op: OpKind,
    pub inputs: Vec<TensorValue>,
    /// The node's static parameters — `axis`, `perm`, `to`, and friends.
    ///
    /// Empty for the elementwise operators, which take none. Whether a given parameter is
    /// an attribute or an *input* is per-operator and per-opset, and it changes between
    /// versions; see [`crate::attrs`].
    #[serde(default)]
    pub attrs: Attrs,
}

impl OnnxCase {
    pub fn new(op: OpKind, opset: i64, inputs: Vec<TensorValue>) -> Self {
        Self {
            opset,
            op,
            inputs,
            attrs: Attrs::new(),
        }
    }

    /// The same case with attributes attached.
    ///
    /// `#[serde(default)]` on the field means a finding stored before attributes existed
    /// still deserializes, with none. That is the backward-compatible-deserializer rule
    /// from `08-RISKS.md` §11: an earlier domain broke every stored finding by widening a
    /// field's type, and the fix was to make old records keep loading.
    #[must_use]
    pub fn with_attrs(mut self, attrs: Attrs) -> Self {
        self.attrs = attrs;
        self
    }

    /// The name the single output is given in the graph.
    pub const OUTPUT_NAME: &'static str = "out";

    /// The inputs a runtime must be **fed** — everything that is not an initializer.
    ///
    /// Every runtime and the reference boundary use this rather than `inputs`. Feeding an
    /// initializer would be an error: it is already in the model, and a runtime that received
    /// it twice would reject the case.
    pub fn fed_inputs(&self) -> impl Iterator<Item = &TensorValue> {
        self.inputs.iter().filter(|t| !t.is_initializer())
    }

    /// The inputs baked into the model as constants.
    pub fn initializers(&self) -> impl Iterator<Item = &TensorValue> {
        self.inputs.iter().filter(|t| t.is_initializer())
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
            let (min, max) = op.arity_range();
            assert!(min >= 1, "{op:?} must take at least one input");
            assert!(max >= min, "{op:?} has an inverted arity range");
            assert!(!op.onnx_name().is_empty());
        }
    }

    /// Operator names must be distinct and must match the ONNX spelling exactly — a typo
    /// here builds a node no runtime recognises, which would read as universal
    /// non-support rather than as our error.
    #[test]
    fn operator_names_are_distinct() {
        let mut names: Vec<&str> = OpKind::ALL.iter().map(|o| o.onnx_name()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count, "two operators share an ONNX name");
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
