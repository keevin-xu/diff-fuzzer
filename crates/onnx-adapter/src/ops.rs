//! What each operator *is* — arity, accepted types, output type, output shape, and the
//! minimal valid model that probes it.
//!
//! # Why this module exists before the generator
//!
//! The capability census (PHASE-N2) must answer *"does runtime R support operator O at
//! element type D?"*, and the only honest way to answer it is to **build a minimal valid
//! model and attempt it**. `08-RISKS.md` §3 names the alternative — reading a support table
//! — as trusting a claim about intent rather than measuring behaviour.
//!
//! That means per-operator knowledge has to exist before the census can run. It is the same
//! knowledge N3's generator needs, so it is built once, here.
//!
//! # Everything here was retrieved, not recalled
//!
//! The signatures, type constraints and `since` versions come from
//! `onnx.defs.get_schema(name, max_inclusive_version=22)` against the pinned `onnx` 1.22.0 —
//! the specification's own schema registry. They are recorded with the retrieval date in
//! `SPECS.md` §2.1.
//!
//! Three of those facts contradict what a reasonable person would assume, and each would
//! have produced invalid models read as capability gaps:
//!
//! - **`Squeeze` and `Unsqueeze` take `axes` as an *input*, not an attribute.** It was an
//!   attribute through opset 12 and moved at 13.
//! - **The comparisons return `Bool` whatever they were given**, and `Shape`/`Size` return
//!   `I64`. An output type assumed equal to the input is wrong for five operators.
//! - **`Round` first appears at opset 22**, the ceiling this domain currently builds at.
//!
//! # The output shape is computed, not stored
//!
//! `onnx.checker` requires a graph output to declare a `shape` (measured — `SPECS.md` §2.2).
//! It accepts symbolic dimensions, so this could have declared "unknown". It declares the
//! **exact** dimensions instead, computed per operator by [`output_spec`], because a wrong
//! shape is then *rejected by the checker* rather than silently tolerated. The stricter
//! option is the one that can fail, and a check that cannot fail is not evidence.

use crate::attrs::Attrs;
use crate::case::{ElemType, OnnxCase, OpKind, TensorData, TensorValue};

/// How tightly an operator's *value* semantics are specified.
///
/// The ordering that defends against legal-difference noise — the failure that consumed the
/// SQL domain. Tier A admits no rounding argument at all; Tier B is IEEE-754 governed and
/// should be bit-exact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Structural, integer, or logical. **Any divergence is a bug.**
    A,
    /// IEEE-754 elementwise float. Bit-exact expected.
    B,
    /// Quantization. **The strongest oracle in the domain**, and the reason is structural:
    /// `QuantizeLinear`, `MatMulInteger` and `DynamicQuantizeLinear` produce **integer**
    /// outputs, so comparison is exact with no tolerance, no rounding argument and no special
    /// values. `SPECS.md` §2q.
    ///
    /// A separate tier rather than folding into A, because N9.7 has to report this surface's
    /// yield *against* the Tier A/B baseline — and two things reported as one number cannot be
    /// compared with each other.
    Q,
}

/// The shape of an operator's signature, which is what drives output inference.
///
/// Grouping by family rather than writing a 31-arm match in every function: operators in a
/// family share their shape rule, so the rule is written once and a new operator joins a
/// family rather than adding another place to get it wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// One input, same shape and type out. `Identity`, `Abs`, `Sqrt`, `Not`.
    UnaryElementwise,
    /// Two inputs of one shape and type, same out. `Add`, `And`.
    BinaryElementwise,
    /// Two inputs, **`Bool` out**. `Equal`, `Greater`, `Less`.
    Comparison,
    /// `cond: Bool`, `x`, `y` — shape and type of `x`.
    Select,
    /// One input, output type given by the `to` attribute.
    Cast,
    /// data + shape input; output dims are the shape input's *values*.
    Reshape,
    /// One input, dims permuted by `perm`.
    Transpose,
    /// Inputs joined along `axis`.
    Concat,
    /// data + indices; `axis` replaced by the indices' shape.
    Gather,
    /// data + axes input; the listed dimensions removed.
    Squeeze,
    /// data + axes input; ones inserted at the listed positions.
    Unsqueeze,
    /// One input, **`I64` rank-1** out.
    Shape,
    /// One input, **`I64` scalar** out.
    Size,
    /// data + starts + ends.
    Slice,
    /// data + pads.
    Pad,
    /// `x`, `y_scale`, `y_zero_point` → quantized integer of the zero-point's type.
    Quantize,
    /// `x`, `x_scale`, `x_zero_point` → float of the scale's type.
    Dequantize,
    /// Two 8-bit matrices plus optional zero-points → **`int32`**, always.
    MatMulInteger,
    /// One float tensor → `uint8` output, plus the scale and zero-point it derived.
    DynamicQuantize,
}

/// Everything the harness needs to know about one operator.
#[derive(Debug, Clone)]
pub struct OpSpec {
    pub kind: OpKind,
    pub tier: Tier,
    pub family: Family,
    /// The operator revision in force at opset 22, from the schema registry.
    pub since: i64,
    /// Element types the **primary data input** accepts, restricted to those this adapter
    /// can build. Retrieved from the schema's type constraints.
    pub data_types: &'static [ElemType],
    /// Whether the output depends on the input **values** rather than only on their shapes.
    ///
    /// Load-bearing for the N2 go/no-go: the agreed minimum requires ≥8 qualifying operators
    /// to be value-dependent, so that the bar cannot be cleared by operators which cannot
    /// exercise the adversarial-value thesis at all.
    pub value_dependent: bool,
}

const FLOATS: &[ElemType] = &[ElemType::F32, ElemType::F64];
const NUMERIC: &[ElemType] = &[ElemType::F32, ElemType::F64, ElemType::I32, ElemType::I64];
const BOOL_ONLY: &[ElemType] = &[ElemType::Bool];
/// The 8-bit quantization targets. `SPECS.md` §2q.1.
const QUANTIZED: &[ElemType] = &[ElemType::I8, ElemType::U8];
/// `QuantizeLinear` and `DynamicQuantizeLinear` take a float *input*; the quantized type is on
/// the output. Keyed on the input, like every other row, so `data_elem_type` stays one rule.
const F32_ONLY: &[ElemType] = &[ElemType::F32];
const ANY: &[ElemType] = &[
    ElemType::F32,
    ElemType::F64,
    ElemType::I32,
    ElemType::I64,
    ElemType::Bool,
];

/// The catalog. One row per operator; see `SPECS.md` §2.1 for the retrieval.
pub fn spec(kind: OpKind) -> OpSpec {
    use Family as F;
    use OpKind as O;
    use Tier::{A, B, Q};

    let (tier, family, since, data_types, value_dependent) = match kind {
        // ── Tier A: structural. Output depends on shapes, not values. ──────────────
        O::Identity => (A, F::UnaryElementwise, 21, ANY, false),
        O::Reshape => (A, F::Reshape, 21, ANY, false),
        O::Transpose => (A, F::Transpose, 21, ANY, false),
        O::Concat => (A, F::Concat, 13, ANY, false),
        O::Squeeze => (A, F::Squeeze, 21, ANY, false),
        O::Unsqueeze => (A, F::Unsqueeze, 21, ANY, false),
        O::Shape => (A, F::Shape, 21, ANY, false),
        O::Size => (A, F::Size, 21, ANY, false),
        O::Slice => (A, F::Slice, 13, ANY, false),
        O::Pad => (A, F::Pad, 21, ANY, false),

        // ── Tier A: value-dependent. Discrete answers, no rounding argument. ───────
        // `Gather` and `Where` read values (indices, condition) to decide the answer.
        O::Gather => (A, F::Gather, 13, ANY, true),
        O::Where => (A, F::Select, 16, ANY, true),
        O::Cast => (A, F::Cast, 21, ANY, true),
        O::Equal => (A, F::Comparison, 19, ANY, true),
        O::Greater | O::Less => (A, F::Comparison, 13, NUMERIC, true),
        O::And | O::Or | O::Xor => (A, F::BinaryElementwise, 7, BOOL_ONLY, true),
        O::Not => (A, F::UnaryElementwise, 1, BOOL_ONLY, true),

        // ── Tier B: IEEE-754 elementwise. The densest special-value surface. ───────
        O::Add | O::Sub | O::Mul | O::Div => (B, F::BinaryElementwise, 14, NUMERIC, true),
        O::Min | O::Max => (B, F::BinaryElementwise, 13, NUMERIC, true),
        O::Abs | O::Neg | O::Sign => (B, F::UnaryElementwise, 13, NUMERIC, true),
        // Float-only: the schema's type constraint excludes integers for these.
        O::Sqrt | O::Floor | O::Ceil => (B, F::UnaryElementwise, 13, FLOATS, true),
        // `Round` does not exist below opset 22.
        O::Round => (B, F::UnaryElementwise, 22, FLOATS, true),

        // ── Tier Q: quantization. `SPECS.md` §2q, all four retrieved before any code. ──
        // `since` values are the operator revisions in force at opset 22.
        O::QuantizeLinear => (Q, F::Quantize, 21, F32_ONLY, true),
        O::DequantizeLinear => (Q, F::Dequantize, 21, QUANTIZED, true),
        O::MatMulInteger => (Q, F::MatMulInteger, 10, QUANTIZED, true),
        O::DynamicQuantizeLinear => (Q, F::DynamicQuantize, 11, F32_ONLY, true),
    };

    OpSpec {
        kind,
        tier,
        family,
        since,
        data_types,
        value_dependent,
    }
}

/// How many inputs an operator accepts, as an inclusive `(min, max)` range.
///
/// A **range**, not a number, because several of these are variadic or have optional
/// inputs — `Concat`, `Min` and `Max` take one or more, `Squeeze` takes one or two,
/// `Slice` three to five, `Pad` two to four. A single expected arity would have been wrong
/// for six operators, and `validate` would have rejected perfectly legal models.
///
/// The upper bound for the variadic operators is capped at what this domain actually builds
/// rather than at the schema's `2147483647`: a bound nothing enforces is not a bound, and a
/// case with a thousand inputs is not something the generator should be free to emit.
pub fn arity_range(kind: OpKind) -> (usize, usize) {
    match spec(kind).family {
        Family::UnaryElementwise
        | Family::Shape
        | Family::Size
        | Family::Cast
        | Family::Transpose => (1, 1),
        Family::BinaryElementwise
        | Family::Comparison
        | Family::Reshape
        | Family::Gather
        | Family::Unsqueeze
        | Family::Pad => (2, 2),
        Family::Select => (3, 3),
        // Variadic in the schema; capped here at what is built.
        Family::Concat => (1, 8),
        // `axes` is optional: omitting it squeezes every length-1 dimension.
        Family::Squeeze => (1, 2),
        // data, starts, ends, then optional axes and steps.
        Family::Slice => (3, 5),
        // x, scale, zero_point — the zero-point is optional in the schema but always supplied
        // here, because omitting it means a *default* zero-point and the default differs by
        // output type. Supplying it makes the case say what it means.
        Family::Quantize | Family::Dequantize => (3, 3),
        // A, B, then the two optional zero-points. Always supplied, same reasoning.
        Family::MatMulInteger => (4, 4),
        // One input and no parameters at all: it derives its own. `SPECS.md` §2q.4.
        Family::DynamicQuantize => (1, 1),
    }
}

/// What an operator requires of its inputs *relative to each other*.
///
/// The elementwise operators need every input to share a shape and an element type. Most
/// others do **not**, and applying the elementwise rule to them is wrong in a way that looks
/// like a validator working: `Reshape`'s second input is an `I64` shape vector, `Gather`'s
/// second is an `I64` index vector, and `Where`'s first is a `Bool` condition. Each is
/// deliberately a different shape and type from the data input.
///
/// This existed as an unconditional rule until the operator catalog was added, at which
/// point it rejected six perfectly legal probes. Recorded because the failure mode is the
/// project's most common one — a rule that was right for the constructs it was written
/// against, and silent about the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputAgreement {
    /// Every input must have the same dimensions.
    pub shapes: bool,
    /// Every input must have the same element type.
    pub elem_types: bool,
}

pub fn input_agreement(kind: OpKind) -> InputAgreement {
    let (shapes, elem_types) = match spec(kind).family {
        // Elementwise: one shape, one type, throughout.
        Family::UnaryElementwise | Family::BinaryElementwise | Family::Comparison => (true, true),
        // `Concat` joins along an axis, so shapes differ there by design — but every input
        // must still be the same type.
        Family::Concat => (false, true),
        // `Where` is `cond: Bool, x: T, y: T` — one shape, two types.
        Family::Select => (true, false),
        // Quantization parameters are scalars of a different type from the data, and the
        // zero-point's type is the *output* type rather than the input's. Nothing agrees.
        Family::Quantize | Family::Dequantize | Family::MatMulInteger => (false, false),
        // One input, so agreement is vacuous.
        Family::DynamicQuantize => (true, true),
        // The rest take structural inputs (shape vectors, index vectors, pad amounts) whose
        // shape and type are deliberately unlike the data input's.
        _ => (false, false),
    };
    InputAgreement { shapes, elem_types }
}

/// The element type and shape of a case's single output.
///
/// This is genuine shape inference over the subset of ONNX this domain builds. It is
/// deliberately **not** defensive: if a case is malformed the result is meaningless, and
/// `validate` plus the reference implementation are what catch that. Making this function
/// tolerant would hide the very errors those gates exist to report.
/// The operator's **additional** outputs, beyond the first.
///
/// Almost every operator here produces exactly one output, and `output_spec` describes it. The
/// exception is `DynamicQuantizeLinear`, which returns the quantized tensor *and* the scale and
/// zero-point it derived (`SPECS.md` §2q.4).
///
/// A separate function rather than making `output_spec` return a list, because the single-output
/// case is overwhelmingly the common one and every existing caller wants exactly one answer.
/// Returning a list everywhere would push an `unwrap`-shaped decision into a dozen call sites.
///
/// **The extra outputs are oracle surface, not overhead.** All three must agree across runtimes,
/// so an implementation that quantizes correctly but derives the wrong scale is caught.
pub fn extra_outputs(case: &OnnxCase) -> Vec<(&'static str, ElemType, Vec<i64>)> {
    match spec(case.op).family {
        // y_scale is a scalar float; y_zero_point is a scalar uint8.
        Family::DynamicQuantize => vec![
            ("out_scale", ElemType::F32, vec![]),
            ("out_zero_point", ElemType::U8, vec![]),
        ],
        _ => Vec::new(),
    }
}

pub fn output_spec(case: &OnnxCase) -> (ElemType, Vec<i64>) {
    let spec = spec(case.op);
    let first = case.inputs.first();
    let dims = first.map(|t| t.dims.clone()).unwrap_or_default();
    let elem = first.map_or(ElemType::F32, TensorValue::elem_type);

    match spec.family {
        Family::UnaryElementwise | Family::BinaryElementwise => (elem, dims),

        // Whatever went in, a truth value comes out.
        Family::Comparison => (ElemType::Bool, dims),

        // ── Quantization. `SPECS.md` §2q. ────────────────────────────────────────────
        // The output type is the **zero-point's**, not the input's: `y_zero_point` and `y`
        // share a data type (§2q.1). Shape is the input's — quantization is elementwise.
        Family::Quantize => case
            .inputs
            .get(2)
            .map_or((ElemType::I8, dims.clone()), |zp| {
                (zp.elem_type(), dims.clone())
            }),

        // The mirror: `y = (x - x_zero_point) * x_scale`, and at opset 22 there is no
        // `output_dtype` attribute, so the output type is the **scale's** (§2q.2).
        Family::Dequantize => case
            .inputs
            .get(1)
            .map_or((ElemType::F32, dims.clone()), |s| {
                (s.elem_type(), dims.clone())
            }),

        // **`int32` only**, whatever the 8-bit inputs were (§2q.3). Shape follows matmul:
        // `[m, k] x [k, n]` → `[m, n]`.
        Family::MatMulInteger => {
            let n = case
                .inputs
                .get(1)
                .and_then(|b| b.dims.last().copied())
                .unwrap_or(1);
            let mut shape = dims.clone();
            if let Some(last) = shape.last_mut() {
                *last = n;
            }
            (ElemType::I32, shape)
        }

        // Three outputs; `output_spec` describes the first, which is the quantized tensor.
        // `uint8` always (§2q.4).
        Family::DynamicQuantize => (ElemType::U8, dims),

        // `cond, x, y` — the answer has x's shape and type, not the condition's.
        Family::Select => case
            .inputs
            .get(1)
            .map_or((elem, dims.clone()), |x| (x.elem_type(), x.dims.clone())),

        // The `to` attribute names the output type directly.
        Family::Cast => {
            let target = match case.attrs.get("to") {
                Some(crate::attrs::AttrValue::Int(wire)) => {
                    ElemType::from_wire(*wire as i32).unwrap_or(elem)
                }
                _ => elem,
            };
            (target, dims)
        }

        // The *values* of the second input are the output shape — but a `0` means "copy the
        // input's dimension here" unless `allowzero=1` says to take it literally. Reading a `0`
        // as literal when the attribute is absent computes an output shape ONNX would not, and
        // the declared shape then disagrees with what the operator produces.
        Family::Reshape => {
            let allow_zero = matches!(
                case.attrs.get("allowzero"),
                Some(crate::attrs::AttrValue::Int(1))
            );
            let target = case
                .inputs
                .get(1)
                .and_then(|t| match &t.data {
                    TensorData::I64(v) => Some(v.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| dims.clone());
            let resolved = target
                .iter()
                .enumerate()
                .map(|(index, extent)| {
                    if *extent == 0 && !allow_zero {
                        dims.get(index).copied().unwrap_or(0)
                    } else {
                        *extent
                    }
                })
                .collect();
            (elem, resolved)
        }

        Family::Transpose => {
            let permuted = match case.attrs.get("perm") {
                Some(crate::attrs::AttrValue::Ints(perm)) => perm
                    .iter()
                    .map(|p| dims.get(*p as usize).copied().unwrap_or(0))
                    .collect(),
                // ONNX default: reverse the dimensions.
                _ => dims.iter().rev().copied().collect(),
            };
            (elem, permuted)
        }

        Family::Concat => {
            let axis = normalized_axis(&case.attrs, dims.len());
            let mut joined = dims.clone();
            if let Some(slot) = joined.get_mut(axis) {
                *slot = case
                    .inputs
                    .iter()
                    .map(|t| t.dims.get(axis).copied().unwrap_or(0))
                    .sum();
            }
            (elem, joined)
        }

        // data[..axis] ++ indices.dims ++ data[axis+1..]
        Family::Gather => {
            let axis = normalized_axis(&case.attrs, dims.len());
            let index_dims = case
                .inputs
                .get(1)
                .map(|t| t.dims.clone())
                .unwrap_or_default();
            let mut out = dims[..axis.min(dims.len())].to_vec();
            out.extend(index_dims);
            if axis < dims.len() {
                out.extend_from_slice(&dims[axis + 1..]);
            }
            (elem, out)
        }

        Family::Squeeze => {
            let axes = axes_input(case);
            let out = dims
                .iter()
                .enumerate()
                .filter(|(index, _)| !axes.contains(&(*index as i64)))
                .map(|(_, d)| *d)
                .collect();
            (elem, out)
        }

        Family::Unsqueeze => {
            let axes = axes_input(case);
            let mut out = dims.clone();
            // Ascending, so each insertion index still means what it said.
            let mut sorted = axes.clone();
            sorted.sort_unstable();
            for axis in sorted {
                let at = (axis as usize).min(out.len());
                out.insert(at, 1);
            }
            (elem, out)
        }

        // The shape *of* a tensor is a rank-1 int64 vector.
        Family::Shape => (ElemType::I64, vec![dims.len() as i64]),
        // The size is a scalar: rank 0, not rank 1 with one element.
        Family::Size => (ElemType::I64, Vec::new()),

        Family::Slice => {
            let starts = i64_input(case, 1);
            let ends = i64_input(case, 2);
            let mut out = dims.clone();
            for (index, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
                if let Some(slot) = out.get_mut(index) {
                    let extent = (*slot).min(*end) - *start;
                    *slot = extent.max(0);
                }
            }
            (elem, out)
        }

        // `pads` is [begin_0..begin_n, end_0..end_n].
        Family::Pad => {
            let pads = i64_input(case, 1);
            let rank = dims.len();
            let out = dims
                .iter()
                .enumerate()
                .map(|(index, d)| {
                    let before = pads.get(index).copied().unwrap_or(0);
                    let after = pads.get(index + rank).copied().unwrap_or(0);
                    d + before + after
                })
                .collect();
            (elem, out)
        }
    }
}

/// The `axis` attribute, with ONNX's negative-index convention resolved.
fn normalized_axis(attrs: &Attrs, rank: usize) -> usize {
    let raw = match attrs.get("axis") {
        Some(crate::attrs::AttrValue::Int(v)) => *v,
        _ => 0,
    };
    // A negative axis counts from the end, which is legal ONNX everywhere it appears.
    if raw < 0 {
        (rank as i64 + raw).max(0) as usize
    } else {
        raw as usize
    }
}

/// The `axes` **input** (not attribute) of `Squeeze`/`Unsqueeze`.
fn axes_input(case: &OnnxCase) -> Vec<i64> {
    i64_input(case, 1)
}

fn i64_input(case: &OnnxCase, index: usize) -> Vec<i64> {
    match case.inputs.get(index).map(|t| &t.data) {
        Some(TensorData::I64(v)) => v.clone(),
        _ => Vec::new(),
    }
}

/// The rank of the tensor whose shape the operator actually works on.
///
/// The companion to [`data_elem_type`], and keyed the same way for the same reason: for the
/// shape-input operators, `inputs[0]` is not always the tensor the operator's support depends
/// on. Defined once so the census and the capability lookup cannot compute it differently —
/// which they did for the element type, and it took a `Where` misclassification to notice.
pub fn data_rank(case: &OnnxCase) -> usize {
    case.inputs
        .iter()
        .find(|input| !input.is_initializer())
        .map_or(0, |input| input.dims.len())
}

/// The element type a case is *about* — the one the census keys on.
///
/// # Why this is a function and not `inputs[0].elem_type()`
///
/// For almost every operator the first input carries the data type. **`Where` is the
/// exception**: its first input is the boolean *condition*, and its data type is on the second.
///
/// The census keys each cell on the data type, so a lookup reading `inputs[0]` would ask about
/// `Where`/`Bool` while the census recorded `Where`/`F32`. Both answers are wrong in different
/// directions, and neither failure is visible — a capability lookup that silently consults the
/// wrong key returns a confident `false` and reclassifies a real disagreement as a gap.
///
/// `02-METHODOLOGY.md`: *a value matched by equality needs a single definition.* This is that
/// definition; the census and the capability model both call it, so they cannot disagree.
pub fn data_elem_type(case: &OnnxCase) -> ElemType {
    let index = match spec(case.op).family {
        // `cond, x, y` — the data type is on `x`.
        Family::Select => 1,
        _ => 0,
    };
    case.inputs
        .get(index)
        .or_else(|| case.inputs.first())
        .map_or(ElemType::F32, TensorValue::elem_type)
}

/// Every element type a case **requires** — its inputs and its output.
///
/// # Why the output type matters, and how missing it produced fake divergences
///
/// The capability model keys on the type a case is *about* ([`data_elem_type`]). For most
/// operators that is enough, because the output is the same type as the input. It is not enough
/// for three families:
///
/// - **`Cast`** produces whatever its `to` attribute names, which may be a type the runtime
///   cannot represent at all;
/// - **the comparisons** produce `Bool` whatever they were given;
/// - **`Shape`/`Size`** produce `I64`.
///
/// Measured consequence: asked to `Cast` an `f32` tensor to `int32`, `candle` returns **`int64`**
/// — it has no `int32` type. The lookup asked "does candle support `Cast` at `f32`?", got yes,
/// and the wrong-typed result was reported as a **divergence against two runtimes that agreed**.
/// It is not a divergence; it is a capability limit the census had already measured, consulted
/// through the wrong key.
///
/// Returning every required type lets the capability model ask the question that actually
/// decides the case: *can this runtime represent all of it?*
pub fn required_elem_types(case: &OnnxCase) -> Vec<ElemType> {
    let mut types: Vec<ElemType> = case.inputs.iter().map(TensorValue::elem_type).collect();
    types.push(output_spec(case).0);
    types.sort_unstable();
    types.dedup();
    types
}

/// Build the minimal valid model that probes `op` at `elem`.
///
/// `None` when the operator does not accept that element type, which is a *specification*
/// fact rather than a capability one — it must not be confused with a runtime declining the
/// operator. Feeding an operator a type its schema forbids would produce an invalid model,
/// and a runtime rejecting *that* says nothing about the runtime.
///
/// Values are ordinary. A probe answers "is this operator implemented", and hostile values
/// would confuse that with "does it handle `NaN` correctly" — a different question, asked by
/// the generator at N3.
pub fn probe(op: OpKind, elem: ElemType, opset: i64) -> Option<OnnxCase> {
    probe_at(op, elem, opset, 2)
}

/// The ranks the census probes.
///
/// # Why more than one
///
/// The census originally probed every operator at exactly `[2, 3]` — one shape, rank 2. That
/// made *support* look like a property of (operator, element type), and it is not: candle
/// implements `Neg` at `i64` **above rank 0 and fails at rank 0**, so the census recorded a
/// claim, the generator reached rank 0, and the refusal was reported as a divergence against
/// three runtimes that agreed. It looked exactly like a finding.
///
/// This is the same class the census already fixed once for element types: **support is a
/// property of the combination**, and a census scoped more narrowly than the generator
/// misclassifies whatever the generator reaches and the census did not. `PENDING` 1.14.
///
/// Rank 0 and rank 1 are the boundaries where implementations actually differ; rank 3 checks
/// that nothing degrades above the probed shape.
pub const PROBED_RANKS: [usize; 4] = [0, 1, 2, 3];

/// A representative shape at the given rank.
fn shape_at(rank: usize) -> Vec<i64> {
    match rank {
        0 => vec![],
        1 => vec![3],
        2 => vec![2, 3],
        _ => vec![2, 3, 2],
    }
}

/// Build a probe at a specific rank.
///
/// `None` means the combination cannot be built — either the schema forbids the element type,
/// or the family is not expressible at that rank (`Transpose` of a scalar, `Concat` with no
/// axis to join along). Both are **specification** facts, not capability ones.
pub fn probe_at(op: OpKind, elem: ElemType, opset: i64, rank: usize) -> Option<OnnxCase> {
    let spec = spec(op);
    if !spec.data_types.contains(&elem) || opset < spec.since {
        return None;
    }
    let dims = shape_at(rank);
    let count: i64 = dims.iter().product::<i64>().max(1);

    // Families that need an axis, a permutation, or a dimension to remove cannot be built
    // against a scalar. Declining to build is the honest answer; inventing a rank-1 stand-in
    // would record a rank-0 claim that was never measured at rank 0.
    let needs_an_axis = matches!(
        spec.family,
        Family::Transpose
            | Family::Concat
            | Family::Gather
            | Family::Squeeze
            | Family::Slice
            | Family::Pad
    );
    if rank == 0 && needs_an_axis {
        return None;
    }

    let case = match spec.family {
        Family::UnaryElementwise => OnnxCase::new(op, opset, vec![data("a", &dims, elem, 0)]),
        Family::BinaryElementwise | Family::Comparison => OnnxCase::new(
            op,
            opset,
            vec![data("a", &dims, elem, 0), data("b", &dims, elem, 1)],
        ),
        Family::Select => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, ElemType::Bool, 0),
                data("b", &dims, elem, 1),
                data("c", &dims, elem, 2),
            ],
        ),
        Family::Cast => {
            // Cast to something genuinely different, or the probe would not exercise a
            // conversion at all.
            let target = if elem == ElemType::I64 {
                ElemType::F32
            } else {
                ElemType::I64
            };
            OnnxCase::new(op, opset, vec![data("a", &dims, elem, 0)])
                .with_attrs(Attrs::new().int("to", i64::from(target.wire())))
        }
        Family::Reshape => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                // Flattened to rank 1, so the element count matches at any input rank.
                TensorValue::new("b", vec![1], TensorData::I64(vec![count])).as_initializer(),
            ],
        ),
        Family::Transpose => OnnxCase::new(op, opset, vec![data("a", &dims, elem, 0)])
            .with_attrs(Attrs::new().ints("perm", (0..dims.len() as i64).rev().collect())),
        Family::Concat => OnnxCase::new(
            op,
            opset,
            vec![data("a", &dims, elem, 0), data("b", &dims, elem, 1)],
        )
        .with_attrs(Attrs::new().int("axis", 0)),
        Family::Gather => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                // `Gather`'s indices are genuinely data — they select, and their *values*
                // decide the answer — so they stay a fed input. Index 0 exists at any rank.
                TensorValue::new("b", vec![1], TensorData::I64(vec![0])),
            ],
        )
        .with_attrs(Attrs::new().int("axis", 0)),
        Family::Squeeze => OnnxCase::new(
            op,
            opset,
            vec![
                // A length-1 dimension, since that is the only kind `Squeeze` may remove.
                data("a", &squeezable(&dims), elem, 0),
                TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
            ],
        ),
        Family::Unsqueeze => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
            ],
        ),
        Family::Shape | Family::Size => OnnxCase::new(op, opset, vec![data("a", &dims, elem, 0)]),
        Family::Slice => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                // A slice of [0, 1) along axis 0 is non-empty at every rank the probe builds.
                TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
                TensorValue::new("c", vec![1], TensorData::I64(vec![1])).as_initializer(),
            ],
        ),
        // ── Quantization probes. `SPECS.md` §2q. ─────────────────────────────────────
        // Scale and zero-point are **scalars** — per-tensor granularity, the simplest of the
        // three the spec allows (§2q.1). Per-axis and blocked are deliberately not probed:
        // support for one does not imply support for the others, and the census should not
        // claim what it did not measure.
        Family::Quantize => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                TensorValue::new("scale", vec![], TensorData::F32(vec![0.5])),
                // The zero-point's type IS the output type.
                quantized_scalar("zp", ElemType::I8, 0),
            ],
        ),
        Family::Dequantize => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                TensorValue::new("scale", vec![], TensorData::F32(vec![0.5])),
                quantized_scalar("zp", elem, 0),
            ],
        ),
        // `[m, k] x [k, n]`, with a deliberately small `k`: §2q.3 permits the int32
        // accumulation to overflow, and keeping the contracted dimension small makes that
        // impossible — so every probe has one determined answer.
        Family::MatMulInteger => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &[2, 3], elem, 0),
                data("b", &[3, 2], elem, 1),
                quantized_scalar("a_zp", elem, 0),
                quantized_scalar("b_zp", elem, 1),
            ],
        ),
        // No parameters at all — it derives its own (§2q.4).
        Family::DynamicQuantize => OnnxCase::new(op, opset, vec![data("a", &dims, elem, 0)]),

        Family::Pad => OnnxCase::new(
            op,
            opset,
            vec![
                data("a", &dims, elem, 0),
                // `pads` is [begin.., end..], so 2 * rank entries.
                TensorValue::new(
                    "b",
                    vec![2 * dims.len() as i64],
                    TensorData::I64(vec![1; 2 * dims.len()]),
                )
                .as_initializer(),
            ],
        )
        .with_attrs(Attrs::new().string("mode", "constant")),
    };
    Some(case)
}

/// A shape with a length-1 dimension, which is the only kind `Squeeze` may remove.
fn squeezable(dims: &[i64]) -> Vec<i64> {
    let mut shape = dims.to_vec();
    shape[0] = 1;
    shape
}

/// A scalar of a quantized type, for a zero-point input.
///
/// Zero-points are kept **in range and small** rather than at the saturation boundary: a
/// zero-point at `127` combined with a positive input saturates every output element, and a
/// probe whose every answer is the same boundary value cannot tell a working implementation
/// from a broken one. The census asks "is this operator implemented", not "does it saturate".
fn quantized_scalar(name: &str, elem: ElemType, offset: i64) -> TensorValue {
    let payload = match elem {
        ElemType::U8 => TensorData::U8(vec![(128 + offset) as u8]),
        _ => TensorData::I8(vec![offset as i8]),
    };
    TensorValue::new(name, vec![], payload)
}

/// Ordinary, distinct values of the requested type.
///
/// `offset` differs per input so a probe cannot pass by symmetry — a runtime that swapped
/// its operands would go unnoticed on `Add` and be caught on `Sub`.
fn data(name: &str, dims: &[i64], elem: ElemType, offset: usize) -> TensorValue {
    let count = dims.iter().product::<i64>().max(0) as usize;
    let base = (offset as i64 + 1) * 10;
    let payload = match elem {
        ElemType::F32 => TensorData::F32((0..count).map(|i| (base + i as i64) as f32).collect()),
        ElemType::F64 => TensorData::F64((0..count).map(|i| (base + i as i64) as f64).collect()),
        ElemType::I32 => TensorData::I32((0..count).map(|i| (base + i as i64) as i32).collect()),
        ElemType::I64 => TensorData::I64((0..count).map(|i| base + i as i64).collect()),
        ElemType::Bool => {
            TensorData::Bool((0..count).map(|i| (i + offset).is_multiple_of(2)).collect())
        }
        ElemType::I8 => TensorData::I8(
            (0..count)
                .map(|i| ((base + i as i64) % 127) as i8)
                .collect(),
        ),
        ElemType::U8 => TensorData::U8(
            (0..count)
                .map(|i| ((base + i as i64) % 255) as u8)
                .collect(),
        ),
    };
    TensorValue::new(name, dims.to_vec(), payload)
}

/// Every (operator, element type) pair the specification permits at `opset`.
///
/// This is the census's candidate list: the surface that *could* be supported, against which
/// what each runtime *does* support is measured.
pub fn candidates(opset: i64) -> Vec<(OpKind, ElemType)> {
    let mut pairs = Vec::new();
    for op in OpKind::ALL {
        for elem in ElemType::ALL {
            if PROBED_RANKS
                .iter()
                .any(|rank| probe_at(op, elem, opset, *rank).is_some())
            {
                pairs.push((op, elem));
            }
        }
    }
    pairs
}

/// Every (operator, element type, rank) the specification permits at `opset`.
///
/// The census's real candidate list. Wider than [`candidates`] because support is a property
/// of the combination — see [`PROBED_RANKS`].
pub fn candidates_by_rank(opset: i64) -> Vec<(OpKind, ElemType, usize)> {
    let mut cells = Vec::new();
    for op in OpKind::ALL {
        for elem in ElemType::ALL {
            for rank in PROBED_RANKS {
                if probe_at(op, elem, opset, rank).is_some() {
                    cells.push((op, elem, rank));
                }
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validate;

    const OPSET: i64 = 22;

    /// Every operator must be probeable at **some** element type, or it is in the catalog
    /// while nothing can ever test it.
    #[test]
    fn every_operator_has_at_least_one_probe() {
        for op in OpKind::ALL {
            let any = ElemType::ALL
                .into_iter()
                .any(|e| probe(op, e, OPSET).is_some());
            assert!(any, "{op:?} has no probe at any element type");
        }
    }

    /// Every probe must satisfy our own validator. A probe that fails here would be *our*
    /// invalid model, and a runtime rejecting it would be read as a capability gap.
    #[test]
    fn every_probe_is_well_formed() {
        for (op, elem) in candidates(OPSET) {
            let case = probe(op, elem, OPSET).expect("candidates only yields buildable pairs");
            let problems = validate(&case);
            assert!(
                problems.is_empty(),
                "{op:?} at {elem:?} produced an invalid probe: {problems:?}"
            );
        }
    }

    /// The type constraints must actually exclude something, or `data_types` is decoration.
    #[test]
    fn type_constraints_are_enforced() {
        // Retrieved facts, from SPECS.md §2.1.
        assert!(
            probe(OpKind::Sqrt, ElemType::I64, OPSET).is_none(),
            "Sqrt is float-only"
        );
        assert!(
            probe(OpKind::And, ElemType::F32, OPSET).is_none(),
            "And is bool-only"
        );
        assert!(
            probe(OpKind::Not, ElemType::I32, OPSET).is_none(),
            "Not is bool-only"
        );
        assert!(
            probe(OpKind::Add, ElemType::Bool, OPSET).is_none(),
            "Add excludes bool"
        );
        assert!(probe(OpKind::Greater, ElemType::Bool, OPSET).is_none());
        // ...and permit what the schema permits.
        assert!(probe(OpKind::Add, ElemType::I64, OPSET).is_some());
        assert!(probe(OpKind::Equal, ElemType::Bool, OPSET).is_some());
        assert!(probe(OpKind::Identity, ElemType::Bool, OPSET).is_some());
    }

    /// `Round` does not exist below opset 22 — a retrieved fact that would otherwise
    /// produce a model no runtime can load.
    #[test]
    fn an_operator_below_its_since_version_has_no_probe() {
        assert!(probe(OpKind::Round, ElemType::F32, 21).is_none());
        assert!(probe(OpKind::Round, ElemType::F32, 22).is_some());
        assert!(
            probe(OpKind::Equal, ElemType::F32, 18).is_none(),
            "Equal is since 19"
        );
    }

    /// The five operators whose output type is **not** the input type. Getting this wrong
    /// would declare a graph output the checker rejects.
    #[test]
    fn output_types_that_differ_from_the_input() {
        let bool_out = [OpKind::Equal, OpKind::Greater, OpKind::Less];
        for op in bool_out {
            let case = probe(op, ElemType::F32, OPSET).unwrap();
            assert_eq!(output_spec(&case).0, ElemType::Bool, "{op:?} returns bool");
        }
        for op in [OpKind::Shape, OpKind::Size] {
            let case = probe(op, ElemType::F32, OPSET).unwrap();
            assert_eq!(output_spec(&case).0, ElemType::I64, "{op:?} returns int64");
        }
        // Cast returns whatever `to` names.
        let case = probe(OpKind::Cast, ElemType::F32, OPSET).unwrap();
        assert_eq!(output_spec(&case).0, ElemType::I64);
    }

    /// Shape inference, checked against hand-computed answers for each structural family.
    #[test]
    fn output_shapes_are_inferred_correctly() {
        let expect = |op: OpKind, dims: Vec<i64>| {
            let case = probe(op, ElemType::F32, OPSET).unwrap();
            assert_eq!(output_spec(&case).1, dims, "{op:?}");
        };

        expect(OpKind::Identity, vec![2, 3]);
        expect(OpKind::Add, vec![2, 3]);
        expect(OpKind::Transpose, vec![3, 2]); // perm [1,0]
        expect(OpKind::Reshape, vec![6]); // flattened, so the probe builds at any rank
        expect(OpKind::Concat, vec![4, 3]); // two [2,3] along axis 0
        expect(OpKind::Gather, vec![1, 3]); // [2,3] indexed by [1] on axis 0
        expect(OpKind::Squeeze, vec![3]); // [1,3] minus axis 0
        expect(OpKind::Unsqueeze, vec![1, 2, 3]); // [2,3] plus a 1 at axis 0
        expect(OpKind::Shape, vec![2]); // rank of [2,3]
        expect(OpKind::Size, vec![]); // a scalar, rank 0
        expect(OpKind::Slice, vec![1, 3]); // [2,3] sliced 0..1 on axis 0
        expect(OpKind::Pad, vec![4, 5]); // [2,3] padded 1 on every side
        expect(OpKind::Where, vec![2, 3]);
    }

    /// The N2 go/no-go requires ≥8 **value-dependent** operators. If the catalog cannot
    /// supply that many, the bar is unmeetable by construction and the failure should be
    /// visible here rather than at the gate.
    #[test]
    fn enough_operators_are_value_dependent_for_the_go_no_go_bar() {
        let count = OpKind::ALL
            .into_iter()
            .filter(|op| spec(*op).value_dependent)
            .count();
        assert!(
            count >= 8,
            "only {count} value-dependent operators; the agreed N2 minimum needs 8"
        );
    }

    /// Structural operators must **not** be counted as value-dependent. `Transpose` moves
    /// numbers around without reading them, so it cannot exercise the adversarial-value
    /// thesis and must not help clear a bar that exists to measure exactly that.
    #[test]
    fn structural_operators_are_not_value_dependent() {
        for op in [
            OpKind::Identity,
            OpKind::Transpose,
            OpKind::Reshape,
            OpKind::Shape,
            OpKind::Size,
            OpKind::Squeeze,
            OpKind::Unsqueeze,
            OpKind::Concat,
        ] {
            assert!(!spec(op).value_dependent, "{op:?} does not read its values");
        }
        // But these do read values to decide the answer.
        for op in [OpKind::Gather, OpKind::Where, OpKind::Add, OpKind::Equal] {
            assert!(spec(op).value_dependent, "{op:?} reads its values");
        }
    }

    /// **The gate on this module's correctness.**
    ///
    /// Our own `validate` only checks what we thought to check. `onnx.checker`, reached
    /// through the reference implementation, checks the model against the *specification* —
    /// including that the declared output type and shape match what the operator actually
    /// produces. Every wrong entry in `output_spec` shows up here as a rejection.
    ///
    /// This is the second of the two validity gates from `06-ORACLES` §2, and it is the one
    /// that matters: a probe the reference rejects is **our** invalid model, and a runtime
    /// failing on it would be recorded as a capability gap that does not exist.
    #[test]
    fn every_probe_is_accepted_by_the_reference_implementation() {
        use crate::outcome::OnnxOutcome;
        use crate::reference::Reference;

        let mut reference = Reference::start().expect("the reference worker must start");
        let mut rejected = Vec::new();

        for (op, elem) in candidates(OPSET) {
            let case = probe(op, elem, OPSET).unwrap();
            let bytes = crate::model::build_bytes(&case);
            // Only the **fed** inputs cross the wire; the initializers are already constants
            // inside the model bytes. Sending them as well is refused by the runner, which
            // checks every name against the graph's declared inputs — and that check caught
            // this exact mistake the moment the initializer split was introduced.
            let fed: Vec<TensorValue> = case.fed_inputs().cloned().collect();
            match reference.run(&bytes, &fed).expect("the worker must reply") {
                OnnxOutcome::Ok(_) => {}
                OnnxOutcome::Rejected { detail } => {
                    // The *last* line of a Python traceback is the exception; the first is
                    // just the word "Traceback". Reporting the first made this failure
                    // unreadable the first time it fired.
                    let reason = detail
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("");
                    rejected.push(format!("{op:?}/{elem:?}: {reason}"));
                }
                other => rejected.push(format!("{op:?}/{elem:?}: {other}")),
            }
        }

        assert!(
            rejected.is_empty(),
            "{} of {} probes were rejected by the specification's own checker — these are \
             OUR invalid models, not capability gaps:\n{}",
            rejected.len(),
            candidates(OPSET).len(),
            rejected.join("\n")
        );
    }

    #[test]
    fn the_candidate_surface_is_substantial() {
        let pairs = candidates(OPSET);
        assert!(
            pairs.len() > 80,
            "only {} operator/type pairs — the census would be measuring very little",
            pairs.len()
        );
        // Every element type must appear, or a type is in the enum untested.
        for elem in ElemType::ALL {
            assert!(
                pairs.iter().any(|(_, e)| *e == elem),
                "{elem:?} appears in no candidate pair"
            );
        }
    }

    #[test]
    fn every_operator_is_assigned_a_tier() {
        let (a, b): (Vec<_>, Vec<_>) = OpKind::ALL
            .into_iter()
            .partition(|op| spec(*op).tier == Tier::A);
        assert!(
            !a.is_empty() && !b.is_empty(),
            "both tiers must be populated"
        );
        assert_eq!(a.len() + b.len(), OpKind::ALL.len());
    }
}
