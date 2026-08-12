//! Building a single-node ONNX model and serializing it to bytes.
//!
//! # The shape of a model
//!
//! ```text
//! ModelProto
//! ├── ir_version                       which revision of the *container* format
//! ├── opset_import [{domain, version}] which revision of the *operators* applies
//! └── graph: GraphProto
//!     ├── node   [NodeProto]           exactly one, for this domain
//!     ├── input  [ValueInfoProto]      typed AND shaped
//!     └── output [ValueInfoProto]
//! ```
//!
//! Two version numbers, and they are not the same thing. `ir_version` versions the
//! protobuf container — what fields a `ModelProto` may contain. The **opset** versions the
//! operator semantics: `Add` at opset 7 and `Add` at opset 14 are different specifications,
//! and ONNX publishes per-operator diffs between them. Getting these confused is the
//! easiest way to test something other than what you meant to.
//!
//! # Why serialize once
//!
//! The bytes produced here go to every runtime unchanged. Building the model separately
//! per runtime would silently destroy the comparison: a difference in results could then
//! be a difference in what each one was asked to compute, and no amount of care in the
//! oracle recovers from that.
//!
//! **The honest limit of that claim**, worth stating because it is easy to overstate: the
//! *model* is byte-identical everywhere, but the input *values* are not part of the model.
//! They are fed through each runtime's own API, so each one decodes the same `f32` buffer
//! its own way. That conversion is a small surface this domain does not control, and it is
//! the reason values are compared after normalization rather than assumed equal on entry.

use prost::Message;

use crate::pb::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
    ValueInfoProto, tensor_proto, tensor_shape_proto, type_proto,
};

/// The IR version written into every model this crate builds.
///
/// Deliberately **not** the newest the pinned schema supports. `onnx.proto` from onnx
/// 1.22.0 defines IR versions up to 13, but a runtime refuses to load a model whose IR
/// version it does not know, and the three runtimes under test move at different speeds.
/// Writing the newest would turn "this runtime is behind on the container format" into a
/// load failure on every case, which would look like a capability gap and hide the
/// operator behaviour this domain is trying to measure.
///
/// 10 is chosen as a conservative floor and is expected to be revisited at PHASE-N2, when
/// the capability census measures what each runtime actually accepts. That measurement,
/// not this comment, is what should eventually justify the number.
pub const IR_VERSION: i64 = 10;

/// The default opset these models are built against.
///
/// The `ai.onnx` domain reaches version 27 in onnx 1.22.0, but the opset a *runtime*
/// supports lags the one the specification has published. Like [`IR_VERSION`] this is a
/// provisional floor pending the PHASE-N2 census; opset becomes a generation axis later
/// (`PENDING` 2.6) rather than staying a constant.
pub const DEFAULT_OPSET: i64 = 22;

/// A tensor's element type, as ONNX numbers them.
///
/// Only the types PHASE-N0 needs. The schema defines 27; the rest arrive with the
/// operators that use them, because a type this crate can name but never generates is a
/// type nothing tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElemType {
    /// 32-bit IEEE-754 binary floating point.
    F32,
}

impl ElemType {
    /// The integer ONNX uses for this type on the wire.
    ///
    /// The generated `DataType` enum is cast to `i32` because the protobuf field is a
    /// plain `int32` — proto2 enums are open, so the field's type cannot be the enum.
    fn wire(self) -> i32 {
        match self {
            ElemType::F32 => tensor_proto::DataType::Float as i32,
        }
    }
}

/// Declares one graph input or output: its name, element type, and shape.
///
/// ONNX calls this a `ValueInfoProto`. Every graph input and output needs one, and it
/// must be **both typed and shaped** — some runtimes will load a model with an unshaped
/// input and then fail at execution, which is a confusing failure to debug and an easy one
/// to avoid.
fn value_info(name: &str, elem: ElemType, dims: &[i64]) -> ValueInfoProto {
    let shape = TensorShapeProto {
        dim: dims
            .iter()
            .map(|d| tensor_shape_proto::Dimension {
                // A dimension is a `oneof`: either a fixed number or a symbolic name for
                // a dynamic dimension. Every shape this domain builds is fully static,
                // because a dynamic dimension would leave the output shape undetermined
                // and an undetermined answer is exactly what the generator must refuse to
                // produce.
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*d)),
                denotation: None,
            })
            .collect(),
    };

    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto {
            // `r#type` because `type` is a Rust keyword. The `r#` prefix says "this is an
            // identifier, not the keyword" — prost adds it automatically when a protobuf
            // field name collides with one.
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: Some(elem.wire()),
                shape: Some(shape),
            })),
            denotation: None,
        }),
        doc_string: None,
        metadata_props: Vec::new(),
    }
}

/// Build a single-node model: one operator, `inputs` in, one output.
///
/// `..Default::default()` fills in every field not named — and for these types that is
/// most of them, since a `ModelProto` has a dozen fields this domain never sets. Listing
/// them all as `None` would bury the four that matter.
pub fn single_node_model(
    op_type: &str,
    inputs: &[(&str, ElemType, Vec<i64>)],
    output: (&str, ElemType, Vec<i64>),
    opset: i64,
) -> ModelProto {
    let node = NodeProto {
        input: inputs
            .iter()
            .map(|(name, _, _)| (*name).to_owned())
            .collect(),
        output: vec![output.0.to_owned()],
        name: Some(format!("{}_0", op_type.to_lowercase())),
        op_type: Some(op_type.to_owned()),
        ..Default::default()
    };

    let graph = GraphProto {
        node: vec![node],
        name: Some("g".to_owned()),
        input: inputs
            .iter()
            .map(|(name, elem, dims)| value_info(name, *elem, dims))
            .collect(),
        output: vec![value_info(output.0, output.1, &output.2)],
        ..Default::default()
    };

    ModelProto {
        ir_version: Some(IR_VERSION),
        // An empty domain string means `ai.onnx`, the main operator set. The other
        // domain, `ai.onnx.ml`, holds the traditional-ML operators and is out of scope.
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(opset),
        }],
        // Recorded in the model itself so that a `.onnx` file recovered from a findings
        // directory says where it came from without needing the log beside it.
        producer_name: Some("diff-fuzzer".to_owned()),
        graph: Some(graph),
        ..Default::default()
    }
}

/// The PHASE-N0 model: `Add` over two `f32` tensors of the given shape.
///
/// `Add` is the smallest useful choice — Tier B, elementwise, two inputs so broadcasting
/// rules are in play, and specified precisely enough by IEEE-754 that the four
/// participants should agree bit-for-bit.
pub fn add_model(dims: &[i64], opset: i64) -> ModelProto {
    single_node_model(
        "Add",
        &[
            ("a", ElemType::F32, dims.to_vec()),
            ("b", ElemType::F32, dims.to_vec()),
        ],
        ("c", ElemType::F32, dims.to_vec()),
        opset,
    )
}

/// Serialize a model to the bytes every runtime will be handed.
///
/// `encode_to_vec` comes from `prost::Message`, which is a **trait**: the generated types
/// implement it, and importing the trait is what brings the method into scope. That is
/// why the `use prost::Message;` at the top of this file looks unused but is not.
pub fn to_bytes(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_add_model_has_the_structure_onnx_expects() {
        let model = add_model(&[2, 3], DEFAULT_OPSET);

        let graph = model.graph.as_ref().expect("a model must carry a graph");
        assert_eq!(graph.node.len(), 1, "this domain builds single-node graphs");
        assert_eq!(graph.node[0].op_type.as_deref(), Some("Add"));
        assert_eq!(graph.node[0].input, vec!["a", "b"]);
        assert_eq!(graph.node[0].output, vec!["c"]);
        assert_eq!(graph.input.len(), 2);
        assert_eq!(graph.output.len(), 1);
    }

    #[test]
    fn every_graph_input_is_typed_and_shaped() {
        // Not pedantry: a runtime will load a model with an unshaped input and then fail
        // at execution, which reads as a runtime bug rather than as our omission.
        let model = add_model(&[2, 3], DEFAULT_OPSET);
        let graph = model.graph.expect("a model must carry a graph");

        for info in graph.input.iter().chain(graph.output.iter()) {
            // `TypeProto.value` is a `oneof` over five kinds — tensor, sequence, map,
            // optional, and sparse tensor. This domain builds only plain tensors, so
            // anything else is a bug in the builder rather than a case to handle.
            let Some(type_proto::Value::TensorType(tensor)) = info
                .r#type
                .as_ref()
                .expect("every value info must carry a type")
                .value
                .as_ref()
            else {
                panic!("this domain builds tensor types only");
            };

            assert_eq!(tensor.elem_type, Some(ElemType::F32.wire()));
            let shape = tensor.shape.as_ref().expect("every input must be shaped");
            assert_eq!(shape.dim.len(), 2);
            for dim in &shape.dim {
                assert!(
                    matches!(
                        dim.value,
                        Some(tensor_shape_proto::dimension::Value::DimValue(_))
                    ),
                    "dimensions must be static, never symbolic"
                );
            }
        }
    }

    /// Serializing must be deterministic: the same model must produce the same bytes
    /// every time, or "every runtime saw byte-identical input" is not a claim we can make.
    #[test]
    fn serialization_is_deterministic() {
        let first = to_bytes(&add_model(&[2, 3], DEFAULT_OPSET));
        let second = to_bytes(&add_model(&[2, 3], DEFAULT_OPSET));

        assert_eq!(first, second);
        assert!(!first.is_empty());
    }

    #[test]
    fn the_opset_is_recorded_on_the_model() {
        let model = add_model(&[2, 3], 17);
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].version, Some(17));
        // Empty domain == `ai.onnx`, the main operator set.
        assert_eq!(model.opset_import[0].domain.as_deref(), Some(""));
    }

    /// A round-trip through the wire format must preserve the model. This is really a
    /// test that the generated schema is coherent — if the build script produced types
    /// from a mismatched `.proto`, this is where it would show.
    #[test]
    fn a_model_survives_a_round_trip_through_bytes() {
        use prost::Message;

        let original = add_model(&[4, 5, 6], DEFAULT_OPSET);
        let decoded = ModelProto::decode(to_bytes(&original).as_slice())
            .expect("bytes we just wrote must decode");

        assert_eq!(original, decoded);
    }
}
