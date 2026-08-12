//! Turning an [`OnnxCase`] into the bytes every runtime is handed.
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
//! and ONNX publishes per-operator diffs between them. Confusing the two is the easiest way
//! to test something other than what you meant to.
//!
//! # Why serialize once
//!
//! The bytes produced here go to every runtime unchanged. Building the model separately per
//! runtime would silently destroy the comparison: a difference in results could then be a
//! difference in what each one was asked to compute, and no care in the oracle recovers
//! from that.
//!
//! **The honest limit of that claim**, worth stating because it is easy to overstate: the
//! *model* is byte-identical everywhere, but the input **values are not part of the
//! model**. They are declared as graph inputs and fed through each runtime's own API, so
//! each one decodes the same buffer its own way. Values are graph inputs rather than
//! baked-in `initializer` constants deliberately — an all-initializer graph can be
//! constant-folded at load time, which would test the optimizer while appearing to test the
//! operator.

use prost::Message;

use crate::case::{ElemType, OnnxCase, OpKind, TensorValue};
use crate::pb::{
    GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorShapeProto, TypeProto,
    ValueInfoProto, tensor_shape_proto, type_proto,
};

/// The IR version written into every model this crate builds.
///
/// Deliberately **not** the newest the pinned schema supports. `onnx.proto` from onnx
/// 1.22.0 defines IR versions up to 13, but a runtime refuses to load a model whose IR
/// version it does not know, and the runtimes under test move at different speeds. Writing
/// the newest would turn "this runtime is behind on the container format" into a load
/// failure on every case — which would present as a capability gap and hide the operator
/// behaviour this domain is trying to measure.
///
/// 10 is a conservative floor, to be revisited at PHASE-N2 when the census measures what
/// each runtime actually accepts. That measurement, not this comment, is what should
/// eventually justify the number.
pub const IR_VERSION: i64 = 10;

/// The default opset models are built against.
///
/// The `ai.onnx` domain reaches 27 in onnx 1.22.0, but the opset a *runtime* supports lags
/// what the specification has published. Provisional, like [`IR_VERSION`]; opset becomes a
/// generation axis later (`PENDING` 2.6) rather than staying a constant.
pub const DEFAULT_OPSET: i64 = 22;

/// Declare one graph input or output: name, element type, shape.
///
/// ONNX calls this a `ValueInfoProto`. Every graph input and output needs one, and it must
/// be **both typed and shaped** — some runtimes load a model with an unshaped input and
/// then fail at execution, which is a confusing failure to debug and an easy one to avoid.
fn value_info(name: &str, elem: ElemType, dims: &[i64]) -> ValueInfoProto {
    let shape = TensorShapeProto {
        dim: dims
            .iter()
            .map(|d| tensor_shape_proto::Dimension {
                // A dimension is a `oneof`: either a fixed number or a symbolic name for a
                // dynamic one. Every shape here is static, because a dynamic dimension
                // leaves the output shape undetermined — and an undetermined answer is
                // exactly what the generator must refuse to produce.
                value: Some(tensor_shape_proto::dimension::Value::DimValue(*d)),
                denotation: None,
            })
            .collect(),
    };

    ValueInfoProto {
        name: Some(name.to_owned()),
        r#type: Some(TypeProto {
            // `r#type` because `type` is a Rust keyword. The `r#` prefix says "identifier,
            // not keyword" — prost adds it whenever a protobuf field name collides.
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

/// Build the `ModelProto` for a case.
///
/// `..Default::default()` fills in every field not named — most of them, since a
/// `ModelProto` has a dozen this domain never sets. Listing them all as `None` would bury
/// the four that matter.
pub fn build(case: &OnnxCase) -> ModelProto {
    let node = NodeProto {
        input: case.inputs.iter().map(|t| t.name.clone()).collect(),
        output: vec![OnnxCase::OUTPUT_NAME.to_owned()],
        name: Some(format!("{}_0", case.op.onnx_name().to_lowercase())),
        op_type: Some(case.op.onnx_name().to_owned()),
        // Order is preserved from the case, which is what keeps serialization
        // byte-identical for the same case. See `attrs.rs`.
        attribute: case.attrs.to_protos(),
        ..Default::default()
    };

    let elem_type = case
        .inputs
        .first()
        .map_or(ElemType::F32, TensorValue::elem_type);

    let graph = GraphProto {
        node: vec![node],
        name: Some("g".to_owned()),
        input: case
            .inputs
            .iter()
            .map(|t| value_info(&t.name, t.elem_type(), &t.dims))
            .collect(),
        output: vec![value_info(
            OnnxCase::OUTPUT_NAME,
            elem_type,
            &case.output_dims(),
        )],
        ..Default::default()
    };

    ModelProto {
        ir_version: Some(IR_VERSION),
        // An empty domain string means `ai.onnx`, the main operator set. The other domain,
        // `ai.onnx.ml`, holds traditional-ML operators and is out of scope.
        opset_import: vec![OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(case.opset),
        }],
        // Recorded in the model itself so a `.onnx` file recovered from a findings
        // directory says where it came from without needing the log beside it.
        producer_name: Some("diff-fuzzer".to_owned()),
        graph: Some(graph),
        ..Default::default()
    }
}

/// Build and serialize in one step — the bytes every runtime receives.
pub fn build_bytes(case: &OnnxCase) -> Vec<u8> {
    to_bytes(&build(case))
}

/// Serialize a model.
///
/// `encode_to_vec` comes from `prost::Message`, which is a **trait**: the generated types
/// implement it, and importing the trait is what brings the method into scope. That is why
/// the `use prost::Message;` above looks unused but is not.
pub fn to_bytes(model: &ModelProto) -> Vec<u8> {
    model.encode_to_vec()
}

/// Build a bare `Add` model over two `f32` tensors — kept for the N0 smoke example, which
/// demonstrates the plumbing without involving the case type.
pub fn add_model(dims: &[i64], opset: i64) -> ModelProto {
    build(&crate::validation::well_formed(OpKind::Add, dims, opset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::well_formed;

    #[test]
    fn a_case_becomes_a_single_node_graph() {
        for op in OpKind::ALL {
            let case = well_formed(op, &[2, 3], DEFAULT_OPSET);
            let model = build(&case);
            let graph = model.graph.as_ref().expect("a model must carry a graph");

            assert_eq!(graph.node.len(), 1, "this domain builds single-node graphs");
            assert_eq!(graph.node[0].op_type.as_deref(), Some(op.onnx_name()));
            assert_eq!(
                graph.node[0].input.len(),
                op.arity(),
                "{op:?} node arity must match the case"
            );
            assert_eq!(graph.input.len(), op.arity());
            assert_eq!(graph.output.len(), 1);
        }
    }

    /// The node's input names must be the graph's input names, in the same order. `tract`
    /// feeds inputs **positionally**, so a mismatch here would silently swap operands —
    /// invisible for `Add`, wrong for `Sub`.
    #[test]
    fn node_inputs_match_the_graph_inputs_in_order() {
        let case = well_formed(OpKind::Sub, &[2], DEFAULT_OPSET);
        let graph = build(&case).graph.expect("graph");

        let declared: Vec<&str> = graph
            .input
            .iter()
            .filter_map(|i| i.name.as_deref())
            .collect();
        let referenced: Vec<&str> = graph.node[0].input.iter().map(String::as_str).collect();

        assert_eq!(declared, referenced);
        assert_eq!(declared, vec!["a", "b"]);
    }

    #[test]
    fn every_graph_value_is_typed_and_shaped() {
        let case = well_formed(OpKind::Add, &[2, 3], DEFAULT_OPSET);
        let graph = build(&case).graph.expect("graph");

        for info in graph.input.iter().chain(graph.output.iter()) {
            // `TypeProto.value` is a `oneof` over five kinds — tensor, sequence, map,
            // optional, sparse tensor. This domain builds only plain tensors, so anything
            // else is a bug in the builder rather than a case to handle.
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
            let shape = tensor.shape.as_ref().expect("every value must be shaped");
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

    /// Same case in, same bytes out — every time. Without this, "every runtime saw
    /// byte-identical input" is not a claim that can be made.
    #[test]
    fn serialization_is_deterministic() {
        for op in OpKind::ALL {
            let case = well_formed(op, &[2, 3], DEFAULT_OPSET);
            assert_eq!(build_bytes(&case), build_bytes(&case), "{op:?}");
            assert!(!build_bytes(&case).is_empty());
        }
    }

    #[test]
    fn the_case_opset_reaches_the_model() {
        for opset in [7, 13, 22, 27] {
            let model = build(&well_formed(OpKind::Add, &[2], opset));
            assert_eq!(model.opset_import.len(), 1);
            assert_eq!(model.opset_import[0].version, Some(opset));
            // Empty domain == `ai.onnx`.
            assert_eq!(model.opset_import[0].domain.as_deref(), Some(""));
        }
    }

    /// A round trip through the wire format must preserve the model. Really a test that
    /// the generated schema is coherent — a build script fed a mismatched `.proto` would
    /// show up here.
    #[test]
    fn a_model_survives_a_round_trip_through_bytes() {
        let original = build(&well_formed(OpKind::Mul, &[4, 5, 6], DEFAULT_OPSET));
        let decoded = ModelProto::decode(to_bytes(&original).as_slice())
            .expect("bytes we just wrote must decode");
        assert_eq!(original, decoded);
    }

    /// Attributes must reach the node, in order, with their tags intact. An attribute that
    /// does not arrive is an operator silently running with its default parameter.
    #[test]
    fn attributes_reach_the_node() {
        let case = well_formed(OpKind::Identity, &[2, 3], DEFAULT_OPSET).with_attrs(
            crate::attrs::Attrs::new()
                .int("axis", 1)
                .ints("perm", vec![1, 0]),
        );
        let graph = build(&case).graph.expect("graph");

        let names: Vec<&str> = graph.node[0]
            .attribute
            .iter()
            .filter_map(|a| a.name.as_deref())
            .collect();
        assert_eq!(names, vec!["axis", "perm"]);
        assert_eq!(graph.node[0].attribute[0].i, Some(1));
        assert_eq!(graph.node[0].attribute[1].ints, vec![1, 0]);
    }

    /// Changing an attribute must change the bytes, or the attribute is not part of what
    /// the runtimes were asked to compute.
    #[test]
    fn attributes_change_the_serialized_bytes() {
        let base = well_formed(OpKind::Identity, &[2], DEFAULT_OPSET);
        let with_axis = base
            .clone()
            .with_attrs(crate::attrs::Attrs::new().int("axis", 0));
        let other_axis = base
            .clone()
            .with_attrs(crate::attrs::Attrs::new().int("axis", 1));

        assert_ne!(build_bytes(&base), build_bytes(&with_axis));
        assert_ne!(build_bytes(&with_axis), build_bytes(&other_axis));
    }

    /// Two different cases must not produce the same bytes. A builder that ignored part of
    /// the case would otherwise pass every test above.
    #[test]
    fn different_cases_produce_different_bytes() {
        let add = build_bytes(&well_formed(OpKind::Add, &[2], DEFAULT_OPSET));
        let sub = build_bytes(&well_formed(OpKind::Sub, &[2], DEFAULT_OPSET));
        let wider = build_bytes(&well_formed(OpKind::Add, &[3], DEFAULT_OPSET));
        let older = build_bytes(&well_formed(OpKind::Add, &[2], 13));

        assert_ne!(add, sub, "operator is not reaching the model");
        assert_ne!(add, wider, "shape is not reaching the model");
        assert_ne!(add, older, "opset is not reaching the model");
    }
}
