//! Write each finding's model to disk exactly as this project builds it.
//!
//! # Why
//!
//! The final reports tell a maintainer to build a model with a short Python script. Everything
//! this project has proven — that the divergence is real, that it minimises to these values —
//! was proven against models **this code** built. Those are not automatically the same artifact.
//!
//! So: dump ours, parse both with `onnx`, and compare the graphs. If they agree, the divergence
//! demonstrated by `verify_findings` transfers to the model the report actually asks for. If they
//! do not, the report is telling maintainers to build something we never tested.

use onnx_adapter::attrs::Attrs;
use onnx_adapter::case::{OnnxCase, OpKind, TensorData, TensorValue};

const OPSET: i64 = 22;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/repro/ours".to_string());
    std::fs::create_dir_all(&out).expect("create output directory");

    let reshape = || {
        OnnxCase::new(
            OpKind::Reshape,
            OPSET,
            vec![
                TensorValue::new("a", vec![3, 0], TensorData::F32(vec![])),
                TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
            ],
        )
        .with_attrs(Attrs::new().int("allowzero", 1))
    };

    let cases: Vec<(&str, OnnxCase)> = vec![
        (
            "sign",
            OnnxCase::new(
                OpKind::Sign,
                OPSET,
                vec![TensorValue::new("a", vec![1], TensorData::F32(vec![-0.0]))],
            ),
        ),
        ("reshape", reshape()),
        (
            "dql",
            OnnxCase::new(
                OpKind::DynamicQuantizeLinear,
                OPSET,
                vec![TensorValue::new(
                    "a",
                    vec![3],
                    TensorData::F32(vec![-127.0, 128.0, 0.5]),
                )],
            ),
        ),
        (
            "div",
            OnnxCase::new(
                OpKind::Div,
                OPSET,
                vec![
                    TensorValue::new("a", vec![1], TensorData::I32(vec![i32::MIN])),
                    TensorValue::new("b", vec![1], TensorData::I32(vec![-1])),
                ],
            ),
        ),
        (
            "where",
            OnnxCase::new(
                OpKind::Where,
                OPSET,
                vec![
                    TensorValue::new("c", vec![1], TensorData::Bool(vec![true])),
                    TensorValue::new("x", vec![1], TensorData::F32(vec![-0.0])),
                    TensorValue::new("y", vec![1], TensorData::F32(vec![0.0])),
                ],
            ),
        ),
    ];

    for (name, case) in cases {
        let bytes = onnx_adapter::model::build_bytes(&case);
        let path = format!("{out}/{name}.onnx");
        std::fs::write(&path, &bytes).expect("write model");
        println!("{path}  ({} bytes)", bytes.len());
    }
}
