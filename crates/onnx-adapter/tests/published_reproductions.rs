//! Every reproduction printed in `issues/onnx-runtime/final/` must still reproduce.
//!
//! # Why this is a test and not a one-off check
//!
//! A finished report is a claim made to somebody else, and it goes stale silently. Bump a runtime
//! version, change the model builder, adjust how outputs are decoded — the report on disk still
//! reads as true, and the first person to discover otherwise is the maintainer we sent it to.
//!
//! `06-ORACLES-AND-LEGAL-DIFFERENCES.md` makes the same argument about the legal-difference
//! catalog rotting; this is that argument applied one level out, to the artifacts the project
//! exists to produce.
//!
//! So the exact byte patterns quoted in each final report are asserted here. If a runtime is
//! upgraded and the behaviour changes, this fails, and the report gets corrected or withdrawn
//! **before** it is sent rather than after.
use diff_fuzzer_core::traits::Implementation;
use onnx_adapter::case::{OnnxCase, OpKind, TensorData, TensorValue};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

/// The first output tensor's bit patterns.
fn bits(outcome: &OnnxOutcome) -> Vec<u64> {
    match outcome {
        OnnxOutcome::Ok(tensors) => match &tensors[0].data {
            TensorData::F32(v) => v.iter().map(|x| u64::from(x.to_bits())).collect(),
            TensorData::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
            other => panic!("expected float output, got {other:?}"),
        },
        other => panic!("expected a result, got {other:?}"),
    }
}

const NEG_ZERO_F32: u64 = 0x8000_0000;
const POS_ZERO: u64 = 0x0;
const NEG_ZERO_F64: u64 = 0x8000_0000_0000_0000;

/// **`final/tract-001-body.md`** — the table of four values, exactly as printed.
#[test]
fn tract_001_sign_of_negative_zero_still_reproduces() {
    let reference = ReferenceRuntime::start().expect("reference");
    let case = OnnxCase::new(
        OpKind::Sign,
        22,
        vec![TensorValue::f32("a", vec![4], vec![-0.0, 0.0, -2.0, 2.0])],
    );

    assert_eq!(
        bits(&TractRuntime.run(&case).expect("never Err")),
        vec![NEG_ZERO_F32, POS_ZERO, 0xbf80_0000, 0x3f80_0000],
        "the tract row of the report has changed"
    );
    assert_eq!(
        bits(&reference.run(&case).expect("never Err")),
        vec![POS_ZERO, POS_ZERO, 0xbf80_0000, 0x3f80_0000],
        "the reference row of the report has changed"
    );
    assert_eq!(
        bits(&OrtRuntime.run(&case).expect("never Err"))[0],
        POS_ZERO,
        "the onnxruntime row of the report has changed"
    );
}

/// **`final/tract-001-body.md`** — the minimised case, and the `float64` claim.
#[test]
fn tract_001_minimised_case_and_the_f64_claim_still_hold() {
    let reference = ReferenceRuntime::start().expect("reference");

    let minimal = OnnxCase::new(
        OpKind::Sign,
        22,
        vec![TensorValue::f32("a", vec![1, 1, 1], vec![-0.0])],
    );
    assert_eq!(
        bits(&TractRuntime.run(&minimal).expect("never Err")),
        vec![NEG_ZERO_F32]
    );
    assert_eq!(
        bits(&reference.run(&minimal).expect("never Err")),
        vec![POS_ZERO]
    );

    let wide = OnnxCase::new(
        OpKind::Sign,
        22,
        vec![TensorValue::new("a", vec![1], TensorData::F64(vec![-0.0]))],
    );
    assert_eq!(
        bits(&TractRuntime.run(&wide).expect("never Err")),
        vec![NEG_ZERO_F64],
        "the report's float64 claim has changed"
    );
    assert_eq!(
        bits(&reference.run(&wide).expect("never Err")),
        vec![POS_ZERO]
    );
}

/// **`final/onnxruntime-001-body.md`** — the `X` branch loses the sign.
#[test]
fn onnxruntime_001_x_branch_still_reproduces() {
    let reference = ReferenceRuntime::start().expect("reference");
    let case = OnnxCase::new(
        OpKind::Where,
        22,
        vec![
            TensorValue::new("c", vec![1, 1], TensorData::Bool(vec![true])),
            TensorValue::f32("x", vec![1, 1], vec![-0.0]),
            TensorValue::f32("y", vec![1, 1], vec![0.0]),
        ],
    );
    assert_eq!(
        bits(&OrtRuntime.run(&case).expect("never Err")),
        vec![POS_ZERO],
        "onnxruntime no longer loses the sign — the report must be withdrawn or corrected"
    );
    assert_eq!(
        bits(&reference.run(&case).expect("never Err")),
        vec![NEG_ZERO_F32]
    );
    assert_eq!(
        bits(&TractRuntime.run(&case).expect("never Err")),
        vec![NEG_ZERO_F32]
    );
}

/// **`final/onnxruntime-001-body.md`** — and the `Y` branch is correct, which is the report's
/// argument that the Rust binding is not responsible. If this ever fails, that argument is gone
/// and the report should not be sent as written.
#[test]
fn onnxruntime_001_y_branch_is_still_correct() {
    let case = OnnxCase::new(
        OpKind::Where,
        22,
        vec![
            TensorValue::new("c", vec![1], TensorData::Bool(vec![false])),
            TensorValue::f32("x", vec![1], vec![1.0]),
            TensorValue::f32("y", vec![1], vec![-0.0]),
        ],
    );
    assert_eq!(
        bits(&OrtRuntime.run(&case).expect("never Err")),
        vec![NEG_ZERO_F32],
        "the Y branch is the report's evidence that the binding is not at fault"
    );
}

/// **`final/onnxruntime-001-body.md`** — the `float64` claim.
#[test]
fn onnxruntime_001_f64_claim_still_holds() {
    let reference = ReferenceRuntime::start().expect("reference");
    let case = OnnxCase::new(
        OpKind::Where,
        22,
        vec![
            TensorValue::new("c", vec![1], TensorData::Bool(vec![true])),
            TensorValue::new("x", vec![1], TensorData::F64(vec![-0.0])),
            TensorValue::new("y", vec![1], TensorData::F64(vec![0.0])),
        ],
    );
    assert_eq!(
        bits(&OrtRuntime.run(&case).expect("never Err")),
        vec![POS_ZERO]
    );
    assert_eq!(
        bits(&reference.run(&case).expect("never Err")),
        vec![NEG_ZERO_F64]
    );
}
