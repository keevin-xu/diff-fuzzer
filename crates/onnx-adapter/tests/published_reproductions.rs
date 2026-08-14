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
use onnx_adapter::attrs::Attrs;
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

/// The `Reshape` model both F-002 reports are built on.
fn zero_size_reshape() -> OnnxCase {
    OnnxCase::new(
        OpKind::Reshape,
        22,
        vec![
            TensorValue::f32("a", vec![3, 0], vec![]),
            TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
        ],
    )
    .with_attrs(Attrs::new().int("allowzero", 1))
}

/// **`final/tract-002-body.md`** — the reference and ONNX Runtime accept it; `tract` does not.
#[test]
fn tract_002_zero_size_reshape_still_reproduces() {
    let reference = ReferenceRuntime::start().expect("reference");
    let case = zero_size_reshape();

    assert!(
        matches!(reference.run(&case).expect("never Err"), OnnxOutcome::Ok(_)),
        "the report rests on the reference accepting this model"
    );
    assert!(matches!(
        OrtRuntime.run(&case).expect("never Err"),
        OnnxOutcome::Ok(_)
    ));
    assert!(
        matches!(
            TractRuntime.run(&case).expect("never Err"),
            OnnxOutcome::Rejected { .. }
        ),
        "tract now accepts this — the report must be withdrawn or corrected"
    );
}

/// **`final/candle-001-body.md`** — and the error text is the report's entire argument.
///
/// The claim is that candle reports `rhs: [3]`, which is the `allowzero=0` reading of the target.
/// If that message ever changes, the inference in the report no longer follows from it, and the
/// report should not be sent as written.
#[cfg(feature = "candle")]
#[test]
fn candle_001_error_still_names_the_copied_dimension() {
    use onnx_adapter::runtimes::CandleRuntime;
    let case = zero_size_reshape();
    match CandleRuntime.run(&case).expect("never Err") {
        OnnxOutcome::Rejected { detail } => {
            assert!(
                detail.contains("[3]"),
                "the report argues from candle reporting rhs: [3]; it now says: {detail}"
            );
        }
        other => panic!("candle now accepts this — the report must be corrected: {other:?}"),
    }
}

/// The minimised `DynamicQuantizeLinear` model F-008's report is built on.
///
/// **Three elements, and it cannot be fewer.** The operator derives its own scale and zero-point
/// from the data, so a one-element tensor's scale is a function of that very element and no tie
/// can survive it. Two of these three exist to pin the range — and therefore the scale and
/// zero-point — while the third lands on the tie:
///
/// ```text
///   scale = (max(0,max) - min(0,min)) / 255 = (128 - -127) / 255 = 1.0   exactly
///   zero_point = round(-min / scale) = 127                               odd
///   0.5 / 1.0 = 0.5                                                      an exact tie
/// ```
///
/// The scale is exactly `1.0` on purpose: any other value risks the tie being destroyed by the
/// division itself, and then the test would pass for the wrong reason.
fn dynamic_quantize_tie() -> OnnxCase {
    OnnxCase::new(
        OpKind::DynamicQuantizeLinear,
        22,
        vec![TensorValue::new(
            "a",
            vec![3],
            TensorData::F32(vec![-127.0, 128.0, 0.5]),
        )],
    )
}

/// The first output tensor as `u8`, which is what a quantized result is.
fn quantized_bytes(outcome: &OnnxOutcome) -> Vec<u8> {
    match outcome {
        OnnxOutcome::Ok(tensors) => match &tensors[0].data {
            TensorData::U8(v) => v.clone(),
            other => panic!("expected a uint8 output, got {other:?}"),
        },
        other => panic!("expected a result, got {other:?}"),
    }
}

/// **`final/tract-003-body.md`** — the minimised reproduction, and the whole of the report's claim.
///
/// ONNX specifies `y = saturate(round(x / y_scale) + y_zero_point)` with **round-half-to-even**.
/// `tract` uses Rust's `f32::round()`, which is **round-half-away-from-zero**:
///
/// ```text
///   spec :  round_half_even(0.5) = 0   ->  0 + 127 = 127
///   tract:  f32::round(0.5)      = 1   ->  1 + 127 = 128
/// ```
///
/// **This is the entire finding**, so if it stops reproducing the report must not be sent.
#[test]
fn tract_003_dynamic_quantize_rounding_mode_still_reproduces() {
    let case = dynamic_quantize_tie();

    let reference = ReferenceRuntime::start()
        .expect("reference")
        .run(&case)
        .expect("never Err");
    let ort = OrtRuntime.run(&case).expect("never Err");
    let tract = TractRuntime.run(&case).expect("never Err");

    // The reference is the specification's own executable implementation, so it defines the
    // expected answer rather than merely voting with ONNX Runtime.
    assert_eq!(
        quantized_bytes(&reference),
        vec![0, 255, 127],
        "the specification's own implementation rounds half to even: round(0.5) = 0"
    );
    assert_eq!(
        quantized_bytes(&ort),
        vec![0, 255, 127],
        "ONNX Runtime agrees with the reference"
    );
    assert_eq!(
        quantized_bytes(&tract),
        vec![0, 255, 128],
        "tract rounds half away from zero: f32::round(0.5) = 1, so 1 + 127 = 128"
    );
}

/// **The derived parameters are identical across all three**, which is what makes the difference
/// attributable to the rounding order rather than to deriving the scale.
///
/// Without this the report would have a hole a maintainer would find immediately: a disagreement
/// about `y_scale` or `y_zero_point` would produce the same wrong byte for an entirely different
/// reason.
#[test]
fn tract_003_the_derived_parameters_agree() {
    let case = dynamic_quantize_tie();

    let scale_and_zero = |outcome: &OnnxOutcome| -> (u32, u8) {
        match outcome {
            OnnxOutcome::Ok(tensors) => {
                assert_eq!(tensors.len(), 3, "y, y_scale, y_zero_point");
                let scale = match &tensors[1].data {
                    TensorData::F32(v) => v[0].to_bits(),
                    other => panic!("y_scale should be float32, got {other:?}"),
                };
                let zero_point = match &tensors[2].data {
                    TensorData::U8(v) => v[0],
                    other => panic!("y_zero_point should be uint8, got {other:?}"),
                };
                (scale, zero_point)
            }
            other => panic!("expected a result, got {other:?}"),
        }
    };

    let reference = scale_and_zero(
        &ReferenceRuntime::start()
            .expect("reference")
            .run(&case)
            .expect("never Err"),
    );
    let ort = scale_and_zero(&OrtRuntime.run(&case).expect("never Err"));
    let tract = scale_and_zero(&TractRuntime.run(&case).expect("never Err"));

    assert_eq!(reference, ort);
    assert_eq!(
        reference, tract,
        "all three must derive the same scale and zero-point, or the finding is not about \
         rounding order"
    );
    assert_eq!(reference.0, 1.0f32.to_bits(), "scale is exactly 1.0");
    assert_eq!(
        reference.1, 127,
        "zero-point is odd, which is what exposes the order"
    );
}

/// **The case that told two explanations apart, and overturned the first one.**
///
/// F-008 was originally written as *"`tract` adds the zero-point before rounding"*. That theory
/// fit every observation available at the time: the 240-element campaign case, and the minimised
/// three-element one above. Both predicted exactly the bytes `tract` produced.
///
/// A competing theory fit them equally well — **the order is right and the rounding mode is
/// wrong**, since Rust's `f32::round()` is half-away-from-zero where ONNX specifies half-to-even.
/// The two are indistinguishable whenever the zero-point is odd, which both earlier cases had.
///
/// This case separates them. With an **even** zero-point and `x / scale = 2.5`:
///
/// ```text
///   add-first, half-to-even :  round(2.5 + 2)  = round(4.5) = 4
///   right order, half-away  :  f32::round(2.5) = 3, then 3 + 2 = 5
/// ```
///
/// `tract` returns **5**, so the rounding mode is the defect and the operation order is fine.
/// `tract`'s own source agrees — `(((x / scale).round() as i32) + zero_point as i32)` adds
/// afterwards, exactly as specified.
///
/// **Kept as a test because the report's root cause depends on it.** Without this case the
/// project would have filed a wrong diagnosis that happened to predict the right bytes.
#[test]
fn tract_003_an_even_zero_point_shows_it_is_the_rounding_mode_not_the_order() {
    let case = OnnxCase::new(
        OpKind::DynamicQuantizeLinear,
        22,
        vec![TensorValue::new(
            "a",
            vec![3],
            // scale = (253 - -2)/255 = 1.0 exactly; zero_point = 2, which is EVEN.
            TensorData::F32(vec![-2.0, 253.0, 2.5]),
        )],
    );

    let reference = ReferenceRuntime::start()
        .expect("reference")
        .run(&case)
        .expect("never Err");
    let tract = TractRuntime.run(&case).expect("never Err");

    assert_eq!(quantized_bytes(&reference), vec![0, 255, 4]);
    assert_eq!(
        quantized_bytes(&tract),
        vec![0, 255, 5],
        "5 means half-away-from-zero rounding; 4 would have meant the zero-point was added first"
    );
}
