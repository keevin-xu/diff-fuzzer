//! **N3.6 — the validity stress test, done against the specification.**
//!
//! # Why this is a separate file from the unit tests
//!
//! `generator.rs` already asserts that thousands of generated cases satisfy `validate()`. That
//! passed 5,000 seeds while **13% of generated `Reshape` cases were invalid models**, and the
//! reason is worth stating plainly:
//!
//! > **A validator weaker than the specification cannot detect that it is weaker.**
//!
//! `validate()` knew nothing about `Reshape`'s element-count rule, so it approved cases the
//! specification rejects. The generator emitted `0` in a shape input meaning "a zero-length
//! dimension", while ONNX reads `0` as "copy the input's dimension here" unless `allowzero=1`.
//! `[5,8,6,0] -> [0]` therefore asked for five elements out of a zero-element tensor.
//!
//! G-N3's criterion is *"thousands of cases across widely separated seeds run on **every
//! participating runtime** with no invalid-model rejections"* — which requires asking something
//! other than ourselves. This file asks `onnx.reference`, whose acceptance is the practical
//! definition of validity (`06-ORACLES` §2).
//!
//! It is slower than a unit test, because every case crosses a process boundary. That is the
//! price of the check being worth anything.

use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::{Generator, Implementation};

use onnx_adapter::gen_shape::Bounds;
use onnx_adapter::generator::OnnxGenerator;
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;

/// Seeds spread across the whole `u64` range rather than a tidy `0..n`.
///
/// A generator with a hidden dependence on seed magnitude passes a sequential run and fails in a
/// campaign that uses a resumable range.
fn wide_seeds(count: u64) -> impl Iterator<Item = u64> {
    (0..count).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Every generated case must be accepted by the **specification's own implementation**.
///
/// A case the reference rejects is *our* invalid model. Reporting a runtime's behaviour on one
/// would be reporting our own bug as theirs — `08-RISKS.md` §2, and the failure that produced
/// 825 findings from invalid queries in an earlier domain.
#[test]
fn every_generated_case_is_accepted_by_the_specification() {
    let generator = OnnxGenerator::default();
    let reference = ReferenceRuntime::start().expect("the reference must start");

    let mut rejected: Vec<String> = Vec::new();
    let mut checked = 0;

    for seed in wide_seeds(3_000) {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        // Our own validator first: if it and the reference ever disagree about a case, that
        // disagreement is itself the finding — one of them is wrong about ONNX.
        let ours = onnx_adapter::validation::validate(&case);
        assert!(
            ours.is_empty(),
            "seed {seed}: our validator rejected a generated case: {ours:?}"
        );

        checked += 1;
        match reference.run(&case).expect("the worker must reply") {
            OnnxOutcome::Ok(_) => {}
            OnnxOutcome::Rejected { detail } => {
                if rejected.len() < 10 {
                    let reason = detail
                        .lines()
                        .rev()
                        .find(|l| !l.trim().is_empty())
                        .unwrap_or("");
                    rejected.push(format!(
                        "seed {seed}: {} at {:?} — {reason}",
                        case.op.onnx_name(),
                        onnx_adapter::ops::data_elem_type(&case)
                    ));
                }
            }
            other => panic!("seed {seed}: the reference did something unexpected: {other}"),
        }
    }

    assert!(
        checked > 2_500,
        "only {checked} cases reached the reference"
    );
    assert!(
        rejected.is_empty(),
        "the specification rejected {} of {checked} generated cases — these are OUR invalid \
         models, not findings:\n{}",
        rejected.len(),
        rejected.join("\n")
    );
}

/// The same, with special values on. They change what is generated, so they change what can be
/// invalid — a corpus validated only at the default configuration says nothing about this one.
#[test]
fn special_values_do_not_produce_invalid_models() {
    let generator = OnnxGenerator::new(Bounds::default().with_special_values());
    let reference = ReferenceRuntime::start().expect("the reference must start");
    let mut rejected: Vec<String> = Vec::new();

    for seed in wide_seeds(1_500) {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        if let OnnxOutcome::Rejected { detail } = reference.run(&case).expect("reply")
            && rejected.len() < 10
        {
            let reason = detail
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("");
            rejected.push(format!("seed {seed}: {} — {reason}", case.op.onnx_name()));
        }
    }
    assert!(
        rejected.is_empty(),
        "the specification rejected {} cases with special values on:\n{}",
        rejected.len(),
        rejected.join("\n")
    );
}

/// **Proof the check above could fail.**
///
/// A validity test that can only pass is not evidence. This hands the reference a model that is
/// deliberately invalid — `Reshape` asking for an element count the input cannot supply, which
/// is precisely the class that slipped through — and asserts it is refused.
#[test]
fn the_specification_check_can_actually_fail() {
    use onnx_adapter::case::{OnnxCase, OpKind, TensorData, TensorValue};

    let reference = ReferenceRuntime::start().expect("the reference must start");

    // 6 elements reshaped to a shape demanding 4. Invalid, and our validator now says so too.
    let broken = OnnxCase::new(
        OpKind::Reshape,
        22,
        vec![
            TensorValue::new("a", vec![2, 3], TensorData::F32(vec![1.0; 6])),
            TensorValue::new("b", vec![2], TensorData::I64(vec![2, 2])).as_initializer(),
        ],
    );

    assert!(
        !onnx_adapter::validation::is_valid(&broken),
        "our validator must now catch a Reshape that does not preserve the element count"
    );
    assert!(
        matches!(
            reference.run(&broken).expect("reply"),
            OnnxOutcome::Rejected { .. }
        ),
        "the specification must reject a Reshape that does not preserve the element count"
    );
}
