//! The systems under test, bound to the [`diff_fuzzer_core::Implementation`] seam.
//!
//! # Every failure is a value
//!
//! `Implementation::run` returns `Result<Out, RunError>`, and the engine treats `Err` as
//! *"this participant could not be compared"*. **Nothing here ever returns `Err`.** Every
//! runtime returns `Ok(OnnxOutcome)`, including when it panics — because a crash is a
//! finding in this domain, and an `Err` would file it under "not evidence of being wrong".
//! See `outcome.rs` for the full argument.
//!
//! # Catching a panic
//!
//! The two Rust runtimes are wrapped in [`std::panic::catch_unwind`], which turns a panic
//! into a value instead of unwinding out of the harness. That is what makes
//! [`OnnxOutcome::Crashed`] reachable at all for them.
//!
//! It does **not** work for ONNX Runtime: `ort` is a binding to C++, and a genuine
//! `abort()` or segfault there takes the process down with no Rust frame to catch. The
//! strategy for that is `PENDING` 1.4, decided below and implemented properly at N5 — for
//! now the case is written to disk before execution so a fatal crash leaves evidence.
//!
//! **Classification, not detection, is what waits for N2.** A panic is unambiguous. Telling
//! a returned error that means "I do not implement this" from one that means "I implement
//! this and it went wrong" needs the capability census, so until then every returned error
//! is conservatively [`OnnxOutcome::Rejected`] — the variant that accuses nobody.
//! Over-reporting crashes before the model exists would manufacture findings.
//!
//! # Why each runtime runs unoptimized
//!
//! `ort` defaults to `GraphOptimizationLevel::Level3` — every optimization including
//! memory-layout rewrites. ONNX Runtime documents only the *basic* level as
//! semantics-preserving, so the default is the one level a conformance comparison must not
//! use. `tract` gets the matching treatment (`into_typed()`, not `into_optimized()`), which
//! keeps the comparison about operator kernels rather than about two different optimizers.

use std::panic::{AssertUnwindSafe, catch_unwind};

use diff_fuzzer_core::traits::{Implementation, RunError};

use crate::case::{OnnxCase, TensorValue};
use crate::model::build_bytes;
use crate::outcome::OnnxOutcome;

/// Run `body`, turning a panic into [`OnnxOutcome::Crashed`].
///
/// `AssertUnwindSafe` is needed because the closure borrows data across a potential unwind
/// boundary. It is sound here: on a panic the borrowed values are only read, and the
/// results are discarded rather than observed in a half-modified state.
pub(crate) fn catching_panics(body: impl FnOnce() -> OnnxOutcome) -> OnnxOutcome {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(outcome) => outcome,
        Err(payload) => OnnxOutcome::Crashed {
            detail: panic_message(&payload),
        },
    }
}

/// Recover the panic message. It is the most useful line in a crash report.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panicked with a non-string payload".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────
// ONNX Runtime
// ─────────────────────────────────────────────────────────────────────────────────────

/// ONNX Runtime, through the `ort` bindings. The maturity anchor.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrtRuntime;

pub const ORT_NAME: &str = "onnxruntime";

impl OrtRuntime {
    fn execute(case: &OnnxCase) -> OnnxOutcome {
        use ort::session::Session;
        use ort::session::builder::GraphOptimizationLevel;
        use ort::value::TensorRef;

        let bytes = build_bytes(case);

        let builder = match Session::builder() {
            Ok(builder) => builder,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("session builder: {e}"),
                };
            }
        };
        // Separate step because `with_optimization_level` returns ort's own
        // recoverable-builder result type, which does not chain with `and_then`.
        let mut builder = match builder.with_optimization_level(GraphOptimizationLevel::Disable) {
            Ok(builder) => builder,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("setting optimization level: {e}"),
                };
            }
        };

        let mut session = match builder.commit_from_memory(&bytes) {
            Ok(session) => session,
            // Load failure. Conservatively `Rejected` rather than `Unsupported`: telling
            // "I do not implement this operator" from "this model is malformed" needs the
            // N2 capability census, and guessing would either hide a real skip or
            // manufacture an accusation.
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("loading model: {e}"),
                };
            }
        };

        let mut feeds: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = Vec::with_capacity(case.inputs.len());
        for input in &case.inputs {
            match TensorRef::from_array_view((input.dims.clone(), input.values.as_slice())) {
                Ok(tensor) => feeds.push((input.name.clone().into(), tensor.into())),
                Err(e) => {
                    return OnnxOutcome::Rejected {
                        detail: format!("building input {}: {e}", input.name),
                    };
                }
            }
        }

        let outputs = match session.run(feeds) {
            Ok(outputs) => outputs,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("running: {e}"),
                };
            }
        };

        let mut tensors = Vec::new();
        for (name, value) in outputs.iter() {
            match value.try_extract_tensor::<f32>() {
                Ok((shape, data)) => tensors.push(TensorValue::f32(
                    name,
                    shape.iter().copied().collect(),
                    data.to_vec(),
                )),
                Err(e) => {
                    return OnnxOutcome::Rejected {
                        detail: format!("extracting output {name}: {e}"),
                    };
                }
            }
        }
        OnnxOutcome::Ok(tensors)
    }
}

impl Implementation for OrtRuntime {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        ORT_NAME
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        // `catch_unwind` catches a Rust panic raised inside the bindings. It does **not**
        // catch a C++ `abort()` or a segfault — those end the process, and surviving them
        // is `PENDING` 1.4, implemented at N5.
        Ok(catching_panics(|| Self::execute(input)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────
// tract
// ─────────────────────────────────────────────────────────────────────────────────────

/// `tract`, the primary target. Pure Rust, so its panics are catchable.
#[derive(Debug, Clone, Copy, Default)]
pub struct TractRuntime;

pub const TRACT_NAME: &str = "tract";

impl TractRuntime {
    fn execute(case: &OnnxCase) -> OnnxOutcome {
        use tract_onnx::prelude::*;

        let bytes = build_bytes(case);
        let mut reader = std::io::Cursor::new(bytes.as_slice());

        let plan = match tract_onnx::onnx()
            .model_for_read(&mut reader)
            .and_then(|m| m.into_typed())
            .and_then(|m| m.into_runnable())
        {
            Ok(plan) => plan,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("loading model: {e}"),
                };
            }
        };

        // `tract` feeds inputs **positionally**, not by name, so this order must match the
        // order the graph declares them in. `model::build` writes them in case order, and
        // a test in that module pins the correspondence — a mismatch would silently swap
        // operands, which is invisible for `Add` and wrong for `Sub`.
        let mut feeds: TVec<TValue> = tvec!();
        for input in &case.inputs {
            let shape: Vec<usize> = input.dims.iter().map(|d| *d as usize).collect();
            match tract_ndarray::ArrayD::from_shape_vec(shape, input.values.clone()) {
                Ok(array) => feeds.push(Tensor::from(array).into()),
                Err(e) => {
                    return OnnxOutcome::Rejected {
                        detail: format!("building input {}: {e}", input.name),
                    };
                }
            }
        }

        let outputs = match plan.run(feeds) {
            Ok(outputs) => outputs,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("running: {e}"),
                };
            }
        };

        let mut tensors = Vec::new();
        for (index, output) in outputs.iter().enumerate() {
            match output.to_plain_array_view::<f32>() {
                Ok(array) => tensors.push(TensorValue::f32(
                    // tract does not carry graph output names through the plan. The oracle
                    // compares positionally, and the canonical form drops names, so this is
                    // a label for humans rather than part of the comparison.
                    OnnxCase::OUTPUT_NAME,
                    array.shape().iter().map(|d| *d as i64).collect(),
                    array.iter().copied().collect(),
                )),
                Err(e) => {
                    return OnnxOutcome::Rejected {
                        detail: format!("extracting output {index}: {e}"),
                    };
                }
            }
        }
        OnnxOutcome::Ok(tensors)
    }
}

impl Implementation for TractRuntime {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        TRACT_NAME
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        Ok(catching_panics(|| Self::execute(input)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────
// candle-onnx
// ─────────────────────────────────────────────────────────────────────────────────────

/// `candle-onnx`, the secondary target. Behind the `candle` cargo feature.
#[cfg(feature = "candle")]
#[derive(Debug, Clone, Copy, Default)]
pub struct CandleRuntime;

#[cfg(feature = "candle")]
pub const CANDLE_NAME: &str = "candle";

#[cfg(feature = "candle")]
impl CandleRuntime {
    fn execute(case: &OnnxCase) -> OnnxOutcome {
        use candle_core::{Device, Tensor};
        use prost::Message;
        use std::collections::HashMap;

        let bytes = build_bytes(case);

        // `candle-onnx` exposes only `read_file(path)`, whose body is a prost decode. The
        // bytes are decoded directly into candle's own generated type instead — the same
        // byte string every other runtime gets, parsed by candle's schema rather than ours.
        let model = match candle_onnx::onnx::ModelProto::decode(bytes.as_slice()) {
            Ok(model) => model,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("parsing model: {e}"),
                };
            }
        };

        let mut feeds: HashMap<String, Tensor> = HashMap::new();
        for input in &case.inputs {
            let shape: Vec<usize> = input.dims.iter().map(|d| *d as usize).collect();
            match Tensor::from_vec(input.values.clone(), shape, &Device::Cpu) {
                Ok(tensor) => {
                    feeds.insert(input.name.clone(), tensor);
                }
                Err(e) => {
                    return OnnxOutcome::Rejected {
                        detail: format!("building input {}: {e}", input.name),
                    };
                }
            }
        }

        let outputs = match candle_onnx::simple_eval(&model, feeds) {
            Ok(outputs) => outputs,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("running: {e}"),
                };
            }
        };

        match outputs.get(OnnxCase::OUTPUT_NAME) {
            Some(tensor) => match tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                Ok(values) => OnnxOutcome::Ok(vec![TensorValue::f32(
                    OnnxCase::OUTPUT_NAME,
                    tensor.dims().iter().map(|d| *d as i64).collect(),
                    values,
                )]),
                Err(e) => OnnxOutcome::Rejected {
                    detail: format!("extracting output: {e}"),
                },
            },
            None => OnnxOutcome::Rejected {
                detail: format!("no output named {}", OnnxCase::OUTPUT_NAME),
            },
        }
    }
}

#[cfg(feature = "candle")]
impl Implementation for CandleRuntime {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        CANDLE_NAME
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        Ok(catching_panics(|| Self::execute(input)))
    }
}

/// The in-process runtimes compiled into this build.
///
/// Reported rather than assumed: which participants exist depends on cargo features, and a
/// campaign that silently ran three where it claimed four would overstate its evidence.
pub fn compiled_runtime_names() -> Vec<&'static str> {
    #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
    let mut names = vec![ORT_NAME, TRACT_NAME];
    #[cfg(feature = "candle")]
    names.push(CANDLE_NAME);
    names
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::OpKind;
    use crate::validation::well_formed;

    const OPSET: i64 = 22;

    fn in_process() -> Vec<Box<dyn Implementation<In = OnnxCase, Out = OnnxOutcome>>> {
        #[cfg_attr(not(feature = "candle"), allow(unused_mut))]
        let mut runtimes: Vec<Box<dyn Implementation<In = OnnxCase, Out = OnnxOutcome>>> =
            vec![Box::new(OrtRuntime), Box::new(TractRuntime)];
        #[cfg(feature = "candle")]
        runtimes.push(Box::new(CandleRuntime));
        runtimes
    }

    /// Every compiled runtime runs every operator this domain builds, and they all agree.
    ///
    /// Not an oracle test — it is the claim that the plumbing carries the same computation
    /// everywhere. Iterating `OpKind::ALL` means a newly added operator is covered here the
    /// moment it exists, rather than whenever someone remembers to extend the list.
    #[test]
    fn every_runtime_runs_every_operator_and_they_agree() {
        for op in OpKind::ALL {
            let case = well_formed(op, &[2, 3], OPSET);

            let results: Vec<(String, OnnxOutcome)> = in_process()
                .iter()
                .map(|r| (r.name().to_string(), r.run(&case).expect("never Err")))
                .collect();

            let first = &results[0];
            let OnnxOutcome::Ok(expected) = &first.1 else {
                panic!("{} could not run {op:?}: {}", first.0, first.1);
            };

            for (name, outcome) in &results[1..] {
                let OnnxOutcome::Ok(actual) = outcome else {
                    panic!("{name} could not run {op:?}: {outcome}");
                };
                assert_eq!(
                    actual[0].values, expected[0].values,
                    "{name} disagreed with {} on {op:?}",
                    first.0
                );
                assert_eq!(actual[0].dims, expected[0].dims, "{name} shape on {op:?}");
            }
        }
    }

    /// `run` must never return `Err`, for any input, valid or not. This is the property the
    /// whole crash thesis depends on: an `Err` would be routed into the engine's skip path
    /// and the outcome would never reach the oracle.
    #[test]
    fn no_runtime_ever_returns_err() {
        // A deliberately invalid case: `Add` with one input. `build` does not validate, so
        // this reaches the runtimes as a malformed model.
        let broken = OnnxCase::new(
            OpKind::Add,
            OPSET,
            vec![TensorValue::f32("a", vec![2], vec![1.0, 2.0])],
        );
        assert!(!crate::validation::is_valid(&broken));

        for runtime in in_process() {
            let result = runtime.run(&broken);
            assert!(
                result.is_ok(),
                "{} returned Err; failures must be values",
                runtime.name()
            );
            // And it must be one of the failure *variants*, not a silent success.
            let outcome = result.unwrap();
            assert!(
                !matches!(outcome, OnnxOutcome::Ok(_)),
                "{} accepted a malformed model: {outcome}",
                runtime.name()
            );
        }
    }

    /// Operand order must reach the runtime intact. `Sub` is the check that `Add` cannot
    /// provide — `tract` feeds inputs positionally, so a swap here would be invisible under
    /// a commutative operator.
    #[test]
    fn operand_order_is_preserved() {
        let case = OnnxCase::new(
            OpKind::Sub,
            OPSET,
            vec![
                TensorValue::f32("a", vec![1], vec![10.0]),
                TensorValue::f32("b", vec![1], vec![3.0]),
            ],
        );
        for runtime in in_process() {
            let OnnxOutcome::Ok(out) = runtime.run(&case).unwrap() else {
                panic!("{} could not run Sub", runtime.name());
            };
            assert_eq!(
                out[0].values,
                vec![7.0],
                "{} computed b - a instead of a - b",
                runtime.name()
            );
        }
    }

    /// Names must be stable and distinct: findings are grouped by them, and a collision
    /// would silently merge two runtimes' results.
    #[test]
    fn runtime_names_are_distinct_and_stable() {
        let names = compiled_runtime_names();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "two runtimes share a name");

        for (runtime, expected) in in_process().iter().zip(names.iter()) {
            assert_eq!(&runtime.name(), expected, "name() disagrees with the list");
        }
    }

    /// Special values must reach a runtime and come back intact. `Identity` performs no
    /// arithmetic, so anything lost here is lost in the plumbing.
    #[test]
    fn special_values_survive_the_round_trip_through_every_runtime() {
        let hostile = vec![f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::MIN_POSITIVE];
        let case = OnnxCase::new(
            OpKind::Identity,
            OPSET,
            vec![TensorValue::f32("a", vec![4], hostile.clone())],
        );

        for runtime in in_process() {
            let OnnxOutcome::Ok(out) = runtime.run(&case).unwrap() else {
                panic!("{} could not run Identity", runtime.name());
            };
            for (sent, received) in hostile.iter().zip(out[0].values.iter()) {
                assert_eq!(
                    sent.to_bits(),
                    received.to_bits(),
                    "{} altered a special value",
                    runtime.name()
                );
            }
        }
    }
}
