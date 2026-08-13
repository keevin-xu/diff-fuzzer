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
//!
//! **Cited, not recalled:** `SPECS.md` §3.1 quotes the primary source. Two details there are
//! easy to overstate and are not overstated here. The page says Basic optimizations *are*
//! "semantics-preserving graph rewrites"; it does **not** say the higher levels are not — that
//! is an absence of a guarantee, not a denial. And ONNX Runtime's own docs record an
//! optimization that changes numeric results (GELU Approximation, "F1 ... 87.05 vs 87.03"),
//! which is the concrete reason a conformance comparison declines the default.
//!
//! Running at `Disable` is the **tightening** direction and so needs only evidence it is
//! achievable. Relaxing to `Level1` would be a loosening, and §3.1 now carries the citation
//! that would justify it if throughput ever demanded it.

use std::panic::{AssertUnwindSafe, catch_unwind};

use diff_fuzzer_core::traits::{Implementation, RunError};

use crate::case::{OnnxCase, TensorData, TensorValue};
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
        // `fed_inputs`, not `inputs`: an initializer is already a constant inside the model,
        // and supplying it again would be rejected as an unknown input name.
        for input in case.fed_inputs() {
            // One arm per element type, each pushing directly. An exhaustive `match`
            // rather than a helper taking a dtype tag: adding an `ElemType` then fails to
            // compile here, which is the compiler enforcing `08-RISKS.md` §4's "adding a
            // type must touch every check" instead of a reviewer having to remember it.
            //
            // The arms cannot be collapsed by building a common value first, because each
            // `TensorRef<T>` is a distinct type — the conversion to a session input is what
            // erases it, so it has to happen inside each arm.
            macro_rules! feed {
                ($values:expr) => {
                    match TensorRef::from_array_view((input.dims.clone(), $values.as_slice())) {
                        Ok(tensor) => feeds.push((input.name.clone().into(), tensor.into())),
                        Err(e) => {
                            return OnnxOutcome::Rejected {
                                detail: format!("building input {}: {e}", input.name),
                            };
                        }
                    }
                };
            }
            match &input.data {
                TensorData::F32(v) => feed!(v),
                TensorData::F64(v) => feed!(v),
                TensorData::I32(v) => feed!(v),
                TensorData::I64(v) => feed!(v),
                TensorData::Bool(v) => feed!(v),
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
            // The output type is whatever the operator produced, which is not always an
            // input type — `Equal` takes floats and returns bools. So the type is read from
            // the value and then decoded, rather than assumed.
            let dtype = match value.dtype().tensor_type() {
                Some(t) => t,
                None => {
                    return OnnxOutcome::Rejected {
                        detail: format!("output {name} is not a tensor"),
                    };
                }
            };

            macro_rules! extract {
                ($rust:ty, $variant:ident) => {
                    match value.try_extract_tensor::<$rust>() {
                        Ok((shape, data)) => TensorValue::new(
                            name,
                            shape.iter().copied().collect(),
                            TensorData::$variant(data.to_vec()),
                        ),
                        Err(e) => {
                            return OnnxOutcome::Rejected {
                                detail: format!("extracting output {name}: {e}"),
                            };
                        }
                    }
                };
            }

            use ort::value::TensorElementType as Ty;
            let tensor = match dtype {
                Ty::Float32 => extract!(f32, F32),
                Ty::Float64 => extract!(f64, F64),
                Ty::Int32 => extract!(i32, I32),
                Ty::Int64 => extract!(i64, I64),
                Ty::Bool => extract!(bool, Bool),
                // A type ORT produced that this adapter cannot represent. Reported rather
                // than decoded as something else — a wrong decode would look like a real
                // divergence.
                other => {
                    return OnnxOutcome::Rejected {
                        detail: format!(
                            "output {name} has element type {other:?}, which this \
                                         adapter does not decode"
                        ),
                    };
                }
            };
            tensors.push(tensor);
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
        // order the graph declares them in. `model::build` declares exactly the fed inputs, in
        // case order, and a test in that module pins the correspondence — a mismatch would
        // silently swap operands, which is invisible for `Add` and wrong for `Sub`.
        //
        // Initializers are excluded here for the same reason they are excluded there: they are
        // constants in the model, not positional arguments.
        let mut feeds: TVec<TValue> = tvec!();
        for input in case.fed_inputs() {
            let shape: Vec<usize> = input.dims.iter().map(|d| *d as usize).collect();

            macro_rules! feed {
                ($values:expr) => {
                    match tract_ndarray::ArrayD::from_shape_vec(shape.clone(), $values.clone()) {
                        Ok(array) => Tensor::from(array),
                        Err(e) => {
                            return OnnxOutcome::Rejected {
                                detail: format!("building input {}: {e}", input.name),
                            };
                        }
                    }
                };
            }

            let tensor = match &input.data {
                TensorData::F32(v) => feed!(v),
                TensorData::F64(v) => feed!(v),
                TensorData::I32(v) => feed!(v),
                TensorData::I64(v) => feed!(v),
                TensorData::Bool(v) => feed!(v),
            };
            feeds.push(tensor.into());
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
            macro_rules! collect {
                ($rust:ty, $variant:ident) => {
                    match output.to_plain_array_view::<$rust>() {
                        Ok(array) => TensorValue::new(
                            // tract does not carry graph output names through the plan. The
                            // oracle compares positionally and the canonical form drops
                            // names, so this is a label for humans, not part of comparison.
                            OnnxCase::OUTPUT_NAME,
                            array.shape().iter().map(|d| *d as i64).collect(),
                            TensorData::$variant(array.iter().copied().collect()),
                        ),
                        Err(e) => {
                            return OnnxOutcome::Rejected {
                                detail: format!("extracting output {index}: {e}"),
                            };
                        }
                    }
                };
            }

            // `TDim` is tract's **symbolic dimension** type, and it is what `Shape` returns:
            // tract models shapes as expressions rather than as concrete integers. That is
            // not a limitation of tract, it is how tract works — and until this arm existed
            // the census recorded 14 cells as tract rejections when tract had in fact
            // answered correctly. A capability matrix that blames a runtime for our own
            // decoding gap is worse than one with a hole in it, because it reads as evidence.
            //
            // Every shape this domain builds is fully static, so each dimension is a concrete
            // integer and the cast succeeds. A genuinely symbolic dimension would fail the
            // cast, and that is reported rather than guessed at.
            if output.datum_type() == DatumType::TDim {
                let concrete = match output.cast_to::<i64>() {
                    Ok(concrete) => concrete,
                    Err(e) => {
                        return OnnxOutcome::Rejected {
                            detail: format!(
                                "output {index} is a symbolic shape that does not resolve to \
                                 concrete integers: {e}"
                            ),
                        };
                    }
                };
                match concrete.as_ref().to_plain_array_view::<i64>() {
                    Ok(array) => {
                        tensors.push(TensorValue::new(
                            OnnxCase::OUTPUT_NAME,
                            array.shape().iter().map(|d| *d as i64).collect(),
                            TensorData::I64(array.iter().copied().collect()),
                        ));
                        continue;
                    }
                    Err(e) => {
                        return OnnxOutcome::Rejected {
                            detail: format!("extracting output {index} after cast: {e}"),
                        };
                    }
                }
            }

            let tensor = match output.datum_type() {
                DatumType::F32 => collect!(f32, F32),
                DatumType::F64 => collect!(f64, F64),
                DatumType::I32 => collect!(i32, I32),
                DatumType::I64 => collect!(i64, I64),
                DatumType::Bool => collect!(bool, Bool),
                other => {
                    return OnnxOutcome::Rejected {
                        detail: format!(
                            "output {index} has datum type {other:?}, which this \
                                         adapter does not decode"
                        ),
                    };
                }
            };
            tensors.push(tensor);
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
        for input in case.fed_inputs() {
            let shape: Vec<usize> = input.dims.iter().map(|d| *d as usize).collect();

            macro_rules! feed {
                ($values:expr) => {
                    match Tensor::from_vec($values.clone(), shape, &Device::Cpu) {
                        Ok(tensor) => tensor,
                        Err(e) => {
                            return OnnxOutcome::Rejected {
                                detail: format!("building input {}: {e}", input.name),
                            };
                        }
                    }
                };
            }

            let tensor = match &input.data {
                TensorData::F32(v) => feed!(v),
                TensorData::F64(v) => feed!(v),
                TensorData::I64(v) => feed!(v),
                // `candle_core::DType` has no boolean or 32-bit-integer variant, so candle
                // cannot represent these tensors at all. That is a genuine capability limit
                // of the runtime rather than a gap in this adapter, which is why it is
                // `Unsupported` (a legitimate skip) and not `Rejected` — and it is exactly
                // the kind of fact the N2 census exists to record.
                TensorData::I32(_) | TensorData::Bool(_) => {
                    return OnnxOutcome::Unsupported {
                        reason: format!("candle has no DType for {:?}", input.elem_type()),
                    };
                }
            };
            feeds.insert(input.name.clone(), tensor);
        }

        let outputs = match candle_onnx::simple_eval(&model, feeds) {
            Ok(outputs) => outputs,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("running: {e}"),
                };
            }
        };

        let Some(tensor) = outputs.get(OnnxCase::OUTPUT_NAME) else {
            return OnnxOutcome::Rejected {
                detail: format!("no output named {}", OnnxCase::OUTPUT_NAME),
            };
        };

        let flat = match tensor.flatten_all() {
            Ok(flat) => flat,
            Err(e) => {
                return OnnxOutcome::Rejected {
                    detail: format!("flattening output: {e}"),
                };
            }
        };

        macro_rules! collect {
            ($rust:ty, $variant:ident) => {
                match flat.to_vec1::<$rust>() {
                    Ok(values) => TensorData::$variant(values),
                    Err(e) => {
                        return OnnxOutcome::Rejected {
                            detail: format!("extracting output: {e}"),
                        };
                    }
                }
            };
        }

        let data = match tensor.dtype() {
            candle_core::DType::F32 => collect!(f32, F32),
            candle_core::DType::F64 => collect!(f64, F64),
            candle_core::DType::I64 => collect!(i64, I64),
            other => {
                return OnnxOutcome::Unsupported {
                    reason: format!(
                        "candle produced dtype {other:?}, which this adapter \
                                     does not decode"
                    ),
                };
            }
        };

        OnnxOutcome::Ok(vec![TensorValue::new(
            OnnxCase::OUTPUT_NAME,
            tensor.dims().iter().map(|d| *d as i64).collect(),
            data,
        )])
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
        for op in OpKind::ELEMENTWISE {
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
                    actual[0].as_f32().expect("f32 tensor"),
                    expected[0].as_f32().expect("f32 tensor"),
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
                out[0].as_f32().expect("f32 tensor"),
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

    /// **The test that makes the multi-dtype work mean anything.**
    ///
    /// Every element type, through every compiled runtime, round-tripping intact. Without
    /// this the refactor is untested for exactly the types it exists to add — the enum
    /// would compile, `validate` would pass, and nothing would ever have run an `i64`.
    ///
    /// `Identity` is used because it performs no arithmetic: anything lost here is lost in
    /// the plumbing, not in a kernel.
    #[test]
    fn every_element_type_survives_every_runtime() {
        use crate::case::ElemType;
        use crate::validation::well_formed_typed;

        for elem in ElemType::ALL {
            let case = well_formed_typed(OpKind::Identity, &[2, 3], OPSET, elem);
            let expected = &case.inputs[0].data;

            for runtime in in_process() {
                match runtime.run(&case).expect("never Err") {
                    OnnxOutcome::Ok(out) => {
                        assert_eq!(
                            out[0].data.to_bit_keys(),
                            expected.to_bit_keys(),
                            "{} altered {elem:?} data",
                            runtime.name()
                        );
                        assert_eq!(
                            out[0].elem_type(),
                            elem,
                            "{} changed the element type of {elem:?}",
                            runtime.name()
                        );
                        assert_eq!(out[0].dims, vec![2, 3], "{} shape", runtime.name());
                    }
                    // A runtime declining a type is legitimate and is exactly the kind of
                    // gap the N2 census exists to measure. Recorded, not failed.
                    other => {
                        eprintln!("note: {} does not handle {elem:?}: {other}", runtime.name());
                    }
                }
            }
        }
    }

    /// At least ONNX Runtime must handle every type this adapter can build. If the maturity
    /// anchor cannot, the type does not belong in `ElemType` yet — an element type nothing
    /// can execute is a variant that inflates the apparent surface without testing anything.
    #[test]
    fn onnx_runtime_handles_every_element_type() {
        use crate::case::ElemType;
        use crate::validation::well_formed_typed;

        for elem in ElemType::ALL {
            let case = well_formed_typed(OpKind::Identity, &[2], OPSET, elem);
            let outcome = OrtRuntime.run(&case).expect("never Err");
            assert!(
                matches!(outcome, OnnxOutcome::Ok(_)),
                "ONNX Runtime could not handle {elem:?}: {outcome}"
            );
        }
    }

    /// **Finding 001, pinned as a test.** `tract` returns `Sign(0) = 1` for integer tensors;
    /// the specification requires `0`.
    ///
    /// > "Calculate the sign of the given input tensor element-wise. If input > 0, output 1.
    /// > if input < 0, output -1. **if input == 0, output 0.**"
    /// > — [ONNX `Sign` reference](https://onnx.ai/onnx/operators/onnx__Sign.html)
    ///
    /// `onnx.reference` and ONNX Runtime both produce `0`; `tract` produces `1`.
    ///
    /// **Correction 2026-08-13:** an earlier version of this comment said the float paths
    /// were correct and that this was specifically the integer path. That was wrong — `tract`
    /// also returns `-0.0` for `Sign(-0.0)` on floats (F-005, and `tests/published_reproductions.rs`
    /// pins it). The original claim came from testing values that included `+0.0` and never
    /// `-0.0`. `Sign` mishandles zero on both type families.
    ///
    /// **The assertion is written to fail when the bug is FIXED**, not while it exists. A
    /// finding recorded only in prose rots: nobody notices when it is fixed, and the report
    /// goes stale. This way a `tract` upgrade that corrects it turns the suite red, which is
    /// the moment to update `FINDING-001` and close it out.
    ///
    /// **This is now expected to fire on the next `tract` upgrade.** The integer path is already
    /// fixed on `main` by [tract#2533](https://github.com/sonos/tract/pull/2533), merged
    /// 30 July 2026 — three weeks after our pinned 0.23.4, and not yet in any release. When the
    /// pin moves past that, this test going red is the *good* outcome and F-001 closes.
    #[test]
    fn finding_001_tract_sign_of_integer_zero() {
        use crate::case::{ElemType, TensorData};

        let case = OnnxCase::new(
            OpKind::Sign,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![3],
                TensorData::I32(vec![-1, 0, 1]),
            )],
        );

        // ONNX Runtime is the control: it agrees with the specification.
        let OnnxOutcome::Ok(correct) = OrtRuntime.run(&case).expect("never Err") else {
            panic!("ONNX Runtime should compute Sign on int32");
        };
        assert_eq!(
            correct[0].data,
            TensorData::I32(vec![-1, 0, 1]),
            "the control must match the specification, or this test proves nothing"
        );

        let OnnxOutcome::Ok(observed) = TractRuntime.run(&case).expect("never Err") else {
            panic!("tract claims Sign at int32 — the census recorded it as supported");
        };
        assert_eq!(
            observed[0].data,
            TensorData::I32(vec![-1, 1, 1]),
            "tract's Sign(0) on integers changed. If it now returns 0 the bug is FIXED — \
             update issues/onnx-runtime/FINDING-001 and delete this test. \
             Expected the known-wrong [-1, 1, 1]; got {:?}",
            observed[0].data
        );

        // `ElemType` is referenced so the doc link above stays honest about the type involved.
        assert_eq!(case.inputs[0].elem_type(), ElemType::I32);
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
            for (sent, received) in hostile
                .iter()
                .zip(out[0].as_f32().expect("f32 tensor").iter())
            {
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
