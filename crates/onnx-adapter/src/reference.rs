//! The specification's own executable definition, reached by subprocess.
//!
//! `onnx.reference` is **ground truth, not a peer**. When a runtime disagrees with it,
//! that is a conformance violation attributable to the runtime — which is the property
//! neither of this project's earlier domains had, and the reason this domain was chosen.
//! It is also the **validity gate**: if the reference accepts a model, the model is valid,
//! so a runtime crashing on it is the runtime's bug rather than ours.
//!
//! # Why a long-lived process
//!
//! The process is kept alive across cases and fed one request after another. Measured
//! 2026-08-12: interpreter startup plus `import onnx` costs **~98 ms**, and the first
//! `ReferenceEvaluator` ever constructed costs another **~55 ms** while it registers
//! roughly 192 operator classes. Both are paid once per *process*. The steady-state cost
//! of a case is **~0.023 ms**.
//!
//! Spawning per case would therefore make the reference about 4,000× more expensive than
//! it needs to be — and that inflated figure is exactly what would justify keeping it out
//! of the loop, which would in turn cost the domain its specification oracle on every
//! agreeing case. See `SPECS.md` §1.1 and `PENDING` 1.2.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use crate::case::{ElemType, OnnxCase, TensorData, TensorValue};
use crate::outcome::OnnxOutcome;

/// A running `onnx.reference` worker.
///
/// Holds the child process and its pipes. Dropping it closes stdin, which the runner
/// treats as a clean end of stream and exits on.
pub struct Reference {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
}

/// What the reference said about one case.
///
/// An alias rather than its own enum: the reference's outcomes are the same kind of thing
/// as any runtime's, and giving it a private vocabulary would mean translating at the
/// oracle boundary — where a translation bug looks exactly like a divergence.
pub type ReferenceOutcome = OnnxOutcome;

impl Reference {
    /// Start the worker.
    ///
    /// The interpreter is the project-local `.venv-onnx`, not whatever `python3` resolves
    /// to. The `onnx` version *is* the specification revision under test, so it must be a
    /// property of the repository rather than of the machine.
    pub fn start() -> Result<Self, String> {
        let root = repo_root();
        let python = root.join(".venv-onnx/bin/python");
        let script = root.join("crates/onnx-adapter/python/reference_runner.py");

        if !python.is_file() {
            return Err(format!(
                "no interpreter at {}. The ONNX domain needs the reference implementation:\n  \
                 python3 -m venv .venv-onnx && ./.venv-onnx/bin/python -m pip install \
                 -r crates/onnx-adapter/requirements.txt",
                python.display()
            ));
        }

        let mut child = Command::new(&python)
            .arg(&script)
            // Keeps Python from writing .pyc files into the source tree.
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr is left inherited on purpose: if the runner dies before it can report
            // a failure as a value, the reason should reach the terminal rather than a pipe
            // nobody drains.
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("spawning {}: {e}", python.display()))?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    /// Run one case and read back what the reference said.
    pub fn run(
        &mut self,
        model_bytes: &[u8],
        inputs: &[TensorValue],
    ) -> Result<OnnxOutcome, String> {
        self.write_request(model_bytes, inputs)
            .map_err(|e| format!("writing request: {e}"))?;
        self.read_response()
            .map_err(|e| format!("reading response: {e}"))
    }

    fn write_request(&mut self, model_bytes: &[u8], inputs: &[TensorValue]) -> std::io::Result<()> {
        let out = &mut self.stdin;
        write_u32(out, model_bytes.len() as u32)?;
        out.write_all(model_bytes)?;

        write_u32(out, inputs.len() as u32)?;
        for input in inputs {
            write_u32(out, input.name.len() as u32)?;
            out.write_all(input.name.as_bytes())?;
            // The element type, as ONNX's own `TensorProto.DataType` integer. A shared
            // vocabulary rather than a private one: both sides already have to agree with
            // ONNX about what `7` means, so inventing a second numbering would add a
            // translation nobody needs and a place for it to be wrong.
            write_u32(out, input.elem_type().wire() as u32)?;
            write_u32(out, input.dims.len() as u32)?;
            for dim in &input.dims {
                out.write_all(&dim.to_le_bytes())?;
            }
            // Raw little-endian bit patterns, preserving NaN payloads and the sign of
            // zero — which a text encoding would destroy.
            let payload = input.data.to_le_bytes();
            write_u32(out, payload.len() as u32)?;
            out.write_all(&payload)?;
        }
        out.flush()
    }

    fn read_response(&mut self) -> std::io::Result<OnnxOutcome> {
        let status = read_u8(&mut self.stdout)?;
        if status == 1 {
            let length = read_u32(&mut self.stdout)? as usize;
            let mut message = vec![0u8; length];
            self.stdout.read_exact(&mut message)?;
            return Ok(OnnxOutcome::Rejected {
                detail: String::from_utf8_lossy(&message).into_owned(),
            });
        }

        let count = read_u32(&mut self.stdout)? as usize;
        let mut tensors = Vec::with_capacity(count);
        for _ in 0..count {
            let name_length = read_u32(&mut self.stdout)? as usize;
            let mut name = vec![0u8; name_length];
            self.stdout.read_exact(&mut name)?;

            let wire_type = read_u32(&mut self.stdout)? as i32;
            let Some(elem_type) = ElemType::from_wire(wire_type) else {
                // A type the reference produced and this adapter cannot represent. Reported
                // as a value rather than silently decoded as something else — guessing here
                // would turn a capability gap into a fabricated divergence.
                return Ok(OnnxOutcome::Rejected {
                    detail: format!(
                        "the reference returned element type {wire_type}, which \
                                     this adapter does not decode"
                    ),
                });
            };

            let rank = read_u32(&mut self.stdout)? as usize;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                let mut raw = [0u8; 8];
                self.stdout.read_exact(&mut raw)?;
                dims.push(i64::from_le_bytes(raw));
            }

            let payload_length = read_u32(&mut self.stdout)? as usize;
            let mut payload = vec![0u8; payload_length];
            self.stdout.read_exact(&mut payload)?;

            tensors.push(TensorValue::new(
                &String::from_utf8_lossy(&name),
                dims,
                TensorData::from_le_bytes(elem_type, &payload),
            ));
        }
        Ok(OnnxOutcome::Ok(tensors))
    }
}

impl Drop for Reference {
    fn drop(&mut self) {
        // The worker exits when stdin closes. Reaping it here keeps a long campaign from
        // accumulating zombie processes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The reference implementation bound to the [`Implementation`] seam.
///
/// # Why the interior mutability
///
/// `Implementation::run` takes `&self`, but the worker owns pipes and must be written to,
/// so the process lives inside a `RefCell`. The alternative — spawning per call — was
/// measured and rejected: it would cost ~150 ms of startup per case against ~0.023 ms of
/// work, and that inflated figure is exactly what would justify demoting the specification
/// oracle to a triage aid. See the module note.
///
/// `RefCell` rather than `Mutex` because the engine drives one case at a time on one
/// thread; a `Mutex` would buy thread-safety nothing here needs and hide that fact.
pub struct ReferenceRuntime {
    worker: std::cell::RefCell<Reference>,
}

pub const REFERENCE_NAME: &str = "onnx.reference";

impl ReferenceRuntime {
    pub fn start() -> Result<Self, String> {
        Ok(Self {
            worker: std::cell::RefCell::new(Reference::start()?),
        })
    }
}

impl diff_fuzzer_core::traits::Implementation for ReferenceRuntime {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        REFERENCE_NAME
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, diff_fuzzer_core::traits::RunError> {
        let bytes = crate::model::build_bytes(input);
        let outcome = self
            .worker
            .borrow_mut()
            .run(&bytes, &input.inputs)
            // A broken pipe means the worker died — which is a *harness* failure, not a
            // statement about the case, so it is reported as `Crashed` against the
            // reference itself rather than silently swallowed. If this ever fires in a
            // campaign it means the specification oracle went offline, and that must be
            // loud.
            .unwrap_or_else(|e| OnnxOutcome::Crashed {
                detail: format!("the reference worker failed: {e}"),
            });
        Ok(outcome)
    }
}

fn write_u32(out: &mut impl Write, value: u32) -> std::io::Result<()> {
    out.write_all(&value.to_le_bytes())
}

fn read_u32(input: &mut impl Read) -> std::io::Result<u32> {
    let mut raw = [0u8; 4];
    input.read_exact(&mut raw)?;
    Ok(u32::from_le_bytes(raw))
}

fn read_u8(input: &mut impl Read) -> std::io::Result<u8> {
    let mut raw = [0u8; 1];
    input.read_exact(&mut raw)?;
    Ok(raw[0])
}

/// The repository root, derived from this crate's location at compile time.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DEFAULT_OPSET, add_model, to_bytes};

    /// A model that is invalid *on purpose*: `Add` declared with a single input.
    ///
    /// Built through the normal path rather than hand-rolled, which is the point —
    /// `model::build` does not validate, so this is exactly the kind of case a buggy
    /// generator or an over-eager shrink step would produce. `validate` rejects it, and
    /// so must the reference.
    fn deliberately_invalid_model() -> Vec<u8> {
        let case = crate::case::OnnxCase::new(
            crate::case::OpKind::Add,
            DEFAULT_OPSET,
            vec![crate::case::TensorValue::f32("a", vec![2], vec![1.0, 2.0])],
        );
        assert!(
            !crate::validation::is_valid(&case),
            "this helper must produce something our own validator rejects"
        );
        crate::model::build_bytes(&case)
    }

    #[test]
    fn the_reference_runs_a_hand_built_model() {
        let mut reference = Reference::start().expect("the reference worker must start");
        let bytes = to_bytes(&add_model(&[2, 3], DEFAULT_OPSET));
        let inputs = vec![
            TensorValue::f32("a", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            TensorValue::f32("b", vec![2, 3], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ];

        match reference
            .run(&bytes, &inputs)
            .expect("the worker must reply")
        {
            OnnxOutcome::Ok(outputs) => {
                assert_eq!(outputs.len(), 1);
                assert_eq!(outputs[0].dims, vec![2, 3]);
                assert_eq!(
                    outputs[0].as_f32().expect("f32 tensor"),
                    vec![11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
                );
            }
            OnnxOutcome::Rejected { detail: why } => {
                panic!("the reference rejected a valid model:\n{why}")
            }
            other => panic!("unexpected outcome from the reference: {other}"),
        }
    }

    /// One worker must serve many cases. This is the property that makes the reference
    /// affordable as a per-case participant rather than only a confirmer.
    #[test]
    fn one_worker_serves_many_cases() {
        let mut reference = Reference::start().expect("the reference worker must start");
        let bytes = to_bytes(&add_model(&[2], DEFAULT_OPSET));

        for round in 0..25 {
            let left = round as f32;
            let inputs = vec![
                TensorValue::f32("a", vec![2], vec![left, left]),
                TensorValue::f32("b", vec![2], vec![1.0, 2.0]),
            ];
            let OnnxOutcome::Ok(out) = reference.run(&bytes, &inputs).expect("reply") else {
                panic!("round {round} was rejected");
            };
            assert_eq!(
                out[0].as_f32().expect("f32 tensor"),
                vec![left + 1.0, left + 2.0]
            );
        }
    }

    /// Special values must survive the process boundary **bit-for-bit**. This is the test
    /// that justifies the binary wire format: a JSON encoding would fail every assertion
    /// below, and would do so silently by turning them into nulls or strings.
    ///
    /// `Identity` is used rather than `Add`, deliberately. Any arithmetic operator would
    /// mix two questions — "did the bytes survive the pipe?" and "what does this operator
    /// do to this value?" — and the first version of this test did exactly that, asserting
    /// that adding `+0.0` preserves the sign of zero. It does not: IEEE-754 gives
    /// `(-0.0) + (+0.0) = +0.0` under round-to-nearest, and only *like*-signed zeros keep
    /// the sign. The test failed, correctly, against a wrong premise. `Identity` performs
    /// no arithmetic, so it tests the boundary and nothing else.
    #[test]
    fn special_values_cross_the_boundary_unchanged() {
        let mut reference = Reference::start().expect("the reference worker must start");
        let hostile = vec![f32::INFINITY, f32::NEG_INFINITY, -0.0, f32::MIN_POSITIVE];
        let case = crate::case::OnnxCase::new(
            crate::case::OpKind::Identity,
            DEFAULT_OPSET,
            vec![crate::case::TensorValue::f32("a", vec![4], hostile.clone())],
        );
        let bytes = crate::model::build_bytes(&case);
        let inputs = vec![TensorValue::f32("a", vec![4], hostile.clone())];

        let OnnxOutcome::Ok(out) = reference.run(&bytes, &inputs).expect("reply") else {
            panic!("the reference rejected an Identity model with special values");
        };

        assert!(
            out[0].as_f32().expect("f32 tensor")[0].is_infinite()
                && out[0].as_f32().expect("f32 tensor")[0].is_sign_positive()
        );
        assert!(
            out[0].as_f32().expect("f32 tensor")[1].is_infinite()
                && out[0].as_f32().expect("f32 tensor")[1].is_sign_negative()
        );
        // `-0.0 == 0.0` is true in IEEE-754, so comparing values would pass even if the
        // sign were lost. Comparing the *bits* is what actually checks this.
        assert_eq!(
            out[0].as_f32().expect("f32 tensor")[2].to_bits(),
            (-0.0f32).to_bits(),
            "the sign of zero was lost crossing the process boundary"
        );
        assert_eq!(
            out[0].as_f32().expect("f32 tensor")[3],
            f32::MIN_POSITIVE,
            "subnormal boundary lost"
        );
    }

    /// What `onnx.reference` actually does with signed zeros under `Add`.
    ///
    /// **Measured, not cited.** This records observed behaviour of onnx 1.22.0 and makes
    /// no claim about what any standard requires — that distinction is load-bearing here
    /// (`02-METHODOLOGY.md`: "measurement is not a substitute for a citation"). The
    /// matching IEEE-754 claim sits in `SPECS.md` §5 until someone retrieves it.
    ///
    /// It is worth pinning now because signed zero was a documented blind spot in the
    /// tensor domain, and `PENDING` 1.6 has to decide at N4 whether `+0.0` and `-0.0`
    /// count as a disagreement. A rule written without knowing this would be written blind.
    #[test]
    fn signed_zero_behaviour_under_add_is_pinned() {
        let mut reference = Reference::start().expect("the reference worker must start");
        let bytes = to_bytes(&add_model(&[2], DEFAULT_OPSET));

        let inputs = vec![
            // opposite-signed zeros, then like-signed zeros
            TensorValue::f32("a", vec![2], vec![-0.0, -0.0]),
            TensorValue::f32("b", vec![2], vec![0.0, -0.0]),
        ];
        let OnnxOutcome::Ok(out) = reference.run(&bytes, &inputs).expect("reply") else {
            panic!("rejected");
        };

        assert_eq!(
            out[0].as_f32().expect("f32 tensor")[0].to_bits(),
            (0.0f32).to_bits(),
            "(-0.0) + (+0.0) should give +0.0"
        );
        assert_eq!(
            out[0].as_f32().expect("f32 tensor")[1].to_bits(),
            (-0.0f32).to_bits(),
            "(-0.0) + (-0.0) should give -0.0"
        );
    }

    /// A NaN must arrive as a NaN. Kept separate from the test above because NaN needs
    /// `is_nan()` rather than equality — `NaN == NaN` is false, so an equality assertion
    /// would fail even on a correct round trip.
    fn nan_survives_the_boundary_impl() {
        let mut reference = Reference::start().expect("the reference worker must start");
        let bytes = to_bytes(&add_model(&[1], DEFAULT_OPSET));
        let inputs = vec![
            TensorValue::f32("a", vec![1], vec![f32::NAN]),
            TensorValue::f32("b", vec![1], vec![0.0]),
        ];

        let OnnxOutcome::Ok(out) = reference.run(&bytes, &inputs).expect("reply") else {
            panic!("the reference rejected a NaN model");
        };
        assert!(
            out[0].as_f32().expect("f32 tensor")[0].is_nan(),
            "NaN did not survive the boundary"
        );
    }

    #[test]
    fn nan_survives_the_boundary() {
        nan_survives_the_boundary_impl();
    }

    /// Every element type must cross the process boundary intact, in both directions.
    ///
    /// The wire format carries the element type explicitly, and this is what proves the
    /// two sides agree about what each code means. A mismatch would decode one type's bits
    /// as another's and produce a divergence that looks entirely real.
    #[test]
    fn every_element_type_survives_the_process_boundary() {
        use crate::case::ElemType;
        use crate::validation::well_formed_typed;

        let mut reference = Reference::start().expect("the reference worker must start");

        for elem in ElemType::ALL {
            let case =
                well_formed_typed(crate::case::OpKind::Identity, &[2, 3], DEFAULT_OPSET, elem);
            let bytes = crate::model::build_bytes(&case);

            let outcome = reference.run(&bytes, &case.inputs).expect("reply");
            let OnnxOutcome::Ok(out) = outcome else {
                panic!("the reference rejected an Identity model at {elem:?}: {outcome}");
            };
            assert_eq!(
                out[0].elem_type(),
                elem,
                "the element type changed crossing the boundary"
            );
            assert_eq!(
                out[0].data.to_bit_keys(),
                case.inputs[0].data.to_bit_keys(),
                "{elem:?} data changed crossing the boundary"
            );
            assert_eq!(out[0].dims, vec![2, 3]);
        }
    }

    /// An invalid model must come back as `Rejected`, not as a panic or a hang. This is
    /// the validity gate working: our own malformed model is our bug, and the reference
    /// saying so is how we find out before blaming a runtime.
    #[test]
    fn an_invalid_model_is_rejected_as_a_value() {
        let mut reference = Reference::start().expect("the reference worker must start");

        // `Add` needs two inputs. Declaring one is a model the checker must refuse.
        let broken = deliberately_invalid_model();
        let inputs = vec![TensorValue::f32("a", vec![2], vec![1.0, 2.0])];

        match reference.run(&broken, &inputs).expect("reply") {
            OnnxOutcome::Rejected { detail: why } => {
                assert!(!why.is_empty(), "a rejection must carry a reason");
            }
            OnnxOutcome::Ok(_) => {
                panic!("the reference accepted an Add node with only one input")
            }
            other => panic!("expected a rejection, got {other}"),
        }
    }

    /// The worker must keep serving after a rejection. Without this, one invalid model
    /// mid-campaign would silently take the specification oracle offline for every case
    /// that followed.
    #[test]
    fn the_worker_survives_a_rejection() {
        let mut reference = Reference::start().expect("the reference worker must start");

        let broken = deliberately_invalid_model();
        let _ = reference.run(&broken, &[TensorValue::f32("a", vec![2], vec![1.0, 2.0])]);

        let good = to_bytes(&add_model(&[2], DEFAULT_OPSET));
        let inputs = vec![
            TensorValue::f32("a", vec![2], vec![1.0, 2.0]),
            TensorValue::f32("b", vec![2], vec![3.0, 4.0]),
        ];
        let OnnxOutcome::Ok(out) = reference.run(&good, &inputs).expect("reply") else {
            panic!("the worker stopped working after a rejection");
        };
        assert_eq!(out[0].as_f32().expect("f32 tensor"), vec![4.0, 6.0]);
    }
}
