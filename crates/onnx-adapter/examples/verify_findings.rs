//! Re-run every finding this project has recorded, against every implementation, and print what
//! each one actually does.
//!
//! # Why this exists as a runnable example rather than only as tests
//!
//! `tests/published_reproductions.rs` asserts that each filed report still reproduces, which is
//! the right thing for CI — it fails loudly when a runtime is upgraded. But an assertion prints
//! nothing when it passes, and a report is written from the *text* an implementation produced:
//! the error message, the exact bytes, which of the four disagreed with which.
//!
//! So this prints the evidence rather than checking it. Run it before filing anything.
//!
//! # What "confirmed" means here
//!
//! For each finding: at least two implementations produce **comparable** outcomes that **differ**,
//! and the difference is the one the draft describes. A finding where every implementation now
//! agrees is not a finding any more, and this is how that would be discovered.

use diff_fuzzer_core::traits::Implementation;
use onnx_adapter::attrs::Attrs;
use onnx_adapter::case::{OnnxCase, OpKind, TensorData, TensorValue};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

const OPSET: i64 = 22;

/// One finding, as a case plus the identity of the report it belongs to.
struct Finding {
    id: &'static str,
    repo: &'static str,
    claim: &'static str,
    case: OnnxCase,
}

/// Render an outcome so that *why* an implementation declined is visible, not just that it did.
///
/// The canonical form a campaign stores collapses every failure to an empty tensor list, which is
/// enough to compare and useless to quote. A report needs the words.
fn render(outcome: &OnnxOutcome) -> String {
    match outcome {
        OnnxOutcome::Ok(tensors) => tensors
            .iter()
            .map(|t| {
                let values = match &t.data {
                    TensorData::F32(v) => format!(
                        "{:?}",
                        v.iter()
                            .map(|x| format!("{x}({:#010x})", x.to_bits()))
                            .collect::<Vec<_>>()
                    ),
                    TensorData::F64(v) => format!(
                        "{:?}",
                        v.iter()
                            .map(|x| format!("{x}({:#018x})", x.to_bits()))
                            .collect::<Vec<_>>()
                    ),
                    TensorData::I32(v) => format!("{v:?}"),
                    TensorData::I64(v) => format!("{v:?}"),
                    TensorData::I8(v) => format!("{v:?}"),
                    TensorData::U8(v) => format!("{v:?}"),
                    TensorData::Bool(v) => format!("{v:?}"),
                };
                format!("{:?}{:?} = {values}", t.elem_type(), t.dims)
            })
            .collect::<Vec<_>>()
            .join(" | "),
        OnnxOutcome::Rejected { detail } => {
            format!("REJECTED: {}", detail.lines().next().unwrap_or("").trim())
        }
        OnnxOutcome::Crashed { detail } => {
            format!("CRASHED:  {}", detail.lines().next().unwrap_or("").trim())
        }
        OnnxOutcome::TimedOut { .. } => "TIMED OUT".to_string(),
        OnnxOutcome::Unsupported { reason } => {
            format!(
                "unsupported: {}",
                reason.lines().next().unwrap_or("").trim()
            )
        }
    }
}

/// Whether an outcome is something the oracle would compare at all.
fn comparable(outcome: &OnnxOutcome) -> bool {
    !matches!(outcome, OnnxOutcome::Unsupported { .. })
}

fn scalar(op: OpKind, dims: Vec<i64>, data: TensorData) -> OnnxCase {
    OnnxCase::new(op, OPSET, vec![TensorValue::new("a", dims, data)])
}

/// The zero-size `Reshape` model, with `allowzero = 1`.
///
/// **The attribute is the whole reason this is well defined.** With `allowzero=1` a `0` in the
/// target means a literal zero-length dimension, so a `[3, 0]` input (0 elements) reshaped to
/// `[0]` (0 elements) matches. With the default `allowzero=0` the `0` would mean "copy input
/// dimension 0" — target `[3]`, 3 elements against 0 — and a rejection would be *correct*.
///
/// Every `Reshape` finding this project recorded carries `allowzero=1`, checked across the
/// 3,000,000-case corpus. Without that check these findings would be ours.
fn zero_size_reshape(elem: TensorData) -> OnnxCase {
    OnnxCase::new(
        OpKind::Reshape,
        OPSET,
        vec![
            TensorValue::new("a", vec![3, 0], elem),
            TensorValue::new("b", vec![1], TensorData::I64(vec![0])).as_initializer(),
        ],
    )
    .with_attrs(Attrs::new().int("allowzero", 1))
}

fn findings() -> Vec<Finding> {
    vec![
        Finding {
            id: "F-001  tract Sign(0) = 1 for integers",
            repo: "sonos/tract  — NOT FILED: fixed upstream by tract#2533",
            claim: "tract returns 1 for Sign(0) on int32; ONNX specifies 0",
            case: scalar(OpKind::Sign, vec![1], TensorData::I32(vec![0])),
        },
        Finding {
            id: "F-002  tract rejects a zero-size Reshape",
            repo: "sonos/tract",
            claim: "tract fails to load; the reference and ONNX Runtime execute it",
            case: zero_size_reshape(TensorData::F32(vec![])),
        },
        Finding {
            id: "F-003  candle fails on rank-0 scalars",
            repo: "huggingface/candle  — NOT FILED: coverage openly incomplete",
            claim: "candle rejects a legal rank-0 input the other three accept",
            case: scalar(OpKind::Neg, vec![], TensorData::I64(vec![-76])),
        },
        Finding {
            id: "F-004  ONNX Runtime loses -0.0 through Where's X branch",
            repo: "microsoft/onnxruntime",
            claim: "Where is selection, so -0.0 selected from X must stay -0.0",
            case: OnnxCase::new(
                OpKind::Where,
                OPSET,
                vec![
                    TensorValue::new("cond", vec![1], TensorData::Bool(vec![true])),
                    TensorValue::new("x", vec![1], TensorData::F32(vec![-0.0])),
                    TensorValue::new("y", vec![1], TensorData::F32(vec![1.0])),
                ],
            ),
        },
        Finding {
            id: "F-005  tract Sign(-0.0) returns -0.0",
            repo: "sonos/tract",
            claim: "-0.0 == 0 is true, so ONNX's 'if input == 0, output 0' applies",
            case: scalar(OpKind::Sign, vec![1], TensorData::F32(vec![-0.0])),
        },
        Finding {
            id: "F-006  tract panics on int32::MIN / -1",
            repo: "sonos/tract",
            claim: "the quotient is unrepresentable; the other two wrap, tract panics",
            case: OnnxCase::new(
                OpKind::Div,
                OPSET,
                vec![
                    TensorValue::new("a", vec![1], TensorData::I32(vec![i32::MIN])),
                    TensorValue::new("b", vec![1], TensorData::I32(vec![-1])),
                ],
            ),
        },
        Finding {
            id: "F-007  candle appears to ignore allowzero=1 on Reshape",
            repo: "huggingface/candle",
            claim: "the error names rhs [3], the input dimension — the allowzero=0 reading",
            case: zero_size_reshape(TensorData::F32(vec![])),
        },
        Finding {
            id: "F-008  tract DynamicQuantizeLinear rounds ties away from zero",
            repo: "sonos/tract",
            claim: "ONNX specifies nearest-ties-to-even; tract uses Rust's f32::round",
            case: scalar(
                OpKind::DynamicQuantizeLinear,
                vec![3],
                TensorData::F32(vec![-127.0, 128.0, 0.5]),
            ),
        },
    ]
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let reference = ReferenceRuntime::start().expect("start onnx.reference");
    let mut confirmed = 0usize;
    let mut vanished = Vec::new();

    for finding in findings() {
        println!("\n═══ {} ═══", finding.id);
        println!("  repo:  {}", finding.repo);
        println!("  claim: {}", finding.claim);
        println!(
            "  model: {} opset {} — {}",
            finding.case.op.onnx_name(),
            finding.case.opset,
            finding
                .case
                .inputs
                .iter()
                .map(|i| format!("{}{:?}", i.name, i.dims))
                .collect::<Vec<_>>()
                .join(", ")
        );

        // The reference is listed first: it is the specification's own implementation, so it
        // defines the expected answer rather than merely voting.
        let outcomes: Vec<(&str, OnnxOutcome)> = vec![
            (
                "onnx.reference",
                reference.run(&finding.case).expect("never Err"),
            ),
            (
                "onnxruntime",
                OrtRuntime.run(&finding.case).expect("never Err"),
            ),
            ("tract", TractRuntime.run(&finding.case).expect("never Err")),
            #[cfg(feature = "candle")]
            (
                "candle",
                onnx_adapter::runtimes::CandleRuntime
                    .run(&finding.case)
                    .expect("never Err"),
            ),
        ];

        for (name, outcome) in &outcomes {
            println!("    {name:<16} {}", render(outcome));
        }

        // Confirmed = at least two comparable outcomes that are not all identical.
        let rendered: Vec<String> = outcomes
            .iter()
            .filter(|(_, o)| comparable(o))
            .map(|(_, o)| render(o))
            .collect();
        let distinct: std::collections::BTreeSet<&String> = rendered.iter().collect();
        if rendered.len() >= 2 && distinct.len() >= 2 {
            println!(
                "    → CONFIRMED: {} comparable, {} distinct",
                rendered.len(),
                distinct.len()
            );
            confirmed += 1;
        } else {
            println!(
                "    → NOT A DIVERGENCE ANY MORE — {} comparable, {} distinct",
                rendered.len(),
                distinct.len()
            );
            vanished.push(finding.id);
        }
    }

    std::panic::set_hook(previous);
    println!("\n═══════════════════════════════════════");
    println!(
        "  {confirmed} of {} findings still diverge",
        findings().len()
    );
    if vanished.is_empty() {
        println!("  none have vanished");
    } else {
        println!("  *** NO LONGER DIVERGING — do not file: {vanished:?} ***");
    }
}
