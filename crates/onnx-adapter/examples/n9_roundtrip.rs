//! **N9.6** — run the quantization round trip against every runtime that can do it.
//!
//! Two single-node models in sequence rather than one two-node graph: this domain tests
//! single-node models, and widening that to make one relation convenient would change what every
//! other number in the project means. The relation is metamorphic across two *runs*, not a graph.
//!
//! Run with:
//!   cargo run --release -p onnx-adapter --example n9_roundtrip --features candle
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Implementation;
use onnx_adapter::case::{ElemType, OnnxCase, OpKind, TensorData, TensorValue};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::roundtrip::{self, Outcome};
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};
use rand::RngExt;

const CASES: u64 = 4000;

fn quantize(x: &[f32], scale: f32, zp: i64, target: ElemType) -> OnnxCase {
    let zp_data = match target {
        ElemType::U8 => TensorData::U8(vec![zp as u8]),
        _ => TensorData::I8(vec![zp as i8]),
    };
    OnnxCase::new(
        OpKind::QuantizeLinear,
        22,
        vec![
            TensorValue::f32("a", vec![x.len() as i64], x.to_vec()),
            TensorValue::new("scale", vec![], TensorData::F32(vec![scale])),
            TensorValue::new("zp", vec![], zp_data),
        ],
    )
}

fn dequantize(q: TensorData, scale: f32, zp: i64, target: ElemType) -> OnnxCase {
    let zp_data = match target {
        ElemType::U8 => TensorData::U8(vec![zp as u8]),
        _ => TensorData::I8(vec![zp as i8]),
    };
    let len = q.len() as i64;
    OnnxCase::new(
        OpKind::DequantizeLinear,
        22,
        vec![
            TensorValue::new("a", vec![len], q),
            TensorValue::new("scale", vec![], TensorData::F32(vec![scale])),
            TensorValue::new("zp", vec![], zp_data),
        ],
    )
}

fn floats(outcome: &OnnxOutcome) -> Option<Vec<f32>> {
    match outcome {
        OnnxOutcome::Ok(t) => match &t[0].data {
            TensorData::F32(v) => Some(v.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn run_all(
    name: &str,
    runtime: &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>,
) -> (Outcome, Vec<String>) {
    let mut outcome = Outcome::default();
    let mut violations = Vec::new();
    let mut rng = SeededRng::from_seed(0);

    for seed in 0..CASES {
        let mut rng = SeededRng::from_seed(seed);
        let target = if rng.random_bool(0.5) {
            ElemType::I8
        } else {
            ElemType::U8
        };
        let scale: f32 = rng.random_range(0.005_f32..2.0);
        let (low, high) = target.saturation_range().expect("quantized");
        let zp = rng.random_range(low..=high);
        let values: Vec<f32> = (0..8).map(|_| rng.random_range(-50.0_f32..50.0)).collect();

        let q_case = quantize(&values, scale, zp, target);
        let Ok(OnnxOutcome::Ok(q_out)) = runtime.run(&q_case) else {
            continue;
        };
        let d_case = dequantize(q_out[0].data.clone(), scale, zp, target);
        let Ok(d_res) = runtime.run(&d_case) else {
            continue;
        };
        let Some(back) = floats(&d_res) else { continue };

        for (x, y) in values.iter().zip(back.iter()) {
            let verdict = roundtrip::holds(*x, *y, scale, zp, target);
            outcome.record(verdict);
            if verdict == Some(false) && violations.len() < 5 {
                violations.push(format!(
                    "seed {seed}: x={x} -> {y} (scale {scale}, zp {zp}, {target:?}), \
                     error {:.6} > bound {:.6}",
                    (y - x).abs(),
                    roundtrip::tolerance(scale)
                ));
            }
        }
    }
    let _ = &mut rng;
    let _ = name;
    (outcome, violations)
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let reference = ReferenceRuntime::start().expect("reference");

    println!("\nquantization round trip — DequantizeLinear(QuantizeLinear(x))");
    println!("bound: scale/2, derived from the round-half-to-even rule (SPECS 2q.1/2q.2)\n");
    println!(
        "{:<14} {:>9} {:>10} {:>18} {:>10}",
        "runtime", "held", "VIOLATED", "not representable", "judged"
    );

    let participants: Vec<(&str, &dyn Implementation<In = OnnxCase, Out = OnnxOutcome>)> = vec![
        ("reference", &reference),
        ("onnxruntime", &OrtRuntime),
        ("tract", &TractRuntime),
    ];

    let mut any = false;
    for (name, runtime) in participants {
        let (outcome, violations) = run_all(name, runtime);
        if outcome.judged() == 0 && outcome.not_representable == 0 {
            println!(
                "{name:<14} {:>9} {:>10} {:>18} {:>10}   (does not run these operators)",
                "-", "-", "-", "-"
            );
            continue;
        }
        any = true;
        println!(
            "{name:<14} {:>9} {:>10} {:>18} {:>10}",
            outcome.held,
            outcome.violated,
            outcome.not_representable,
            outcome.judged()
        );
        for v in &violations {
            println!("      VIOLATION  {v}");
        }
    }
    std::panic::set_hook(previous);
    if !any {
        println!("\nnothing ran — the relation judged nothing at all");
    }
}
