//! **N9.2 smoke test** — do the quantized operators actually run, on every participant?
//!
//! Deliberately separate from the census: this asks the narrower question "is the plumbing
//! right", where the census asks "what does each runtime support". If `int8` were being fed or
//! decoded wrongly, every quantized result would be wrong in the same way on every runtime and
//! the oracle would report agreement — the failure mode a differential oracle cannot see.
//! Comparing against `onnx.reference` is what catches it.
use diff_fuzzer_core::traits::Implementation;
use onnx_adapter::case::{ElemType, OnnxCase, OpKind, TensorData, TensorValue};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime};

fn show(o: &OnnxOutcome) -> String {
    match o {
        OnnxOutcome::Ok(t) => t
            .iter()
            .map(|x| format!("{:?}{:?}", x.dims, x.data))
            .collect::<Vec<_>>()
            .join(" | "),
        other => format!("{other:?}").chars().take(72).collect(),
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let reference = ReferenceRuntime::start().expect("reference");

    // 1. Plain int8 identity — proves feed and decode, with no quantization logic involved.
    let identity = OnnxCase::new(
        OpKind::Identity,
        22,
        vec![TensorValue::new(
            "a",
            vec![4],
            TensorData::I8(vec![-128, -1, 0, 127]),
        )],
    );

    // 2. QuantizeLinear at the rounding boundary. scale=1, zp=0, so x/scale = x and the
    //    halfway values 0.5 and 1.5 probe round-half-to-even directly: 0.5 -> 0, 1.5 -> 2.
    let quantize = OnnxCase::new(
        OpKind::QuantizeLinear,
        22,
        vec![
            TensorValue::f32("a", vec![6], vec![0.5, 1.5, 2.5, -0.5, -1.5, 300.0]),
            TensorValue::new("scale", vec![], TensorData::F32(vec![1.0])),
            TensorValue::new("zp", vec![], TensorData::I8(vec![0])),
        ],
    );

    let dequantize = OnnxCase::new(
        OpKind::DequantizeLinear,
        22,
        vec![
            TensorValue::new("a", vec![3], TensorData::I8(vec![-128, 0, 127])),
            TensorValue::new("scale", vec![], TensorData::F32(vec![0.5])),
            TensorValue::new("zp", vec![], TensorData::I8(vec![0])),
        ],
    );

    let matmul = OnnxCase::new(
        OpKind::MatMulInteger,
        22,
        vec![
            TensorValue::new("a", vec![2, 2], TensorData::I8(vec![1, 2, 3, 4])),
            TensorValue::new("b", vec![2, 2], TensorData::I8(vec![5, 6, 7, 8])),
            TensorValue::new("azp", vec![], TensorData::I8(vec![0])),
            TensorValue::new("bzp", vec![], TensorData::I8(vec![0])),
        ],
    );

    let dynamic = OnnxCase::new(
        OpKind::DynamicQuantizeLinear,
        22,
        vec![TensorValue::f32("a", vec![4], vec![-1.0, 0.0, 2.0, 4.0])],
    );

    for (label, case) in [
        ("Identity int8", identity),
        ("QuantizeLinear (round-half-even probe)", quantize),
        ("DequantizeLinear", dequantize),
        ("MatMulInteger", matmul),
        ("DynamicQuantizeLinear", dynamic),
    ] {
        println!("\n── {label}");
        println!(
            "   valid by our validator: {}",
            onnx_adapter::validation::is_valid(&case)
        );
        println!("   reference   {}", show(&reference.run(&case).unwrap()));
        println!("   onnxruntime {}", show(&OrtRuntime.run(&case).unwrap()));
        println!("   tract       {}", show(&TractRuntime.run(&case).unwrap()));
        #[cfg(feature = "candle")]
        println!(
            "   candle      {}",
            show(&onnx_adapter::runtimes::CandleRuntime.run(&case).unwrap())
        );
    }
    let _ = ElemType::QUANTIZED;
    std::panic::set_hook(previous);
}
