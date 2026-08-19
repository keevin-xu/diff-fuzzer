//! Load a `.onnx` produced by a report's own script and run it through the accused runtime.
//!
//! # Why this is not redundant with `verify_findings`
//!
//! `verify_findings` proves the divergence for models **this project builds**. A report tells a
//! maintainer to build one with a short Python script, and those two artifacts differ in tensor
//! names even when they agree on every semantic field. For a finding that is a *load* failure,
//! assuming a naming difference is harmless is exactly the kind of inference that should be
//! measured instead.
//!
//! So this takes the file the report actually produces and runs it.

use tract_onnx::prelude::*;

fn f32_tensor(shape: &[usize], v: &[f32]) -> Tensor {
    Tensor::from_shape(shape, v).expect("f32 tensor")
}

fn run(path: &str, inputs: Vec<Tensor>) -> String {
    let loaded = tract_onnx::onnx()
        .model_for_path(path)
        .and_then(|m| m.into_typed())
        .and_then(|m| m.into_runnable());
    let plan = match loaded {
        Ok(p) => p,
        Err(e) => {
            return format!(
                "REJECTED AT LOAD: {}",
                e.to_string().lines().next().unwrap_or("")
            );
        }
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        plan.run(inputs.into_iter().map(|t| t.into()).collect::<TVec<_>>())
    }));
    match outcome {
        Err(_) => "CRASHED (panic)".to_string(),
        Ok(Err(e)) => format!(
            "REJECTED AT RUN: {}",
            e.to_string().lines().next().unwrap_or("")
        ),
        Ok(Ok(out)) => format!("{:?}", out[0]),
    }
}

fn main() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let d = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/repro".to_string());

    println!("tract on the models the reports' own scripts produce\n");
    println!(
        "  sign     {}",
        run(&format!("{d}/sign.onnx"), vec![f32_tensor(&[1], &[-0.0])])
    );
    println!(
        "  reshape  {}",
        run(&format!("{d}/reshape.onnx"), vec![f32_tensor(&[3, 0], &[])])
    );
    println!(
        "  dql      {}",
        run(
            &format!("{d}/dql.onnx"),
            vec![f32_tensor(&[3], &[-127.0, 128.0, 0.5])]
        )
    );
    println!(
        "  div      {}",
        run(
            &format!("{d}/div.onnx"),
            vec![
                Tensor::from_shape(&[1], &[i32::MIN]).unwrap(),
                Tensor::from_shape(&[1], &[-1i32]).unwrap()
            ]
        )
    );
    // ONNX's own `zero_and_negative_dim` shape: a 0 and a -1 in one target, allowzero unset.
    println!(
        "  zeroneg  {}",
        run(
            &format!("{d}/zeroneg.onnx"),
            vec![f32_tensor(
                &[2, 3, 1, 4],
                &(0..24).map(|i| i as f32).collect::<Vec<_>>()
            )]
        )
    );
    println!(
        "  where    {}",
        run(
            &format!("{d}/where.onnx"),
            vec![
                Tensor::from_shape(&[1], &[true]).unwrap(),
                f32_tensor(&[1], &[-0.0]),
                f32_tensor(&[1], &[0.0])
            ]
        )
    );

    // candle is the accused in candle-001, on the same Reshape file tract-002 uses.
    #[cfg(feature = "candle")]
    {
        use prost::Message;
        let bytes = std::fs::read(format!("{d}/reshape.onnx")).expect("read reshape.onnx");
        let model = candle_onnx::onnx::ModelProto::decode(bytes.as_slice()).expect("decode");
        let empty = candle_core::Tensor::from_vec(
            Vec::<f32>::new(),
            (3usize, 0usize),
            &candle_core::Device::Cpu,
        )
        .expect("empty tensor");
        let mut feeds = std::collections::HashMap::new();
        feeds.insert("a".to_string(), empty);
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candle_onnx::simple_eval(&model, feeds)
        }));
        let line = match got {
            Err(_) => "CRASHED (panic)".to_string(),
            Ok(Err(e)) => format!("REJECTED: {}", e.to_string().lines().next().unwrap_or("")),
            Ok(Ok(out)) => format!("{:?}", out.values().next()),
        };
        println!("\ncandle on the same Reshape file\n  reshape  {line}");

        // The spec's own `zero_and_negative_dim` shape, with allowzero UNSET.
        let bytes = std::fs::read(format!("{d}/zeroneg.onnx")).expect("read zeroneg.onnx");
        let model = candle_onnx::onnx::ModelProto::decode(bytes.as_slice()).expect("decode");
        let x = candle_core::Tensor::from_vec(
            (0..24).map(|i| i as f32).collect::<Vec<_>>(),
            (2usize, 3usize, 1usize, 4usize),
            &candle_core::Device::Cpu,
        )
        .expect("tensor");
        let mut feeds = std::collections::HashMap::new();
        feeds.insert("a".to_string(), x);
        let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candle_onnx::simple_eval(&model, feeds)
        }));
        let line = match got {
            Err(_) => "CRASHED (panic)".to_string(),
            Ok(Err(e)) => format!("REJECTED: {}", e.to_string().lines().next().unwrap_or("")),
            Ok(Ok(out)) => format!("{:?}", out.values().next().map(|t| t.dims().to_vec())),
        };
        println!(
            "\ncandle on target [2,0,1,-1], allowzero unset (expect [2,3,1,4])\n  zeroneg  {line}"
        );
    }

    std::panic::set_hook(previous);
}
