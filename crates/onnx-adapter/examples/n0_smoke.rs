//! Build one model, run it on every participant, print the results side by side.
//!
//! Originally the PHASE-N0 feasibility demonstration; kept, and rewritten onto the
//! `Implementation` seam at N1 so it exercises the same path a campaign will.
//!
//! **It computes no verdict.** The oracle exists now, but this program deliberately does
//! not call it: what it demonstrates is that the plumbing carries the same computation to
//! four independent implementations, and printing an `Agree` here would make it look like
//! a passing oracle run rather than the plumbing check it is. `n1_oracle` is the example
//! that shows the oracle.
//!
//! Run with:
//!   cargo run -p onnx-adapter --example n0_smoke
//!   cargo run -p onnx-adapter --example n0_smoke --features candle

use diff_fuzzer_core::traits::Implementation;

use onnx_adapter::case::OpKind;
use onnx_adapter::environment;
use onnx_adapter::model::{DEFAULT_OPSET, IR_VERSION, build_bytes};
use onnx_adapter::outcome::OnnxOutcome;
use onnx_adapter::reference::ReferenceRuntime;
use onnx_adapter::runtimes::{OrtRuntime, TractRuntime, compiled_runtime_names};
use onnx_adapter::validation::well_formed;

fn main() {
    let dims = vec![2i64, 3];
    let case = well_formed(OpKind::Add, &dims, DEFAULT_OPSET);
    let bytes = build_bytes(&case);

    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "target/n0_add.onnx".to_string());
    std::fs::write(&out_path, &bytes).expect("writing the model should succeed");

    println!("One operator, four participants");
    println!("═══════════════════════════════════════════════════════════════");
    println!("model      {} f32 {dims:?}", case.op.onnx_name());
    println!("ir_version {IR_VERSION}   opset {}", case.opset);
    println!("bytes      {} (written to {out_path})", bytes.len());
    println!(
        "spec       onnx {} (max opset {})",
        environment::ONNX_PYTHON_VERSION,
        environment::MAX_OPSET
    );
    println!(
        "compiled   {}  + onnx.reference",
        compiled_runtime_names().join(", ")
    );
    if !cfg!(feature = "candle") {
        println!("           (candle-onnx is OFF — rebuild with --features candle)");
    }
    println!();

    println!("inputs");
    for input in &case.inputs {
        println!(
            "  {:<12} {:?}  {:?}",
            input.name,
            input.dims,
            input.as_f32().expect("f32 tensor")
        );
    }
    println!();

    // Every participant behind the same trait object, which is the point: the engine sees
    // no difference between a C++ library, two pure-Rust crates, and a Python subprocess.
    let mut participants: Vec<Box<dyn Implementation<In = _, Out = OnnxOutcome>>> =
        vec![Box::new(OrtRuntime), Box::new(TractRuntime)];
    #[cfg(feature = "candle")]
    participants.push(Box::new(onnx_adapter::runtimes::CandleRuntime));
    match ReferenceRuntime::start() {
        Ok(reference) => participants.push(Box::new(reference)),
        Err(why) => println!("  !! the reference is unavailable: {why}\n"),
    }

    println!("results");
    println!("  {:<16} outcome", "participant");
    println!("  {:-<16} {:-<52}", "", "");
    for participant in &participants {
        // `run` never returns `Err` in this domain — every failure is a value. The
        // `expect` documents that invariant rather than handling a case that cannot occur.
        let outcome = participant
            .run(&case)
            .expect("failures are values, never Err");
        println!("  {:<16} {outcome}", participant.name());
    }

    println!();
    println!("Values are shown as `value#bits` so +0.0 and -0.0 are distinguishable.");
    println!("No verdict is computed here — see the n1_oracle example for that.");
}
