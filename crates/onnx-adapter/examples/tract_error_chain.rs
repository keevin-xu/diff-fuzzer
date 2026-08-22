//! Print tract's *full* error chain for a model on disk, stage by stage.
//!
//! The campaign records only the first line of a rejection, which is enough to compare and
//! useless for finding the code that produced it. This walks the same pipeline the adapter
//! uses and says which stage fails, with `anyhow`'s causes.

use tract_onnx::prelude::*;

fn chain(e: &TractError) {
    // `TractError` is an `anyhow::Error`, whose `chain()` walks the causes it accumulated as the
    // error propagated. The adapter stores only the first line, which is why the code that
    // produced it was never findable from a campaign record.
    for (depth, cause) in e.chain().enumerate() {
        println!("    {depth}: {cause}");
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: tract_error_chain <model.onnx>");

    let inference = match tract_onnx::onnx().model_for_path(&path) {
        Ok(m) => {
            println!("1. model_for_path  OK");
            m
        }
        Err(e) => {
            println!("1. model_for_path  FAILED");
            chain(&e);
            return;
        }
    };

    let typed = match inference.into_typed() {
        Ok(m) => {
            println!("2. into_typed      OK");
            m
        }
        Err(e) => {
            println!("2. into_typed      FAILED");
            chain(&e);
            return;
        }
    };

    match typed.into_runnable() {
        Ok(_) => println!("3. into_runnable   OK"),
        Err(e) => {
            println!("3. into_runnable   FAILED");
            chain(&e);
        }
    }
}
