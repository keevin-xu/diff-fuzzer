//! What can the search actually score against today?
use tensor_adapter::negatives::{self, Pool, SamplingContext};

fn main() {
    let all = negatives::load(tensor_adapter::NEGATIVES_ROOT);
    println!("{} negatives on disk\n", all.len());

    for (label, ctx) in contexts() {
        match Pool::matched(all.clone(), &ctx) {
            Ok(pool) => println!("  {label:<28} {} negatives", pool.len()),
            Err(e) => println!("  {label:<28} REFUSED — {e}"),
        }
    }
}

/// The contexts a campaign might plausibly want to score against.
fn contexts() -> Vec<(String, SamplingContext)> {
    let pair = [tensor_adapter::FLEX_NAME, tensor_adapter::LIBTORCH_NAME];
    vec![
        (
            "fuzzer (current bounds)".to_string(),
            SamplingContext::new(negatives::FUZZER_GENERATOR, &pair),
        ),
        (
            "seeded wide".to_string(),
            SamplingContext::new(
                "Bounds { max_rank: 3, max_dim: 64, magnitude: 1000.0 }",
                &pair,
            ),
        ),
    ]
}
