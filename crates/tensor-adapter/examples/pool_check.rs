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
    // **Both the two- and three-backend sets**, because a pool is scoped to the
    // implementations it was observed on and asking about only one of them reports "no
    // usable negatives" for a pool that is perfectly usable by the other.
    let pair = [
        tensor_adapter::FLEX_NAME,
        tensor_adapter::LIBTORCH_NAME,
        tensor_adapter::WGPU_NAME,
    ];
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
