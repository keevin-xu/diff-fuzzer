//! Write the seed corpus to a directory the fuzzer can start from.
//!
//! Run with:
//! ```text
//! cargo run -p tensor-adapter --features fuzzing --example write_seed_corpus
//! ```
//!
//! Safe to re-run: each seed is named by a hash of its own bytes, so writing again
//! overwrites the same files rather than accumulating near-duplicates. Anything libFuzzer
//! has since discovered and added to the directory is left alone.

use tensor_adapter::seeds::seed_corpus;

const DEFAULT_DIRECTORY: &str = "fuzz/corpus/tensor_diff";

fn main() {
    let directory = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_DIRECTORY.to_string());

    std::fs::create_dir_all(&directory).expect("corpus directory is writable");

    let corpus = seed_corpus();
    let mut written = 0;

    for seed in &corpus {
        // Named by content, so re-running is idempotent.
        let name = format!("seed-{:016x}", digest(seed));
        let path = format!("{directory}/{name}");

        std::fs::write(&path, seed).expect("seed is writable");
        written += 1;
    }

    println!("wrote {written} seeds to {directory}/");
    println!(
        "  covering every operation, ranks 1..={}, degenerate and larger shapes,",
        tensor_adapter::MAX_RANK
    );
    println!("  reductions along each axis, and batched matrix multiplication.");
    println!();
    println!("  The fuzzer will mutate outward from these rather than spending its early");
    println!("  budget rediscovering that zero, one and rank-4 tensors exist.");
}

fn digest(bytes: &[u8]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
