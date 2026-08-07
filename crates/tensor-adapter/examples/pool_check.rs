//! What can the search actually score against today?
//!
//! **Reports the pools that exist, rather than asking about ones it guessed.** This used to
//! construct a `SamplingContext` from a hardcoded backend pair and a compiled-in generator
//! string, then report "no usable negatives" whenever either had moved on — a message that
//! reads as missing data and means a stale constant.
//!
//! Negatives record how they were produced, so grouping by that record answers the real
//! question: which pools are on disk, and how large is each.
use std::collections::BTreeMap;
use tensor_adapter::negatives::{self, Pool, SamplingContext};

fn main() {
    let all = negatives::load(tensor_adapter::NEGATIVES_ROOT);
    println!("{} negatives on disk\n", all.len());

    // Grouped by the configuration that produced them — which is exactly what
    // `Pool::matched` compares, so a group here is a pool that could be scored against.
    let mut pools: BTreeMap<(String, Vec<String>), usize> = BTreeMap::new();
    for negative in &all {
        *pools
            .entry((negative.generator.clone(), negative.backends.clone()))
            .or_default() += 1;
    }

    if pools.is_empty() {
        println!("nothing to score against.");
        return;
    }

    for ((generator, backends), count) in &pools {
        let context = SamplingContext::new(generator.clone(), &[]);
        let usable = Pool::matched(
            all.clone(),
            &SamplingContext {
                generator: generator.clone(),
                backends: backends.clone(),
            },
        );
        let _ = context;
        println!("  {count:>5} negatives");
        println!("        backends:  {backends:?}");
        println!("        generator: {generator}");
        match usable {
            Ok(pool) => println!("        usable:    {} \n", pool.len()),
            Err(error) => println!("        REFUSED:   {error}\n"),
        }
    }
}
