//! What can the search actually score against today?
use tensor_adapter::negatives::{self, Pool, Provenance};

fn main() {
    let all = negatives::load("findings/negatives");
    println!("{} negatives on disk\n", all.len());

    for p in [
        Provenance::Fuzzer,
        Provenance::SeededWide,
        Provenance::Constructed,
        Provenance::Unknown,
    ] {
        let n = all.iter().filter(|x| x.provenance == p).count();
        println!("  {:<16} {n}", p.label());
    }

    println!("\nusable when scoring findings from each generator:");
    for p in [Provenance::Fuzzer, Provenance::SeededWide] {
        match Pool::matched(all.clone(), p) {
            Ok(pool) => println!("  {:<16} {} negatives", p.label(), pool.len()),
            Err(e) => println!("  {:<16} REFUSED — {e}", p.label()),
        }
    }
}
