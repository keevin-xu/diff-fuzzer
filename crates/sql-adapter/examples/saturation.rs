//! Is the generator still producing new cases, or re-covering ground?
//!
//! A campaign's zero means very different things depending on the answer. If 400,000 seeds
//! produce 400,000 distinct cases, the space is far from exhausted and more hours genuinely
//! buy more coverage. If they produce 40,000, then nine of every ten executions re-ran
//! something already tested, and the next move is a *wider* generator or a *different*
//! oracle — not a longer run.
//!
//! Three things are counted, because they saturate at different rates and mean different
//! things:
//!
//! - **Distinct cases** — the whole `SqlCase`, data included. Saturating here means literally
//!   re-running identical work.
//! - **Distinct query shapes** — the clause set, ignoring data and literals. This is the
//!   [`signature`](crate::signature) key's first half, so it is what de-duplication would
//!   collapse findings by. Saturating here means new cases can still differ in data but not
//!   in structure.
//! - **Distinct feature vectors** — the S7 vocabulary. Saturating here means the *predicate
//!   search* can learn nothing more from additional cases, whatever else varies.
//!
//! Run with:
//!   cargo run --release -p sql-adapter --example saturation -- [cases]

use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::Generator;
use sql_adapter::features::extract;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;
use sql_adapter::signature::clause_shape;
use std::collections::HashSet;

fn main() {
    let total: usize = std::env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(200_000);

    let generator = SqlGenerator::new(Bounds::V1_ALL);
    println!("generator: {}", generator.description());
    println!("cases:     {total}\n");

    let mut cases: HashSet<String> = HashSet::new();
    let mut shapes: HashSet<String> = HashSet::new();
    let mut vectors: HashSet<u32> = HashSet::new();

    // Checkpoints, because the *shape of the curve* is the answer. A count at the end says
    // how much was covered; the curve says whether it had stopped growing.
    let checkpoints: Vec<usize> = [1_000, 5_000, 20_000, 50_000, 100_000, total]
        .into_iter()
        .filter(|point| *point <= total)
        .collect();

    println!(
        "  {:>9} {:>12} {:>10} {:>12} {:>10} {:>12}",
        "seeds", "cases", "new/1k", "shapes", "vectors", "case ratio"
    );

    let mut previous_cases = 0usize;
    let mut previous_seeds = 0usize;

    for seed in 0..total as u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));

        cases.insert(serde_json::to_string(&case).expect("serializes"));
        shapes.insert(clause_shape(&case).join("+"));
        vectors.insert(extract(&case).0);

        let seen = seed as usize + 1;
        if checkpoints.contains(&seen) {
            // How many *new* distinct cases the last thousand seeds bought — the number that
            // goes to zero when a generator has nothing left to say.
            let fresh = cases.len() - previous_cases;
            let spent = seen - previous_seeds;
            let per_thousand = 1000.0 * fresh as f64 / spent as f64;

            println!(
                "  {:>9} {:>12} {:>10.0} {:>12} {:>10} {:>11.1}%",
                seen,
                cases.len(),
                per_thousand,
                shapes.len(),
                vectors.len(),
                100.0 * cases.len() as f64 / seen as f64
            );

            previous_cases = cases.len();
            previous_seeds = seen;
        }
    }

    println!("\nreading");
    let ratio = cases.len() as f64 / total as f64;
    if ratio > 0.98 {
        println!(
            "  Cases are still essentially all distinct ({:.1}%). The space is nowhere near\n  \
             exhausted, so longer runs do buy more coverage.",
            100.0 * ratio
        );
    } else {
        println!(
            "  {:.1}% of seeds produced a distinct case — {} executions re-ran something\n  \
             already tested. Longer runs are buying repetition, not coverage.",
            100.0 * ratio,
            total - cases.len()
        );
    }
    println!(
        "  Query shapes: {} distinct. Feature vectors: {} distinct of {} possible.",
        shapes.len(),
        vectors.len(),
        1u64 << sql_adapter::features::FEATURES.len()
    );
    println!(
        "  Feature vectors are what the predicate search sees; once that column stops\n  \
         growing, more cases teach it nothing regardless of how the data varies."
    );
}
