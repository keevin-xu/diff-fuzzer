//! How large a result can one generated case ask an engine to build?
//!
//! A join over two large tables is **quadratic**: `t0 JOIN t1 ON (t0.c0 <> t1.c0)` over two
//! 2,000-row tables asks for up to 4,000,000 rows from a case that looks small on paper. That
//! cost is invisible in the corpus statistics — which count *cases*, not work — and it is
//! invisible in a small throughput sample, because the largest tables are rare.
use diff_fuzzer_core::SeededRng;
use diff_fuzzer_core::traits::Generator;
use sql_adapter::gen_schema::Bounds;
use sql_adapter::generator::SqlGenerator;

fn main() {
    let generator = SqlGenerator::new(Bounds::V1_ALL_LARGE);
    let (mut worst, mut over_1m, mut over_100k, mut total) = (0usize, 0usize, 0usize, 0u128);

    for seed in 0..20_000u64 {
        let case = generator.generate(&mut SeededRng::from_seed(seed));
        let rows = |name: &str| {
            case.data
                .iter()
                .find(|insert| insert.table == name)
                .map_or(0, |insert| insert.rows.len())
        };
        // The cross product the engine may have to materialise before filtering.
        let primary = rows(case.query.primary());
        let product = match &case.query.join {
            Some(join) => primary.saturating_mul(rows(&join.table)),
            None => primary,
        };
        worst = worst.max(product);
        over_1m += usize::from(product > 1_000_000);
        over_100k += usize::from(product > 100_000);
        total += product as u128;
    }

    println!("over 20,000 cases on V1_ALL_LARGE");
    println!("  worst-case join product : {worst}");
    println!("  cases over 1,000,000    : {over_1m}");
    println!("  cases over 100,000      : {over_100k}");
    println!("  mean product            : {}", total / 20_000);
}
