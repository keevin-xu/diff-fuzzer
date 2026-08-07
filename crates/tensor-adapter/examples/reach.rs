//! Which features does the default generator actually produce?
use diff_fuzzer_core::{Generator, SeededRng};
use tensor_adapter::features::{FEATURES, extract};
use tensor_adapter::generator::TensorOpGenerator;
use tensor_adapter::ops::Bounds;

fn main() {
    let bounds = match std::env::args().nth(1).as_deref() {
        // The fuzzer's actual configuration: wide *and* unrestricted. Leaving
        // `restrict_domains` at its default measured a setting nothing runs.
        Some("wide") => Bounds {
            max_rank: 3,
            max_dim: 64,
            magnitude: 1000.0,
            special_value_rate: 0.125,
            restrict_domains: false,
            ..Bounds::default()
        },
        _ => Bounds::default(),
    };
    let generator = TensorOpGenerator::new(bounds);
    let mut rng = SeededRng::from_seed(7);
    let mut counts = vec![0usize; FEATURES.len()];
    let n = 20_000;
    for _ in 0..n {
        let features = extract(&generator.generate(&mut rng));
        for (bit, count) in counts.iter_mut().enumerate() {
            if features.0 & (1 << bit) != 0 {
                *count += 1;
            }
        }
    }
    let mut rows: Vec<_> = FEATURES.iter().zip(&counts).collect();
    rows.sort_by_key(|(_, c)| **c);
    for (name, count) in rows {
        println!("{count:>6} / {n}  {name}");
    }
}
