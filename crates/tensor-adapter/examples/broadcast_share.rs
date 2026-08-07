//! What share of generated binary cases actually broadcast?
use diff_fuzzer_core::SeededRng;
use tensor_adapter::TensorOp;
use tensor_adapter::ops::{Bounds, binary, broadcast, element_count};

fn main() {
    let bounds = Bounds::default();
    let (mut equal, mut stretched, mut total_in, mut total_out) = (0usize, 0usize, 0usize, 0usize);
    let n = 20_000u64;

    for seed in 0..n {
        let mut rng = SeededRng::from_seed(seed);
        if let TensorOp::Binary { lhs, rhs, .. } = binary::generate(&mut rng, &bounds) {
            if lhs.shape() == rhs.shape() {
                equal += 1;
            } else {
                stretched += 1;
            }
            total_in += element_count(lhs.shape()) + element_count(rhs.shape());
            total_out += broadcast::result_count(lhs.shape(), rhs.shape());
        }
    }

    println!("{n} binary cases");
    println!(
        "  equal shapes   {equal:>6}  ({:.0}%)",
        100.0 * equal as f64 / n as f64
    );
    println!(
        "  broadcasting   {stretched:>6}  ({:.0}%)",
        100.0 * stretched as f64 / n as f64
    );
    println!(
        "\n  elements read  {total_in:>9}\n  elements written {total_out:>7}   (ratio {:.2}x)",
        total_out as f64 / total_in as f64
    );
}
