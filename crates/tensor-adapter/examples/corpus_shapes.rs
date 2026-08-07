//! What did a campaign's corpus actually explore?
//!
//! The corpus holds the byte strings libFuzzer kept because they reached new coverage, so
//! decoding it says what the campaign *found interesting* — a stronger statement than what
//! the generator produces on average.
use arbitrary::{Arbitrary, Unstructured};
use tensor_adapter::TensorOp;
use tensor_adapter::ops::broadcast;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: corpus_shapes <corpus-dir>");
    let (mut binary, mut broadcasting, mut whole, mut both) = (0usize, 0usize, 0usize, 0usize);
    let mut ops = std::collections::BTreeMap::<&str, usize>::new();
    let mut total = 0usize;

    for entry in std::fs::read_dir(&dir).expect("corpus dir").flatten() {
        let Ok(bytes) = std::fs::read(entry.path()) else {
            continue;
        };
        let mut u = Unstructured::new(&bytes);
        let Ok(case) = TensorOp::arbitrary(&mut u) else {
            continue;
        };
        total += 1;
        *ops.entry(case.name()).or_default() += 1;

        if let TensorOp::Binary { lhs, rhs, .. } = &case {
            binary += 1;
            let result = broadcast::result_shape(lhs.shape(), rhs.shape()).expect("valid");
            let l = lhs.shape() != result.as_slice();
            let r = rhs.shape() != result.as_slice();
            if l || r {
                broadcasting += 1;
            }
            if (l && lhs.data().len() == 1) || (r && rhs.data().len() == 1) {
                whole += 1;
            }
            if l && r {
                both += 1;
            }
        }
    }

    println!("{total} corpus entries decoded\n");
    for (op, n) in &ops {
        println!("  {op:<8} {n}");
    }
    println!("\n  binary cases          {binary}");
    println!("  of which broadcasting {broadcasting}");
    println!("    whole operand       {whole}");
    println!("    both operands       {both}");
}
