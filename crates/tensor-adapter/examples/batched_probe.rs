//! **Step 7.0.1** — what actually distinguishes a diverging `matmul` overflow from a
//! non-diverging one?
//!
//! # Why this exists
//!
//! The seeded campaign found 68 rank-3 divergences, and the obvious reading — "batched
//! matmul has the same bug" — has already failed once. A constructed batched case where
//! every product overflows returns `inf` on **both** backends.
//!
//! A later hypothesis said the probe only produced *same-sign* overflows, so `inf + inf`
//! stayed `inf` rather than becoming `NaN`. **That was also wrong**: the construction below
//! alternates signs, so each dot product already contains `+1e60` and `−1e60`. Verified
//! independently, the mixed-sign property holds for **41/41** recorded findings — and for
//! these non-diverging probe cases too. It is therefore *necessary but not sufficient*, and
//! cannot be the discriminator.
//!
//! # THE ANSWER (measured 2026-08-04, step 7.0.1)
//!
//! **It is a tile-remainder effect in libtorch's GEMM, and it is self-inconsistency —
//! not a cross-backend disagreement.**
//!
//! Experiment C pins it exactly. The number of disagreeing output elements is
//!
//! ```text
//!     (m mod 4) * (n mod 8)
//! ```
//!
//! which predicts **every case measured**, with no exceptions:
//!
//! | m, n | predicted | observed |
//! |---|---|---|
//! | 16, 32 | 0 | agree |
//! | 17, 32 | 0 | agree |
//! | 16, 33 | 0 | agree |
//! | 17, 33 | **1** | **1** |
//! | 14, 27 | **6** | **6** — at rows 12–13, cols 24–26 |
//! | 1, 1 (burn#5284) | **1** | **1** |
//! | 4, 4 | 0 | agree |
//!
//! So libtorch's matmul uses a **4x8 micro-kernel** that fuses its multiply-add, plus a
//! cleanup path for the trailing corner that does **not**. Divergence appears exactly where
//! *both* dimensions leave a remainder — the bottom-right block that no full tile covers.
//! `matrixmultiply` (via flex) fuses everywhere, so it returns `inf` throughout.
//!
//! **The sharpest consequence: libtorch disagrees with itself.** At `m=14, n=27` it returns
//! `inf` for 372 elements and `NaN` for 6, from arithmetic that is structurally identical —
//! the only difference is where the element sits relative to a tile boundary. That is
//! observable without any second backend, which makes it a much stronger report than
//! "two libraries differ".
//!
//! burn#5284's minimal `[1,2] x [2,1]` is simply the degenerate case: a 1x1 output is
//! *entirely* corner remainder, so it always takes the non-fusing path.
//!
//! Two hypotheses died on the way here, both recorded in `DECISIONS.md`: that the probe
//! generated only same-sign overflows (it never did), and that `rank>=3 AND m=n=1` was the
//! non-diverging cell (rank-2 `m=n=4` also agrees). Rank was never the discriminator.
//!
//! # The experiments
//!
//! **A — isolate rank.** Identical values, `k`, `m`, `n`; only the rank wrapper varies.
//!
//! **B — isolate output shape.** Rank held at 3; `m` and `n` move away from 1.
//!
//! **C — test the tile model.** Sizes chosen on and off multiples of 4 and 8.
//!
//! Every non-diverging case is written to `findings/negatives/` as data. Those are the only
//! known near-misses — cases satisfying a plausible trigger that do *not* diverge — and any
//! future candidate predicate has to survive them.
//!
//! Run with:
//! ```text
//! cargo run --release -p tensor-adapter --example batched_probe
//! ```

use burn::backend::{Flex, LibTorch};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use tensor_adapter::negatives::{self, Source};
use tensor_adapter::{TensorOp, TensorValue};

/// Left operand values: sign alternates along the contraction axis, so every dot product
/// contains both a positively- and a negatively-overflowing product.
///
/// `lhs[i][t] = ±1e30` by parity of `t`; paired with an all-positive `rhs`, the products
/// are `+1e60` and `−1e60`. Their exact sum is zero; in `f32` each overflows.
fn lhs_values(rows: usize, k: usize) -> Vec<f32> {
    (0..rows * k)
        .map(|idx| {
            if (idx % k).is_multiple_of(2) {
                1e30
            } else {
                -1e30
            }
        })
        .collect()
}

fn rhs_values(k: usize, cols: usize) -> Vec<f32> {
    vec![1e30; k * cols]
}

/// One measurement: both backends' results, and whether they disagree.
struct Outcome {
    flex: Vec<f32>,
    libtorch: Vec<f32>,
}

impl Outcome {
    /// Disagreement counted exactly — `NaN` is distinct from any number, and two numbers
    /// differ if they are not bit-equal. No tolerance: this experiment is about `inf`
    /// versus `NaN`, which no tolerance may absorb.
    fn differing(&self) -> usize {
        self.flex
            .iter()
            .zip(&self.libtorch)
            .filter(|(a, b)| a.is_nan() != b.is_nan() || (!a.is_nan() && a != b))
            .count()
    }

    fn summary(values: &[f32]) -> String {
        let infs = values.iter().filter(|v| v.is_infinite()).count();
        let nans = values.iter().filter(|v| v.is_nan()).count();
        let finite = values.len() - infs - nans;
        format!("{infs} inf, {nans} NaN, {finite} finite")
    }
}

/// Run a rank-2 matmul on both backends.
fn run_rank2(m: usize, k: usize, n: usize) -> Outcome {
    fn go<B: Backend>(m: usize, k: usize, n: usize) -> Vec<f32> {
        let device = Default::default();
        let a = Tensor::<B, 2>::from_data(TensorData::new(lhs_values(m, k), [m, k]), &device);
        let b = Tensor::<B, 2>::from_data(TensorData::new(rhs_values(k, n), [k, n]), &device);
        a.matmul(b).into_data().to_vec::<f32>().expect("f32 read")
    }
    Outcome {
        flex: go::<Flex<f32>>(m, k, n),
        libtorch: go::<LibTorch<f32>>(m, k, n),
    }
}

/// Run a rank-3 (batched) matmul on both backends.
fn run_rank3(batch: usize, m: usize, k: usize, n: usize) -> Outcome {
    fn go<B: Backend>(batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let device = Default::default();
        let lhs: Vec<f32> = (0..batch).flat_map(|_| lhs_values(m, k)).collect();
        let rhs: Vec<f32> = (0..batch).flat_map(|_| rhs_values(k, n)).collect();
        let a = Tensor::<B, 3>::from_data(TensorData::new(lhs, [batch, m, k]), &device);
        let b = Tensor::<B, 3>::from_data(TensorData::new(rhs, [batch, k, n]), &device);
        a.matmul(b).into_data().to_vec::<f32>().expect("f32 read")
    }
    Outcome {
        flex: go::<Flex<f32>>(batch, m, k, n),
        libtorch: go::<LibTorch<f32>>(batch, m, k, n),
    }
}

/// Run a rank-4 (doubly batched) matmul on both backends.
fn run_rank4(b0: usize, b1: usize, m: usize, k: usize, n: usize) -> Outcome {
    fn go<B: Backend>(b0: usize, b1: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let device = Default::default();
        let lhs: Vec<f32> = (0..b0 * b1).flat_map(|_| lhs_values(m, k)).collect();
        let rhs: Vec<f32> = (0..b0 * b1).flat_map(|_| rhs_values(k, n)).collect();
        let a = Tensor::<B, 4>::from_data(TensorData::new(lhs, [b0, b1, m, k]), &device);
        let b = Tensor::<B, 4>::from_data(TensorData::new(rhs, [b0, b1, k, n]), &device);
        a.matmul(b).into_data().to_vec::<f32>().expect("f32 read")
    }
    Outcome {
        flex: go::<Flex<f32>>(b0, b1, m, k, n),
        libtorch: go::<LibTorch<f32>>(b0, b1, m, k, n),
    }
}

/// The same case expressed as a `TensorOp`, so a non-diverging one can be recorded as data
/// that future feature extraction can read.
fn as_case(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> TensorOp {
    let batch: usize = lhs_shape[..lhs_shape.len() - 2].iter().product();
    let (m, k) = (
        lhs_shape[lhs_shape.len() - 2],
        lhs_shape[lhs_shape.len() - 1],
    );
    let n = rhs_shape[rhs_shape.len() - 1];

    let lhs: Vec<f32> = (0..batch).flat_map(|_| lhs_values(m, k)).collect();
    let rhs: Vec<f32> = (0..batch).flat_map(|_| rhs_values(k, n)).collect();

    TensorOp::matmul(
        TensorValue::new(lhs_shape, lhs),
        TensorValue::new(rhs_shape, rhs),
    )
}

fn row(label: &str, outcome: &Outcome, cases: &mut Vec<(String, TensorOp)>, case: TensorOp) {
    let differing = outcome.differing();
    println!(
        "  {label:<28} nd: {:<26} tch: {:<26} {}",
        Outcome::summary(&outcome.flex),
        Outcome::summary(&outcome.libtorch),
        if differing > 0 {
            format!("◄ DIVERGES ({differing})")
        } else {
            "agree".to_string()
        }
    );
    if differing == 0 {
        cases.push((label.to_string(), case));
    }
}

fn main() {
    // Only non-diverging cases are collected: they are the near-misses a candidate
    // predicate must survive. A diverging case is already a finding.
    let mut negatives: Vec<(String, TensorOp)> = Vec::new();

    println!("\nEXPERIMENT A — isolate rank");
    println!("  identical values, k, m, n. Only the rank wrapper varies.\n");

    for k in [2usize, 4] {
        let a2 = run_rank2(1, k, 1);
        row(
            &format!("rank 2  [1,{k}]x[{k},1]"),
            &a2,
            &mut negatives,
            as_case(vec![1, k], vec![k, 1]),
        );

        let a3 = run_rank3(1, 1, k, 1);
        row(
            &format!("rank 3  [1,1,{k}]x[1,{k},1]"),
            &a3,
            &mut negatives,
            as_case(vec![1, 1, k], vec![1, k, 1]),
        );

        let a4 = run_rank4(1, 1, 1, k, 1);
        row(
            &format!("rank 4  [1,1,1,{k}]x[1,1,{k},1]"),
            &a4,
            &mut negatives,
            as_case(vec![1, 1, 1, k], vec![1, 1, k, 1]),
        );
        println!();
    }

    println!("EXPERIMENT B — isolate output shape");
    println!("  rank held at 3, batch 2, k = 4. Only m and n vary.\n");

    for (m, n) in [(1usize, 1usize), (1, 4), (4, 1), (4, 4), (14, 27)] {
        let outcome = run_rank3(2, m, 4, n);
        row(
            &format!("rank 3  m={m}, n={n}"),
            &outcome,
            &mut negatives,
            as_case(vec![2, m, 4], vec![2, 4, n]),
        );
    }

    println!("\nCONTROL — the rank-2 shape at larger m, n");
    for (m, n) in [(4usize, 4usize), (14, 27)] {
        let outcome = run_rank2(m, 4, n);
        row(
            &format!("rank 2  m={m}, n={n}"),
            &outcome,
            &mut negatives,
            as_case(vec![m, 4], vec![4, n]),
        );
    }

    // If the boundary is a tile remainder, shapes that divide evenly should agree entirely
    // and shapes one past a multiple should disagree in exactly one trailing strip.
    println!("\nEXPERIMENT C — is the boundary a tile remainder?");
    println!("  rank 2, k = 4. Sizes chosen around plausible micro-kernel multiples.\n");
    for (m, n) in [
        (16usize, 32usize),
        (17, 32),
        (16, 33),
        (17, 33),
        (8, 16),
        (12, 24),
    ] {
        let outcome = run_rank2(m, 4, n);
        let differing = outcome.differing();
        // Predicted remainder, if the micro-kernel tile is 4 rows x 8 columns.
        let predicted = (m % 4) * (n % 8);
        println!(
            "  m={m:<3} n={n:<3}  {:<28} {:<20}  predicted {predicted}",
            Outcome::summary(&outcome.libtorch),
            if differing > 0 {
                format!("◄ {differing} differ")
            } else {
                "agree".to_string()
            }
        );
    }

    // Only a fraction of elements disagree in the large cases, which is the strongest clue
    // available: a whole-kernel difference would affect every element. Where they sit says
    // whether the boundary is a tile edge.
    println!("\nWHICH elements disagree at m=14, k=4, n=27 (rank 2)?");
    let outcome = run_rank2(14, 4, 27);
    let differing: Vec<(usize, usize)> = outcome
        .flex
        .iter()
        .zip(&outcome.libtorch)
        .enumerate()
        .filter(|(_, (a, b))| a.is_nan() != b.is_nan())
        .map(|(index, _)| (index / 27, index % 27))
        .collect();
    println!(
        "  output is 14 rows x 27 cols; {} disagree",
        differing.len()
    );
    println!("  (row, col): {differing:?}");

    write_negatives(&negatives);
}

/// Record the non-diverging cases where a future search can score against them.
fn write_negatives(cases: &[(String, TensorOp)]) {
    let dir = "findings/negatives";
    if let Err(error) = std::fs::create_dir_all(dir) {
        eprintln!("\ncould not create {dir}: {error}");
        return;
    }

    let only_cases: Vec<TensorOp> = cases.iter().map(|(_, case)| case.clone()).collect();
    let path = format!("{dir}/batched_probe.json");
    // Recorded as `Constructed`: built by hand to probe a specific hypothesis, which makes
    // them stronger evidence than anything sampled and worth distinguishing in a report.
    match negatives::save_batch(
        &path,
        &only_cases,
        Source::Constructed,
        negatives::Provenance::Constructed,
    ) {
        Ok(()) => println!("\n{} non-diverging case(s) written to {path}", cases.len()),
        Err(error) => eprintln!("\ncould not write {path}: {error}"),
    }
}
