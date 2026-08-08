//! Running results along an axis — one output per element.
//!
//! # Why this operation is worth its own module
//!
//! Every other operation tested has all three backends performing the **same additions in the
//! same order**, which is why 3.9 million cases of ordinary arithmetic produced no numeric
//! disagreement. A scan is different by construction: a sequential implementation keeps a
//! running total, while a parallel one uses a prefix-scan algorithm — Hillis–Steele or
//! Blelloch — that **associates the additions differently**.
//!
//! Floating-point addition is not associative, so two correct scans can return different last
//! bits. That makes this the strongest available candidate for a *numeric* divergence, as
//! opposed to the structural ones every finding so far has been.
//!
//! # What the generator aims at
//!
//! **Long axes**, because the accumulated difference grows with the number of terms and a
//! two-element scan cannot express an association difference at all.
//!
//! **Values that cancel**, because catastrophic cancellation is what turns a last-bit
//! difference in association into a visible one: summing `+1e30, -1e30, 1.0` in different
//! orders gives `1.0` or `0.0`.

use crate::input::{ScanOp, TensorOp, TensorValue};
use crate::ops::{Bounds, Domain, element_count, shape, values};
use diff_fuzzer_core::SeededRng;
use rand::RngExt;

/// Every scan the generator may pick.
pub const ALL: [ScanOp; 2] = [ScanOp::CumSum, ScanOp::CumProd];

/// How often the values are built to cancel against each other.
///
/// Cancellation is what makes an association difference observable rather than last-bit, so
/// it is over-sampled — but kept a minority, since a generator producing only pathological
/// values stops testing the ordinary path.
const CANCELLING_SHARE: f64 = 0.3;

/// Build a valid scan case.
pub fn generate(rng: &mut SeededRng, bounds: &Bounds) -> TensorOp {
    let kind = ALL[rng.random_range(0..ALL.len())];
    let shape = shape(rng, bounds);
    // Prefer the longest axis: the bound and the accumulated difference both scale with it,
    // and a short axis cannot express an association difference at all.
    let dim = longest_axis(&shape);

    let count = element_count(&shape);
    let data = if rng.random_bool(CANCELLING_SHARE) {
        cancelling_values(rng, count, bounds)
    } else {
        values(rng, count, Domain::Any, bounds)
    };

    TensorOp::scan(kind, TensorValue::new(shape, data), dim)
}

/// The axis with the most elements, first one wins on a tie.
fn longest_axis(shape: &[usize]) -> usize {
    let mut best = 0;
    for (index, extent) in shape.iter().enumerate() {
        if *extent > shape[best] {
            best = index;
        }
    }
    let _ = shape;
    best
}

/// Values built to cancel: large magnitudes of alternating sign, with small ones between.
///
/// A running sum over these depends sharply on association order — `(a + -a) + b` is `b`
/// while `a + (-a + b)` may be `a - a` if `b` is lost to rounding first.
fn cancelling_values(rng: &mut SeededRng, count: usize, bounds: &Bounds) -> Vec<f32> {
    let large = bounds.magnitude * 1e6;
    (0..count)
        .map(|index| match index % 3 {
            0 => large,
            1 => -large,
            _ => rng.random_range(-1.0..1.0),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::tests::for_many_seeds;

    #[test]
    fn the_generator_never_emits_an_out_of_range_dim() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let case = generate(rng, &bounds);
            let TensorOp::Scan { arg, dim, .. } = &case else {
                panic!("scan generator produced {case:?}");
            };
            assert!(*dim < arg.rank());
        });
    }

    /// **The scanned axis must be the long one**, or the operation cannot express the
    /// association difference it exists to look for.
    #[test]
    fn the_scanned_axis_is_the_longest_one() {
        let bounds = Bounds::default();
        for_many_seeds(|rng| {
            let TensorOp::Scan { arg, dim, .. } = generate(rng, &bounds) else {
                unreachable!()
            };
            let longest = *arg.shape().iter().max().expect("non-empty shape");
            assert_eq!(
                arg.shape()[dim],
                longest,
                "scanned {:?} along axis {dim}, which is not its longest",
                arg.shape()
            );
        });
    }

    /// Cancelling cases must actually occur — they are what make an ordering difference
    /// visible rather than last-bit.
    #[test]
    fn cancelling_values_are_produced() {
        let bounds = Bounds::default();
        let mut cancelling = 0;
        for seed in 0..2_000u64 {
            let mut rng = SeededRng::from_seed(seed);
            let TensorOp::Scan { arg, .. } = generate(&mut rng, &bounds) else {
                unreachable!()
            };
            if arg.data().iter().any(|v| v.abs() > bounds.magnitude * 1e5) {
                cancelling += 1;
            }
        }
        assert!(cancelling > 100, "too few cancelling cases: {cancelling}");
    }
}
