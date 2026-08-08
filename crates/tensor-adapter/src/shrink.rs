//! Simpler versions of a tensor case.
//!
//! Two kinds of move, and they attack different things.
//!
//! **Shape reductions** make the case *smaller* — halve a dimension, collapse one to a
//! single element, drop one entirely. These do the most work: a divergence found on a
//! `[7, 6, 5]` tensor is unreadable, while the same one on `[2]` can be pasted into a
//! bug report.
//!
//! **Value reductions** make the case *simpler to read* — all zeros, all ones, whole
//! numbers, smaller magnitudes. A reproduction whose input is `[1.0, 1.0]` is far more
//! convincing than one whose input is `[-7.3418274, 2.9917533]`, because a reader can
//! see at a glance that nothing about the specific digits matters.
//!
//! Every candidate must remain a **valid** case, which is what makes this domain
//! knowledge rather than generic byte-shrinking: halving a matrix multiplication's inner
//! dimension means changing *both* operands, and dropping a dimension from a reduction
//! may put its axis out of range. Generic shrinkers produce invalid cases and waste the
//! search on them.

use crate::input::{ActivationOp, BinaryOp, ScanOp, TensorOp, TensorValue, UnaryOp};
use diff_fuzzer_core::Shrink;

impl Shrink for TensorOp {
    fn candidates(&self) -> Vec<Self> {
        match self {
            TensorOp::Unary { kind, arg } => unary_candidates(*kind, arg),
            TensorOp::Binary { kind, lhs, rhs } => binary_candidates(*kind, lhs, rhs),
            TensorOp::Reduce { kind, arg, axis } => reduce_candidates(*kind, arg, *axis),
            TensorOp::Matmul { lhs, rhs } => matmul_candidates(lhs, rhs),
            TensorOp::Activation { kind, arg, dim } => activation_candidates(*kind, arg, *dim),
            TensorOp::Scan { kind, arg, dim } => scan_candidates(*kind, arg, *dim),
            // **7G.7 owns this.** Shrinking a convolution is harder than anything here so
            // far, because every candidate must stay dimensionally valid: reducing
            // `in_channels` has to keep it divisible by `groups`, and reducing a spatial
            // extent has to keep the window fitting. An invalid candidate would turn a real
            // finding into a panic during minimisation.
            //
            // Returning no candidates is the safe interim: minimisation reports the case as
            // already minimal, which is honest — this shrinker cannot currently improve it —
            // rather than emitting candidates that might crash.
            TensorOp::Conv2d { .. } => Vec::new(),
        }
    }
}

fn unary_candidates(kind: UnaryOp, arg: &TensorValue) -> Vec<TensorOp> {
    let mut out = Vec::new();

    for shape in smaller_shapes(arg.shape()) {
        out.push(TensorOp::unary(kind, arg.resized(&shape)));
    }
    for data in simpler_values(arg.data(), value_rules(kind)) {
        out.push(TensorOp::unary(
            kind,
            TensorValue::new(arg.shape().to_vec(), data),
        ));
    }

    out
}

fn binary_candidates(kind: BinaryOp, lhs: &TensorValue, rhs: &TensorValue) -> Vec<TensorOp> {
    let mut out = Vec::new();

    // **Shape reduction works on the *result*, not on either operand.** Since PHASE-7C the
    // two shapes may differ, and shrinking one alone can leave a pair that does not combine —
    // a case no backend can run, which the constructor would panic on. Deriving both from a
    // smaller result keeps every candidate valid by construction.
    let result = crate::ops::broadcast::result_shape(lhs.shape(), rhs.shape())
        .expect("a constructed case always has combinable operands");
    let lhs_stretched = stretched_axes(lhs.shape(), &result);
    let rhs_stretched = stretched_axes(rhs.shape(), &result);

    for (smaller, kept) in smaller_results(&result) {
        let l = derive_operand(&smaller, &lhs_stretched, &kept);
        let r = derive_operand(&smaller, &rhs_stretched, &kept);
        out.push(TensorOp::binary(kind, lhs.resized(&l), rhs.resized(&r)));
    }

    // **Stretching one more axis is itself a reduction**, and it is new at PHASE-7C.
    // Collapsing an axis to 1 on one side removes elements from that operand while leaving
    // the result shape untouched: `[4,4] x [4,4]` is 32 elements, `[4,1] x [4,4]` is 20.
    // Strictly simpler, so the `Shrink` contract holds and the search still terminates.
    //
    // The reverse move — un-broadcasting, giving both operands the full result shape — is
    // deliberately **not** offered: it *increases* the element count, which would break that
    // contract and could make minimisation loop.
    for axis in 0..result.len() {
        if result[axis] <= 1 {
            continue;
        }
        if !lhs_stretched[axis] {
            let mut shape = lhs.shape().to_vec();
            shape[axis] = 1;
            out.push(TensorOp::binary(kind, lhs.resized(&shape), rhs.clone()));
        }
        if !rhs_stretched[axis] {
            let mut shape = rhs.shape().to_vec();
            shape[axis] = 1;
            out.push(TensorOp::binary(kind, lhs.clone(), rhs.resized(&shape)));
        }
    }

    // Values, though, shrink independently — often only one operand matters to the
    // failure, and simplifying the other makes that visible.
    for data in simpler_values(lhs.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::binary(
            kind,
            TensorValue::new(lhs.shape().to_vec(), data),
            rhs.clone(),
        ));
    }
    for data in simpler_values(rhs.data(), right_operand_rules(kind)) {
        out.push(TensorOp::binary(
            kind,
            lhs.clone(),
            TensorValue::new(rhs.shape().to_vec(), data),
        ));
    }

    out
}

/// Candidates for a normalisation along one axis.
///
/// Mirrors `reduce_candidates`, with one addition that matters for `softmax` specifically:
/// **moving the normalised dimension to the last position is offered as a simplification.**
/// `burn-flex` takes a different code path when `dim != rank - 1` — it transposes, normalises
/// the last axis, and transposes back — so a case that only diverges off the last axis is a
/// case about that transpose. Offering the move lets minimisation *discover* that: if the
/// divergence survives, the transpose was irrelevant; if it vanishes, the transpose is the
/// story, and the minimised case says so by still pointing at the original dimension.
fn activation_candidates(kind: ActivationOp, arg: &TensorValue, dim: usize) -> Vec<TensorOp> {
    let mut out = Vec::new();

    for shape in smaller_shapes(arg.shape()) {
        // Dropping a dimension can leave `dim` past the end; clamping keeps the candidate
        // runnable, since burn panics rather than erroring on an out-of-range dimension.
        let clamped = dim.min(shape.len() - 1);
        out.push(TensorOp::activation(kind, arg.resized(&shape), clamped));
    }

    // The last axis is the simplest form to read *and* the one that avoids flex's transpose.
    if dim != arg.rank() - 1 {
        out.push(TensorOp::activation(kind, arg.clone(), arg.rank() - 1));
    }

    for data in simpler_values(arg.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::activation(
            kind,
            TensorValue::new(arg.shape().to_vec(), data),
            dim,
        ));
    }

    out
}

/// Candidates for a scan.
///
/// Mirrors the reduction shrinkers, with one difference that matters: **shortening the
/// scanned axis is the highest-value move**, because the bound and the accumulated error
/// both scale with it. A scan whose disagreement survives down to two elements is a very
/// different claim from one that needs sixty-four.
fn scan_candidates(kind: ScanOp, arg: &TensorValue, dim: usize) -> Vec<TensorOp> {
    let mut out = Vec::new();

    for shape in smaller_shapes(arg.shape()) {
        let clamped = dim.min(shape.len() - 1);
        out.push(TensorOp::scan(kind, arg.resized(&shape), clamped));
    }

    if dim != 0 {
        out.push(TensorOp::scan(kind, arg.clone(), 0));
    }

    for data in simpler_values(arg.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::scan(
            kind,
            TensorValue::new(arg.shape().to_vec(), data),
            dim,
        ));
    }

    out
}

fn reduce_candidates(
    kind: crate::input::ReduceOp,
    arg: &TensorValue,
    axis: usize,
) -> Vec<TensorOp> {
    let mut out = Vec::new();

    for shape in smaller_shapes(arg.shape()) {
        // Dropping a dimension can leave the axis pointing past the end. Clamping keeps
        // the candidate valid; the axis it lands on is still a real reduction, just not
        // the original one — and if that changes the outcome, the predicate rejects it.
        let clamped = axis.min(shape.len() - 1);
        out.push(TensorOp::reduce(kind, arg.resized(&shape), clamped));
    }

    // Reducing along the first axis is the simplest form to read, so try it early.
    if axis != 0 {
        out.push(TensorOp::reduce(kind, arg.clone(), 0));
    }

    for data in simpler_values(arg.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::reduce(
            kind,
            TensorValue::new(arg.shape().to_vec(), data),
            axis,
        ));
    }

    out
}

/// Matrix multiplication needs its own treatment, because its three free dimensions are
/// shared between the operands: `[batch.., m, k]` times `[batch.., k, n]`. Shrinking `k`
/// in one operand without the other produces a case that cannot run at all.
fn matmul_candidates(lhs: &TensorValue, rhs: &TensorValue) -> Vec<TensorOp> {
    let mut out = Vec::new();

    let ls = lhs.shape();
    let rs = rhs.shape();
    let rank = ls.len();
    let batch = &ls[..rank - 2];
    let (m, k, n) = (ls[rank - 2], ls[rank - 1], rs[rank - 1]);

    let build = |batch: &[usize], m: usize, k: usize, n: usize, out: &mut Vec<TensorOp>| {
        let mut lhs_shape = batch.to_vec();
        lhs_shape.extend([m, k]);
        let mut rhs_shape = batch.to_vec();
        rhs_shape.extend([k, n]);
        out.push(TensorOp::matmul(
            lhs.resized(&lhs_shape),
            rhs.resized(&rhs_shape),
        ));
    };

    // Drop a batch dimension entirely — the largest available reduction.
    if rank > 2 {
        build(&batch[1..], m, k, n, &mut out);
    }
    // Collapse each batch dimension to one.
    for (index, size) in batch.iter().enumerate() {
        if *size > 1 {
            let mut smaller = batch.to_vec();
            smaller[index] = 1;
            build(&smaller, m, k, n, &mut out);
        }
    }
    // Shrink each of the three free dimensions, most aggressively first. Collapsing to
    // one comes before halving, since it reaches a readable case in a single step.
    for dimension in [Dimension::Rows, Dimension::Inner, Dimension::Columns] {
        let current = dimension.of(m, k, n);
        for candidate in [1, current / 2] {
            if candidate >= 1 && candidate < current {
                let (nm, nk, nn) = dimension.replaced(m, k, n, candidate);
                build(batch, nm, nk, nn, &mut out);
            }
        }
    }

    for data in simpler_values(lhs.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::matmul(
            TensorValue::new(ls.to_vec(), data),
            rhs.clone(),
        ));
    }
    for data in simpler_values(rhs.data(), ValueRules::unrestricted()) {
        out.push(TensorOp::matmul(
            lhs.clone(),
            TensorValue::new(rs.to_vec(), data),
        ));
    }

    out
}

/// Which of matrix multiplication's three free dimensions is being shrunk.
///
/// Named rather than indexed, because `apply == 1` at a call site says nothing while
/// `Dimension::Inner` says exactly which constraint is in play — the shared one.
#[derive(Debug, Clone, Copy)]
enum Dimension {
    /// The `m` in `[.., m, k] x [.., k, n]` — rows of the result.
    Rows,
    /// The shared `k`. Changing it must change *both* operands.
    Inner,
    /// The `n` — columns of the result.
    Columns,
}

impl Dimension {
    fn of(self, m: usize, k: usize, n: usize) -> usize {
        match self {
            Dimension::Rows => m,
            Dimension::Inner => k,
            Dimension::Columns => n,
        }
    }

    fn replaced(self, m: usize, k: usize, n: usize, with: usize) -> (usize, usize, usize) {
        match self {
            Dimension::Rows => (with, k, n),
            Dimension::Inner => (m, with, n),
            Dimension::Columns => (m, k, with),
        }
    }
}

/// Smaller shapes derived from one shape, most aggressive first.
///
/// Every result has strictly fewer elements than the input, which is what guarantees the
/// search terminates rather than cycling.
/// Which axes of an operand are being stretched to reach the result.
///
/// An axis stretches when the operand holds 1 there and the result holds more. An axis that
/// is 1 on *both* sides is not stretched — it is simply small.
fn stretched_axes(operand: &[usize], result: &[usize]) -> Vec<bool> {
    operand
        .iter()
        .zip(result)
        .map(|(&o, &r)| o == 1 && r > 1)
        .collect()
}

/// Smaller result shapes, each paired with the original axis indices it kept.
///
/// The indices are what let the stretch pattern survive a dropped axis: after removing the
/// leading dimension, axis 0 of the new shape was axis 1 of the old one, and applying the
/// old flags positionally would silently stretch the wrong axis.
fn smaller_results(result: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = result.len();
    let mut out = Vec::new();

    if n > 1 {
        out.push((result[1..].to_vec(), (1..n).collect()));
        out.push((result[..n - 1].to_vec(), (0..n - 1).collect()));
    }

    let all: Vec<usize> = (0..n).collect();
    for (axis, &size) in result.iter().enumerate() {
        if size > 1 {
            let mut smaller = result.to_vec();
            smaller[axis] = 1;
            out.push((smaller, all.clone()));
        }
    }
    for (axis, &size) in result.iter().enumerate() {
        if size > 2 {
            let mut smaller = result.to_vec();
            smaller[axis] = size / 2;
            out.push((smaller, all.clone()));
        }
    }

    out
}

/// Rebuild one operand against a smaller result, keeping the axes it was stretching.
fn derive_operand(result: &[usize], stretched: &[bool], kept: &[usize]) -> Vec<usize> {
    kept.iter()
        .enumerate()
        .map(|(new_axis, &old_axis)| {
            if stretched[old_axis] {
                1
            } else {
                result[new_axis]
            }
        })
        .collect()
}

fn smaller_shapes(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();

    // Dropping a dimension removes the most at once.
    if shape.len() > 1 {
        out.push(shape[1..].to_vec());
        out.push(shape[..shape.len() - 1].to_vec());
    }

    // Then collapsing a dimension to a single element.
    for (index, size) in shape.iter().enumerate() {
        if *size > 1 {
            let mut smaller = shape.to_vec();
            smaller[index] = 1;
            out.push(smaller);
        }
    }

    // Then halving, which converges quickly on large dimensions without the abruptness
    // of going straight to one.
    for (index, size) in shape.iter().enumerate() {
        if *size > 2 {
            let mut smaller = shape.to_vec();
            smaller[index] = size / 2;
            out.push(smaller);
        }
    }

    out
}

/// Restrictions on what values may be substituted into an operand.
///
/// Shrinking must not push a case outside an operation's domain. Replacing a divisor
/// with zeros would turn a numeric divergence into a division by zero — a *different*
/// case, which might fail for a different reason and would make the reproduction
/// misleading rather than smaller.
#[derive(Debug, Clone, Copy)]
struct ValueRules {
    allow_zero: bool,
    allow_negative: bool,
}

impl ValueRules {
    fn unrestricted() -> Self {
        Self {
            allow_zero: true,
            allow_negative: true,
        }
    }
}

/// What a unary operation permits in its argument.
fn value_rules(kind: UnaryOp) -> ValueRules {
    match kind {
        // Substituting negatives into `sqrt` would produce NaN — a different failure
        // from whatever was being shrunk.
        UnaryOp::Sqrt => ValueRules {
            allow_zero: true,
            allow_negative: false,
        },
        // `log` rejects negatives for the same reason, and **also excludes zero**: shrinking
        // a value to 0.0 would turn the result into `-inf`, which is a different failure
        // from the one being minimised rather than a simpler version of it.
        UnaryOp::Log => ValueRules {
            allow_zero: false,
            allow_negative: false,
        },
        UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Exp | UnaryOp::Erf => ValueRules::unrestricted(),
    }
}

/// What a binary operation permits in its *right* operand.
fn right_operand_rules(kind: BinaryOp) -> ValueRules {
    match kind {
        BinaryOp::Div => ValueRules {
            allow_zero: false,
            allow_negative: true,
        },
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => ValueRules::unrestricted(),
    }
}

/// Simpler value sets of the same length, most aggressive first.
///
/// "Simpler" means easier to read in a bug report: all the same, whole numbers, small.
/// Ordering runs from most drastic (every value identical) to least (magnitudes reduced
/// but structure kept), so the search reaches a readable case quickly and only falls
/// back to gentler moves when the drastic ones destroy the failure.
fn simpler_values(data: &[f32], rules: ValueRules) -> Vec<Vec<f32>> {
    let mut out: Vec<Vec<f32>> = Vec::new();

    if rules.allow_zero {
        out.push(vec![0.0; data.len()]);
    }
    out.push(vec![1.0; data.len()]);
    if rules.allow_negative {
        out.push(vec![-1.0; data.len()]);
    }

    // Round to whole numbers, which removes incidental digits while keeping the shape of
    // the data. A value that rounds into a forbidden region is left alone.
    let rounded: Vec<f32> = data
        .iter()
        .map(|v| {
            let r = v.round();
            if (r == 0.0 && !rules.allow_zero) || (r < 0.0 && !rules.allow_negative) {
                *v
            } else {
                r
            }
        })
        .collect();
    if rounded != data {
        out.push(rounded);
    }

    // Reduce magnitudes without changing signs. Useful when the failure depends on scale
    // rather than on particular values.
    let smaller: Vec<f32> = data
        .iter()
        .map(|v| {
            let s = v / 10.0;
            if s == 0.0 && !rules.allow_zero { *v } else { s }
        })
        .collect();
    if smaller != data {
        out.push(smaller);
    }

    // **Keep only what is strictly simpler.** `Shrink` requires it, and the moves above
    // do not guarantee it on their own: offered against data that is already all zeros,
    // "replace with ones" proposes something *more* complex, and since zeros are then
    // offered straight back the search oscillates between the two until a budget stops
    // it. Discarding an identical candidate is not enough — the cycle is between two
    // different values.
    //
    // Found by the step budget firing on a case that should have finished, which is
    // precisely the failure the budget exists to make visible rather than silent.
    let current = complexity(data);
    out.retain(|candidate| complexity(candidate) < current);

    out
}

/// How complex a set of values is, for deciding what counts as simpler.
///
/// Ordered lexicographically: **form first, then magnitude.** Form is what a reader
/// notices — `[0.0, 0.0]` is obviously incidental in a way `[0.25, -3.5]` is not — and
/// magnitude breaks ties so that reducing scale still counts as progress among values of
/// the same form.
///
/// Making this explicit is the point. "Strictly simpler" was previously an assumption
/// each move was trusted to honour, and one of them did not.
fn complexity(data: &[f32]) -> (u32, f64) {
    let form: u32 = data.iter().map(|v| form_rank(*v)).sum();
    let magnitude: f64 = data
        .iter()
        .map(|v| if v.is_finite() { v.abs() as f64 } else { 0.0 })
        .sum();

    (form, magnitude)
}

/// How complicated a single value looks, from simplest to least.
fn form_rank(value: f32) -> u32 {
    if !value.is_finite() {
        return 5;
    }
    if value == 0.0 {
        0
    } else if value == 1.0 {
        1
    } else if value == -1.0 {
        2
    } else if value.fract() == 0.0 {
        3
    } else {
        4
    }
}

impl TensorValue {
    /// The same values reshaped to `shape`, taking as many as it needs from the front.
    ///
    /// Truncating rather than resampling keeps shrinking **deterministic** — no new
    /// randomness enters, so a minimised case is reproducible in the same way the
    /// original was. Whether the retained values still trigger the failure is exactly
    /// what the search's predicate decides.
    ///
    /// # Panics
    ///
    /// If `shape` describes more elements than this value holds — shrinking only ever
    /// goes smaller.
    pub fn resized(&self, shape: &[usize]) -> TensorValue {
        let wanted: usize = shape.iter().product();
        assert!(
            wanted <= self.len(),
            "resize is for shrinking only: {wanted} wanted, {} available",
            self.len()
        );

        TensorValue::new(shape.to_vec(), self.data()[..wanted].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::ReduceOp;

    fn value(shape: &[usize]) -> TensorValue {
        let count: usize = shape.iter().product();
        TensorValue::new(shape.to_vec(), (0..count).map(|i| i as f32 + 0.5).collect())
    }

    fn element_count(op: &TensorOp) -> usize {
        op.element_count()
    }

    /// Every candidate must be constructible — which, since the constructors assert
    /// their constraints, means every candidate is a *valid* case. This is the property
    /// that separates domain-aware shrinking from generic byte-shrinking, and simply
    /// producing the candidates exercises it.
    #[test]
    fn every_candidate_is_valid() {
        let cases = [
            TensorOp::unary(UnaryOp::Exp, value(&[3, 4])),
            TensorOp::binary(BinaryOp::Add, value(&[2, 3]), value(&[2, 3])),
            TensorOp::reduce(ReduceOp::Sum, value(&[2, 3, 4]), 2),
            TensorOp::matmul(value(&[2, 3]), value(&[3, 4])),
            TensorOp::matmul(value(&[2, 3, 4]), value(&[2, 4, 5])),
        ];

        for case in cases {
            let candidates = case.candidates();
            assert!(!candidates.is_empty(), "{} produced none", case.name());
            // Construction already asserted validity; this confirms the operation kind
            // is preserved, so shrinking never turns one case into a different one.
            for candidate in candidates {
                assert_eq!(candidate.name(), case.name());
            }
        }
    }

    /// Nothing may grow. If a candidate could be as large as its parent, the search
    /// could cycle forever rather than terminate.
    #[test]
    fn no_candidate_is_larger_than_its_parent() {
        let cases = [
            TensorOp::unary(UnaryOp::Neg, value(&[3, 4])),
            TensorOp::binary(BinaryOp::Mul, value(&[2, 5]), value(&[2, 5])),
            TensorOp::reduce(ReduceOp::Sum, value(&[4, 4]), 1),
            TensorOp::matmul(value(&[3, 4]), value(&[4, 5])),
        ];

        for case in cases {
            let size = element_count(&case);
            for candidate in case.candidates() {
                assert!(
                    element_count(&candidate) <= size,
                    "{} grew from {size} to {}",
                    case.name(),
                    element_count(&candidate)
                );
            }
        }
    }

    /// A smallest case must have no shape reductions left to offer, or the search would
    /// never stop.
    #[test]
    fn a_minimal_case_offers_no_shape_reductions() {
        let smallest = TensorOp::unary(UnaryOp::Abs, TensorValue::new(vec![1], vec![1.0]));

        for candidate in smallest.candidates() {
            let TensorOp::Unary { arg, .. } = &candidate else {
                unreachable!()
            };
            assert_eq!(arg.shape(), &[1], "a rank-1 single element was reshaped");
        }
    }

    /// Elementwise operands must stay **combinable** through every reduction. Since PHASE-7C
    /// they need not be identical — but a candidate whose operands do not broadcast could not
    /// run at all, and `TensorOp::binary` would panic building it.
    #[test]
    fn binary_candidates_always_combine() {
        let cases = [
            TensorOp::binary(BinaryOp::Sub, value(&[2, 6]), value(&[2, 6])),
            TensorOp::binary(BinaryOp::Sub, value(&[2, 1]), value(&[2, 6])),
            TensorOp::binary(BinaryOp::Sub, value(&[1, 6]), value(&[2, 6])),
            TensorOp::binary(BinaryOp::Add, value(&[1, 1, 1]), value(&[3, 2, 4])),
        ];

        for case in cases {
            for candidate in case.candidates() {
                let TensorOp::Binary { lhs, rhs, .. } = &candidate else {
                    unreachable!()
                };
                assert!(
                    crate::ops::broadcast::compatible(lhs.shape(), rhs.shape()),
                    "shrinking {case:?} produced an unrunnable pair: {:?} and {:?}",
                    lhs.shape(),
                    rhs.shape()
                );
            }
        }
    }

    /// **No candidate may grow.** Simplicity here is two-dimensional — a candidate either
    /// holds fewer elements or holds the same elements with simpler values — so "strictly
    /// fewer elements" is the wrong assertion and passes only by accident on shape moves.
    ///
    /// What must never happen is *growth*, and that is exactly what un-broadcasting would do:
    /// giving both operands the full result shape turns `[2,1] x [2,6]` (14 elements) into
    /// `[2,6] x [2,6]` (24). Offering it would break the `Shrink` contract and could make
    /// minimisation loop, which is why the reverse move is not generated.
    #[test]
    fn no_binary_candidate_grows() {
        let cases = [
            TensorOp::binary(BinaryOp::Sub, value(&[2, 1]), value(&[2, 6])),
            TensorOp::binary(BinaryOp::Sub, value(&[2, 6]), value(&[2, 6])),
            TensorOp::binary(BinaryOp::Add, value(&[1, 4]), value(&[3, 4])),
        ];

        for case in cases {
            let before = case_elements(&case);
            for candidate in case.candidates() {
                assert!(
                    case_elements(&candidate) <= before,
                    "candidate grew from {before}: {candidate:?}"
                );
                assert_ne!(
                    candidate, case,
                    "a candidate equal to its parent loops forever"
                );
            }
        }
    }

    /// And a shape move specifically *must* shrink, or the shape search would not converge.
    #[test]
    fn every_shape_changing_candidate_holds_fewer_elements() {
        let case = TensorOp::binary(BinaryOp::Sub, value(&[2, 1]), value(&[2, 6]));
        let before = case_elements(&case);
        let (lhs_shape, rhs_shape) = shapes(&case);

        for candidate in case.candidates() {
            if shapes(&candidate) != (lhs_shape.clone(), rhs_shape.clone()) {
                assert!(
                    case_elements(&candidate) < before,
                    "a shape move did not shrink: {candidate:?}"
                );
            }
        }
    }

    fn shapes(case: &TensorOp) -> (Vec<usize>, Vec<usize>) {
        let TensorOp::Binary { lhs, rhs, .. } = case else {
            unreachable!()
        };
        (lhs.shape().to_vec(), rhs.shape().to_vec())
    }

    /// The move PHASE-7C adds: shrinking an operand *into* a broadcast, which reaches minimal
    /// cases the old shrinker could not express.
    #[test]
    fn an_axis_can_shrink_to_one_on_a_single_side() {
        let case = TensorOp::binary(BinaryOp::Sub, value(&[2, 6]), value(&[2, 6]));

        let reached = case.candidates().into_iter().any(|c| {
            let TensorOp::Binary { lhs, rhs, .. } = &c else {
                unreachable!()
            };
            lhs.shape() != rhs.shape()
        });
        assert!(reached, "no candidate stretches an axis on one side only");
    }

    /// Total elements across both operands.
    fn case_elements(case: &TensorOp) -> usize {
        let TensorOp::Binary { lhs, rhs, .. } = case else {
            unreachable!()
        };
        lhs.data().len() + rhs.data().len()
    }

    /// Matrix multiplication's shared inner dimension must stay shared.
    #[test]
    fn matmul_candidates_keep_their_dimensions_compatible() {
        for case in [
            TensorOp::matmul(value(&[3, 4]), value(&[4, 5])),
            TensorOp::matmul(value(&[2, 3, 4]), value(&[2, 4, 5])),
        ] {
            for candidate in case.candidates() {
                let TensorOp::Matmul { lhs, rhs } = &candidate else {
                    unreachable!()
                };
                let (ls, rs) = (lhs.shape(), rhs.shape());
                assert_eq!(ls.len(), rs.len());
                assert_eq!(ls[ls.len() - 1], rs[rs.len() - 2]);
                assert_eq!(ls[..ls.len() - 2], rs[..rs.len() - 2]);
            }
        }
    }

    /// A reduction's axis must remain within its argument's rank after any reshape.
    #[test]
    fn reduce_candidates_keep_their_axis_in_range() {
        let case = TensorOp::reduce(ReduceOp::Sum, value(&[2, 3, 4]), 2);

        for candidate in case.candidates() {
            let TensorOp::Reduce { arg, axis, .. } = &candidate else {
                unreachable!()
            };
            assert!(*axis < arg.rank(), "axis {axis} for rank {}", arg.rank());
        }
    }

    /// **Shrinking must not walk a case out of its domain.** Substituting zeros into a
    /// divisor would turn a numeric divergence into a division by zero — a different
    /// failure, which would make the "minimised" reproduction misleading rather than
    /// smaller.
    #[test]
    fn division_candidates_never_introduce_a_zero_divisor() {
        let case = TensorOp::binary(BinaryOp::Div, value(&[4]), value(&[4]));

        for candidate in case.candidates() {
            let TensorOp::Binary { rhs, .. } = &candidate else {
                unreachable!()
            };
            assert!(
                rhs.data().iter().all(|v| *v != 0.0),
                "a zero divisor was introduced: {rhs:?}"
            );
        }
    }

    /// Likewise, `sqrt` must never be handed a negative by the shrinker.
    #[test]
    fn sqrt_candidates_never_introduce_a_negative_argument() {
        let case = TensorOp::unary(UnaryOp::Sqrt, value(&[4]));

        for candidate in case.candidates() {
            let TensorOp::Unary { arg, .. } = &candidate else {
                unreachable!()
            };
            assert!(
                arg.data().iter().all(|v| *v >= 0.0),
                "a negative argument was introduced: {arg:?}"
            );
        }
    }

    /// The readable forms must be offered, since they are what makes a reproduction
    /// convincing — a reader can see at a glance that the specific digits do not matter.
    #[test]
    fn simple_value_substitutions_are_offered() {
        let case = TensorOp::unary(UnaryOp::Neg, value(&[3]));
        let candidates = case.candidates();

        let has_all = |wanted: f32| {
            candidates.iter().any(|c| {
                let TensorOp::Unary { arg, .. } = c else {
                    return false;
                };
                arg.data().iter().all(|v| *v == wanted)
            })
        };

        assert!(has_all(0.0), "no all-zero candidate");
        assert!(has_all(1.0), "no all-one candidate");
    }

    /// **No candidate may equal its parent.** The `Shrink` contract requires strict
    /// simplification, and an unchanged candidate breaks the search: it is accepted as
    /// progress, lands back where it started, and repeats until a budget intervenes.
    ///
    /// The case that exposed this was one already reduced to all zeros, where "replace
    /// the values with zeros" proposes exactly what is already there. Cases at rest on
    /// each simple form are checked here for that reason.
    #[test]
    fn no_candidate_equals_its_parent() {
        let settled = [
            TensorOp::unary(UnaryOp::Abs, TensorValue::new(vec![2], vec![0.0, 0.0])),
            TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![1], vec![1.0])),
            TensorOp::binary(
                BinaryOp::Add,
                TensorValue::new(vec![1], vec![0.0]),
                TensorValue::new(vec![1], vec![0.0]),
            ),
            TensorOp::binary(
                BinaryOp::Div,
                TensorValue::new(vec![1], vec![1.0]),
                TensorValue::new(vec![1], vec![1.0]),
            ),
            TensorOp::reduce(ReduceOp::Sum, TensorValue::new(vec![1], vec![0.0]), 0),
            TensorOp::matmul(
                TensorValue::new(vec![1, 1], vec![1.0]),
                TensorValue::new(vec![1, 1], vec![1.0]),
            ),
        ];

        for case in settled {
            for candidate in case.candidates() {
                assert_ne!(
                    candidate,
                    case,
                    "{} proposed itself, which would loop the search",
                    case.name()
                );
            }
        }
    }

    /// A fully settled case must offer nothing at all, which is what lets the search
    /// finish rather than exhaust a budget.
    #[test]
    fn a_fully_reduced_case_offers_no_candidates() {
        let settled = TensorOp::unary(UnaryOp::Abs, TensorValue::new(vec![1], vec![0.0]));
        assert!(
            settled.candidates().is_empty(),
            "{:?}",
            settled.candidates()
        );
    }

    /// Shrinking introduces no randomness, so the same case always yields the same
    /// candidates — without which a minimised reproduction would not be reproducible.
    #[test]
    fn shrinking_is_deterministic() {
        let case = TensorOp::matmul(value(&[3, 4]), value(&[4, 5]));
        assert_eq!(case.candidates(), case.candidates());
    }
}
