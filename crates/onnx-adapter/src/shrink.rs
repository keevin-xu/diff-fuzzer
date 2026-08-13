//! Reducing a divergence to a case a maintainer can act on.
//!
//! # Two gates, and why the second one is what makes this tractable
//!
//! Every candidate must be **strictly simpler** than its parent and must **`validate()`**.
//!
//! The simplicity gate is what guarantees termination — the search can never revisit a case.
//! The validity gate is what makes *non-local* reductions safe, and it is the reason this
//! shrinker can be written at all. A single-node ONNX model is full of constraints that couple
//! parts of the case together: `Transpose`'s `perm` must be a permutation of the input's rank,
//! `Reshape`'s target must multiply out to the input's element count, `Concat`'s inputs must
//! agree on every axis but one. A reduction that changes a shape can violate any of them.
//!
//! Rather than encoding each constraint into each move — which would mean writing the operator
//! catalog a second time, in a second place, where it could drift — the moves are proposed
//! **liberally** and `validate()` throws away whatever is inconsistent. A move that cannot be
//! made valid for some operator simply produces no candidate there.
//!
//! > **The validator is already the single definition of what a legal case is. Reductions borrow
//! > it rather than re-deriving it.**
//!
//! # Preserving the finding
//!
//! Shrinking is driven by a predicate: "does this candidate still fail?". The dangerous version
//! of that predicate is "does this candidate still diverge?", because a case can shrink out of one
//! bug and into a different one, and the report then describes a case that never demonstrated what
//! it claims. [`still_shows`] is the predicate that avoids it — it requires the **same signature**,
//! not merely some divergence.

use diff_fuzzer_core::Shrink;

use crate::attrs::{AttrValue, Attrs};
use crate::case::{OnnxCase, TensorData};
use crate::validation::is_valid;

/// How complex a case is, as a lexicographic ordering.
///
/// # Why a tuple rather than a weighted sum
///
/// A weighted sum needs the weights to be separated far enough that a small term can never
/// outweigh a large one — and getting that wrong silently makes some reductions impossible,
/// which looks exactly like a shrinker that has finished. Ordering the components explicitly
/// says what matters more than what, and cannot be knocked out of order by a large case.
///
/// The order is deliberate: **fewer elements** beats **lower rank** beats **fewer inputs** beats
/// **simpler values** beats **lower opset**. Element count comes first because it dominates how
/// readable the reproduction is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Complexity {
    /// Total elements across every input.
    pub elements: usize,
    /// Sum of the ranks of every input.
    pub rank_sum: usize,
    /// How many inputs the node takes.
    pub inputs: usize,
    /// Values that are neither `0` nor `1` — the ones a reader has to think about.
    pub nontrivial_values: usize,
    /// Total magnitude of the integer attributes, so `axis: 3` is more complex than `axis: 0`.
    pub attribute_weight: i64,
    /// The opset. Last, because lowering it is the least valuable simplification.
    pub opset: i64,
}

/// Measure a whole case.
///
/// **Every part of the case must contribute**, or reductions to the parts it ignores become
/// invisible to the search and are silently impossible. `N7.3` calls for testing this against
/// the structure rather than against a hand count, which the tests below do by mutating each
/// component in turn and requiring the measure to respond.
pub fn complexity(case: &OnnxCase) -> Complexity {
    let mut elements = 0;
    let mut rank_sum = 0;
    let mut nontrivial_values = 0;

    for input in &case.inputs {
        elements += input.data.len();
        rank_sum += input.dims.len();
        nontrivial_values += count_nontrivial(&input.data);
    }

    let attribute_weight = case
        .attrs
        .iter()
        .map(|(_, value)| match value {
            AttrValue::Int(v) => v.abs(),
            AttrValue::Ints(vs) => vs.iter().map(|v| v.abs()).sum(),
            _ => 0,
        })
        .sum();

    Complexity {
        elements,
        rank_sum,
        inputs: case.inputs.len(),
        nontrivial_values,
        attribute_weight,
        opset: case.opset,
    }
}

/// Values a reader has to think about: anything that is not `0` or `1`.
///
/// Counted on the **bit pattern** for floats, so `-0.0` counts as non-trivial. It is exactly the
/// kind of value a reproduction should keep only if it matters — and F-004 is a finding *about*
/// `-0.0`, so a shrinker that quietly normalised it away would destroy the case it was
/// minimising.
fn count_nontrivial(data: &TensorData) -> usize {
    match data {
        TensorData::F32(v) => v
            .iter()
            .filter(|x| x.to_bits() != 0.0f32.to_bits() && x.to_bits() != 1.0f32.to_bits())
            .count(),
        TensorData::F64(v) => v
            .iter()
            .filter(|x| x.to_bits() != 0.0f64.to_bits() && x.to_bits() != 1.0f64.to_bits())
            .count(),
        TensorData::I32(v) => v.iter().filter(|x| **x != 0 && **x != 1).count(),
        TensorData::I64(v) => v.iter().filter(|x| **x != 0 && **x != 1).count(),
        TensorData::Bool(v) => v.iter().filter(|x| **x).count(),
    }
}

/// Above this many candidates, stop enumerating and keep the most aggressive ones.
///
/// **`candidates()` returns a `Vec`, so every candidate is built before any is tried.** The SQL
/// domain measured that cost at 1,526 ms of construction against 1.3 ms of execution — the
/// minimiser dominating the campaign it was supposed to serve. Here each candidate is a full
/// case clone, and a rank-4 tensor of thousands of values can propose a great many.
///
/// The cap keeps the *front* of the list, which is where the aggressive moves are, so capping
/// costs the fine-grained reductions rather than the big ones. `PENDING` 2.5.
const MAX_CANDIDATES: usize = 48;

impl Shrink for OnnxCase {
    fn candidates(&self) -> Vec<Self> {
        let mut out: Vec<OnnxCase> = Vec::new();

        // ── Most aggressive first: whole-case value collapse ────────────────────────
        // A greedy search takes the first candidate that still fails, so the biggest
        // reductions belong at the front.
        out.push(map_values(self, |_| Simplify::Zero));
        out.push(map_values(self, |_| Simplify::One));

        // ── Shape reductions, applied consistently across every data input ──────────
        // Applied to *all* data inputs at once because elementwise operators require their
        // operands to agree — halving one alone produces a case that cannot run.
        let max_rank = self
            .inputs
            .iter()
            .filter(|i| !i.is_initializer())
            .map(|i| i.dims.len())
            .max()
            .unwrap_or(0);
        for axis in 0..max_rank {
            // **Both halves, for every reduction.** Truncating to the front only would destroy
            // any defect that lives past the cut, the candidate would fail the predicate, and
            // the search would report a local minimum while the case was still enormous. That
            // is exactly what happened: `Where` bottomed out at 144 elements for a defect in a
            // single element, because the interesting value was never in the front slice.
            for keep in [Keep::Front, Keep::Back] {
                out.push(reshape_axis(self, axis, keep, |extent| extent / 2));
                out.push(reshape_axis(self, axis, keep, |_| 1));
            }
            out.push(drop_axis(self, axis));
        }

        // ── Drop a trailing variadic input ─────────────────────────────────────────
        // Only meaningful above two, since every operator here needs at least one and the
        // binary families need two.
        if self.inputs.len() > 2 {
            let mut shorter = self.clone();
            shorter.inputs.pop();
            out.push(shorter);
        }

        // ── Simplify individual values ─────────────────────────────────────────────
        // Last, because they are the least aggressive. Each one zeroes a single element,
        // which is what eventually isolates "this one value is what matters".
        for (index, input) in self.inputs.iter().enumerate() {
            if input.is_initializer() {
                // An initializer is configuration — a `Reshape` target, a `Squeeze` axis. Its
                // values are structure, not data, and zeroing them changes what the case *is*
                // rather than simplifying it.
                continue;
            }
            for position in 0..input.data.len().min(8) {
                out.push(zero_element(self, index, position));
            }
        }

        // ── Lower the opset ────────────────────────────────────────────────────────
        let since = crate::ops::spec(self.op).since;
        if self.opset > since {
            let mut lower = self.clone();
            lower.opset = since;
            out.push(lower);
        }

        // ── The two gates ──────────────────────────────────────────────────────────
        // Strictly simpler, and valid. Everything above proposes liberally; this is where
        // anything that broke a coupling between parts of the case is discarded.
        let parent = complexity(self);
        out.retain(|candidate| complexity(candidate) < parent && is_valid(candidate));

        // Distinct, so the search does not spend its budget re-testing the same case.
        out.dedup_by(|a, b| a == b);
        out.truncate(MAX_CANDIDATES);
        out
    }
}

/// What to replace a value with.
#[derive(Debug, Clone, Copy)]
enum Simplify {
    Zero,
    One,
}

/// Rewrite every value in every **data** input.
fn map_values(case: &OnnxCase, choose: impl Fn(usize) -> Simplify) -> OnnxCase {
    let mut out = case.clone();
    for input in out.inputs.iter_mut() {
        if input.is_initializer() {
            continue;
        }
        let simplify = choose(input.data.len());
        input.data = filled(&input.data, simplify);
    }
    out
}

fn filled(data: &TensorData, simplify: Simplify) -> TensorData {
    let n = data.len();
    match (data, simplify) {
        (TensorData::F32(_), Simplify::Zero) => TensorData::F32(vec![0.0; n]),
        (TensorData::F32(_), Simplify::One) => TensorData::F32(vec![1.0; n]),
        (TensorData::F64(_), Simplify::Zero) => TensorData::F64(vec![0.0; n]),
        (TensorData::F64(_), Simplify::One) => TensorData::F64(vec![1.0; n]),
        (TensorData::I32(_), Simplify::Zero) => TensorData::I32(vec![0; n]),
        (TensorData::I32(_), Simplify::One) => TensorData::I32(vec![1; n]),
        (TensorData::I64(_), Simplify::Zero) => TensorData::I64(vec![0; n]),
        (TensorData::I64(_), Simplify::One) => TensorData::I64(vec![1; n]),
        (TensorData::Bool(_), _) => TensorData::Bool(vec![false; n]),
    }
}

/// Zero one element of one input.
fn zero_element(case: &OnnxCase, input: usize, position: usize) -> OnnxCase {
    let mut out = case.clone();
    let data = &mut out.inputs[input].data;
    match data {
        TensorData::F32(v) => v[position] = 0.0,
        TensorData::F64(v) => v[position] = 0.0,
        TensorData::I32(v) => v[position] = 0,
        TensorData::I64(v) => v[position] = 0,
        TensorData::Bool(v) => v[position] = false,
    }
    out
}

/// Which end of an axis to keep when shortening it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Keep {
    Front,
    Back,
}

/// Change the extent of one axis in every data input that has it, **slicing along that axis**.
///
/// # Why this is not a flat truncation
///
/// A tensor's values are stored flat, and the obvious implementation keeps the first *n* of them.
/// That is only equivalent to slicing when the axis being shortened is the leading one. For any
/// other axis it scrambles the tensor — it keeps whole leading rows rather than a slice through
/// each of them — so the resulting case is not a reduction of the original at all, and whether it
/// preserves the defect is luck.
fn reshape_axis(case: &OnnxCase, axis: usize, keep: Keep, extent: impl Fn(i64) -> i64) -> OnnxCase {
    let mut out = case.clone();
    for input in out.inputs.iter_mut() {
        if input.is_initializer() || axis >= input.dims.len() {
            continue;
        }
        let before = input.dims[axis];
        let updated = extent(before).clamp(0, before);
        if updated == before {
            continue;
        }
        let start = match keep {
            Keep::Front => 0,
            Keep::Back => (before - updated) as usize,
        };
        input.data = slice_axis(&input.data, &input.dims, axis, start, updated as usize);
        input.dims[axis] = updated;
    }
    out
}

/// Take `len` positions starting at `start` along `axis`, in row-major order.
///
/// The flat index of an element decomposes into `outer · extent · inner`, where `outer` is the
/// product of the dimensions before the axis and `inner` the product of those after it. Keeping
/// a slice means walking those three loops rather than cutting the flat array.
fn slice_axis(
    data: &TensorData,
    dims: &[i64],
    axis: usize,
    start: usize,
    len: usize,
) -> TensorData {
    let outer: usize = dims[..axis].iter().product::<i64>().max(0) as usize;
    let extent: usize = dims[axis].max(0) as usize;
    let inner: usize = dims[axis + 1..].iter().product::<i64>().max(0) as usize;

    let mut keep = Vec::with_capacity(outer * len * inner);
    for o in 0..outer {
        for e in start..(start + len).min(extent) {
            for i in 0..inner {
                keep.push((o * extent + e) * inner + i);
            }
        }
    }
    gather(data, &keep)
}

/// Build a new tensor from the given flat indices.
fn gather(data: &TensorData, indices: &[usize]) -> TensorData {
    fn pick<T: Clone + Default>(values: &[T], indices: &[usize]) -> Vec<T> {
        indices
            .iter()
            .map(|i| values.get(*i).cloned().unwrap_or_default())
            .collect()
    }
    match data {
        TensorData::F32(v) => TensorData::F32(pick(v, indices)),
        TensorData::F64(v) => TensorData::F64(pick(v, indices)),
        TensorData::I32(v) => TensorData::I32(pick(v, indices)),
        TensorData::I64(v) => TensorData::I64(pick(v, indices)),
        TensorData::Bool(v) => TensorData::Bool(pick(v, indices)),
    }
}

/// Remove an axis entirely, lowering the rank — and fix up the attributes that name axes.
fn drop_axis(case: &OnnxCase, axis: usize) -> OnnxCase {
    let mut out = case.clone();
    for input in out.inputs.iter_mut() {
        if input.is_initializer() || axis >= input.dims.len() || input.dims.len() <= 1 {
            continue;
        }
        // Removing an axis keeps the slice at index 0 along it — the same reasoning as
        // `reshape_axis`: a flat truncation would scramble anything but the leading axis.
        input.data = slice_axis(&input.data, &input.dims, axis, 0, 1);
        input.dims.remove(axis);
    }

    // `perm` is a permutation of the *rank*, so dropping an axis makes the old one invalid.
    // Rebuilt rather than left to fail the validity gate, because otherwise `Transpose` could
    // never lose a dimension — a whole class of reduction silently unavailable.
    let rank = out
        .inputs
        .iter()
        .find(|i| !i.is_initializer())
        .map_or(0, |i| i.dims.len());
    if let Some(AttrValue::Ints(_)) = out.attrs.get("perm") {
        let mut rebuilt = Attrs::new();
        for (name, value) in out.attrs.iter() {
            if name == "perm" {
                rebuilt = rebuilt.ints("perm", (0..rank as i64).rev().collect());
            } else {
                rebuilt = rebuilt.with(name, value.clone());
            }
        }
        out.attrs = rebuilt;
    }
    out
}

/// The predicate a minimisation must be driven by: **the same finding, not merely a finding**.
///
/// # Bug hijacking
///
/// A case can shrink out of one divergence and into another. The report then shows a minimised
/// case that demonstrates something other than what it claims — and it is not detectable by
/// reading the result, because a minimised case that diverges looks exactly right.
///
/// So the predicate compares signatures rather than asking "did anything diverge". Anything
/// coarser risks the report describing the wrong bug; anything finer would reject legitimate
/// reductions that leave the finding intact.
pub fn still_shows<F>(target: &str, mut signature_of: F) -> impl FnMut(&OnnxCase) -> bool
where
    F: FnMut(&OnnxCase) -> Option<String>,
{
    let target = target.to_string();
    move |case: &OnnxCase| signature_of(case).as_deref() == Some(target.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{ElemType, OpKind, TensorValue};
    use crate::gen_shape::Bounds;
    use crate::generator::OnnxGenerator;
    use crate::validation::well_formed;
    use diff_fuzzer_core::rng::SeededRng;
    use diff_fuzzer_core::traits::Generator;

    /// **N7.3, and the way the step asks for it.** Not a hand-written expected number — every
    /// component of the case is mutated in turn, and the measure must respond to each. A
    /// component the measure ignores is one the shrinker can never reduce, and that failure is
    /// invisible: the search simply reports a local minimum sooner.
    #[test]
    fn complexity_responds_to_every_part_of_the_case() {
        let base = well_formed(OpKind::Add, &[2, 3], 22);
        let start = complexity(&base);

        let mut more_elements = base.clone();
        more_elements.inputs[0].dims = vec![4, 3];
        more_elements.inputs[0].data = TensorData::F32(vec![1.0; 12]);
        assert!(complexity(&more_elements) > start, "elements ignored");

        let mut higher_rank = base.clone();
        higher_rank.inputs[0].dims = vec![1, 2, 3];
        assert!(complexity(&higher_rank) > start, "rank ignored");

        let mut extra_input = base.clone();
        extra_input
            .inputs
            .push(TensorValue::f32("c", vec![1], vec![0.0]));
        assert!(complexity(&extra_input) > start, "input count ignored");

        let mut bigger_values = base.clone();
        bigger_values.inputs[0].data = TensorData::F32(vec![7.0; 6]);
        let mut trivial_values = base.clone();
        trivial_values.inputs[0].data = TensorData::F32(vec![0.0; 6]);
        assert!(
            complexity(&bigger_values) > complexity(&trivial_values),
            "value simplicity ignored"
        );

        let mut with_attr = base.clone();
        with_attr.attrs = Attrs::new().int("axis", 3);
        let mut small_attr = base.clone();
        small_attr.attrs = Attrs::new().int("axis", 0);
        assert!(
            complexity(&with_attr) > complexity(&small_attr),
            "attributes ignored"
        );

        let mut higher_opset = base.clone();
        higher_opset.opset = 23;
        assert!(complexity(&higher_opset) > start, "opset ignored");
    }

    /// `-0.0` must count as a value worth simplifying away, which means counting bit patterns.
    /// F-004 is a finding *about* `-0.0`; a measure that treated it as trivial would let the
    /// shrinker normalise away the thing being reported.
    #[test]
    fn negative_zero_is_not_a_trivial_value() {
        let mut case = well_formed(OpKind::Add, &[1], 22);
        case.inputs[0].data = TensorData::F32(vec![-0.0]);
        let mut positive = case.clone();
        positive.inputs[0].data = TensorData::F32(vec![0.0]);
        assert!(
            complexity(&case) > complexity(&positive),
            "-0.0 must be reducible to +0.0"
        );
    }

    /// **Both gates, on real generated cases.** Every candidate of every case must be strictly
    /// simpler and must validate — the two properties the search's termination and safety rest
    /// on.
    #[test]
    fn every_candidate_is_simpler_and_valid() {
        let generator = OnnxGenerator::new(Bounds::default().with_special_values());
        for seed in 0..400u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !is_valid(&case) {
                continue;
            }
            let parent = complexity(&case);
            for candidate in case.candidates() {
                assert!(
                    complexity(&candidate) < parent,
                    "seed {seed}: candidate not strictly simpler ({:?} vs {:?})",
                    complexity(&candidate),
                    parent
                );
                assert!(
                    is_valid(&candidate),
                    "seed {seed}: candidate is invalid: {:?}",
                    crate::validation::validate(&candidate)
                );
            }
        }
    }

    /// The candidate list must be capped, or the minimiser dominates the campaign it serves.
    #[test]
    fn the_candidate_list_is_bounded() {
        let generator = OnnxGenerator::new(Bounds::default().with_special_values());
        for seed in 0..300u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            assert!(
                case.candidates().len() <= MAX_CANDIDATES,
                "seed {seed} proposed {} candidates",
                case.candidates().len()
            );
        }
    }

    /// A shrinker that proposes nothing cannot shrink anything, and would pass every test above
    /// vacuously. Most non-trivial cases must offer at least one reduction.
    #[test]
    fn the_shrinker_actually_proposes_reductions() {
        let generator = OnnxGenerator::new(Bounds::default().with_special_values());
        let mut productive = 0;
        let mut considered = 0;
        for seed in 0..300u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !is_valid(&case) || complexity(&case).elements <= 1 {
                continue;
            }
            considered += 1;
            if !case.candidates().is_empty() {
                productive += 1;
            }
        }
        assert!(considered > 100, "not enough cases considered");
        assert!(
            productive * 10 >= considered * 9,
            "only {productive} of {considered} non-trivial cases could be shrunk"
        );
    }

    /// The search must terminate and must actually reduce. Driven by a predicate that keeps any
    /// case retaining a non-zero value, so there is a real floor to reach.
    #[test]
    fn minimisation_terminates_and_reduces() {
        use diff_fuzzer_core::minimize;
        let case = well_formed(OpKind::Add, &[4, 4], 22);
        let before = complexity(&case);

        let result = minimize(case, |candidate: &OnnxCase| {
            candidate.inputs.iter().any(|i| !i.data.is_empty())
        });

        assert!(complexity(&result.input) < before, "nothing was reduced");
        assert!(result.is_minimal(), "stopped early: {:?}", result.stopped);
    }

    /// **Bug hijacking.** The predicate must reject a case that diverges *differently*, not just
    /// accept anything that diverges.
    #[test]
    fn the_predicate_requires_the_same_signature() {
        let case = well_formed(OpKind::Add, &[2], 22);
        let mut predicate = still_shows("Add/22/value", |c: &OnnxCase| {
            // Stand-in signature: the real one comes from `signature.rs`.
            Some(format!("{}/{}/value", c.op.onnx_name(), c.opset))
        });
        assert!(
            predicate(&case),
            "the original must satisfy its own signature"
        );

        let mut different = case.clone();
        different.op = OpKind::Sub;
        assert!(
            !predicate(&different),
            "a different operator is a different finding"
        );
    }

    /// Initializers are configuration rather than data — a `Reshape` target or a `Squeeze` axis.
    /// Zeroing them changes what the case *is*, and for `Reshape` a zero means "copy this
    /// dimension", so it would silently alter the semantics rather than simplify them.
    #[test]
    fn initializer_values_are_never_simplified() {
        let generator = OnnxGenerator::new(Bounds::default());
        for seed in 0..300u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            if !case.inputs.iter().any(|i| i.is_initializer()) {
                continue;
            }
            let originals: Vec<&TensorValue> =
                case.inputs.iter().filter(|i| i.is_initializer()).collect();
            for candidate in case.candidates() {
                let shrunk: Vec<&TensorValue> = candidate
                    .inputs
                    .iter()
                    .filter(|i| i.is_initializer())
                    .collect();
                if shrunk.len() != originals.len() {
                    continue; // an input was dropped entirely, which is a different move
                }
                for (before, after) in originals.iter().zip(shrunk.iter()) {
                    assert_eq!(
                        before.data, after.data,
                        "seed {seed}: an initializer's values were altered"
                    );
                }
            }
        }
    }

    /// Dropping an axis must rebuild `perm`, or `Transpose` can never lose a dimension and a
    /// whole class of reduction is silently unavailable.
    #[test]
    fn dropping_an_axis_rebuilds_the_permutation() {
        let case = OnnxCase::new(
            OpKind::Transpose,
            22,
            vec![TensorValue::f32("a", vec![2, 3], vec![1.0; 6])],
        )
        .with_attrs(Attrs::new().ints("perm", vec![1, 0]));
        assert!(is_valid(&case));

        let reduced = drop_axis(&case, 0);
        assert_eq!(reduced.inputs[0].dims.len(), 1);
        assert_eq!(reduced.attrs.get("perm"), Some(&AttrValue::Ints(vec![0])));
        assert!(
            is_valid(&reduced),
            "{:?}",
            crate::validation::validate(&reduced)
        );
    }

    /// Every element type must be reducible, or findings at some types cannot be minimised and
    /// the gap is invisible.
    #[test]
    fn every_element_type_can_be_simplified() {
        for elem in ElemType::ALL {
            let case = crate::validation::well_formed_typed(OpKind::Identity, &[4], 22, elem);
            assert!(
                !case.candidates().is_empty(),
                "{elem:?} produced no candidates"
            );
        }
    }
}
