//! How much difference each operation is allowed.
//!
//! Every number here is **derived, then checked against measurement** — never fitted to
//! it. That ordering is the whole point. A threshold set just above the largest
//! observed noise has no argument behind it: it is guaranteed to produce no false
//! positives on the data it was fitted to, no margin for cases not yet generated, and
//! it would silently absorb a real bug that happened to be smaller than noise already
//! seen. A threshold derived from how floating-point arithmetic works, which then turns
//! out to cover the observed noise with room to spare, is a claim that can be defended.
//!
//! Three classes, for three genuinely different reasons.
//!
//! # Exactly equal: `add`, `sub`, `mul`, `div`, `sqrt`, `neg`, `abs`
//!
//! Not "very close" — identical, and provably so. IEEE-754 **requires** addition,
//! subtraction, multiplication, division and square root to be *correctly rounded*: the
//! result must be the representable number nearest the true answer. There is exactly
//! one such number, so any two conforming implementations must produce the same bits.
//! `neg` and `abs` only touch the sign bit. Measurement agrees: zero error across
//! 14,000 cases.
//!
//! Holding these to exact equality is therefore not strictness for its own sake. A
//! difference here would be a genuine violation, and giving them slack would only hide
//! it.
//!
//! # One rounding step: `exp`
//!
//! `exp` is conspicuously *not* in the list IEEE-754 requires to be correctly rounded —
//! doing so is expensive, and libraries choose their own approximations. So two
//! correct implementations may land on adjacent representable numbers. Measurement
//! shows exactly that: a hard ceiling at `1.192e-7`, which is precisely one unit in the
//! last place. Two units are allowed, since each side may round its own approximation.
//!
//! # Accumulated error: `sum`, `matmul`
//!
//! Adding many numbers is where implementations legitimately part company, because
//! floating-point addition is not associative — a different summation order gives a
//! different answer, and neither is wrong. The standard bound for summing `n` terms is
//!
//! ```text
//!     |computed - exact|  <=  n * eps * sum|x_i|
//! ```
//!
//! and two implementations may sit on opposite sides of the true value, so the gap
//! between *them* can reach twice that. This is computed **per case** from the actual
//! shapes and values rather than from a global worst case, which keeps the tolerance
//! tight on small inputs instead of applying the loosest case everywhere.
//!
//! The absolute term matters more than it looks here. Summing mixed-sign values can
//! land near zero while the terms are large — cancellation — and a tiny absolute error
//! then becomes an enormous *relative* one. Measurement shows `sum` reaching `1.2e-3`
//! relative while its absolute error stays at `7.6e-6`: the error did not grow, the
//! denominator shrank.

use crate::input::{BinaryOp, TensorOp, UnaryOp};
use diff_fuzzer_core::{Tolerance, TolerancePolicy};

/// One rounding step for `f32`, as a `f64` so the arithmetic below does not itself
/// round.
const EPSILON: f64 = f32::EPSILON as f64;

/// Why an operation is allowed the tolerance it gets.
///
/// Named for the *reason* rather than the operations, since the reason is what a new
/// operation has to be classified by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpClass {
    /// IEEE-754 requires a correctly rounded result, so implementations must agree
    /// bit-for-bit.
    CorrectlyRounded,
    /// Approximated by each library independently; results may differ by a rounding
    /// step.
    Approximated,
    /// Sums many terms, so results depend on summation order.
    Accumulating,
    /// A **composition** of the classes above, whose bound is derived from its parts.
    ///
    /// `softmax` is the first: an approximated `exp`, an accumulation over the normalised
    /// axis, and a division. None of the three classes describes it, and picking the loosest
    /// of them would be a guess rather than a derivation — the parts compose in a specific
    /// way, and the composition is what has to be bounded.
    Composed,
}

impl TensorOp {
    /// Which tolerance class this case falls into.
    pub fn class(&self) -> OpClass {
        use crate::input::{ReduceOp, UnaryOp};

        match self {
            TensorOp::Unary { kind, .. } => match kind {
                // Neither is required to be correctly rounded, and their error grows in
                // opposite directions — see `approximated_tolerance`.
                UnaryOp::Exp | UnaryOp::Log => OpClass::Approximated,
                // `sqrt` is correctly rounded by IEEE-754; `neg` and `abs` only touch
                // the sign bit.
                UnaryOp::Neg | UnaryOp::Abs | UnaryOp::Sqrt => OpClass::CorrectlyRounded,
            },
            // Every elementwise binary operation is one correctly rounded arithmetic
            // operation per element, with nothing accumulated.
            TensorOp::Binary { .. } => OpClass::CorrectlyRounded,
            TensorOp::Reduce { kind, .. } => match kind {
                ReduceOp::Sum => OpClass::Accumulating,
                // A sum plus a division: the accumulation dominates, and the extra rounding
                // is added inside `accumulating_tolerance` rather than by changing class.
                ReduceOp::Mean => OpClass::Accumulating,
                // **Exact, and for a stronger reason than the other members of this class.**
                // `sqrt` is correctly *rounded*; `max` and `min` do no arithmetic whatever —
                // they select one of their inputs and return it untouched. Nothing can round
                // differently because nothing is rounded.
                //
                // A disagreement is therefore never precision. It would be a semantic
                // difference — most plausibly over `NaN`, which IEEE-754's `maxNum` ignores
                // and which implementations genuinely handle differently — and that is a
                // finding, not noise.
                ReduceOp::Max | ReduceOp::Min => OpClass::CorrectlyRounded,
            },
            TensorOp::Matmul { .. } => OpClass::Accumulating,
            // See `composed_tolerance`: `exp` over a sum, then a division.
            TensorOp::Activation { .. } => OpClass::Composed,
        }
    }
}

/// Chooses a tolerance from the operation and the size of its arguments.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorTolerancePolicy;

/// Units in the last place that Metal permits for `f32` division with fast math enabled.
///
/// **Quoted, not chosen.** *Metal Shading Language Specification* (2026-06-04) Table 8.2:
/// `x / y` → "`<= 2.5 ulp` for y in the domain of 2⁻¹²⁶ to 2¹²⁶". Fast math is the default
/// unless `-fno-fast-math` is passed. See `SPECS.md` §4.1.
///
/// Measured difference between CPU and GPU: **1 ULP**, so the derived bound clears the
/// observation by 2.5x. Had it not, that would be a finding about the GPU.
const METAL_DIV_ULPS: f64 = 2.5;

/// Units in the last place derived for `f32` `sqrt` on Metal with fast math enabled.
///
/// **Composed, not quoted.** Table 8.2 gives `sqrt` no figure of its own; it states
/// `sqrt(x)` is "Implemented as `x * rsqrt(x)` with special cases handled correctly", and
/// gives `rsqrt <= 2 ulp`. The multiply is correctly rounded and contributes at most a
/// further half ULP, so the composition permits **3**. Measured: 2 ULP, a 1.5x margin.
const METAL_SQRT_ULPS: f64 = 3.0;

/// The largest magnitude Metal is permitted to flush to zero.
///
/// **Derived, and exactly bounded.** *Metal Shading Language Specification* (2026-06-04)
/// §8.1: "Denormalized single-precision … numbers passed as input to or produced as the
/// output of … arithmetic operations **may be flushed to zero**."
///
/// A denormal is precisely a value with magnitude below `f32::MIN_POSITIVE`, so an absolute
/// tolerance of exactly that covers every permitted flush **and nothing else**: anything at
/// or above it is a normal number, which the specification gives no licence to discard.
///
/// This has to be an *absolute* tolerance. A subnormal becoming zero is a **relative error
/// of 1.0** — the largest a relative measure can express short of a sign change — so no
/// `rtol` short of absurd would absorb it, and an absurd one would hide everything else.
const METAL_SUBNORMAL_FLOOR: f64 = f32::MIN_POSITIVE as f64;

/// Whether any operand carries a subnormal value.
///
/// Checked on the **input**, not the result. Metal §8.1 licenses flushing denormals
/// "passed as input to *or* produced as the output of" an arithmetic operation, and the
/// input case is the one no tolerance can handle: a subnormal input flushed to zero can
/// produce a perfectly normal-sized difference in the output.
fn has_subnormal_input(case: &TensorOp) -> bool {
    let operands: [&crate::input::TensorValue; 2] = match case {
        TensorOp::Unary { arg, .. }
        | TensorOp::Reduce { arg, .. }
        | TensorOp::Activation { arg, .. } => [arg, arg],
        TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => [lhs, rhs],
    };

    operands.iter().any(|operand| {
        operand
            .data()
            .iter()
            .any(|value| *value != 0.0 && value.abs() < f32::MIN_POSITIVE)
    })
}

/// Whether a comparison involves a GPU backend.
///
/// Matched on the name because that is what the policy is handed. Crude, and deliberately
/// so: the alternative is threading a backend *capability* description through the engine,
/// which would be real machinery in service of one boolean.
fn involves_gpu(implementations: (&str, &str)) -> bool {
    let (left, right) = implementations;
    left.contains("wgpu") || right.contains("wgpu")
}

impl TolerancePolicy<TensorOp> for TensorTolerancePolicy {
    fn tolerance_for(&self, input: &TensorOp, implementations: (&str, &str)) -> Tolerance {
        let base = self.without_hardware_allowances(input, implementations);

        if !involves_gpu(implementations) {
            return base;
        }

        // Metal may flush denormals (§8.1). Raise the absolute floor to cover exactly that
        // — `max`, not `+`, so a class that already derives a larger absolute term keeps
        // its own bound rather than silently gaining a little more.
        Tolerance {
            atol: base.atol.max(METAL_SUBNORMAL_FLOOR),
            ..base
        }
    }

    fn known_legal(
        &self,
        input: &TensorOp,
        implementations: (&str, &str),
    ) -> Option<(String, String)> {
        self.licensed_difference(input, implementations)
    }
}

impl TensorTolerancePolicy {
    /// **The one licensed difference, and it is licensed by a quoted clause.**
    ///
    /// Metal Shading Language Specification (2026-06-04) §8.1: "Denormalized
    /// single-precision … numbers passed as **input to** or produced as the output of …
    /// arithmetic operations may be flushed to zero."
    ///
    /// The *output* half is handled by an absolute tolerance ([`METAL_SUBNORMAL_FLOOR`]).
    /// The **input** half cannot be: flushing a subnormal input can move the output by an
    /// unbounded amount. `sqrt(1.4e-45)` is `3.7e-23` on a CPU and `0` on the GPU —
    /// fifteen orders of magnitude apart, from a difference the specification permits.
    ///
    /// So such a case is **not compared** against a GPU. That discards evidence, which is
    /// why it is confined to exactly this condition rather than applied to the operation
    /// or the class: a policy that declares its awkward cases legal finds nothing and
    /// looks flawless.
    ///
    /// **CPU pairs are unaffected** — nothing licenses a CPU to discard a subnormal, and
    /// a divergence there would be a real finding.
    fn licensed_difference(
        &self,
        input: &TensorOp,
        implementations: (&str, &str),
    ) -> Option<(String, String)> {
        if involves_gpu(implementations) && has_subnormal_input(input) {
            return Some((
                "subnormal input on a GPU".to_string(),
                "Metal Shading Language Specification §8.1 permits denormals passed as \
                 input to be flushed to zero; the resulting output difference is unbounded \
                 and no tolerance can distinguish it from a defect"
                    .to_string(),
            ));
        }
        None
    }
}

impl TensorTolerancePolicy {
    /// The bound before any hardware-specific allowance.
    fn without_hardware_allowances(
        &self,
        input: &TensorOp,
        implementations: (&str, &str),
    ) -> Tolerance {
        match input.class() {
            // **Exact stays exact even against the GPU** for most of this class. Metal's
            // Table 8.2 lists `x + y`, `x - y`, `x * y` as *correctly rounded* and `fabs`
            // at `0 ulp`, and measurement found them bit-identical on 200/200 cases. Two
            // operations are the exception, and only when a GPU is involved.
            OpClass::CorrectlyRounded => match input {
                TensorOp::Binary {
                    kind: BinaryOp::Div,
                    ..
                } if involves_gpu(implementations) => ulps(METAL_DIV_ULPS),

                TensorOp::Unary {
                    kind: UnaryOp::Sqrt,
                    ..
                } if involves_gpu(implementations) => ulps(METAL_SQRT_ULPS),

                _ => Tolerance::EXACT,
            },

            OpClass::Approximated => approximated_tolerance(input),

            OpClass::Accumulating => accumulating_tolerance(input),

            OpClass::Composed => composed_tolerance(input, implementations),
        }
    }
}

/// A relative tolerance of `n` units in the last place.
///
/// `f32::EPSILON` is the gap between 1.0 and the next representable value, which is the
/// width of one ULP in relative terms — so `n` ULPs is `n * EPSILON` of relative error.
fn ulps(n: f64) -> Tolerance {
    Tolerance {
        rtol: n * EPSILON,
        atol: 0.0,
    }
}

/// Tolerance for an approximated function, scaled by how hard the function is to
/// evaluate at the argument it was given.
///
/// The governing idea is the **condition number**: how much a small relative
/// perturbation of the input is magnified in the output. For `exp(x)` it is `|x|`,
/// because `exp(x + d) = exp(x) * e^d` — a tiny error in the argument becomes a relative
/// error of roughly `d` in the result. Implementations reduce the argument before
/// approximating (`x = k*ln2 + r`), and the error in that reduction grows with `|x|`, so
/// two libraries drift further apart the larger the argument.
///
/// Hence `(1 + |x|) * eps` for one implementation — a rounding step plus the
/// condition-number term — doubled because two implementations may sit on opposite sides
/// of the true value.
///
/// **This replaces a fixed `2 * eps`, which was wrong in an instructive way.** That
/// constant was derived from data measured with `|x| <= 10`, and it held perfectly
/// there. Run at `|x| <= 1000` it produced 235 false positives, because it did not
/// scale with the thing that actually drives the error. *Fixed thresholds inherit the
/// scope of the evidence they were derived from* — the same trap the accumulating class
/// avoided by being computed per case from the outset.
///
/// The bound is deliberately looser than measurement at small arguments (roughly 20x the
/// worst observed at `|x| <= 10`). That gap is honest: the model bounds what is
/// *permissible* for a function the standard does not require to be correctly rounded,
/// while measurement shows what these two particular libraries *happen* to do today. A
/// threshold tightened to the latter would be fitted to an implementation detail.
fn approximated_tolerance(input: &TensorOp) -> Tolerance {
    let (kind, arg) = match input {
        TensorOp::Unary { kind, arg } => (*kind, arg),
        // Every approximated operation is unary; anything else is misclassified.
        other => unreachable!("{} is not an approximated unary operation", other.name()),
    };

    // **The two approximated functions need different condition numbers, and using one for
    // both would be wrong in opposite directions.** `exp` is hardest at large arguments;
    // `log` is hardest near 1. Applying `exp`'s model to `log` would be tightest exactly
    // where `log` is loosest.
    let amplification = match kind {
        UnaryOp::Exp => largest_magnitude(arg.data()).min(EXP_SATURATION),
        UnaryOp::Log => log_amplification(arg.data()),
        other => unreachable!("{other:?} is not approximated"),
    };

    // **An absolute floor at the smallest normal, and it is not decoration.**
    //
    // Capping the relative term at saturation broke a measured case: `exp` showed a relative
    // error of 1.633e-4 against a capped bound of 2.5e-5, in 34 elements out of 2,603 cases.
    //
    // The diagnosis is that those are not condition-number effects at all. No argument can
    // produce them — `exp` overflows above ~88.7, so the amplification term cannot legally
    // reach that far. They are **subnormal results**: `exp(-100)` is `3.7e-44`, and relative
    // precision below `f32::MIN_POSITIVE` degrades to a few significant bits whatever the
    // implementation does.
    //
    // A relative bound is the wrong instrument there. `atol` at the smallest normal covers
    // exactly the region where relative error stops meaning anything, and costs no
    // sensitivity above it — which is why this is still far tighter than the 2.4e23 it
    // replaced.
    Tolerance::new(
        2.0 * (1.0 + amplification) * EPSILON,
        f32::MIN_POSITIVE as f64,
    )
}

/// How much `log` magnifies a relative perturbation of its argument, over a whole tensor.
///
/// # The derivation
///
/// `log(x(1 + d)) = log(x) + log(1 + d) ≈ log(x) + d`. The *absolute* error in the result is
/// therefore about `d`, and the **relative** error is `d / |log(x)|` — so the condition
/// number is `1 / |ln x|`.
///
/// **This diverges as `x` approaches 1**, because `log(1)` is exactly zero and a relative
/// error measured against zero is unbounded. That is the opposite shape from `exp`, whose
/// condition number is `|x|` and which is therefore benign near its own zero.
///
/// # Why it is capped
///
/// Taken literally the bound is infinite at `x = 1`, which would make every case containing
/// a 1 unjudgeable — the tolerance would excuse anything. But `log(1)` is a value every
/// implementation gets *exactly* right: it is a special case in every library, and the
/// result is exactly `0.0` with no rounding at all.
///
/// So the relative bound is capped, and near-1 arguments are instead covered by an absolute
/// term of one `EPSILON` — the largest absolute error a correctly-behaved implementation can
/// have when the true answer is near zero. Capping without that absolute term would be the
/// unsafe direction; capping *with* it bounds the same cases by a different measure rather
/// than by a looser one.
///
/// `LOG_MAX_AMPLIFICATION` is a judgment about where "near 1" starts, not a fitted value:
/// at `|ln x| = 1/64`, `x` is within about 1.6% of 1.
fn log_amplification(data: &[f32]) -> f64 {
    data.iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| {
            let ln = (v as f64).ln().abs();
            if ln == 0.0 {
                LOG_MAX_AMPLIFICATION
            } else {
                (1.0 / ln).min(LOG_MAX_AMPLIFICATION)
            }
        })
        .fold(0.0f64, f64::max)
}

/// The ceiling on `log`'s condition number, reached when the argument is within ~1.6% of 1.
///
/// See [`log_amplification`] for why a cap is needed and what covers the cases it excludes.
const LOG_MAX_AMPLIFICATION: f64 = 64.0;

/// Beyond this argument magnitude, `exp` has **saturated** and carries no uncertainty.
///
/// `exp(88.8)` overflows `f32` to infinity and `exp(-104)` underflows to zero. Both are
/// exact: every implementation returns the same thing, and there is nothing left for a
/// condition number to bound.
///
/// **Capping here is a tightening, not a relaxation, and it fixes a measured blindness.**
/// The condition number of `exp` is `|x|`, so an unbounded model applied at `x = 1e30` — a
/// value the special-value table injects deliberately — produced `rtol = 2.4e23`. That
/// accepts any answer, and **81% of `exp` cases carried such a bound**, meaning the
/// operation could not report a divergence at all. The uncapped model was correct about
/// worst-case sensitivity and irrelevant to a case whose answer is exactly `inf` or exactly
/// `0`.
///
/// 104 is where the *smaller* of the two saturations occurs, so it bounds both.
const EXP_SATURATION: f64 = 104.0;

/// Tolerance for an operation that sums `terms` values of magnitude up to `largest`.
///
/// The factor of two accounts for the two implementations sitting on opposite sides of
/// the true value: each may be off by the bound, so the gap between them can be twice
/// it.
fn bound(terms: usize, largest: f64) -> Tolerance {
    let terms = terms as f64;
    // Relative component: covers results that are large, where the error scales with
    // the answer.
    let rtol = 2.0 * terms * EPSILON;
    // Absolute component: covers results near zero, where cancellation has destroyed
    // the scale that a relative tolerance would need. `terms * largest` bounds the sum
    // of magnitudes.
    let atol = 2.0 * terms * EPSILON * (terms * largest);
    Tolerance::new(rtol, atol)
}

/// Tolerance for `softmax`, derived from the parts it is built out of.
///
/// # The composition
///
/// `softmax(x)_i = exp(x_i - m) / sum_j exp(x_j - m)`, where `m` is the maximum along the
/// normalised axis. Four steps contribute error, and **relative errors add through a
/// quotient**, so they sum:
///
/// | step | contribution, in units of `EPSILON` | why |
/// |---|---|---|
/// | `max` | 0 | returns one of its inputs unchanged; nothing is rounded |
/// | `x_i - m` then `exp` | `1 + range` | the `Approximated` model, at argument `x_i - m`. `exp`'s condition number is the magnitude of its argument, and since `m` is the maximum, that magnitude is at most the **range** of values along the axis |
/// | `sum_j` | `n` | the `Accumulating` model over `n` terms |
/// | `/` | 1, or [`METAL_DIV_ULPS`] against a GPU | one correctly rounded division; Metal permits 2.5 ULP (`SPECS.md` §4.1) |
///
/// Doubled, as everywhere else here, because two implementations may sit on opposite sides
/// of the true value.
///
/// # Two deliberate conservatisms, stated rather than hidden
///
/// **The range is taken over the whole tensor, not per slice.** Each output element depends
/// only on its own slice along `dim`, so the exact bound would use that slice's range. The
/// whole-tensor range is an upper bound on every slice's, which errs toward a looser
/// tolerance — the direction that costs sensitivity rather than the one that hides defects.
/// Computing it per slice is a refinement worth making only if this proves too loose.
///
/// **No absolute term.** The `Accumulating` class carries one because cancellation can
/// destroy the scale a relative tolerance needs. Here it cannot: every `exp(x_i - m)` is
/// positive and no term cancels another, so the sum keeps its scale and a relative bound is
/// well founded. Against a GPU an absolute floor is still added upstream, for denormal
/// flushing rather than for cancellation.
///
/// # What this does *not* assume
///
/// It does not assume how a backend organises the work. `burn-wgpu` composes five separate
/// operations and `burn-flex` fuses three passes into one kernel; both perform the same
/// mathematical steps, and the bound above covers either. A fused implementation should land
/// *inside* it with margin, which is the point — a bound that only the composed version
/// satisfied would be fitted to one implementation.
fn composed_tolerance(input: &TensorOp, implementations: (&str, &str)) -> Tolerance {
    let TensorOp::Activation { arg, dim, .. } = input else {
        unreachable!("{} is not a composed operation", input.name());
    };

    let terms = arg.shape()[*dim] as f64;
    // Capped for the same reason as `exp`'s, and it is the same cap: the shifted argument
    // `x_i - m` is what `exp` receives, so once it falls below `-EXP_SATURATION` the term is
    // exactly zero and contributes no uncertainty. Uncapped, a tensor spanning ±1e30 — which
    // the special-value table produces routinely — gave `rtol = 4.8e23`, and **65% of
    // `softmax` cases became unjudgeable while reporting agreement**.
    let range = value_range(arg.data()).min(EXP_SATURATION);
    let division = if involves_gpu(implementations) {
        METAL_DIV_ULPS
    } else {
        1.0
    };

    Tolerance::new(2.0 * ((1.0 + range) + terms + division) * EPSILON, 0.0)
}

/// The spread of finite values, which bounds `exp`'s condition number after the max-shift.
///
/// Non-finite values are skipped: they are handled by the special-value policy (§5), not by
/// a tolerance, and letting an infinity into this arithmetic would produce a `NaN` bound.
fn value_range(data: &[f32]) -> f64 {
    let finite = data.iter().copied().filter(|v| v.is_finite());
    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for value in finite {
        low = low.min(value);
        high = high.max(value);
    }
    if low > high {
        return 0.0; // nothing finite; the bound is irrelevant to such a case
    }
    (high - low) as f64
}

fn accumulating_tolerance(input: &TensorOp) -> Tolerance {
    use crate::input::ReduceOp;

    match input {
        TensorOp::Reduce { kind, arg, axis } => {
            // Each output element sums exactly the values along the collapsed axis.
            let terms = arg.shape()[*axis];
            let summed = bound(terms, largest_magnitude(arg.data()));

            match kind {
                ReduceOp::Sum => summed,
                // **`mean` is a sum and then a division, and the division is not free.**
                // Reusing `sum`'s entry unchanged would understate the bound by one rounding
                // — small, but the kind of omission that spreads silently once a class is
                // shared. Relative errors add through a quotient, so one `EPSILON` is added
                // to the relative term. The absolute term is scaled down by the same divisor
                // the values are.
                ReduceOp::Mean => {
                    Tolerance::new(summed.rtol + 2.0 * EPSILON, summed.atol / terms as f64)
                }
                // Classified `CorrectlyRounded`; never routed here.
                ReduceOp::Max | ReduceOp::Min => {
                    unreachable!("{} does not accumulate", input.name())
                }
            }
        }
        TensorOp::Matmul { lhs, rhs } => {
            // Each output element sums `k` products, where `k` is the shared inner
            // dimension. A product of two values is bounded by the product of their
            // largest magnitudes.
            let shape = lhs.shape();
            let terms = shape[shape.len() - 1];
            let largest = largest_magnitude(lhs.data()) * largest_magnitude(rhs.data());
            bound(terms, largest)
        }
        // Every accumulating operation is handled above; anything else is misclassified.
        other => unreachable!("{} is not an accumulating operation", other.name()),
    }
}

/// Largest absolute value present, ignoring anything not finite.
///
/// A non-finite input would make the bound meaningless rather than merely large, so it
/// is excluded; such cases are judged by the NaN and infinity rules instead.
fn largest_magnitude(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| v.abs() as f64)
        .filter(|v| v.is_finite())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{FLEX_NAME, LIBTORCH_NAME, WGPU_NAME};
    use crate::input::{BinaryOp, ReduceOp, TensorValue, UnaryOp};

    fn value(shape: &[usize], fill: f32) -> TensorValue {
        let count = shape.iter().product();
        TensorValue::new(shape.to_vec(), vec![fill; count])
    }

    fn tolerance_for(op: &TensorOp) -> Tolerance {
        // The CPU pair: these tests state what the *specification* requires of conforming
        // implementations, which is the bound before any hardware-specific relaxation.
        //
        // **Named by constant, not by literal.** These said `"burn-ndarray"` — a backend
        // removed at PHASE-7A — for three phases, and passed the whole time, because the
        // policy only inspects a name to decide whether it is a GPU and `"burn-ndarray"`
        // correctly is not one. Right by accident, while asserting about something that no
        // longer existed. A constant cannot go stale that way.
        TensorTolerancePolicy.tolerance_for(op, (FLEX_NAME, LIBTORCH_NAME))
    }

    fn gpu_tolerance(op: &TensorOp) -> Tolerance {
        TensorTolerancePolicy.tolerance_for(op, (FLEX_NAME, WGPU_NAME))
    }

    /// **The filed bug must still be reportable after the vacuity rule.**
    ///
    /// `matmul` over `1e30` terms derives an enormous absolute bound — correctly, since such
    /// products genuinely cancel to anything — and 96% of its cases are now skipped rather
    /// than falsely passed. That is the honest outcome for a *numeric* comparison.
    ///
    /// But burn#5284, the one issue filed upstream, lives in exactly that region: `inf` on
    /// one backend against `NaN` on another. If the vacuity rule swallowed it, the fix would
    /// have blinded the tool to its own best result.
    ///
    /// It does not, and the reason is structural: `inf` versus `NaN` is settled before any
    /// tolerance is consulted. This test exists so that stays true.
    #[test]
    fn a_vacuous_bound_does_not_hide_the_inf_versus_nan_class() {
        use diff_fuzzer_core::{Agreement, ApproxEq};

        let overflowing = TensorOp::matmul(
            TensorValue::new(vec![1, 2], vec![1e30, -1e30]),
            TensorValue::new(vec![2, 1], vec![1e30, 1e30]),
        );

        // The bound really is vacuous for this case — that is the premise.
        let tolerance = tolerance_for(&overflowing);
        assert!(
            tolerance.is_vacuous(),
            "premise failed: this case was expected to derive an unusable bound, got {tolerance:?}"
        );

        // And the disagreement survives it anyway, because it is not numeric.
        let inf = crate::CanonicalTensor {
            shape: vec![1, 1],
            dtype: "F32".to_string(),
            values: vec![f32::INFINITY],
        };
        let nan = crate::CanonicalTensor {
            shape: vec![1, 1],
            dtype: "F32".to_string(),
            values: vec![f32::NAN],
        };

        assert!(
            !matches!(inf.approx_compare(&nan, tolerance), Agreement::Agree(_)),
            "inf versus NaN was absorbed by a vacuous bound"
        );
    }

    /// **The bound must come from the specification, not the measurement.** Pinned so that
    /// a future adjustment has to be deliberate, and so nobody quietly widens it to make a
    /// campaign quieter.
    ///
    /// Metal Shading Language Specification (2026-06-04) Table 8.2, fast math enabled:
    /// `x / y` → `<= 2.5 ulp`. See `SPECS.md` §4.1.
    #[test]
    fn division_against_the_gpu_gets_exactly_the_bound_metal_permits() {
        let op = TensorOp::binary(BinaryOp::Div, value(&[4], 1.0), value(&[4], 2.0));
        let tolerance = gpu_tolerance(&op);

        assert_eq!(
            tolerance.rtol,
            2.5 * EPSILON,
            "Metal Table 8.2: x / y <= 2.5 ulp"
        );
        // The absolute term is the separate subnormal allowance from §8.1, not part of the
        // ULP bound — two permissions from two clauses, deliberately kept distinct.
        assert_eq!(tolerance.atol, METAL_SUBNORMAL_FLOOR);
    }

    /// `sqrt` is the one composed bound: Table 8.2 gives it no figure, stating it is
    /// "Implemented as `x * rsqrt(x)`" with `rsqrt <= 2 ulp`, and the correctly-rounded
    /// multiply adds at most half a ULP.
    #[test]
    fn sqrt_against_the_gpu_gets_the_composed_bound() {
        let op = TensorOp::unary(UnaryOp::Sqrt, value(&[4], 4.0));
        assert_eq!(gpu_tolerance(&op).rtol, 3.0 * EPSILON);
    }

    /// **The relaxation applies only where the specification permits it.** Metal lists
    /// `x + y`, `x - y`, `x * y` as *correctly rounded* and `fabs` at `0 ulp`, and
    /// measurement found them bit-identical on 200/200 cases — so exactness holds even
    /// against the GPU. Widening the whole class would have been the easy, wrong move.
    #[test]
    fn the_gpu_relaxation_does_not_leak_to_operations_metal_rounds_correctly() {
        for op in [
            TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::binary(BinaryOp::Sub, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::binary(BinaryOp::Mul, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::unary(UnaryOp::Neg, value(&[4], 1.0)),
            TensorOp::unary(UnaryOp::Abs, value(&[4], 1.0)),
        ] {
            assert_eq!(
                gpu_tolerance(&op).rtol,
                0.0,
                "Metal rounds this correctly; a GPU pair earns no *relative* slack: {op:?}"
            );
        }
    }

    /// **And it applies only to pairs involving the GPU.** A CPU-versus-CPU division is
    /// still held to IEEE-754's correctly-rounded requirement; loosening it would give away
    /// sensitivity on hardware that has no excuse for the difference.
    #[test]
    fn two_cpu_backends_get_no_gpu_slack() {
        let op = TensorOp::binary(BinaryOp::Div, value(&[4], 1.0), value(&[4], 2.0));

        assert_eq!(
            TensorTolerancePolicy.tolerance_for(&op, (FLEX_NAME, LIBTORCH_NAME)),
            Tolerance::EXACT
        );
    }

    /// **The subnormal floor is bounded exactly by what the specification permits.** Metal
    /// §8.1 licenses flushing *denormals*, so the absolute allowance is exactly
    /// `f32::MIN_POSITIVE` — every permitted flush, and nothing above it.
    #[test]
    fn the_gpu_gets_an_absolute_floor_at_exactly_the_smallest_normal() {
        let op = TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0));
        let tolerance = gpu_tolerance(&op);

        assert_eq!(tolerance.atol, f32::MIN_POSITIVE as f64);
        assert_eq!(tolerance.rtol, 0.0, "add is still correctly rounded");
    }

    /// A flushed subnormal must be absorbed; a difference one step above must not. This is
    /// the boundary the derivation claims, asserted from both sides.
    #[test]
    fn the_floor_absorbs_a_flushed_subnormal_and_nothing_larger() {
        let op = TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0));
        let atol = gpu_tolerance(&op).atol;

        let largest_subnormal = f32::from_bits(0x007f_ffff) as f64;
        assert!(atol > largest_subnormal, "a flushed denormal is permitted");

        let smallest_normal = f32::MIN_POSITIVE as f64;
        assert!(
            atol <= smallest_normal,
            "a normal value vanishing is not permitted and must still be reported"
        );
    }

    /// CPU pairs get no such floor — nothing in IEEE-754 licenses a CPU to discard a
    /// subnormal, and granting it anyway would give away sensitivity for free.
    #[test]
    fn cpu_pairs_get_no_subnormal_floor() {
        let op = TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0));

        assert_eq!(
            TensorTolerancePolicy.tolerance_for(&op, (FLEX_NAME, LIBTORCH_NAME)),
            Tolerance::EXACT
        );
    }

    /// **The licensed difference fires only where the specification licenses it.**
    #[test]
    fn a_subnormal_input_against_the_gpu_is_licensed() {
        let op = TensorOp::unary(UnaryOp::Sqrt, value(&[2], 1e-45));

        let licensed = TensorTolerancePolicy.known_legal(&op, (FLEX_NAME, WGPU_NAME));
        let (class, detail) = licensed.expect("Metal §8.1 permits flushing a denormal input");

        assert!(class.contains("subnormal"));
        assert!(
            detail.contains("§8.1"),
            "the licence must cite its clause, since it discards evidence: {detail}"
        );
    }

    /// **Nothing licenses a CPU to discard a subnormal**, so the same case is still judged.
    /// Getting this wrong would silently stop testing two conforming implementations
    /// against each other.
    #[test]
    fn the_same_case_between_two_cpus_is_not_licensed() {
        let op = TensorOp::unary(UnaryOp::Sqrt, value(&[2], 1e-45));

        assert!(
            TensorTolerancePolicy
                .known_legal(&op, (FLEX_NAME, LIBTORCH_NAME))
                .is_none()
        );
    }

    /// The licence is confined to the condition that earns it — not the operation, and not
    /// the class. A GPU comparison on ordinary values is judged normally.
    #[test]
    fn ordinary_values_against_the_gpu_are_still_judged() {
        let op = TensorOp::unary(UnaryOp::Sqrt, value(&[2], 4.0));

        assert!(
            TensorTolerancePolicy
                .known_legal(&op, (FLEX_NAME, WGPU_NAME))
                .is_none()
        );
    }

    /// A zero is not a subnormal, and treating it as one would license a large share of
    /// every campaign for no reason.
    #[test]
    fn zero_is_not_a_subnormal() {
        let op = TensorOp::unary(UnaryOp::Sqrt, value(&[2], 0.0));

        assert!(
            TensorTolerancePolicy
                .known_legal(&op, (FLEX_NAME, WGPU_NAME))
                .is_none()
        );
    }

    /// The derived bounds must cover what was actually measured, with margin. If a future
    /// device exceeded them, this is where it would show — and that would be a finding
    /// about the device, not a reason to widen the number.
    #[test]
    fn the_derived_bounds_cover_the_measured_error_with_margin() {
        let measured_div = EPSILON; // 1 ULP, `examples/gpu_numerics.rs`
        let measured_sqrt = 2.0 * EPSILON; // 2 ULP

        let div = gpu_tolerance(&TensorOp::binary(
            BinaryOp::Div,
            value(&[4], 1.0),
            value(&[4], 2.0),
        ));
        let sqrt = gpu_tolerance(&TensorOp::unary(UnaryOp::Sqrt, value(&[4], 4.0)));

        assert!(div.rtol >= 2.0 * measured_div, "div margin too thin");
        assert!(sqrt.rtol >= 1.4 * measured_sqrt, "sqrt margin too thin");
    }

    #[test]
    fn correctly_rounded_operations_get_no_slack() {
        for op in [
            TensorOp::binary(BinaryOp::Add, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::binary(BinaryOp::Div, value(&[4], 1.0), value(&[4], 2.0)),
            TensorOp::unary(UnaryOp::Sqrt, value(&[4], 4.0)),
            TensorOp::unary(UnaryOp::Neg, value(&[4], 1.0)),
        ] {
            assert_eq!(
                tolerance_for(&op),
                Tolerance::EXACT,
                "{} should be exact",
                op.name()
            );
        }
    }

    #[test]
    fn exp_is_allowed_at_least_two_rounding_steps() {
        let tolerance = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1.0)));

        // **`atol` is no longer zero, and that is the point.** It was, and a subnormal result
        // — `exp(-100)` is `3.7e-44` — then had to be judged by a relative bound, where a few
        // significant bits is all the format offers. The floor is the smallest normal, so it
        // costs no sensitivity for any result above that.
        assert_eq!(tolerance.atol, f32::MIN_POSITIVE as f64);

        // Comfortably above the measured ceiling of one unit in the last place, and still
        // tight where a relative bound is the right instrument.
        assert!(tolerance.rtol > f32::EPSILON as f64);
        assert!(tolerance.rtol < 1e-6);
    }

    /// The fix for the 235 false positives found at wide bounds: `exp`'s allowance must
    /// grow with the argument, because that is what drives its error. A fixed constant
    /// is only valid over the range of arguments it was measured on.
    #[test]
    fn exp_tolerance_scales_with_argument_magnitude() {
        // **Both arguments below saturation.** The scaling claim holds where the condition
        // number is meaningful; past `exp`'s overflow point the result is exactly `inf` on
        // every backend and there is nothing left to scale. This compared 1 against 1000,
        // which now sits beyond the cap and made the test assert growth in a region where
        // growth would be fictitious.
        let small = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1.0)));
        let large = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 80.0)));

        assert!(large.rtol > small.rtol * 20.0, "{large:?} vs {small:?}");
    }

    /// **The bound stops growing once `exp` has saturated**, which is what keeps it usable.
    ///
    /// Uncapped, an argument of `1e30` — injected routinely by the special-value table —
    /// produced `rtol = 2.4e23`, a bound nothing could fail. 81% of `exp` cases carried one.
    #[test]
    fn the_exp_bound_stops_growing_past_saturation() {
        let saturated = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1e30)));
        let at_cap = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 200.0)));

        assert_eq!(saturated.rtol, at_cap.rtol, "the cap is not binding");
        assert!(
            !saturated.is_vacuous(),
            "an argument of 1e30 still yields an unusable bound: {saturated:?}"
        );
    }

    /// The derived bound must cover what was actually measured at wide bounds, with
    /// margin. If this fails, either the model is wrong or something is happening that
    /// is not rounding.
    #[test]
    fn the_exp_bound_covers_the_measured_worst_case_at_large_arguments() {
        let tolerance = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[4], 1000.0)));

        // **Measured worst relative error: 1.633e-4** — and it is *not* covered by the
        // relative term, which is the discovery that reshaped this bound.
        //
        // No argument can produce that through the condition number: `exp` overflows above
        // ~88.7, so the amplification term cannot legally reach far enough. Those elements
        // are **subnormal results** — `exp(-100)` is `3.7e-44` — where relative precision
        // degrades to a few bits regardless of implementation. The absolute floor covers
        // them; a wider relative bound would have been the wrong instrument and would have
        // cost sensitivity everywhere else.
        //
        // Verified end to end by `examples/exp_cap_check.rs`: 0 of 2,603 cases exceed the
        // full rule, worst at 0.5% of the allowance.
        let measured_worst_value = 3.7e-44f64; // a subnormal result of that shape
        let allowed = tolerance.atol + tolerance.rtol * measured_worst_value;
        assert!(
            allowed > measured_worst_value,
            "the floor does not cover a subnormal result: allowance {allowed:e}"
        );

        // And the relative term stays tight where it does apply.
        assert!(tolerance.rtol < 1e-4, "{tolerance:?}");
    }

    /// The argument's magnitude drives the allowance, not the tensor's size — twice as
    /// many values of the same magnitude are no harder to evaluate.
    #[test]
    fn exp_tolerance_ignores_how_many_values_there_are() {
        let few = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[2], 5.0)));
        let many = tolerance_for(&TensorOp::unary(UnaryOp::Exp, value(&[64], 5.0)));

        assert_eq!(few.rtol, many.rtol);
    }

    /// The derived bound must cover what was actually measured, with margin. If this
    /// ever fails, either the derivation is wrong or something is happening that is not
    /// rounding — both worth stopping for.
    #[test]
    fn the_derived_bound_covers_the_measured_worst_case_for_sum() {
        // Worst case within the generator's limits: eight terms of magnitude ten.
        let op = TensorOp::reduce(ReduceOp::Sum, value(&[8], 10.0), 0);
        let tolerance = tolerance_for(&op);

        // Measured worst absolute error for `sum` across 20,000 cases.
        let measured_worst = 7.63e-6;
        assert!(
            tolerance.atol > measured_worst,
            "derived atol {:e} does not cover measured {:e}",
            tolerance.atol,
            measured_worst
        );
        // ... and is not absurdly loose either. Ten times the observed worst is margin;
        // ten thousand times would be a licence to miss real bugs.
        assert!(tolerance.atol < measured_worst * 1_000.0);
    }

    #[test]
    fn the_derived_bound_covers_the_measured_worst_case_for_matmul() {
        let op = TensorOp::matmul(value(&[8, 8], 10.0), value(&[8, 8], 10.0));
        let tolerance = tolerance_for(&op);

        let measured_worst = 3.05e-5;
        assert!(
            tolerance.atol > measured_worst,
            "derived atol {:e} does not cover measured {:e}",
            tolerance.atol,
            measured_worst
        );
        assert!(tolerance.atol < measured_worst * 1_000.0);
    }

    /// The point of computing per case rather than from a global worst case: a small
    /// input must not inherit the tolerance a large one needs.
    #[test]
    fn smaller_inputs_get_a_tighter_tolerance() {
        let small = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[2], 1.0), 0));
        let large = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[8], 10.0), 0));

        assert!(
            small.atol < large.atol,
            "small {:e} vs large {:e}",
            small.atol,
            large.atol
        );
        assert!(small.rtol < large.rtol);
    }

    /// Values, not just shapes, must move the bound — the error depends on the
    /// magnitude of what is being added.
    #[test]
    fn larger_values_get_a_looser_absolute_tolerance() {
        let modest = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[4], 1.0), 0));
        let big = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, value(&[4], 1000.0), 0));

        assert!(big.atol > modest.atol);
        // The relative component depends only on how many terms are summed, so it is
        // unchanged by their size.
        assert_eq!(big.rtol, modest.rtol);
    }

    /// Reducing a different axis sums a different number of terms, so the tolerance
    /// must follow the axis rather than the tensor's total size.
    #[test]
    fn the_tolerance_follows_the_reduced_axis() {
        let arg = value(&[2, 8], 1.0);
        let short = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, arg.clone(), 0));
        let long = tolerance_for(&TensorOp::reduce(ReduceOp::Sum, arg, 1));

        assert!(short.atol < long.atol, "axis 0 sums 2, axis 1 sums 8");
    }

    #[test]
    fn every_operation_class_is_reachable() {
        assert_eq!(
            TensorOp::unary(UnaryOp::Exp, value(&[2], 1.0)).class(),
            OpClass::Approximated
        );
        assert_eq!(
            TensorOp::binary(BinaryOp::Mul, value(&[2], 1.0), value(&[2], 1.0)).class(),
            OpClass::CorrectlyRounded
        );
        assert_eq!(
            TensorOp::matmul(value(&[2, 2], 1.0), value(&[2, 2], 1.0)).class(),
            OpClass::Accumulating
        );
    }
}
