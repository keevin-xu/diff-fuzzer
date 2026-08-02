//! Deciding whether two floating-point results are close enough.
//!
//! Exact equality is the wrong question to ask of floating-point arithmetic. Two
//! correct implementations routinely produce different final bits, because addition is
//! not associative — `(a + b) + c` and `a + (b + c)` genuinely differ — so any two
//! systems that accumulate in different orders will disagree slightly on any sum.
//! Measured on this project's own operations: seven elementwise operations agree
//! bit-for-bit every time, while summation, matrix multiplication and `exp` disagree
//! on 16% to 73% of cases. None of that is a bug.
//!
//! So the question becomes "close enough", and the honest difficulty is that
//! **loosening the threshold to silence noise also hides real bugs**. A tolerance wide
//! enough to make every complaint go away makes the tool useless while looking like it
//! is working. The threshold therefore has to be argued for, not tuned until green.

/// How much difference is acceptable.
///
/// Two components, because error comes in two flavours. A *relative* tolerance handles
/// large numbers, where a difference of 0.001 between values around a million is
/// nothing. An *absolute* tolerance handles values near zero, where relative error is
/// meaningless — the relative difference between `1e-30` and `2e-30` is 100%, yet both
/// are indistinguishable from zero for any practical purpose.
///
/// Using only one is the classic mistake. Relative alone reports noise around zero as
/// catastrophic; absolute alone lets genuinely wrong large numbers through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Scaled by the magnitude of the values being compared.
    pub rtol: f64,
    /// A flat floor, dominating when values are near zero.
    pub atol: f64,
}

impl Tolerance {
    pub const fn new(rtol: f64, atol: f64) -> Self {
        Self { rtol, atol }
    }

    /// Demands bit-for-bit equality. Useful for the operations measured to produce it,
    /// and for tests that need to prove a comparison is not silently permissive.
    pub const EXACT: Self = Self::new(0.0, 0.0);

    /// Do these two values agree?
    ///
    /// The rule is `|a - b| <= atol + rtol * max(|a|, |b|)`.
    ///
    /// Note the `max`. The familiar form of this test, from `numpy.allclose`, scales by
    /// the magnitude of the *second* argument alone — which makes it asymmetric, so
    /// `close(a, b)` and `close(b, a)` can disagree. That is defensible when comparing
    /// a result against a known-good reference, since the reference is the meaningful
    /// scale. Here neither side is a reference: two backends are being compared, and
    /// which one is passed first is an accident of how the list was ordered. A verdict
    /// that changed with argument order would be indefensible in a bug report, so the
    /// larger magnitude sets the scale and the comparison is symmetric.
    pub fn agree(&self, a: f32, b: f32) -> bool {
        match Special::classify(a, b) {
            Some(special) => special.agrees(),
            None => self.finite_agree(a, b),
        }
    }

    /// Compare two values already known to be finite.
    fn finite_agree(&self, a: f32, b: f32) -> bool {
        // Widened to f64 for the arithmetic. The difference between two nearly equal
        // f32 values loses precision when computed in f32, which would make the error
        // measurement itself unreliable — a poor property for the number a bug report
        // quotes.
        let (a, b) = (a as f64, b as f64);
        let difference = (a - b).abs();
        difference <= self.atol + self.rtol * a.abs().max(b.abs())
    }
}

/// How a pair of values was judged when at least one of them was not finite.
///
/// Broken out as a named type rather than left as branches inside the comparison,
/// because these are **policy decisions, not arithmetic**, and policy must be auditable.
/// Naming each case lets the comparison count how often each was taken — and that
/// matters, because two of them are effectively *exclusions*: a pair where both sides
/// produced `NaN` is recorded as agreement without any numeric comparison having
/// happened. Exclusions that nobody counts are how a tool quietly stops testing
/// anything while continuing to report success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Special {
    /// Both undefined. **Treated as agreement**: each system was asked something with no
    /// answer and both said so, which is consistent behaviour.
    ///
    /// Note this is deliberately *not* what `==` does — `NaN != NaN` — so relying on
    /// equality would report two backends correctly producing `NaN` as a disagreement.
    ///
    /// It is also the weakest form of agreement there is, and worth counting: a result
    /// that is entirely `NaN` on both sides has told us nothing about either
    /// implementation.
    BothUndefined,
    /// One produced a number, the other did not. **A real disagreement** — they differ
    /// about whether an answer exists at all, which is more fundamental than differing
    /// about its value.
    OneUndefined,
    /// Both overflowed the same way. **Agreement**, and equally uninformative: two
    /// systems agreeing that a result is beyond representation says little about their
    /// arithmetic.
    SameInfinity,
    /// Infinities of opposite sign, or an infinity against a finite number. **A real
    /// disagreement that no tolerance may absorb** — and one the arithmetic could not
    /// judge anyway, since `inf - inf` is `NaN`.
    ConflictingInfinity,
}

impl Special {
    /// Classify a pair, or `None` if both are finite and ordinary comparison applies.
    pub fn classify(a: f32, b: f32) -> Option<Self> {
        match (a.is_nan(), b.is_nan()) {
            (true, true) => return Some(Special::BothUndefined),
            (true, false) | (false, true) => return Some(Special::OneUndefined),
            (false, false) => {}
        }

        if a.is_infinite() || b.is_infinite() {
            // Equality here is exact and correct: it requires the same sign as well as
            // both being infinite.
            return Some(if a == b {
                Special::SameInfinity
            } else {
                Special::ConflictingInfinity
            });
        }

        None
    }

    /// Whether this outcome counts as the two systems agreeing.
    pub fn agrees(self) -> bool {
        matches!(self, Special::BothUndefined | Special::SameInfinity)
    }

    /// Whether this outcome was reached *without comparing any arithmetic*.
    ///
    /// Both forms of agreement here are vacuous: nothing was learned about either
    /// implementation's numerics. A result made entirely of these has been checked in
    /// name only, which is worth being able to detect.
    pub fn is_vacuous(self) -> bool {
        self.agrees()
    }
}

/// A result that can be compared against another of its kind within a tolerance.
///
/// The engine knows nothing about tensors, rows, or whatever else a result might be —
/// only that two of them can be held up against each other. Implementing this is how a
/// domain says "here is what comparing my results means".
pub trait ApproxEq {
    fn approx_compare(&self, other: &Self, tolerance: Tolerance) -> Agreement;
}

/// The outcome of comparing two results.
///
/// Structural disagreement is a separate variant rather than an extreme numeric one,
/// and that separation is the point. Two results of different shapes do not differ *by
/// an amount* — they differ about what the operation produced. No tolerance should ever
/// absorb that, however loose, so it must not be expressible on the same scale.
#[derive(Debug, Clone, PartialEq)]
pub enum Agreement {
    /// Same structure, values within tolerance. Carries the comparison anyway, so a
    /// result that only just squeaked through is still visible.
    Agree(Comparison),
    /// Same structure, values outside tolerance.
    Disagree(Comparison),
    /// The results are not comparable — different shapes, sizes, or types.
    Structural { reason: String },
}

/// How much tolerance to allow for a given test case.
///
/// A single number cannot serve every operation. Summing a thousand values accumulates
/// far more rounding error than negating one, so holding both to the same standard
/// means either flagging correct sums or missing wrong negations. The engine defines
/// the question; only the domain knows enough to answer it, since only the domain knows
/// what kind of operation a case represents.
pub trait TolerancePolicy<In> {
    fn tolerance_for(&self, input: &In) -> Tolerance;
}

/// The same tolerance regardless of the case.
///
/// Useful for tests, and as a baseline to measure a smarter policy against.
#[derive(Debug, Clone, Copy)]
pub struct FixedTolerance(pub Tolerance);

impl<In> TolerancePolicy<In> for FixedTolerance {
    fn tolerance_for(&self, _input: &In) -> Tolerance {
        self.0
    }
}

/// Comparing plain lists of numbers, which is what the engine's own tests use.
impl ApproxEq for Vec<f32> {
    fn approx_compare(&self, other: &Self, tolerance: Tolerance) -> Agreement {
        if self.len() != other.len() {
            return Agreement::Structural {
                reason: format!("lengths differ: {} vs {}", self.len(), other.len()),
            };
        }

        let comparison = compare(self, other, tolerance);
        if comparison.agrees() {
            Agreement::Agree(comparison)
        } else {
            Agreement::Disagree(comparison)
        }
    }
}

/// The worst single disagreement found.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mismatch {
    /// Position in the flattened values, so the offending element can be located.
    pub index: usize,
    pub left: f32,
    pub right: f32,
    /// `|left - right|`.
    pub absolute_error: f64,
    /// `|left - right| / max(|left|, |right|)`, or zero when both are zero.
    pub relative_error: f64,
}

/// What comparing two sets of values found.
///
/// Carries the magnitude of the difference, not merely whether there was one. "These
/// disagree" is not actionable; "these disagree by 3e-7 relative, at element 41 of 64"
/// is the difference between a report a maintainer can act on and one they cannot.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// How many elements fell outside the tolerance.
    pub mismatches: usize,
    pub total: usize,
    /// Largest absolute difference anywhere, whether or not it exceeded the tolerance.
    pub max_absolute_error: f64,
    /// Largest relative difference anywhere, likewise.
    pub max_relative_error: f64,
    /// The single worst element that exceeded the tolerance.
    pub worst: Option<Mismatch>,
    /// How many elements agreed **without any arithmetic being compared** — both `NaN`,
    /// or both the same infinity.
    ///
    /// Counted rather than silently passed. These are exclusions wearing the costume of
    /// agreement, and a comparison made mostly of them has verified very little.
    pub vacuous_agreements: usize,
}

impl Comparison {
    pub fn agrees(&self) -> bool {
        self.mismatches == 0
    }

    /// Did this comparison actually examine any arithmetic?
    ///
    /// False when every element was excluded by a special-value rule — a result that is
    /// entirely `NaN` or entirely infinite on both sides. Such a case *agrees*, but the
    /// agreement is empty: neither implementation was tested. Distinguishing it from a
    /// genuine pass is what stops a run of undefined results from being counted as
    /// evidence that anything works.
    pub fn examined_any_arithmetic(&self) -> bool {
        self.total > self.vacuous_agreements
    }
}

/// Compare two equally sized sets of values elementwise.
///
/// # Panics
///
/// If the two slices differ in length. Length is a structural property settled before
/// any arithmetic — two results of different sizes do not disagree *numerically*, they
/// disagree about what the operation produced — so callers are expected to have
/// established it, and reaching here with a mismatch is a defect in the caller.
pub fn compare(left: &[f32], right: &[f32], tolerance: Tolerance) -> Comparison {
    assert_eq!(
        left.len(),
        right.len(),
        "compare expects equally sized results; shape must be checked before values"
    );

    let mut comparison = Comparison {
        mismatches: 0,
        total: left.len(),
        max_absolute_error: 0.0,
        max_relative_error: 0.0,
        worst: None,
        vacuous_agreements: 0,
    };

    // Tracked separately from the maxima above: those describe the whole result, while
    // this picks the worst element that actually *failed*, which is the one to report.
    let mut worst_failing_error = f64::NEG_INFINITY;

    for (index, (&a, &b)) in left.iter().zip(right).enumerate() {
        // Record when an element was settled by a special-value rule rather than by
        // comparing numbers, so the vacuous portion of a result stays visible.
        if let Some(special) = Special::classify(a, b)
            && special.is_vacuous()
        {
            comparison.vacuous_agreements += 1;
        }

        let (absolute_error, relative_error) = errors(a, b);

        // Non-finite errors would poison the maxima, so they are excluded from the
        // summary statistics while still being judged by `agree` below.
        if absolute_error.is_finite() {
            comparison.max_absolute_error = comparison.max_absolute_error.max(absolute_error);
        }
        if relative_error.is_finite() {
            comparison.max_relative_error = comparison.max_relative_error.max(relative_error);
        }

        if !tolerance.agree(a, b) {
            comparison.mismatches += 1;

            // A non-finite error still needs to be reportable, so rank by absolute
            // error treating non-finite as maximally bad.
            let rank = if absolute_error.is_finite() {
                absolute_error
            } else {
                f64::MAX
            };
            if rank > worst_failing_error {
                worst_failing_error = rank;
                comparison.worst = Some(Mismatch {
                    index,
                    left: a,
                    right: b,
                    absolute_error,
                    relative_error,
                });
            }
        }
    }

    comparison
}

/// Absolute and relative difference between two values.
///
/// Both zero when the values are identical, including when both are NaN or matching
/// infinities — those agree, so describing them as infinitely far apart would be
/// misleading in a report.
fn errors(a: f32, b: f32) -> (f64, f64) {
    if (a.is_nan() && b.is_nan()) || a == b {
        return (0.0, 0.0);
    }

    let (wide_a, wide_b) = (a as f64, b as f64);
    let absolute = (wide_a - wide_b).abs();
    let scale = wide_a.abs().max(wide_b.abs());

    let relative = if scale == 0.0 {
        // Only reachable if one value is a zero of one sign and the other of the
        // opposite sign, since `a == b` was handled above. They are equal numerically.
        0.0
    } else {
        absolute / scale
    };

    (absolute, relative)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOOSE: Tolerance = Tolerance::new(1e-5, 1e-8);

    #[test]
    fn identical_values_agree_at_any_tolerance() {
        assert!(Tolerance::EXACT.agree(1.5, 1.5));
        assert!(LOOSE.agree(1.5, 1.5));
    }

    #[test]
    fn a_tiny_difference_is_within_a_loose_tolerance_but_not_an_exact_one() {
        let (a, b) = (1.0_f32, 1.000_001_f32);
        assert!(LOOSE.agree(a, b));
        assert!(!Tolerance::EXACT.agree(a, b));
    }

    #[test]
    fn a_large_difference_is_outside_tolerance() {
        assert!(!LOOSE.agree(1.0, 1.1));
    }

    /// The relative term must scale with magnitude: the same *proportional* error is
    /// acceptable at any size, which is the entire point of having it.
    #[test]
    fn relative_tolerance_scales_with_magnitude() {
        assert!(LOOSE.agree(1_000_000.0, 1_000_001.0));
        // The same absolute gap of 1.0 is enormous for small values.
        assert!(!LOOSE.agree(1.0, 2.0));
    }

    /// The absolute term must rescue values near zero, where relative error explodes
    /// while the numbers remain indistinguishable in practice.
    #[test]
    fn absolute_tolerance_covers_values_near_zero() {
        let tolerance = Tolerance::new(1e-5, 1e-8);
        // A relative difference of 100%, but both are effectively zero.
        assert!(tolerance.agree(1e-30, 2e-30));
        // With no absolute floor, that same pair would be reported as wrong.
        assert!(!Tolerance::new(1e-5, 0.0).agree(1e-30, 2e-30));
    }

    /// The property that motivated departing from the usual formula: swapping the
    /// arguments must never change the answer, since neither backend is a reference.
    #[test]
    fn comparison_is_symmetric() {
        let tolerance = Tolerance::new(0.1, 0.0);
        // Values chosen so the asymmetric form would disagree with itself: scaling by
        // the smaller magnitude fails while scaling by the larger passes.
        let (a, b) = (1.0_f32, 1.09_f32);
        assert_eq!(tolerance.agree(a, b), tolerance.agree(b, a));

        for (a, b) in [(0.0, 1e-9), (1e6, 1.000_01e6), (-3.0, -3.000_01)] {
            assert_eq!(
                tolerance.agree(a, b),
                tolerance.agree(b, a),
                "asymmetric on {a} vs {b}"
            );
        }
    }

    /// Every combination of special values, in one table.
    ///
    /// Table-driven on purpose: the point is *exhaustiveness*. A policy about undefined
    /// results is exactly the kind that gets one case wrong and stays wrong, because the
    /// wrong case is rare in practice and silent when it happens.
    #[test]
    fn every_special_value_combination_is_classified() {
        let nan = f32::NAN;
        let pos = f32::INFINITY;
        let neg = f32::NEG_INFINITY;

        let cases: [(f32, f32, Option<Special>); 14] = [
            // Both undefined — agreement, and vacuous.
            (nan, nan, Some(Special::BothUndefined)),
            // One undefined — a real disagreement, whichever side it is on.
            (nan, 1.0, Some(Special::OneUndefined)),
            (1.0, nan, Some(Special::OneUndefined)),
            (nan, 0.0, Some(Special::OneUndefined)),
            // Undefined against infinite is still "one has no answer". The NaN rule is
            // checked first, deliberately: not-a-number is a stronger statement than
            // out-of-range.
            (nan, pos, Some(Special::OneUndefined)),
            (neg, nan, Some(Special::OneUndefined)),
            // Same overflow — agreement, and equally vacuous.
            (pos, pos, Some(Special::SameInfinity)),
            (neg, neg, Some(Special::SameInfinity)),
            // Opposite overflow, or overflow against a number — real disagreements.
            (pos, neg, Some(Special::ConflictingInfinity)),
            (neg, pos, Some(Special::ConflictingInfinity)),
            (pos, 1.0, Some(Special::ConflictingInfinity)),
            (1.0, neg, Some(Special::ConflictingInfinity)),
            // Ordinary values are not special at all.
            (1.0, 1.0, None),
            (0.0, -0.0, None),
        ];

        for (a, b, expected) in cases {
            assert_eq!(
                Special::classify(a, b),
                expected,
                "classifying {a} against {b}"
            );
        }
    }

    /// Which classifications count as agreement, stated once so the policy can be read
    /// off in one place rather than inferred from the comparison's control flow.
    #[test]
    fn the_agreement_policy_is_explicit() {
        assert!(Special::BothUndefined.agrees());
        assert!(Special::SameInfinity.agrees());
        assert!(!Special::OneUndefined.agrees());
        assert!(!Special::ConflictingInfinity.agrees());

        // Both forms of agreement are reached without comparing arithmetic.
        assert!(Special::BothUndefined.is_vacuous());
        assert!(Special::SameInfinity.is_vacuous());
        assert!(!Special::OneUndefined.is_vacuous());
        assert!(!Special::ConflictingInfinity.is_vacuous());
    }

    /// No tolerance, however enormous, may absorb a disagreement about whether a result
    /// exists or is in range. These are not differences of degree.
    #[test]
    fn special_disagreements_survive_any_tolerance() {
        let enormous = Tolerance::new(1e30, 1e30);

        assert!(!enormous.agree(f32::NAN, 1.0));
        assert!(!enormous.agree(1.0, f32::NAN));
        assert!(!enormous.agree(f32::INFINITY, f32::NEG_INFINITY));
        assert!(!enormous.agree(f32::INFINITY, 1.0));
        assert!(!enormous.agree(f32::NAN, f32::INFINITY));
    }

    /// And no tolerance, however strict, may *break* an agreement between two systems
    /// that both correctly reported no answer.
    #[test]
    fn special_agreements_survive_exact_comparison() {
        assert!(Tolerance::EXACT.agree(f32::NAN, f32::NAN));
        assert!(Tolerance::EXACT.agree(f32::INFINITY, f32::INFINITY));
        assert!(Tolerance::EXACT.agree(f32::NEG_INFINITY, f32::NEG_INFINITY));
    }

    /// The auditability requirement: a result made entirely of undefined values
    /// *agrees*, but nothing was actually tested. That has to be distinguishable from a
    /// genuine pass, or a run where every case overflowed would look like success.
    #[test]
    fn an_entirely_undefined_result_agrees_but_examines_nothing() {
        let undefined = [f32::NAN, f32::NAN, f32::INFINITY];
        let comparison = compare(&undefined, &undefined, Tolerance::EXACT);

        assert!(comparison.agrees());
        assert_eq!(comparison.vacuous_agreements, 3);
        assert!(
            !comparison.examined_any_arithmetic(),
            "a result of nothing but undefined values verified nothing"
        );
    }

    #[test]
    fn a_partly_undefined_result_still_examines_the_rest() {
        let left = [f32::NAN, 1.0, 2.0];
        let right = [f32::NAN, 1.0, 2.0];
        let comparison = compare(&left, &right, Tolerance::EXACT);

        assert_eq!(comparison.vacuous_agreements, 1);
        assert!(comparison.examined_any_arithmetic());
    }

    #[test]
    fn an_ordinary_result_has_no_vacuous_agreements() {
        let comparison = compare(&[1.0, 2.0], &[1.0, 2.0], Tolerance::EXACT);
        assert_eq!(comparison.vacuous_agreements, 0);
        assert!(comparison.examined_any_arithmetic());
    }

    /// An empty result examines nothing either — there was nothing to examine. Worth
    /// pinning so the check is about *content*, not just about special values.
    #[test]
    fn an_empty_result_examines_nothing() {
        let comparison = compare(&[], &[], Tolerance::EXACT);
        assert!(comparison.agrees());
        assert!(!comparison.examined_any_arithmetic());
    }

    #[test]
    fn two_nans_agree() {
        assert!(Tolerance::EXACT.agree(f32::NAN, f32::NAN));
        assert!(LOOSE.agree(f32::NAN, f32::NAN));
    }

    #[test]
    fn a_nan_and_a_number_disagree() {
        assert!(!LOOSE.agree(f32::NAN, 1.0));
        assert!(!LOOSE.agree(1.0, f32::NAN));
    }

    #[test]
    fn infinities_must_match_including_sign() {
        assert!(LOOSE.agree(f32::INFINITY, f32::INFINITY));
        assert!(LOOSE.agree(f32::NEG_INFINITY, f32::NEG_INFINITY));
        assert!(!LOOSE.agree(f32::INFINITY, f32::NEG_INFINITY));
    }

    #[test]
    fn an_infinity_and_a_finite_number_disagree_at_any_tolerance() {
        let enormous = Tolerance::new(1e30, 1e30);
        assert!(!enormous.agree(f32::INFINITY, 1.0));
    }

    #[test]
    fn positive_and_negative_zero_agree() {
        assert!(Tolerance::EXACT.agree(0.0, -0.0));
    }

    /// The comparison is `<=`, so a difference landing exactly on the threshold agrees.
    /// Pinning which side of the boundary is inclusive matters: an off-by-one-bit change
    /// here would shift every borderline verdict in the project without failing anything
    /// else.
    #[test]
    fn a_difference_exactly_at_the_threshold_agrees() {
        // Purely absolute, so the arithmetic is exact and the test is not itself subject
        // to rounding: threshold is atol = 0.5, difference is exactly 0.5.
        let tolerance = Tolerance::new(0.0, 0.5);
        assert!(tolerance.agree(1.0, 1.5));
        // Just beyond it does not.
        assert!(!tolerance.agree(1.0, 1.5000001));
    }

    /// Under exact comparison, adjacent representable numbers must be reported as
    /// different — one unit in the last place is the smallest disagreement that exists,
    /// and exact comparison exists precisely to catch it.
    #[test]
    fn one_unit_in_the_last_place_is_caught_by_exact_comparison() {
        let value = 1.0_f32;
        let next = f32::from_bits(value.to_bits() + 1);

        assert_ne!(value, next);
        assert!(!Tolerance::EXACT.agree(value, next));
        // And the smallest tolerance that covers it is one epsilon.
        assert!(Tolerance::new(f32::EPSILON as f64, 0.0).agree(value, next));
    }

    /// Subnormals — numbers too small for the normal representation, which trade
    /// precision for range. They appear in practice: `exp` of a large negative argument
    /// lands here. Relative comparison degrades in this region, which is another reason
    /// an absolute floor is not optional.
    #[test]
    fn subnormal_values_are_handled() {
        let tiny = 4.5e-39_f32; // below f32::MIN_POSITIVE (~1.18e-38)
        let next = f32::from_bits(tiny.to_bits() + 1);

        assert!(
            tiny > 0.0,
            "the test value must actually be subnormal-range"
        );
        assert!(!Tolerance::EXACT.agree(tiny, next));
        // An absolute floor far below anything meaningful still covers them, because
        // the gap between adjacent subnormals is minuscule in absolute terms.
        assert!(Tolerance::new(0.0, 1e-40).agree(tiny, next));
    }

    /// Values of opposite sign. The scale is the larger magnitude, so this must not
    /// accidentally cancel into a small denominator.
    #[test]
    fn values_of_opposite_sign_are_compared_by_magnitude() {
        let tolerance = Tolerance::new(0.5, 0.0);
        // Difference is 2.0, larger magnitude is 1.0, so the threshold is 0.5.
        assert!(!tolerance.agree(1.0, -1.0));
        // Both near zero and both tiny: still a real difference relative to their size.
        assert!(!tolerance.agree(1e-10, -1e-10));
    }

    /// Two zeros are identical, and the relative error of identical values is zero
    /// rather than an undefined division.
    #[test]
    fn two_zeros_agree_with_no_undefined_arithmetic() {
        let comparison = compare(&[0.0, -0.0], &[0.0, 0.0], Tolerance::EXACT);
        assert!(comparison.agrees());
        assert_eq!(comparison.max_relative_error, 0.0);
        assert!(comparison.max_relative_error.is_finite());
    }

    /// Very large finite values must still be comparable — the relative term has to do
    /// the work there, since no sane absolute floor would.
    #[test]
    fn very_large_values_are_compared_relatively() {
        let huge = 3.0e38_f32; // near f32::MAX
        let slightly_more = huge * (1.0 + 1e-7);

        assert!(slightly_more.is_finite(), "test value must not overflow");
        assert!(Tolerance::new(1e-6, 0.0).agree(huge, slightly_more));
        assert!(!Tolerance::new(1e-9, 0.0).agree(huge, slightly_more));
    }

    // `ApproxEq` for `Vec<f32>` is what the engine's own tests compare through, so its
    // three outcomes are worth exercising directly rather than only via the oracle.

    #[test]
    fn approx_compare_reports_agreement() {
        let a = vec![1.0, 2.0];
        assert!(matches!(
            a.approx_compare(&a.clone(), Tolerance::EXACT),
            Agreement::Agree(_)
        ));
    }

    #[test]
    fn approx_compare_reports_disagreement_with_the_comparison() {
        let outcome = vec![1.0, 2.0].approx_compare(&vec![1.0, 9.0], Tolerance::EXACT);

        let Agreement::Disagree(comparison) = outcome else {
            panic!("expected disagreement, got {outcome:?}");
        };
        assert_eq!(comparison.mismatches, 1);
        assert_eq!(comparison.worst.expect("a worst element").index, 1);
    }

    /// Different lengths are structural, and no tolerance may absorb them.
    #[test]
    fn approx_compare_reports_a_length_difference_as_structural() {
        let outcome = vec![1.0].approx_compare(&vec![1.0, 2.0], Tolerance::new(1e30, 1e30));

        let Agreement::Structural { reason } = outcome else {
            panic!("a length difference was absorbed by tolerance: {outcome:?}");
        };
        assert!(reason.contains("lengths differ"), "{reason}");
    }

    #[test]
    fn comparing_equal_results_reports_no_mismatches() {
        let values = [1.0, 2.0, 3.0];
        let comparison = compare(&values, &values, Tolerance::EXACT);

        assert!(comparison.agrees());
        assert_eq!(comparison.mismatches, 0);
        assert_eq!(comparison.total, 3);
        assert_eq!(comparison.max_absolute_error, 0.0);
        assert!(comparison.worst.is_none());
    }

    #[test]
    fn comparing_reports_the_worst_element_and_its_position() {
        let left = [1.0, 5.0, 3.0];
        let right = [1.0, 5.5, 3.1];
        let comparison = compare(&left, &right, Tolerance::EXACT);

        assert_eq!(comparison.mismatches, 2);
        let worst = comparison.worst.expect("a mismatch was found");
        // Element 1 differs by 0.5, element 2 by 0.1 — the larger is reported.
        assert_eq!(worst.index, 1);
        assert!((worst.absolute_error - 0.5).abs() < 1e-6);
    }

    /// The maxima describe the whole result, including differences that stayed within
    /// tolerance. That matters for judging whether a threshold is close to the edge:
    /// a run agreeing everywhere with a maximum error just under the limit is a very
    /// different situation from one where errors are a thousand times smaller.
    #[test]
    fn maximum_errors_are_reported_even_when_everything_agrees() {
        let left = [1.0, 2.0];
        // A relative difference of about 1e-6, comfortably inside `LOOSE`'s 1e-5.
        // Sitting nearer the threshold makes the test depend on the exact f32
        // representation of the literal, which is a fragile thing to assert.
        let right = [1.000_001, 2.0];
        let comparison = compare(&left, &right, LOOSE);

        assert!(comparison.agrees());
        assert!(comparison.max_absolute_error > 0.0);
        assert!(comparison.max_relative_error > 0.0);
    }

    #[test]
    fn comparing_empty_results_agrees() {
        let comparison = compare(&[], &[], Tolerance::EXACT);
        assert!(comparison.agrees());
        assert_eq!(comparison.total, 0);
    }

    #[test]
    #[should_panic(expected = "equally sized")]
    fn comparing_different_lengths_is_a_caller_error() {
        compare(&[1.0], &[1.0, 2.0], Tolerance::EXACT);
    }

    #[test]
    fn nan_against_a_number_is_counted_as_a_mismatch() {
        let comparison = compare(&[f32::NAN], &[1.0], LOOSE);
        assert_eq!(comparison.mismatches, 1);
        assert!(comparison.worst.is_some());
    }

    #[test]
    fn matching_nans_are_not_counted_as_mismatches() {
        let comparison = compare(&[f32::NAN, 1.0], &[f32::NAN, 1.0], Tolerance::EXACT);
        assert!(comparison.agrees());
        assert_eq!(comparison.max_absolute_error, 0.0);
    }
}
