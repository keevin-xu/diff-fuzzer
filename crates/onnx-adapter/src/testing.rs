//! Implementations that are **wrong on purpose**.
//!
//! # Why this exists, and why it is built before any campaign
//!
//! A fuzzing campaign that finds nothing has two explanations, and they look identical from
//! the outside:
//!
//! - the implementations agree, or
//! - **the detector is broken and could never have fired.**
//!
//! Nothing in a clean run distinguishes them. This module is what does: it provides
//! implementations whose output is deliberately corrupted, so a test can assert that the
//! oracle *catches* them. A "found nothing" result is worth something only when paired with
//! evidence that something could have been found — and `docs/handoff/05-MEASUREMENT-AND-
//! CAMPAIGNS.md` makes fault injection the **first** of the four pre-campaign checks for
//! exactly this reason.
//!
//! Both prior domains had periods where the finding count was large and the true-positive
//! count was zero. One of them ran a six-hour campaign that found nothing in `softmax` —
//! not because the implementations agreed, but because the tolerance was so wide that
//! nothing could fail it, and the tool reported that as agreement.
//!
//! # The three kinds of wrongness, and why three
//!
//! | wrapper | breaks | catches an oracle that… |
//! |---|---|---|
//! | [`WrongValues`] | one element's value | ignores values, or compares with a bound that accepts anything |
//! | [`WrongShape`] | the output shape | compares values only, element by element, without checking shape |
//! | [`Panicking`] | the process | routes crashes into the skip path — **the domain's thesis** |
//!
//! One wrapper would not be enough. An oracle that compares values but ignores shape passes
//! a `WrongValues` test and fails silently on real shape bugs, and that is a plausible bug
//! to write.
//!
//! # The classification that fault injection must respect
//!
//! A corrupted run has **three** outcomes, not two:
//!
//! - **exercised** — the fault changed the normalized output, so the oracle *must* diverge;
//! - **inert** — the fault changed nothing, so agreement is **correct**, not a miss;
//! - **unrunnable** — the case did not produce a result to corrupt.
//!
//! The detection rate is computed over **exercised only**. Skipping that distinction is not
//! pedantry: with 44.4% of SQL results empty, dropping a row changed nothing, and the naive
//! rate read **55.7%** for an oracle that caught everything it was actually shown.
//!
//! Two further details, both learned the hard way: compare the **normalized** forms when
//! deciding whether the fault did anything, because a difference the normalizer legitimately
//! erases is one the oracle is right to ignore; and a **`Skipped`** verdict counts as a
//! miss, or a normalizer that declined every case would score perfect.

use diff_fuzzer_core::traits::{Implementation, RunError};

use crate::case::{OnnxCase, TensorData};
use crate::outcome::OnnxOutcome;

/// Wraps a real implementation and perturbs one output value.
///
/// The perturbation is a **single element**, not the whole tensor, because that is the
/// hardest case for an oracle to catch and therefore the one worth testing. An oracle that
/// only compares, say, the first element or a summary statistic would pass a
/// whole-tensor corruption and fail this.
pub struct WrongValues<I> {
    inner: I,
    name: String,
    /// Added to the element at [`Self::index`]. Must be large enough to survive any
    /// tolerance the oracle might eventually apply, and is not a small epsilon for that
    /// reason.
    delta: f32,
    index: usize,
}

impl<I> WrongValues<I> {
    pub fn new(inner: I, delta: f32) -> Self
    where
        I: Implementation,
    {
        Self {
            name: format!("{}-wrong-values", inner.name()),
            inner,
            delta,
            index: 0,
        }
    }

    /// Corrupt a different element — useful for checking that an oracle inspects the whole
    /// tensor rather than only its first element.
    pub fn at_index(mut self, index: usize) -> Self {
        self.index = index;
        self
    }
}

impl<I> Implementation for WrongValues<I>
where
    I: Implementation<In = OnnxCase, Out = OnnxOutcome>,
{
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        let outcome = self.inner.run(input)?;
        // Only an `Ok` result can be corrupted. Anything else passes through unchanged and
        // the case will be classified *inert* — which is the honest classification, not a
        // failure of the oracle.
        let OnnxOutcome::Ok(mut tensors) = outcome else {
            return Ok(outcome);
        };
        if let Some(tensor) = tensors.first_mut() {
            corrupt_element(&mut tensor.data, self.index, self.delta);
        }
        Ok(OnnxOutcome::Ok(tensors))
    }
}

/// Perturb one element, whatever type it is.
///
/// Each type needs its own notion of "different": adding a float delta to a boolean is not
/// a thing, and adding it to an integer would truncate to no change for any delta below 1.
/// Out-of-range indices are a no-op, which the caller sees as an **inert** fault rather
/// than a miss.
fn corrupt_element(data: &mut TensorData, index: usize, delta: f32) {
    match data {
        TensorData::F32(v) => {
            if let Some(x) = v.get_mut(index) {
                *x += delta;
            }
        }
        TensorData::F64(v) => {
            if let Some(x) = v.get_mut(index) {
                *x += f64::from(delta);
            }
        }
        // `wrapping_add` rather than `+`: a corruptor must never panic on overflow, or a
        // deliberate fault would look like a crash in the code under test.
        TensorData::I32(v) => {
            if let Some(x) = v.get_mut(index) {
                *x = x.wrapping_add(1);
            }
        }
        TensorData::I64(v) => {
            if let Some(x) = v.get_mut(index) {
                *x = x.wrapping_add(1);
            }
        }
        // Saturating, so corrupting a value already at its boundary is *inert* rather than
        // wrapping to the opposite extreme — which would be a corruption so large it proves
        // less than a small one. `classify_fault` reports inert honestly.
        TensorData::I8(v) => {
            if let Some(x) = v.get_mut(index) {
                *x = x.saturating_add(delta as i8);
            }
        }
        TensorData::U8(v) => {
            if let Some(x) = v.get_mut(index) {
                *x = x.saturating_add(delta as u8);
            }
        }
        TensorData::Bool(v) => {
            if let Some(x) = v.get_mut(index) {
                *x = !*x;
            }
        }
    }
}

/// Wraps a real implementation and reports a different shape for the same data.
///
/// Catches an oracle that walks values element by element without ever comparing `dims`.
/// The values are left untouched, so **only** the shape check can catch this.
pub struct WrongShape<I> {
    inner: I,
    name: String,
}

impl<I> WrongShape<I> {
    pub fn new(inner: I) -> Self
    where
        I: Implementation,
    {
        Self {
            name: format!("{}-wrong-shape", inner.name()),
            inner,
        }
    }
}

impl<I> Implementation for WrongShape<I>
where
    I: Implementation<In = OnnxCase, Out = OnnxOutcome>,
{
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        let OnnxOutcome::Ok(mut tensors) = self.inner.run(input)? else {
            return self.inner.run(input);
        };
        if let Some(tensor) = tensors.first_mut() {
            // Flatten to rank 1 while keeping the element count identical. A shape-blind
            // oracle sees exactly the same values in the same order and reports agreement.
            let total: i64 = tensor.dims.iter().product::<i64>().max(0);
            tensor.dims = vec![total];
        }
        Ok(OnnxOutcome::Ok(tensors))
    }
}

/// An implementation that panics.
///
/// This is the fault injection for the **domain's thesis**. `06-ORACLES-AND-LEGAL-
/// DIFFERENCES.md` §2 argues that a crash must be a divergence rather than a skip; without
/// this wrapper, a campaign finding no crashes would be indistinguishable from a crash path
/// that never runs.
pub struct Panicking {
    name: String,
    message: &'static str,
}

impl Default for Panicking {
    fn default() -> Self {
        Self::new()
    }
}

impl Panicking {
    pub fn new() -> Self {
        Self {
            name: "panicking".to_string(),
            message: "deliberate panic from testing.rs",
        }
    }
}

impl Implementation for Panicking {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        // Returned as a value rather than by actually unwinding. The real runtimes wrap
        // themselves in `catch_unwind`, and this stands in for what that produces — so the
        // test exercises the oracle's handling of a crash without filling test output with
        // panic backtraces, and without depending on panic-hook behaviour.
        //
        // That a real panic genuinely becomes `Crashed` is tested separately, in
        // `catch_unwind_turns_a_real_panic_into_a_value` below.
        Ok(OnnxOutcome::Crashed {
            detail: self.message.to_string(),
        })
    }
}

/// An implementation that declares everything unsupported.
///
/// The **negative control** for the crash oracle. `Unsupported` and `Crashed` must be
/// treated differently, and a test asserting only that crashes are caught would also pass
/// if the oracle flagged *every* non-answer — which would bury real findings under
/// unimplemented operators.
pub struct AlwaysUnsupported {
    name: String,
}

impl Default for AlwaysUnsupported {
    fn default() -> Self {
        Self::new()
    }
}

impl AlwaysUnsupported {
    pub fn new() -> Self {
        Self {
            name: "always-unsupported".to_string(),
        }
    }
}

impl Implementation for AlwaysUnsupported {
    type In = OnnxCase;
    type Out = OnnxOutcome;

    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, _input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
        Ok(OnnxOutcome::Unsupported {
            reason: "this implementation does nothing at all".to_string(),
        })
    }
}

/// How a fault-injected case turned out.
///
/// Three variants rather than two — see the module note. Computing a detection rate over
/// anything but [`Self::Exercised`] produces a number that understates the oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultClass {
    /// The fault changed the normalized output. The oracle **must** diverge.
    Exercised,
    /// The fault changed nothing. Agreement is **correct** here, not a miss.
    Inert,
    /// No result was produced to corrupt.
    Unrunnable,
}

/// Classify what an injected fault actually did.
///
/// Compares the **normalized** forms, not the raw outcomes: a difference the normalizer
/// legitimately erases is a difference the oracle is right to ignore, and counting it as
/// exercised would blame the oracle for the normalizer's correct behaviour.
pub fn classify_fault(clean: &OnnxOutcome, faulty: &OnnxOutcome) -> FaultClass {
    use crate::normalize::{OnnxNormalizer, equivalent};
    use diff_fuzzer_core::Normalizer;

    if !matches!(clean, OnnxOutcome::Ok(_)) {
        return FaultClass::Unrunnable;
    }
    let clean_canon = OnnxNormalizer.normalize(clean.clone());
    let faulty_canon = OnnxNormalizer.normalize(faulty.clone());

    if equivalent(&clean_canon, &faulty_canon) {
        FaultClass::Inert
    } else {
        FaultClass::Exercised
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::normalize::{Canonical, OnnxNormalizer};
    use crate::oracle::OnnxOracle;
    use crate::runtimes::{OrtRuntime, TractRuntime};
    use crate::validation::well_formed;
    use diff_fuzzer_core::Normalizer;
    use diff_fuzzer_core::traits::{NamedOutput, Oracle, Verdict};

    const OPSET: i64 = 22;

    /// Run a set of implementations and ask the oracle for a verdict — the same path the
    /// engine's driver takes, assembled here so a test can control the participants.
    fn judge(
        case: &OnnxCase,
        implementations: &[&dyn Implementation<In = OnnxCase, Out = OnnxOutcome>],
    ) -> Verdict {
        let outputs: Vec<NamedOutput<Canonical>> = implementations
            .iter()
            .map(|imp| NamedOutput {
                implementation: imp.name().to_string(),
                output: OnnxNormalizer.normalize(imp.run(case).expect("never Err")),
            })
            .collect();
        OnnxOracle.check(case, &outputs)
    }

    /// **The control.** Real implementations, no fault: the oracle must stay silent.
    ///
    /// Without this, a test proving the oracle fires would be satisfied by an oracle that
    /// fires on everything — which detects nothing and reports everything.
    #[test]
    fn the_oracle_is_silent_when_nothing_is_wrong() {
        for op in OpKind::ELEMENTWISE {
            let case = well_formed(op, &[2, 3], OPSET);
            assert_eq!(
                judge(&case, &[&OrtRuntime, &TractRuntime]),
                Verdict::Agree,
                "{op:?}: real implementations should agree"
            );
        }
    }

    /// **The thing this module exists for.** A corrupted value must be caught, on every
    /// operator, and the corrupted implementation must be the one named.
    #[test]
    fn the_oracle_catches_a_corrupted_value() {
        let wrong = WrongValues::new(TractRuntime, 1.0);

        for op in OpKind::ELEMENTWISE {
            let case = well_formed(op, &[2, 3], OPSET);

            // Only meaningful if the fault actually changed something. Classified rather
            // than assumed — an inert fault agreeing is correct behaviour, not a miss.
            let clean = TractRuntime.run(&case).unwrap();
            let faulty = wrong.run(&case).unwrap();
            assert_eq!(
                classify_fault(&clean, &faulty),
                FaultClass::Exercised,
                "{op:?}: the fault must change the output for this test to mean anything"
            );

            let Verdict::Diverged(divergence) = judge(
                &case,
                &[&OrtRuntime, &wrong, &crate::runtimes::TractRuntime],
            ) else {
                panic!("{op:?}: the oracle missed a corrupted value");
            };
            assert!(
                divergence.summary.contains("wrong-values"),
                "{op:?}: the corrupted implementation must be named: {}",
                divergence.summary
            );
        }
    }

    /// An oracle comparing only the first element would pass the test above and miss this.
    #[test]
    fn corruption_is_caught_anywhere_in_the_tensor() {
        let case = well_formed(OpKind::Add, &[2, 3], OPSET);
        for index in 0..6 {
            let wrong = WrongValues::new(TractRuntime, 1.0).at_index(index);
            assert!(
                matches!(judge(&case, &[&OrtRuntime, &wrong]), Verdict::Diverged(_)),
                "corruption at element {index} was missed"
            );
        }
    }

    /// A shape-blind oracle passes every value test and misses real shape bugs.
    #[test]
    fn the_oracle_catches_a_wrong_shape() {
        let case = well_formed(OpKind::Add, &[2, 3], OPSET);
        let wrong = WrongShape::new(TractRuntime);

        // The values are untouched, so only the shape check can catch this.
        let OnnxOutcome::Ok(clean) = TractRuntime.run(&case).unwrap() else {
            panic!("tract should run Add");
        };
        let OnnxOutcome::Ok(faulty) = wrong.run(&case).unwrap() else {
            panic!("the wrapper should still produce a result");
        };
        assert_eq!(
            clean[0].as_f32().expect("f32 tensor"),
            faulty[0].as_f32().expect("f32 tensor"),
            "values must be identical"
        );
        assert_ne!(clean[0].dims, faulty[0].dims, "shape must differ");

        assert!(matches!(
            judge(&case, &[&OrtRuntime, &wrong]),
            Verdict::Diverged(_)
        ));
    }

    /// **The domain's thesis, as an executable check.** A crash must be reported as a
    /// divergence, not swallowed by the skip path.
    #[test]
    fn the_oracle_reports_a_crash_rather_than_skipping_it() {
        let case = well_formed(OpKind::Add, &[2], OPSET);
        let crasher = Panicking::new();

        let Verdict::Diverged(divergence) = judge(&case, &[&OrtRuntime, &crasher, &TractRuntime])
        else {
            panic!("a crash must be a finding, not a skip — this is the whole thesis");
        };
        assert!(
            divergence.summary.contains("panicking"),
            "the crasher must be named: {}",
            divergence.summary
        );
    }

    /// **The negative control for the thesis.** An unsupported operator must *not* be
    /// reported. A test asserting only that crashes are caught would also pass if the
    /// oracle flagged every non-answer, which would bury findings under unimplemented ops.
    #[test]
    fn an_unsupported_implementation_is_not_reported() {
        let case = well_formed(OpKind::Add, &[2], OPSET);
        let abstainer = AlwaysUnsupported::new();

        assert_eq!(
            judge(&case, &[&OrtRuntime, &TractRuntime, &abstainer]),
            Verdict::Agree,
            "declaring an operator unimplemented is not a bug"
        );
    }

    /// `catch_unwind` really does convert a panic into `Crashed`.
    ///
    /// `Panicking` returns the value directly, so it exercises the *oracle* but not the
    /// *catching*. This test covers the other half, with the panic hook silenced so the
    /// backtrace does not pollute test output.
    #[test]
    fn catch_unwind_turns_a_real_panic_into_a_value() {
        struct ReallyPanics;
        impl Implementation for ReallyPanics {
            type In = OnnxCase;
            type Out = OnnxOutcome;
            fn name(&self) -> &str {
                "really-panics"
            }
            fn run(&self, _input: &OnnxCase) -> Result<OnnxOutcome, RunError> {
                Ok(crate::runtimes::catching_panics(|| {
                    panic!("a genuine panic");
                }))
            }
        }

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = ReallyPanics.run(&well_formed(OpKind::Add, &[1], OPSET));
        std::panic::set_hook(previous);

        let Ok(OnnxOutcome::Crashed { detail }) = outcome else {
            panic!("a panic must become Crashed, and must not propagate as Err");
        };
        assert!(
            detail.contains("a genuine panic"),
            "the panic message is the most useful part of a crash report, got: {detail}"
        );
    }

    /// The three-way classification, each branch reached deliberately.
    #[test]
    fn faults_are_classified_three_ways() {
        let answered =
            |v: Vec<f32>| OnnxOutcome::Ok(vec![TensorValue::f32("out", vec![v.len() as i64], v)]);

        assert_eq!(
            classify_fault(&answered(vec![1.0]), &answered(vec![2.0])),
            FaultClass::Exercised
        );
        assert_eq!(
            classify_fault(&answered(vec![1.0]), &answered(vec![1.0])),
            FaultClass::Inert
        );
        assert_eq!(
            classify_fault(
                &OnnxOutcome::Rejected {
                    detail: "no".into()
                },
                &answered(vec![1.0])
            ),
            FaultClass::Unrunnable
        );
    }

    /// An inert fault is a real situation, not a hypothetical: corrupting element 0 of an
    /// empty tensor changes nothing. Agreement there is **correct**, and a detection rate
    /// that counted it as a miss would understate the oracle — which is exactly how a
    /// naive rate read 55.7% for an oracle that caught everything it was shown.
    #[test]
    fn an_inert_fault_is_not_a_miss() {
        let case = well_formed(OpKind::Identity, &[0], OPSET);
        let wrong = WrongValues::new(TractRuntime, 1.0);

        let clean = TractRuntime.run(&case).unwrap();
        let faulty = wrong.run(&case).unwrap();

        assert_eq!(
            classify_fault(&clean, &faulty),
            FaultClass::Inert,
            "corrupting element 0 of an empty tensor should change nothing"
        );

        // The oracle not diverging here is **correct behaviour**, so a detection rate must
        // exclude this case rather than score it as a miss.
        assert!(
            !matches!(judge(&case, &[&OrtRuntime, &wrong]), Verdict::Diverged(_)),
            "an inert fault must not be counted against the oracle"
        );
    }
}
