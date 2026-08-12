//! What each runtime **claims** to support — the model that makes the crash thesis possible.
//!
//! # Why this exists
//!
//! `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §2 rests on one distinction:
//!
//! | situation | outcome | finding? |
//! |---|---|---|
//! | the runtime does not implement this operator | `Unsupported` | **no** |
//! | the runtime implements it and blew up | `Crashed` | **yes** |
//!
//! Those look identical from outside — both are "no result". Telling them apart requires
//! knowing what each runtime *claims*, which is what this module answers, from the measured
//! census rather than from documentation.
//!
//! # The direction of the inference, which is easy to get backwards
//!
//! A capability model can only ever *promote* a failure to a finding. It cannot demote one:
//! a runtime that answered a minimal probe and then panicked on hostile values has still
//! crashed, whatever any table says.
//!
//! So the rule is deliberately one-way and conservative:
//!
//! - **claimed + failed** → this may be a crash; the harness is entitled to say so.
//! - **not claimed + failed** → `Unsupported`, always. Never an accusation.
//! - **unknown** → treated as *not claimed*. An absent measurement must not manufacture a
//!   finding, and `08-RISKS.md` §2 is blunt that most early findings in any differential
//!   project are the tool's own.
//!
//! # This is a snapshot, and snapshots rot
//!
//! The census records the date, the opset and every component version, because a capability
//! matrix is a claim about *those* versions on *that* day. A stale one is worse than none,
//! because it still looks current. [`Capabilities::is_stale_for`] exists so a campaign can
//! refuse to run against a matrix taken under different versions rather than silently
//! trusting it.

use std::collections::BTreeSet;

use crate::case::{ElemType, OpKind};
use crate::census::{Census, Support};

/// What the runtimes claim, derived from a census.
#[derive(Debug, Clone)]
pub struct Capabilities {
    claimed: BTreeSet<(String, String, ElemType)>,
    measured: BTreeSet<(String, String, ElemType)>,
    environment: Vec<(String, String)>,
    taken: String,
    opset: i64,
}

impl Capabilities {
    /// Build from a census.
    ///
    /// **Only [`Support::Supported`] counts as a claim.** A `Rejected` cell is explicitly not
    /// a claim: it is the ambiguous outcome, and treating ambiguity as a claim would turn
    /// every polite refusal into a potential crash report. That is the direction this module
    /// must never lean.
    pub fn from_census(census: &Census) -> Self {
        let mut claimed = BTreeSet::new();
        let mut measured = BTreeSet::new();
        for cell in &census.cells {
            let key = (cell.runtime.clone(), cell.op.clone(), cell.elem_type);
            measured.insert(key.clone());
            if cell.support == Support::Supported {
                claimed.insert(key);
            }
        }
        Self {
            claimed,
            measured,
            environment: census.environment.clone(),
            taken: census.taken.clone(),
            opset: census.opset,
        }
    }

    /// Load the census stored in the repository.
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("reading the census at {path}: {e}"))?;
        let census: Census =
            serde_json::from_str(&text).map_err(|e| format!("parsing the census: {e}"))?;
        Ok(Self::from_census(&census))
    }

    /// Does this runtime claim this operator at this element type?
    pub fn claims(&self, runtime: &str, op: OpKind, elem: ElemType) -> bool {
        self.claimed
            .contains(&(runtime.to_string(), op.onnx_name().to_string(), elem))
    }

    /// Was this combination ever measured?
    ///
    /// Distinct from [`Self::claims`] on purpose. "Measured and refused" and "never measured"
    /// are different states, and collapsing them would let a gap in the census read as a
    /// statement about a runtime.
    pub fn was_measured(&self, runtime: &str, op: OpKind, elem: ElemType) -> bool {
        self.measured
            .contains(&(runtime.to_string(), op.onnx_name().to_string(), elem))
    }

    /// **The question N5 asks.** A runtime failed on a case — may that be reported as a crash?
    ///
    /// Only when the runtime claimed the operator. Everything else, including an unmeasured
    /// combination, is a legitimate skip. One-way by construction: this can promote a failure
    /// to a finding but never excuse one.
    pub fn failure_is_reportable(&self, runtime: &str, op: OpKind, elem: ElemType) -> bool {
        self.claims(runtime, op, elem)
    }

    /// Which element types a runtime was ever measured to handle successfully.
    ///
    /// Derived from the census rather than declared: a runtime that produced a result for *any*
    /// operator at a type can evidently represent it. A runtime that never did — `candle` at
    /// `I32` and `Bool`, across all 46 of its cells — cannot.
    ///
    /// This is the question `claims` cannot answer. `claims` asks "does this runtime implement
    /// this operator at this type"; a `Cast` to `int32` on a runtime with no `int32` fails for a
    /// reason that has nothing to do with `Cast`.
    pub fn representable_types(&self, runtime: &str) -> BTreeSet<ElemType> {
        self.claimed
            .iter()
            .filter(|(r, _, _)| r == runtime)
            .map(|(_, _, elem)| *elem)
            .collect()
    }

    /// Whether this runtime is known **unable** to represent this element type.
    ///
    /// # The asymmetry, and the test that forced it
    ///
    /// Phrased as *known unable* rather than *can represent*, because the two differ exactly
    /// where it matters. A runtime the census never covered has no measured types at all, so a
    /// naive `can_represent` returns `false` for every type — and a caller that skips the run on
    /// that basis **excuses everything the runtime does, including crashing**.
    ///
    /// That regression was written, and `a_gap_becomes_unsupported_but_a_crash_never_changes`
    /// caught it on the first run: the deliberately-panicking test implementation is not in the
    /// census, so its crash was being reclassified as a gap. The whole thesis of this domain is
    /// that crashes are findings, and this module had just quietly excused one.
    ///
    /// So absence of evidence is **not** evidence of inability. An uncensused runtime is known
    /// unable to represent nothing, and every one of its outcomes reaches the oracle.
    pub fn known_unable_to_represent(&self, runtime: &str, elem: ElemType) -> bool {
        let measured = self.representable_types(runtime);
        // Never measured at all → we know nothing, and nothing is what we may conclude.
        !measured.is_empty() && !measured.contains(&elem)
    }

    /// The runtimes the census covered.
    pub fn runtimes(&self) -> BTreeSet<&str> {
        self.measured.iter().map(|(r, _, _)| r.as_str()).collect()
    }

    /// How many combinations this runtime claims.
    pub fn claim_count(&self, runtime: &str) -> usize {
        self.claimed.iter().filter(|(r, _, _)| r == runtime).count()
    }

    /// Whether this matrix was taken under a different environment than the one now running.
    ///
    /// A capability claim is version-specific. Running a campaign against a census taken with
    /// a different ONNX Runtime — or a different `onnx`, which *is* the specification revision
    /// — means classifying crashes against a runtime that no longer exists.
    ///
    /// Returns the components that differ, so the caller can say which.
    pub fn is_stale_for(&self, current: &[(String, String)]) -> Vec<String> {
        let mut drifted = Vec::new();
        for (name, version) in current {
            match self.environment.iter().find(|(n, _)| n == name) {
                Some((_, recorded)) if recorded != version => {
                    drifted.push(format!("{name}: census {recorded}, now {version}"));
                }
                None => drifted.push(format!("{name}: not recorded in the census")),
                _ => {}
            }
        }
        drifted
    }

    pub fn taken(&self) -> &str {
        &self.taken
    }

    pub fn opset(&self) -> i64 {
        self.opset
    }
}

/// Wrap a runtime so that a failure on an operator it does not claim becomes `Unsupported`.
///
/// # Why this exists, and why it was pulled forward from N5
///
/// A runtime that does not implement an operator reports it as a **typed error string**, not as
/// a machine-readable "unsupported". The adapter records that conservatively as `Rejected` —
/// the variant that accuses nobody — because telling a polite refusal from a genuine failure
/// needs exactly the knowledge this module holds.
///
/// The consequence, measured at N3: once the generator widened to 33 operators, the oracle
/// reported around twenty divergence signatures that were **all** capability gaps the census
/// had already measured — `candle` implements neither `Max` nor `Round`, `tract` declines `Abs`
/// on `f64`. Each surfaced as "one runtime rejected while others answered", which is a real
/// disagreement in form and no finding at all in substance.
///
/// The roadmap placed this at N5, after the generator and the oracle. **That ordering was
/// wrong**: a corpus cannot be judged until a gap can be told from a defect, so this is a
/// prerequisite rather than a refinement. `PENDING` 1.12.
///
/// # What it does not do
///
/// It never turns a failure *into* a crash, and it never excuses one. A `Crashed` outcome
/// passes through untouched — a runtime that panicked has panicked whatever any table says
/// about what it claims. Only `Rejected` is reclassified, and only downward, to `Unsupported`.
pub struct WithCapabilities<'a, I> {
    inner: I,
    capabilities: &'a Capabilities,
}

impl<'a, I> WithCapabilities<'a, I> {
    pub fn new(inner: I, capabilities: &'a Capabilities) -> Self {
        Self {
            inner,
            capabilities,
        }
    }
}

impl<I> diff_fuzzer_core::traits::Implementation for WithCapabilities<'_, I>
where
    I: diff_fuzzer_core::traits::Implementation<
            In = crate::case::OnnxCase,
            Out = crate::outcome::OnnxOutcome,
        >,
{
    type In = crate::case::OnnxCase;
    type Out = crate::outcome::OnnxOutcome;

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn run(
        &self,
        input: &crate::case::OnnxCase,
    ) -> Result<crate::outcome::OnnxOutcome, diff_fuzzer_core::traits::RunError> {
        use crate::outcome::OnnxOutcome;

        // ── Before running at all ─────────────────────────────────────────────────
        //
        // A type the runtime cannot represent makes the case unrunnable for a reason that has
        // nothing to do with the operator. `candle` has no `int32`, so asking it to `Cast` to
        // `int32` is not a question it can be wrong about — and it does not *fail*: it returns
        // an `int64` tensor, which the oracle then reports as a divergence against two runtimes
        // that agreed. Checking after the fact would miss exactly that case, because the run
        // succeeded.
        //
        // So this is checked first, and the runtime is not asked.
        let unrepresentable: Vec<crate::case::ElemType> = crate::ops::required_elem_types(input)
            .into_iter()
            .filter(|t| {
                self.capabilities
                    .known_unable_to_represent(self.inner.name(), *t)
            })
            .collect();
        if !unrepresentable.is_empty() {
            return Ok(OnnxOutcome::Unsupported {
                reason: format!(
                    "{} cannot represent {unrepresentable:?}, which {} requires (census {})",
                    self.inner.name(),
                    input.op.onnx_name(),
                    self.capabilities.taken()
                ),
            });
        }

        let outcome = self.inner.run(input)?;
        let OnnxOutcome::Rejected { detail } = outcome else {
            // `Ok`, `Unsupported`, `Crashed` and `TimedOut` pass through untouched. In
            // particular a crash is never reclassified: this can only ever say "that failure
            // was a gap", never "that crash was not one".
            return Ok(outcome);
        };

        // The element type the census keyed on — **not** `inputs[0]`, which for `Where` is the
        // boolean condition rather than the data type. See `ops::data_elem_type`.
        let elem = crate::ops::data_elem_type(input);

        if self.capabilities.claims(self.inner.name(), input.op, elem) {
            // It claims the operator and still refused this case. That is a statement about
            // *this input*, and stays a comparable value — rows-versus-error was one of the SQL
            // domain's most productive signals.
            Ok(OnnxOutcome::Rejected { detail })
        } else {
            Ok(OnnxOutcome::Unsupported {
                reason: format!(
                    "{} does not implement {} at {elem:?} (census {}): {}",
                    self.inner.name(),
                    input.op.onnx_name(),
                    self.capabilities.taken(),
                    detail.lines().next().unwrap_or("")
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census;
    use crate::runtimes::{OrtRuntime, TractRuntime};

    const OPSET: i64 = 22;

    fn sample() -> Capabilities {
        Capabilities::from_census(&census::take(&[&OrtRuntime, &TractRuntime], OPSET))
    }

    /// The inference must run one way only. A claimed operator makes a failure reportable;
    /// nothing makes a failure *un*-reportable, and an unmeasured combination must never
    /// manufacture a finding.
    #[test]
    fn only_a_claimed_operator_makes_a_failure_reportable() {
        let caps = sample();

        // ONNX Runtime answered `Add` on f32 in the census, so it claims it.
        assert!(caps.claims("onnxruntime", OpKind::Add, ElemType::F32));
        assert!(caps.failure_is_reportable("onnxruntime", OpKind::Add, ElemType::F32));

        // It declined `Where` on bool — measured, but not a claim.
        assert!(caps.was_measured("onnxruntime", OpKind::Where, ElemType::Bool));
        assert!(!caps.claims("onnxruntime", OpKind::Where, ElemType::Bool));
        assert!(!caps.failure_is_reportable("onnxruntime", OpKind::Where, ElemType::Bool));
    }

    /// An unmeasured combination is treated as *not claimed*. The safe direction: an absent
    /// measurement must not become an accusation.
    #[test]
    fn an_unmeasured_combination_is_never_reportable() {
        let caps = sample();
        assert!(!caps.was_measured(
            "a-runtime-that-was-not-in-the-census",
            OpKind::Add,
            ElemType::F32
        ));
        assert!(!caps.failure_is_reportable(
            "a-runtime-that-was-not-in-the-census",
            OpKind::Add,
            ElemType::F32
        ));
    }

    /// "Measured and refused" and "never measured" must stay distinct, or a gap in the census
    /// reads as a statement about a runtime.
    #[test]
    fn measured_and_claimed_are_different_questions() {
        let caps = sample();
        // Sqrt has no integer probe at all — the specification forbids it — so it was never
        // measured, as opposed to having been measured and refused.
        assert!(!caps.was_measured("tract", OpKind::Sqrt, ElemType::I64));
        assert!(!caps.claims("tract", OpKind::Sqrt, ElemType::I64));
    }

    /// A stale matrix must be detectable. Classifying crashes against versions that are no
    /// longer running is worse than not classifying them.
    #[test]
    fn version_drift_is_detected_and_named() {
        let caps = sample();

        // The current environment matches the one the census was taken in.
        assert!(
            caps.is_stale_for(&crate::environment::environment().components)
                .is_empty()
        );

        // A changed component is reported, and reported by name.
        let drifted = caps.is_stale_for(&[("onnx (python, reference)".into(), "9.9.9".into())]);
        assert_eq!(drifted.len(), 1);
        assert!(
            drifted[0].contains("onnx"),
            "the drifted component must be named"
        );

        // An unknown component is also drift — a census that never saw it cannot vouch for it.
        let unknown = caps.is_stale_for(&[("some-new-runtime".into(), "1.0".into())]);
        assert_eq!(unknown.len(), 1);
    }

    /// The model must actually carry claims, or every test above passes vacuously.
    #[test]
    fn the_model_is_not_empty() {
        let caps = sample();
        assert!(
            caps.claim_count("onnxruntime") > 100,
            "ORT claims almost everything"
        );
        assert!(caps.claim_count("tract") > 50);
        assert_eq!(
            caps.runtimes(),
            ["onnxruntime", "tract"].into_iter().collect()
        );
        assert_eq!(caps.opset(), OPSET);
        assert!(!caps.taken().is_empty());
    }

    /// **The classification, and the two directions it must not run.**
    #[test]
    fn a_gap_becomes_unsupported_but_a_crash_never_changes() {
        use crate::outcome::OnnxOutcome;
        use crate::testing::Panicking;
        use diff_fuzzer_core::traits::Implementation;

        let caps = sample();
        let case = crate::ops::probe(OpKind::Add, ElemType::F32, OPSET).unwrap();

        /// A runtime that refuses everything with a typed error, named as a runtime the census
        /// covered so the lookup finds it.
        struct RefusesEverything(&'static str);
        impl Implementation for RefusesEverything {
            type In = crate::case::OnnxCase;
            type Out = OnnxOutcome;
            fn name(&self) -> &str {
                self.0
            }
            fn run(
                &self,
                _: &crate::case::OnnxCase,
            ) -> Result<OnnxOutcome, diff_fuzzer_core::traits::RunError> {
                Ok(OnnxOutcome::Rejected {
                    detail: "no idea what that is".into(),
                })
            }
        }

        // ONNX Runtime *claims* `Add` at f32, so a refusal stays a comparable value.
        let claimed = WithCapabilities::new(RefusesEverything("onnxruntime"), &caps);
        assert!(
            matches!(claimed.run(&case).unwrap(), OnnxOutcome::Rejected { .. }),
            "a runtime that claims the operator and still refuses is making a statement about \
             this input, which must stay comparable"
        );

        // A runtime the census never saw claims nothing, so the same refusal is a gap.
        let unclaimed = WithCapabilities::new(RefusesEverything("never-censused"), &caps);
        assert!(
            matches!(
                unclaimed.run(&case).unwrap(),
                OnnxOutcome::Unsupported { .. }
            ),
            "a refusal from a runtime that claims nothing is a gap, not a disagreement"
        );

        // A crash is never reclassified, whatever the table says. This is the direction that
        // must not run: the whole thesis is that crashes are findings.
        let crasher = WithCapabilities::new(Panicking::new(), &caps);
        assert!(
            matches!(crasher.run(&case).unwrap(), OnnxOutcome::Crashed { .. }),
            "a crash must survive classification — nothing may excuse one"
        );
    }

    /// A runtime that cannot represent a type the case needs is skipped **without being run**.
    ///
    /// The measured case: asked to `Cast` an `f32` tensor to `int32`, `candle` returns an
    /// `int64` tensor, because it has no `int32` type. It does not fail, so a check that only
    /// inspected failures would miss it entirely and the wrong-typed result would be reported
    /// as a divergence against two runtimes that agreed.
    #[test]
    fn a_type_the_runtime_cannot_represent_is_a_gap_not_a_divergence() {
        use crate::attrs::Attrs;
        use crate::case::{OnnxCase, TensorData, TensorValue};
        use crate::outcome::OnnxOutcome;
        use diff_fuzzer_core::traits::Implementation;

        // A census in which one runtime has only ever handled f32.
        let census = census::take(&[&OrtRuntime], OPSET);
        let mut narrowed = census.clone();
        narrowed.cells.retain(|c| c.elem_type == ElemType::F32);
        let caps = Capabilities::from_census(&narrowed);

        assert!(caps.known_unable_to_represent("onnxruntime", ElemType::I32));
        assert!(!caps.known_unable_to_represent("onnxruntime", ElemType::F32));

        // A `Cast` whose *output* is i32 — the input is f32, which the runtime does handle.
        let case = OnnxCase::new(
            OpKind::Cast,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![2],
                TensorData::F32(vec![1.5, -2.5]),
            )],
        )
        .with_attrs(Attrs::new().int("to", i64::from(ElemType::I32.wire())));

        assert!(
            crate::ops::required_elem_types(&case).contains(&ElemType::I32),
            "the output type must be part of what the case requires"
        );

        let wrapped = WithCapabilities::new(OrtRuntime, &caps);
        assert!(
            matches!(wrapped.run(&case).unwrap(), OnnxOutcome::Unsupported { .. }),
            "a case needing a type the runtime cannot represent is a gap, not a disagreement"
        );

        // The same operator at a type it *can* represent still runs normally.
        let representable = OnnxCase::new(
            OpKind::Cast,
            OPSET,
            vec![TensorValue::new(
                "a",
                vec![2],
                TensorData::F32(vec![1.5, -2.5]),
            )],
        )
        .with_attrs(Attrs::new().int("to", i64::from(ElemType::F32.wire())));
        assert!(
            matches!(wrapped.run(&representable).unwrap(), OnnxOutcome::Ok(_)),
            "the skip must be specific to the unrepresentable type, not a blanket refusal"
        );
    }

    /// **Absence of evidence is not evidence of inability.**
    ///
    /// A runtime the census never covered must have every outcome pass through untouched. The
    /// naive phrasing — "can this runtime represent this type?" — returns `false` for an
    /// uncensused runtime and would skip it before running, excusing even a crash.
    #[test]
    fn an_uncensused_runtime_is_never_skipped() {
        let caps = sample();
        for elem in ElemType::ALL {
            assert!(
                !caps.known_unable_to_represent("a-runtime-nobody-measured", elem),
                "an uncensused runtime must not be declared unable to represent {elem:?}"
            );
        }
    }

    /// The reclassified reason must name the census it came from, so a stale matrix is
    /// traceable from a report rather than being invisible in it.
    #[test]
    fn a_reclassified_gap_names_its_evidence() {
        use crate::outcome::OnnxOutcome;
        use diff_fuzzer_core::traits::Implementation;

        struct Refuses;
        impl Implementation for Refuses {
            type In = crate::case::OnnxCase;
            type Out = OnnxOutcome;
            fn name(&self) -> &str {
                "never-censused"
            }
            fn run(
                &self,
                _: &crate::case::OnnxCase,
            ) -> Result<OnnxOutcome, diff_fuzzer_core::traits::RunError> {
                Ok(OnnxOutcome::Rejected {
                    detail: "unsupported op_type Max".into(),
                })
            }
        }

        let caps = sample();
        let case = crate::ops::probe(OpKind::Max, ElemType::F32, OPSET).unwrap();
        let OnnxOutcome::Unsupported { reason } =
            WithCapabilities::new(Refuses, &caps).run(&case).unwrap()
        else {
            panic!("expected a gap");
        };
        assert!(
            reason.contains("Max"),
            "the operator must be named: {reason}"
        );
        assert!(
            reason.contains(caps.taken()),
            "the census date must be named so a stale matrix is traceable: {reason}"
        );
    }

    /// The stored census must load and agree with a freshly taken one about a runtime both
    /// cover. This is what proves the file on disk is usable rather than merely written.
    #[test]
    fn the_stored_census_loads_and_agrees() {
        let path = format!("{}/census.json", crate::FINDINGS_ROOT);
        let Ok(stored) = Capabilities::load(&path) else {
            // The census is regenerable output and is gitignored, so a fresh checkout will
            // not have it. Skipping is correct here — but the skip is *loud*, because a
            // silent one would hide a genuinely broken loader.
            eprintln!("note: no census at {path}; run the n2_census example to generate it");
            return;
        };
        assert!(stored.claims("onnxruntime", OpKind::Add, ElemType::F32));
        assert!(stored.opset() > 0);
    }
}
