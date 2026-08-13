//! The findings log: what a divergence has to carry to still be usable in six months.
//!
//! # A seed is not a record
//!
//! The tempting design is to store the seed and regenerate the case on demand. It is wrong, and
//! the reason is worth stating precisely: **a seed identifies a case only in combination with the
//! generator that consumed it.** Change a draw order, add an operator, widen a bound, and seed
//! 4,182 becomes a different case — silently, with no error, and the stored finding now points at
//! something that was never tested. The record does not go stale loudly; it goes stale invisibly.
//!
//! So every finding stores **the whole case**, serialized. It replays without the generator, and
//! it survives a generator rewrite. `03-THE-SEAMS.md` calls this the self-contained artifact.
//!
//! # And the generator description is not optional
//!
//! A finding also records the configuration that produced it — the axis values plus the
//! compile-time fingerprint of the generation logic. Not to replay the case (the case is right
//! there) but to answer *"what was being explored when this turned up?"*, which is what tells a
//! reader whether the absence of other findings means anything.
//!
//! `PENDING` 1.15 held this open as a rule to remember at N7. Remembering is the weak form. Here
//! the constructor **takes it as an argument**, so a finding without one cannot be built — the
//! same move as making a budget an input to shape construction rather than repairing afterwards.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use diff_fuzzer_core::Environment;

use crate::case::{OnnxCase, TensorData};

/// One recorded divergence, self-contained.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredFinding {
    /// The de-duplication key: operator, element type, and who disagreed with whom.
    ///
    /// A campaign that hits one defect ten thousand times must report it once. The signature is
    /// derived from the *shape* of the disagreement rather than its values, so two cases that
    /// differ only in their numbers collapse together.
    pub signature: String,

    /// The human-readable statement of what disagreed.
    pub summary: String,

    /// The kind of disagreement — `value`, `shape`, `crash`, and so on.
    ///
    /// Duplicated out of [`Self::signature_parts`] into a top-level field because it is how the
    /// other two domains name a finding, and because it becomes the filename prefix. A reader
    /// listing a run directory should be able to see what kinds of problem it holds without
    /// opening anything.
    #[serde(default)]
    pub kind: String,

    /// The implementations that disagreed, sorted.
    ///
    /// Sorted so the same disagreement records identically whatever order the runtimes ran in —
    /// the same determinism requirement the oracle's summary and the signature both carry.
    #[serde(default)]
    pub disagreeing: Vec<String>,

    /// The seed, kept for **reproducing the surrounding run**, never as the case itself.
    pub seed: u64,

    /// The generator configuration and logic fingerprint. See the module comment.
    pub generator: String,

    /// Versions of every participant, so "which `tract`?" is answerable later.
    pub environment: Environment,

    /// The case itself — the artifact that survives a generator change.
    pub case: OnnxCase,

    /// The model in words: what a maintainer reads instead of parsing the case.
    ///
    /// The counterpart of the SQL domain's `sql` field, which stores the statements themselves
    /// rather than only the structured query. **A reproduction nobody can read is a reproduction
    /// nobody will act on**, and expecting the recipient of an issue to decode a JSON tensor is
    /// how a report gets closed unread.
    #[serde(default)]
    pub model: String,

    /// What each participant produced, rendered for a human.
    pub outputs: Vec<(String, String)>,

    /// The comparison rules in force when this was judged, including their fingerprint.
    ///
    /// **A finding is a claim relative to a policy.** Loosen a rule and a recorded finding stops
    /// diverging, with nothing to say the tool changed rather than the runtimes. Replay refuses
    /// to claim a verdict when this no longer matches — see [`crate::repro`].
    ///
    /// Defaulted so records written before the field existed still load; an empty policy is
    /// treated as unverifiable, which is the safe reading rather than the convenient one.
    #[serde(default)]
    pub policy: String,

    /// The legal-difference entries consulted, and what each concluded.
    ///
    /// The audit trail behind "this was not excused". Without it a reader cannot tell a
    /// difference nobody considered from one considered and rejected.
    #[serde(default)]
    pub legal_trail: Vec<String>,

    /// What minimisation achieved, when the finding was minimised.
    ///
    /// **Recorded because "minimised to one element" and "stopped at one element with reductions
    /// still untried" are different claims**, and a report that cannot tell them apart overstates
    /// the second as the first.
    #[serde(default)]
    pub minimisation: Option<Minimisation>,

    /// The structured signature, when one could be derived.
    ///
    /// The `signature` field above is the flat de-duplication key; this is the decomposition
    /// behind it, so a report can group by operator or by kind without re-parsing a string.
    #[serde(default)]
    pub signature_parts: Option<crate::signature::Signature>,
}

impl StoredFinding {
    /// Build a finding. **The generator description is a required argument**, not a field to
    /// remember to fill in.
    pub fn new(
        signature: impl Into<String>,
        summary: impl Into<String>,
        seed: u64,
        generator: impl Into<String>,
        case: OnnxCase,
        outputs: Vec<(String, String)>,
    ) -> Self {
        let case_for_description = case.clone();
        Self {
            signature: signature.into(),
            summary: summary.into(),
            seed,
            generator: generator.into(),
            environment: crate::environment::environment(),
            case,
            outputs,
            model: describe_model(&case_for_description),
            kind: String::new(),
            disagreeing: Vec::new(),
            policy: crate::policy::describe(),
            legal_trail: legal_trail_for(),
            minimisation: None,
            signature_parts: None,
        }
    }

    /// Attach the structured signature, deriving the flat key, kind and participant list from it.
    ///
    /// Derived rather than passed separately, so the four cannot disagree about the same finding.
    pub fn with_signature(mut self, signature: crate::signature::Signature) -> Self {
        self.signature = signature.key();
        self.kind = signature.kind.token().to_string();
        let mut disagreeing: Vec<String> = signature
            .participants
            .iter()
            .filter(|(_, outcome)| outcome != "unsupported")
            .map(|(name, _)| name.clone())
            .collect();
        disagreeing.sort();
        self.disagreeing = disagreeing;
        self.signature_parts = Some(signature);
        self
    }

    /// Attach what minimisation achieved.
    pub fn with_minimisation(mut self, minimisation: Minimisation) -> Self {
        self.minimisation = Some(minimisation);
        self
    }

    /// The file this finding is stored as, inside its run directory.
    ///
    /// `{kind}-{hash}.json`, matching the convention the other two domains use. The hash is of
    /// the **signature**, not of the case, so re-running a campaign overwrites the same file
    /// rather than accumulating one copy per occurrence — a directory holding a hundred files for
    /// one problem is not a report.
    pub fn file_name(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in self.signature.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
        let kind = if self.kind.is_empty() {
            "finding"
        } else {
            &self.kind
        };
        format!("{kind}-{hash:016x}.json")
    }

    /// Was this finding judged under the rules currently compiled in?
    pub fn policy_is_current(&self) -> bool {
        !crate::policy::has_drifted(&self.policy)
    }
}

/// The legal-difference entries in force, each with the handling that kept it from becoming noise.
///
/// Recorded per finding rather than assumed from the catalog at read time, because the catalog is
/// code: a finding read next year must say what was consulted *then*.
fn legal_trail_for() -> Vec<String> {
    crate::known::CATALOG
        .iter()
        .map(|entry| {
            let handling = match entry.handling {
                crate::known::Handling::ForgivenByComparison => "forgiven-by-comparison",
                crate::known::Handling::DeclinedByGenerator => "declined-by-generator",
                crate::known::Handling::ExcludedByConfiguration { .. } => {
                    "excluded-by-configuration"
                }
            };
            format!(
                "{}: {handling} (SPECS §{})",
                entry.id, entry.citation.specs_section
            )
        })
        .collect()
}

/// Render a single-node model as one readable line.
///
/// Deliberately compact: a finding's model is a single node with a handful of small tensors once
/// it has been minimised, and a reader wants to see all of it at once rather than scroll.
/// Floating-point values carry their **bit pattern** alongside, because `0` and `-0` print
/// identically and two of this domain's five findings are about exactly that difference.
fn describe_model(case: &OnnxCase) -> String {
    let inputs: Vec<String> = case
        .inputs
        .iter()
        .map(|input| {
            format!(
                "{}{}: {:?}{:?} = {}",
                if input.is_initializer() {
                    "initializer "
                } else {
                    ""
                },
                input.name,
                input.elem_type(),
                input.dims,
                render_values(&input.data)
            )
        })
        .collect();

    let attrs = if case.attrs.is_empty() {
        String::new()
    } else {
        format!(", attributes {{{}}}", case.attrs.describe())
    };

    format!(
        "node {} (opset {}){} with {}",
        case.op.onnx_name(),
        case.opset,
        attrs,
        inputs.join("; ")
    )
}

/// Values, with bit patterns for floats.
fn render_values(data: &TensorData) -> String {
    fn join<T: std::fmt::Debug>(values: &[T], limit: usize) -> String {
        let shown: Vec<String> = values
            .iter()
            .take(limit)
            .map(|v| format!("{v:?}"))
            .collect();
        if values.len() > limit {
            format!("[{}, … {} more]", shown.join(", "), values.len() - limit)
        } else {
            format!("[{}]", shown.join(", "))
        }
    }
    match data {
        TensorData::F32(v) => {
            let shown: Vec<String> = v
                .iter()
                .take(8)
                .map(|x| format!("{x}({:#010x})", x.to_bits()))
                .collect();
            format!(
                "[{}{}]",
                shown.join(", "),
                if v.len() > 8 {
                    format!(", … {} more", v.len() - 8)
                } else {
                    String::new()
                }
            )
        }
        TensorData::F64(v) => {
            let shown: Vec<String> = v
                .iter()
                .take(8)
                .map(|x| format!("{x}({:#018x})", x.to_bits()))
                .collect();
            format!(
                "[{}{}]",
                shown.join(", "),
                if v.len() > 8 {
                    format!(", … {} more", v.len() - 8)
                } else {
                    String::new()
                }
            )
        }
        TensorData::I32(v) => join(v, 12),
        TensorData::I64(v) => join(v, 12),
        TensorData::Bool(v) => join(v, 12),
    }
}

/// What a minimisation achieved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Minimisation {
    pub steps: usize,
    pub candidates_tried: usize,
    /// Whether the search reached a local minimum rather than running out of budget.
    ///
    /// **Only a complete search may be described as minimal.** A budget-exhausted result is a
    /// smaller case, not the smallest one.
    pub complete: bool,
    /// Element count before and after, which is the figure a reader actually wants.
    pub elements_before: usize,
    pub elements_after: usize,
}

/// A campaign run: a directory of findings, one JSON file each.
///
/// # Why one file per finding rather than one file per run
///
/// The other two domains write a file per finding, and it is the better shape for the same reason
/// a stored case beats a stored seed: the unit a human works with is **one problem**. A file can
/// be opened, read, attached to an issue, diffed against a re-run, and deleted when it is fixed.
/// A single appended log makes each of those a text-processing exercise.
///
/// De-duplication falls out of the filename: it is derived from the signature, so the same
/// problem found a hundred times writes the same file a hundred times rather than accumulating a
/// hundred copies.
#[derive(Debug)]
pub struct Run {
    directory: PathBuf,
    written: Vec<String>,
}

impl Run {
    /// Open a run directory under the tree belonging to `oracle`.
    ///
    /// The oracle decides the tree rather than the caller passing a path, so a metamorphic
    /// finding cannot be filed into the differential tree by writing the wrong string.
    pub fn open(oracle: crate::OracleKind, name: &str) -> std::io::Result<Self> {
        let directory = Path::new(oracle.root()).join(name);
        std::fs::create_dir_all(&directory)?;
        Ok(Self {
            directory,
            written: Vec::new(),
        })
    }

    /// Write a finding. Returns whether it was new **to this run**.
    pub fn record(&mut self, finding: &StoredFinding) -> std::io::Result<bool> {
        let name = finding.file_name();
        let fresh = !self.written.contains(&name);
        let json = serde_json::to_string_pretty(finding).map_err(std::io::Error::other)?;
        std::fs::write(self.directory.join(&name), json)?;
        if fresh {
            self.written.push(name);
        }
        Ok(fresh)
    }

    /// How many distinct findings this run has written.
    pub fn distinct(&self) -> usize {
        self.written.len()
    }

    /// Where the findings went, for a log line or a report to point at.
    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Read every finding back out of a run directory.
    pub fn load(oracle: crate::OracleKind, name: &str) -> std::io::Result<Vec<StoredFinding>> {
        let directory = Path::new(oracle.root()).join(name);
        let mut found = Vec::new();
        if !directory.exists() {
            return Ok(found);
        }
        // Sorted, so a report over a run directory does not depend on filesystem order.
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&directory)?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        for path in paths {
            let text = std::fs::read_to_string(&path)?;
            found.push(serde_json::from_str(&text).map_err(std::io::Error::other)?);
        }
        Ok(found)
    }
}

/// A campaign's log: what the run did, as opposed to what it found.
///
/// **A campaign that finds nothing produces no findings and a log that is the entire result.**
/// `05-MEASUREMENT-AND-CAMPAIGNS.md` is clear that a zero is only worth anything alongside the
/// surface it was measured over, and that surface lives here.
///
/// Written through as each line is added, so a campaign killed halfway still leaves the part of
/// the log it had reached — losing the record of a four-hour run because it was interrupted at
/// three would be its own small disaster.
#[derive(Debug)]
pub struct CampaignLog {
    path: PathBuf,
    file: File,
}

impl CampaignLog {
    /// Open (or truncate) the log for a named run.
    pub fn open(name: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(crate::LOGS_ROOT)?;
        let path = Path::new(crate::LOGS_ROOT).join(format!("{name}.log"));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    /// Append a line, flushing immediately.
    pub fn line(&mut self, text: impl AsRef<str>) -> std::io::Result<()> {
        writeln!(self.file, "{}", text.as_ref())?;
        self.file.flush()
    }

    /// Write a line to the log **and** to standard output.
    ///
    /// Campaigns are watched while they run and read afterwards, and a line that goes to only one
    /// of those is a line somebody misses.
    pub fn say(&mut self, text: impl AsRef<str>) -> std::io::Result<()> {
        println!("{}", text.as_ref());
        self.line(text)
    }

    /// Record the header every campaign log must carry.
    ///
    /// The environment and the generator description are what make a later reader able to say
    /// what the run actually covered. A log without them is a number with no surface attached.
    pub fn header(&mut self, name: &str, generator: &str) -> std::io::Result<()> {
        let environment = crate::environment::environment();
        self.say(format!("campaign: {name}"))?;
        self.say(format!("started:  {}", crate::census::env_date()))?;
        self.say(format!("platform: {}", environment.platform))?;
        for (component, version) in &environment.components {
            self.say(format!("  {component}: {version}"))?;
        }
        self.say(format!("generator: {generator}"))?;
        self.say(format!("policy:    {}", crate::policy::describe()))?;
        self.say(String::new())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::gen_shape::Bounds;
    use crate::validation::well_formed;
    use diff_fuzzer_core::axes::GenerationAxes;

    fn finding(signature: &str, case: OnnxCase) -> StoredFinding {
        StoredFinding::new(
            signature,
            "tract disagrees with 2 others",
            4182,
            Bounds::default().description(),
            case,
            vec![("tract".into(), "1".into()), ("ort".into(), "0".into())],
        )
    }

    /// **N7.7: a record written by an older build must still load.**
    ///
    /// This is a *verbatim* finding from before `policy`, `legal_trail` and `signature_parts`
    /// existed — pasted, not regenerated, because a test that regenerates its own fixture proves
    /// only that today's writer agrees with today's reader. Stored findings are the artifact this
    /// project exists to produce, and a schema change that quietly orphans them destroys work
    /// that cannot be recovered.
    #[test]
    fn a_finding_written_by_an_older_build_still_loads() {
        const OLDER: &str = r#"{"signature":"Sign | Sign: 2 distinct results","summary":"tract disagrees with 2 others","seed":17418742259747381416,"generator":"float-elementwise=on special-values=off max-rank=4 max-dim=8 element-budget=256 opset=22 logic=7a41e479","environment":{"tool":"diff-fuzzer 0.1.0","platform":"aarch64-macos","components":[["onnx (python, reference)","1.22.0"],["ort","2.0.0-rc.13"],["tract-onnx","0.23.4"]]},"case":{"opset":22,"op":"Sign","inputs":[{"name":"a","dims":[5],"data":{"I32":[-5,-1,0,1,5]},"role":"Data"}],"attrs":[]},"outputs":[["tract","[-1,-1,1,1,1]"],["onnxruntime","[-1,-1,0,1,1]"]]}"#;

        let finding: StoredFinding =
            serde_json::from_str(OLDER).expect("an older finding must still deserialize");

        assert_eq!(finding.seed, 17_418_742_259_747_381_416);
        assert_eq!(finding.case.op, OpKind::Sign);
        assert_eq!(finding.case.inputs[0].dims, vec![5]);

        // The new fields default rather than failing.
        assert!(finding.policy.is_empty());
        assert!(finding.legal_trail.is_empty());
        assert!(finding.signature_parts.is_none());

        // And an empty policy must read as *unverifiable*, not as current. The convenient
        // default would be to treat a missing policy as "probably fine"; that is exactly how a
        // stale record starts being trusted.
        assert!(
            !finding.policy_is_current(),
            "a finding with no recorded policy must not claim to match the current one"
        );
    }

    /// Every finding must carry the rules it was judged under, or a reader cannot tell whether a
    /// replay that now agrees means the runtimes changed or the tool did.
    #[test]
    fn a_finding_records_the_policy_and_the_legal_trail() {
        let finding = finding("sig-p", well_formed(OpKind::Sign, &[2], 22));
        assert!(finding.policy_is_current());
        assert!(finding.policy.contains("bit-exact"));
        assert!(
            finding.legal_trail.len() >= crate::known::CATALOG.len(),
            "the trail must name every catalog entry consulted"
        );
        assert!(
            finding
                .legal_trail
                .iter()
                .any(|e| e.contains("nan-payload")),
            "the one forgiven rule must appear in the trail"
        );
    }

    /// **The property the module exists for.** A finding must replay from its own record, with
    /// no generator involved — and the values must survive the round trip *bit for bit*, since a
    /// `NaN` that comes back as a different `NaN` is a different test case.
    #[test]
    fn a_finding_replays_from_its_own_record() {
        let case = OnnxCase::new(
            OpKind::Add,
            22,
            vec![
                TensorValue::f32("a", vec![3], vec![f32::NAN, -0.0, f32::INFINITY]),
                TensorValue::f32("b", vec![3], vec![1.0, 0.0, f32::NEG_INFINITY]),
            ],
        );
        let name = format!("test-replay-{}", std::process::id());
        let mut run = Run::open(crate::OracleKind::Differential, &name).unwrap();
        let mut record = finding("Add/22/F32/rank1/value", case.clone());
        record.kind = "value".into();
        run.record(&record).unwrap();

        let loaded = Run::load(crate::OracleKind::Differential, &name).unwrap();
        assert_eq!(loaded.len(), 1);

        // **Not `assert_eq!` on the case.** `PartialEq` is derived, so it compares floats with
        // `==` — wrong in both directions here: it fails on a perfectly round-tripped `NaN` and
        // passes on a corrupted `-0.0`. The bit-pattern serialization is what makes the
        // comparison expressible at all.
        assert_eq!(
            serde_json::to_string(&loaded[0].case).unwrap(),
            serde_json::to_string(&case).unwrap(),
            "the case must survive verbatim"
        );

        let TensorValue { data, .. } = &loaded[0].case.inputs[0];
        let bits: Vec<u32> = match data {
            crate::case::TensorData::F32(values) => values.iter().map(|v| v.to_bits()).collect(),
            other => panic!("expected f32 data, got {other:?}"),
        };
        assert_eq!(bits[0], f32::NAN.to_bits());
        assert_eq!(
            bits[1],
            (-0.0f32).to_bits(),
            "the sign of zero must survive"
        );

        let _ = std::fs::remove_dir_all(run.directory());
    }

    /// A finding's filename must be derived from its **signature**, so re-running a campaign
    /// overwrites one file per problem rather than accumulating one per occurrence. A directory
    /// holding a hundred files for one problem is not a report.
    #[test]
    fn the_file_name_is_stable_per_problem_and_carries_the_kind() {
        let case = well_formed(OpKind::Sign, &[2], 22);
        let signature = crate::signature::Signature {
            operator: "Sign".into(),
            opset: 22,
            elem_type: crate::case::ElemType::I32,
            rank: 1,
            kind: crate::signature::Kind::Value,
            participants: vec![
                ("tract".into(), "ok".into()),
                ("onnxruntime".into(), "ok".into()),
            ],
        };
        let a = finding("ignored", case.clone()).with_signature(signature.clone());
        let b = finding("ignored", case).with_signature(signature);

        assert_eq!(a.file_name(), b.file_name(), "the same problem, twice");
        assert!(
            a.file_name().starts_with("value-"),
            "the kind must be visible without opening the file: {}",
            a.file_name()
        );
        assert!(a.file_name().ends_with(".json"));
        assert_eq!(a.kind, "value");
        assert_eq!(a.disagreeing, vec!["onnxruntime", "tract"]);
    }

    /// Two different problems must not collide onto one file.
    #[test]
    fn different_problems_get_different_files() {
        let case = well_formed(OpKind::Sign, &[2], 22);
        let mut a = finding("Sign/22/I32/rank1/value", case.clone());
        a.kind = "value".into();
        let mut b = finding("Sign/22/F32/rank1/value", case);
        b.kind = "value".into();
        assert_ne!(a.file_name(), b.file_name());
    }

    /// **The round trip that matters for the new layout**: written into a run directory as one
    /// pretty JSON per finding, and read back identically.
    #[test]
    fn a_run_directory_round_trips() {
        let name = format!("test-run-{}", std::process::id());
        let case = well_formed(OpKind::Sign, &[2], 22);
        let mut run = Run::open(crate::OracleKind::Differential, &name).unwrap();

        let mut first = finding("Sign/22/I32/rank1/value", case.clone());
        first.kind = "value".into();
        assert!(run.record(&first).unwrap(), "first write is new");
        assert!(!run.record(&first).unwrap(), "the same problem is not new");
        assert_eq!(run.distinct(), 1);

        let loaded = Run::load(crate::OracleKind::Differential, &name).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].signature, "Sign/22/I32/rank1/value");
        assert_eq!(loaded[0].case.op, OpKind::Sign);

        let _ = std::fs::remove_dir_all(run.directory());
    }

    /// The two oracles must land in different directories even under the same run name — the
    /// exact collision the SQL domain hit.
    #[test]
    fn the_same_run_name_under_two_oracles_does_not_collide() {
        let name = format!("shared-label-{}", std::process::id());
        let differential = Run::open(crate::OracleKind::Differential, &name).unwrap();
        let metamorphic = Run::open(crate::OracleKind::Metamorphic, &name).unwrap();
        assert_ne!(differential.directory(), metamorphic.directory());
        let _ = std::fs::remove_dir_all(differential.directory());
        let _ = std::fs::remove_dir_all(metamorphic.directory());
    }

    /// A campaign log must carry the surface the run covered, or a zero result means nothing.
    #[test]
    fn a_campaign_log_records_the_surface_it_covered() {
        let name = format!("test-log-{}", std::process::id());
        {
            let mut log = CampaignLog::open(&name).unwrap();
            log.header(&name, "float-elementwise=on logic=abcd1234")
                .unwrap();
            log.line("judged 10 cases").unwrap();
        }
        let path = std::path::Path::new(crate::LOGS_ROOT).join(format!("{name}.log"));
        let text = std::fs::read_to_string(&path).unwrap();

        assert!(text.contains("campaign: "), "{text}");
        assert!(text.contains("generator: "), "the surface must be recorded");
        assert!(
            text.contains("policy:"),
            "the rules in force must be recorded"
        );
        assert!(text.contains("tract-onnx"), "versions must be recorded");
        assert!(text.contains("judged 10 cases"));
        let _ = std::fs::remove_file(&path);
    }

    /// The generator description must actually reach the record — including the fingerprint of
    /// the generation logic, which is the half of drift the axis values cannot see.
    #[test]
    fn every_finding_carries_the_generator_that_produced_it() {
        let name = format!("test-generator-{}", std::process::id());
        let mut run = Run::open(crate::OracleKind::Differential, &name).unwrap();
        let mut record = finding(
            "Sign/22/I32/rank1/value",
            well_formed(OpKind::Sign, &[2], 22),
        );
        record.kind = "value".into();
        run.record(&record).unwrap();

        let loaded = Run::load(crate::OracleKind::Differential, &name).unwrap();
        assert!(
            loaded[0].generator.contains("logic="),
            "the logic fingerprint must be recorded: {}",
            loaded[0].generator
        );
        assert!(
            !loaded[0].environment.components.is_empty(),
            "a finding with no version information cannot be re-checked later"
        );
        assert!(
            loaded[0].model.contains("node Sign"),
            "the readable model must be recorded: {}",
            loaded[0].model
        );
        let _ = std::fs::remove_dir_all(run.directory());
    }
}
