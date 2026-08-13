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
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use diff_fuzzer_core::Environment;

use crate::case::OnnxCase;

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

    /// The seed, kept for **reproducing the surrounding run**, never as the case itself.
    pub seed: u64,

    /// The generator configuration and logic fingerprint. See the module comment.
    pub generator: String,

    /// Versions of every participant, so "which `tract`?" is answerable later.
    pub environment: Environment,

    /// The case itself — the artifact that survives a generator change.
    pub case: OnnxCase,

    /// What each participant produced, rendered for a human.
    pub outputs: Vec<(String, String)>,
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
        Self {
            signature: signature.into(),
            summary: summary.into(),
            seed,
            generator: generator.into(),
            environment: crate::environment::environment(),
            case,
            outputs,
        }
    }
}

/// An append-only log of findings, de-duplicated by signature.
///
/// Held in memory and flushed on each write, because a campaign that crashes should not lose the
/// findings it had already made — the whole point of the log is to survive the run that produced
/// it.
#[derive(Debug)]
pub struct FindingsLog {
    path: PathBuf,
    seen: Vec<String>,
}

impl FindingsLog {
    /// Open (or create) a log, loading the signatures already present so a resumed campaign
    /// does not re-report what it found before.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut seen = Vec::new();
        if path.exists() {
            for line in BufReader::new(File::open(&path)?).lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                // A corrupt trailing line — a run killed mid-write — must not stop the log from
                // opening. Skipping it costs one duplicate at worst.
                if let Ok(finding) = serde_json::from_str::<StoredFinding>(&line) {
                    seen.push(finding.signature);
                }
            }
        }
        Ok(Self { path, seen })
    }

    /// Record a finding unless its signature has already been seen.
    ///
    /// Returns whether it was new — the count a campaign should report is distinct findings, not
    /// occurrences, since one defect hit ten thousand times is one defect.
    pub fn record(&mut self, finding: &StoredFinding) -> std::io::Result<bool> {
        if self.seen.iter().any(|s| s == &finding.signature) {
            return Ok(false);
        }
        let line = serde_json::to_string(finding).map_err(std::io::Error::other)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        file.flush()?;
        self.seen.push(finding.signature.clone());
        Ok(true)
    }

    /// How many distinct findings the log holds.
    pub fn distinct(&self) -> usize {
        self.seen.len()
    }

    /// Read every finding back.
    pub fn load(path: impl AsRef<Path>) -> std::io::Result<Vec<StoredFinding>> {
        let mut found = Vec::new();
        if !path.as_ref().exists() {
            return Ok(found);
        }
        for line in BufReader::new(File::open(path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            found.push(serde_json::from_str(&line).map_err(std::io::Error::other)?);
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{OpKind, TensorValue};
    use crate::gen_shape::Bounds;
    use crate::validation::well_formed;
    use diff_fuzzer_core::axes::GenerationAxes;

    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("dffind-{name}-{}.jsonl", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

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

    /// **The property the whole module exists for.** A finding must replay from its own record,
    /// with no generator involved — and the values must survive the round trip *bit for bit*,
    /// since a `NaN` that comes back as a different `NaN` is a different test case.
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
        let path = temp("replay");
        let mut log = FindingsLog::open(&path).unwrap();
        assert!(log.record(&finding("sig-a", case.clone())).unwrap());

        let loaded = FindingsLog::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);

        // **Not `assert_eq!` on the case.** `PartialEq` is derived, so it compares the floats
        // with `==` — and `NaN == NaN` is false, which would fail on a perfectly round-tripped
        // case, while `-0.0 == 0.0` is true, which would pass on a corrupted one. It is wrong
        // in *both* directions. The bit-pattern serialization is what makes the comparison
        // expressible at all, so the round trip is checked on that.
        assert_eq!(
            serde_json::to_string(&loaded[0].case).unwrap(),
            serde_json::to_string(&case).unwrap(),
            "the case must survive verbatim"
        );

        // And spot-check the two values a naive comparison gets wrong.
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
        let _ = std::fs::remove_file(&path);
    }

    /// One defect hit repeatedly is one finding.
    #[test]
    fn repeated_signatures_are_recorded_once() {
        let path = temp("dedup");
        let case = well_formed(OpKind::Sign, &[2], 22);
        let mut log = FindingsLog::open(&path).unwrap();
        assert!(log.record(&finding("sig-x", case.clone())).unwrap());
        assert!(!log.record(&finding("sig-x", case.clone())).unwrap());
        assert!(log.record(&finding("sig-y", case)).unwrap());
        assert_eq!(log.distinct(), 2);
        assert_eq!(FindingsLog::load(&path).unwrap().len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    /// A resumed campaign must not re-report what a previous run already found.
    #[test]
    fn a_reopened_log_remembers_what_it_holds() {
        let path = temp("resume");
        let case = well_formed(OpKind::Sign, &[2], 22);
        let mut log = FindingsLog::open(&path).unwrap();
        log.record(&finding("sig-z", case.clone())).unwrap();

        let mut reopened = FindingsLog::open(&path).unwrap();
        assert_eq!(reopened.distinct(), 1);
        assert!(
            !reopened.record(&finding("sig-z", case)).unwrap(),
            "a signature from a previous run must still de-duplicate"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// The generator description must actually reach the record — including the fingerprint of
    /// the generation logic, which is the half of drift the axis values cannot see.
    #[test]
    fn every_finding_carries_the_generator_that_produced_it() {
        let path = temp("generator");
        let mut log = FindingsLog::open(&path).unwrap();
        log.record(&finding("sig-g", well_formed(OpKind::Sign, &[2], 22)))
            .unwrap();

        let loaded = FindingsLog::load(&path).unwrap();
        assert!(
            loaded[0].generator.contains("logic="),
            "the logic fingerprint must be recorded: {}",
            loaded[0].generator
        );
        assert!(
            !loaded[0].environment.components.is_empty(),
            "a finding with no version information cannot be re-checked later"
        );
        let _ = std::fs::remove_file(&path);
    }
}
