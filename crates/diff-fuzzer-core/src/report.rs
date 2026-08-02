//! What we record when implementations disagree, and how it reaches disk.
//!
//! Two types, and the split between them is the point.
//!
//! [`Divergence`] is what the *oracle* produces: what disagreed. It carries no seed,
//! because an oracle is handed some results and judges them — it has no idea which run
//! produced them or how.
//!
//! [`Finding`] is a divergence plus the run context needed to act on it, assembled by
//! whoever owns those facts. A divergence is a statement about *results*; a seed is a
//! fact about *how the run was produced*. Keeping them in one struct would force
//! whoever knows the first to also know the second.
//!
//! A finding that exists only in a terminal scrollback is not something anyone can act
//! on, so findings are written to disk as **JSON Lines** — one self-contained JSON
//! object per line. That format is chosen deliberately over a single JSON array: a
//! campaign appends as it goes, and a run that is interrupted still leaves every line
//! written so far readable. A truncated JSON array is not parseable at all.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// A record of implementations disagreeing on one input.
///
/// The fields are deliberately plain strings at this stage. Making this generic over the
/// output type would be guessing at requirements that the minimisation work has not
/// produced yet — and a guess baked into the core is harder to remove than to add.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Divergence {
    /// The offending input, as its `Debug` representation.
    pub input: String,
    /// One entry per implementation: its name, and what it produced.
    pub outputs: Vec<(String, String)>,
    /// A human-readable statement of what disagreed.
    pub summary: String,
}

impl Divergence {
    /// A copy with over-long fields elided, for writing to disk.
    ///
    /// A generated case can hold tens of thousands of values, and its `Debug` form runs
    /// to about a megabyte. Storing that verbatim produced a **224 MB log for 235
    /// findings** — a file nobody will open, which is the same as no file at all.
    ///
    /// Elision is lossy, and says so: each truncated field records how much was dropped,
    /// so a reader is never misled into thinking they are looking at the whole input.
    /// Two things make that acceptable *for a log*. The seed is recorded, and
    /// regenerates the case exactly for this version of the generator. And a log is a
    /// triage aid — deciding which findings are worth investigating — whereas the
    /// self-contained artifact that must survive a generator change is built later,
    /// from a case that has first been shrunk to something small enough to store whole.
    pub fn truncated(&self, max_chars: usize) -> Self {
        Self {
            input: elide(&self.input, max_chars),
            outputs: self
                .outputs
                .iter()
                .map(|(name, output)| (name.clone(), elide(output, max_chars)))
                .collect(),
            // The summary is already bounded, and is the part triage actually reads.
            summary: self.summary.clone(),
        }
    }
}

/// Shorten `text` to `max_chars`, stating how much was removed.
fn elide(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let kept: String = text.chars().take(max_chars).collect();
    let dropped = text.chars().count() - max_chars;
    format!("{kept} ...[{dropped} more characters elided]")
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "divergence: {}", self.summary)?;
        writeln!(f, "  input: {}", self.input)?;
        for (name, output) in &self.outputs {
            writeln!(f, "  {name}: {output}")?;
        }
        Ok(())
    }
}

/// A divergence together with everything needed to go back to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// The seed that produced the case.
    ///
    /// Necessary but **not sufficient** to reproduce: a seed only means anything for the
    /// generator that produced it, and adding an operation reassigns what every seed
    /// maps to. That is why `input` is stored in full rather than regenerated on demand.
    pub seed: u64,
    /// A stable label for grouping — the adapter supplies something like an operation
    /// name. The engine never interprets it, only records it, since only the domain
    /// knows what makes two findings "the same kind".
    pub label: String,
    pub divergence: Divergence,
}

impl Finding {
    pub fn new(seed: u64, label: impl Into<String>, divergence: Divergence) -> Self {
        Self {
            seed,
            label: label.into(),
            divergence,
        }
    }
}

/// Appends findings to a JSON Lines file.
///
/// Each write is flushed immediately. A campaign can run for hours, and losing the last
/// buffered findings to an interrupted run would be losing exactly the ones that
/// prompted the interruption.
#[derive(Debug)]
pub struct FindingsLog {
    file: File,
    written: usize,
}

impl FindingsLog {
    /// Open `path` for appending, creating it and any missing parent directories.
    ///
    /// Appending rather than truncating: successive campaigns accumulate into one
    /// history instead of each silently erasing the last.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file, written: 0 })
    }

    /// Write one finding as a single line.
    pub fn append(&mut self, finding: &Finding) -> std::io::Result<()> {
        // `to_string` rather than a pretty form: one object per line is what makes the
        // file streamable and greppable, and pretty-printing would break that.
        let line = serde_json::to_string(finding)?;
        writeln!(self.file, "{line}")?;
        self.file.flush()?;
        self.written += 1;
        Ok(())
    }

    /// How many findings this log has written.
    pub fn written(&self) -> usize {
        self.written
    }
}

/// Read every finding back from a JSON Lines file.
///
/// The counterpart to [`FindingsLog::append`], and the reason the round trip can be
/// tested. A saved finding that cannot be read back is not a finding.
pub fn read_findings(path: impl AsRef<Path>) -> std::io::Result<Vec<Finding>> {
    let file = File::open(path)?;
    let mut findings = Vec::new();

    for line in BufReader::new(file).lines() {
        let line = line?;
        // Tolerate blank lines, which a partially written or hand-edited file may have.
        if line.trim().is_empty() {
            continue;
        }
        findings.push(serde_json::from_str(&line)?);
    }

    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unique path under the system temporary directory. Avoids a dependency for
    /// something this small, and the nanosecond clock plus the test name is enough to
    /// keep parallel tests from colliding.
    fn temp_path(name: &str) -> std::path::PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("diff-fuzzer-{name}-{unique}.jsonl"))
    }

    fn finding(seed: u64, label: &str) -> Finding {
        Finding::new(
            seed,
            label,
            Divergence {
                input: format!("Case {{ seed: {seed} }}"),
                outputs: vec![
                    ("alpha".to_string(), "[1.0, 2.0]".to_string()),
                    ("beta".to_string(), "[1.0, 2.5]".to_string()),
                ],
                summary: "alpha vs beta: 1 of 2 elements differ".to_string(),
            },
        )
    }

    /// The property the whole file exists for: what is written can be read back
    /// unchanged. A report that does not survive the round trip cannot be acted on.
    #[test]
    fn findings_survive_a_round_trip() {
        let path = temp_path("round-trip");

        let mut log = FindingsLog::open(&path).unwrap();
        log.append(&finding(1, "matmul")).unwrap();
        log.append(&finding(42, "sum")).unwrap();
        assert_eq!(log.written(), 2);

        let read_back = read_findings(&path).unwrap();
        assert_eq!(read_back, vec![finding(1, "matmul"), finding(42, "sum")]);

        std::fs::remove_file(&path).ok();
    }

    /// One self-contained object per line is what makes the file streamable: a run cut
    /// short still leaves every completed line readable, which a truncated JSON array
    /// would not.
    #[test]
    fn each_finding_occupies_exactly_one_line() {
        let path = temp_path("one-line");

        let mut log = FindingsLog::open(&path).unwrap();
        log.append(&finding(1, "exp")).unwrap();
        log.append(&finding(2, "exp")).unwrap();
        log.append(&finding(3, "exp")).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents.lines().count(), 3);
        // Every line must parse on its own, without the others.
        for line in contents.lines() {
            serde_json::from_str::<Finding>(line).expect("each line is self-contained");
        }

        std::fs::remove_file(&path).ok();
    }

    /// A second campaign must not erase the first.
    #[test]
    fn reopening_appends_rather_than_truncating() {
        let path = temp_path("append");

        FindingsLog::open(&path)
            .unwrap()
            .append(&finding(1, "first"))
            .unwrap();
        FindingsLog::open(&path)
            .unwrap()
            .append(&finding(2, "second"))
            .unwrap();

        assert_eq!(read_findings(&path).unwrap().len(), 2);
        std::fs::remove_file(&path).ok();
    }

    /// The log is keyed by seed, so going from a line in the file back to the case that
    /// produced it must be direct.
    #[test]
    fn findings_are_keyed_by_seed() {
        let path = temp_path("seeds");

        let mut log = FindingsLog::open(&path).unwrap();
        for seed in [7, 4242, u64::MAX] {
            log.append(&finding(seed, "sum")).unwrap();
        }

        let seeds: Vec<u64> = read_findings(&path)
            .unwrap()
            .iter()
            .map(|f| f.seed)
            .collect();
        assert_eq!(seeds, vec![7, 4242, u64::MAX]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn missing_parent_directories_are_created() {
        let path = temp_path("nested").with_extension("");
        let nested = path.join("deeper").join("findings.jsonl");

        FindingsLog::open(&nested)
            .unwrap()
            .append(&finding(1, "matmul"))
            .unwrap();
        assert_eq!(read_findings(&nested).unwrap().len(), 1);

        std::fs::remove_dir_all(&path).ok();
    }

    /// Without a cap, a single finding can carry a megabyte of tensor values, and a
    /// campaign produces a log too large to open. Truncation must actually bound it.
    #[test]
    fn truncation_bounds_the_size_of_a_finding() {
        let enormous = Divergence {
            input: "x".repeat(500_000),
            outputs: vec![("a".to_string(), "y".repeat(500_000))],
            summary: "1 of 100000 elements differ".to_string(),
        };

        let trimmed = enormous.truncated(200);
        assert!(trimmed.input.len() < 300, "{}", trimmed.input.len());
        assert!(trimmed.outputs[0].1.len() < 300);
        // The summary is what triage reads, so it is never cut.
        assert_eq!(trimmed.summary, enormous.summary);
    }

    /// Lossy is acceptable; *silently* lossy is not. A reader must be able to tell that
    /// they are looking at an excerpt.
    #[test]
    fn truncation_says_how_much_it_dropped() {
        let divergence = Divergence {
            input: "z".repeat(1_000),
            outputs: vec![],
            summary: String::new(),
        };

        let trimmed = divergence.truncated(100);
        assert!(
            trimmed.input.contains("900 more characters elided"),
            "{}",
            trimmed.input
        );
    }

    #[test]
    fn short_fields_are_left_alone() {
        let divergence = finding(1, "exp").divergence;
        assert_eq!(divergence.truncated(10_000), divergence);
    }

    #[test]
    fn blank_lines_are_tolerated() {
        let path = temp_path("blank");
        std::fs::write(
            &path,
            format!("{}\n\n", serde_json::to_string(&finding(1, "exp")).unwrap()),
        )
        .unwrap();

        assert_eq!(read_findings(&path).unwrap().len(), 1);
        std::fs::remove_file(&path).ok();
    }
}
