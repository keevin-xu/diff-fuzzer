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
    /// How the generator was configured.
    ///
    /// Without this the seed is unusable: it identifies a case only in combination with
    /// the configuration that produced it, so a reader with the log alone would have to
    /// be told the bounds out of band — which, in practice, means guessing.
    pub generator: String,
    /// What makes two findings "the same kind", for de-duplication.
    ///
    /// Computed by the caller, because only the domain knows what distinguishes one
    /// underlying problem from another. The engine records and groups by it without
    /// interpreting it.
    pub signature: String,
    pub divergence: Divergence,
}

impl Finding {
    pub fn new(
        seed: u64,
        label: impl Into<String>,
        generator: impl Into<String>,
        signature: impl Into<String>,
        divergence: Divergence,
    ) -> Self {
        Self {
            seed,
            label: label.into(),
            generator: generator.into(),
            signature: signature.into(),
            divergence,
        }
    }
}

/// Tracks which findings have already been seen, so one underlying problem is reported
/// once rather than once per input that triggers it.
///
/// **Why this is necessary rather than tidy.** A single defect is usually reachable from
/// an enormous number of inputs — a fuzzer that finds one will find it again within
/// seconds, and keep finding it. Without collapsing them, a campaign's output is a
/// thousand copies of the same thing, the log grows without bound, and any genuinely
/// *second* problem is invisible in the noise. Reporting volume stops carrying
/// information about how much is wrong.
///
/// The signature is supplied by the caller, since only the domain knows what makes two
/// findings the same. Grouping by it is all the engine does.
#[derive(Debug, Default)]
pub struct Seen {
    counts: std::collections::BTreeMap<String, usize>,
}

impl Seen {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a signature, and say whether it is the first of its kind.
    ///
    /// Counting *every* occurrence while reporting only the first is deliberate: the
    /// count is evidence of how reachable a problem is, which is worth knowing and would
    /// be lost by discarding duplicates outright.
    pub fn is_new(&mut self, signature: &str) -> bool {
        let count = self.counts.entry(signature.to_string()).or_insert(0);
        *count += 1;
        *count == 1
    }

    /// How many times each signature has been seen, in a stable order.
    pub fn counts(&self) -> impl Iterator<Item = (&str, usize)> {
        self.counts.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// How many distinct problems have been seen.
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// How many findings have been seen in total, duplicates included.
    pub fn total(&self) -> usize {
        self.counts.values().sum()
    }
}

/// What produced a finding, so a reader can tell whether it applies to them.
///
/// A divergence is **version-specific**. "These two backends disagree" is not a claim
/// about software in general; it is a claim about particular releases on a particular
/// platform, and a maintainer's first question is which ones. A report that cannot answer
/// that is asking to be dismissed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    /// The version of this tool.
    pub tool: String,
    /// Where it ran — architecture and operating system.
    pub platform: String,
    /// The systems under test and their versions, supplied by the adapter, which is the
    /// only part that knows what they are.
    pub components: Vec<(String, String)>,
}

impl Environment {
    /// Capture what the engine can determine for itself, ready for the adapter to add
    /// the components it knows about.
    pub fn detect() -> Self {
        Self {
            tool: format!("diff-fuzzer {}", env!("CARGO_PKG_VERSION")),
            platform: format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS),
            components: Vec::new(),
        }
    }

    /// Record a system under test and its version.
    pub fn with(mut self, name: impl Into<String>, version: impl Into<String>) -> Self {
        self.components.push((name.into(), version.into()));
        self
    }
}

/// How much the failing case was shrunk, and whether shrinking actually finished.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MinimisationRecord {
    pub steps: usize,
    pub candidates_tried: usize,
    pub stopped: crate::minimize::StopReason,
    /// False when the search ran out of budget — in which case the case is small, but
    /// **not** minimal, and the report must not imply otherwise.
    pub minimal: bool,
}

impl<T> From<&crate::minimize::Minimized<T>> for MinimisationRecord {
    fn from(minimized: &crate::minimize::Minimized<T>) -> Self {
        Self {
            steps: minimized.steps,
            candidates_tried: minimized.candidates_tried,
            stopped: minimized.stopped,
            minimal: minimized.is_minimal(),
        }
    }
}

/// The complete, self-contained record of one divergence.
///
/// This is the artifact a maintainer receives, and it is designed around a single
/// requirement: **it must be actionable without running our generator.** Hence the
/// `input` field holds the case *itself*, fully serialised, rather than only the seed
/// that produced it. A seed is meaningless outside the exact generator that made it —
/// adding one operation reassigns what every seed maps to — so a report carrying only a
/// seed would quietly rot the first time the generator changed.
///
/// The seed is still recorded, because it locates the case within a campaign and lets a
/// run be replayed in context. It is useful, just not sufficient.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DivergenceReport<In> {
    /// The seed that produced the original case. Context, not the means of reproduction.
    pub seed: u64,
    /// A stable label for grouping — typically the operation's name.
    pub label: String,
    /// A description of the generator's configuration. Also context: the input below is
    /// what actually reproduces the failure.
    pub generator: String,
    /// **The minimised case, in full.** Everything needed to reproduce, independent of
    /// the generator.
    pub input: In,
    pub minimisation: MinimisationRecord,
    /// What each implementation produced.
    pub outputs: Vec<(String, String)>,
    /// The tolerance in force. Without it the claim is unfalsifiable — a reader cannot
    /// tell whether the difference was meaningful or the threshold merely tight.
    pub tolerance: crate::tolerance::Tolerance,
    pub environment: Environment,
    /// A human-readable statement of what disagreed and by how much.
    pub summary: String,
}

impl<In: Serialize> DivergenceReport<In> {
    /// Write the report to `path` as indented JSON.
    ///
    /// Indented, unlike the findings log: a log is streamed and scanned in bulk, while a
    /// report is *read by a person* and quite possibly pasted into an issue.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// A stable filename for this report.
    ///
    /// Derived from the label and seed rather than a timestamp or counter, so re-running
    /// the same campaign overwrites the same file instead of accumulating duplicates of
    /// one finding.
    pub fn filename(&self) -> String {
        format!("{}-{}.json", self.label, self.seed)
    }
}

impl<In: std::fmt::Debug> std::fmt::Display for DivergenceReport<In> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "divergence in {} (seed {})", self.label, self.seed)?;
        writeln!(f, "  {}", self.summary)?;
        writeln!(
            f,
            "  tolerance: rtol {:e}, atol {:e}",
            self.tolerance.rtol, self.tolerance.atol
        )?;
        writeln!(
            f,
            "  minimised: {} reductions over {} candidates ({})",
            self.minimisation.steps, self.minimisation.candidates_tried, self.minimisation.stopped
        )?;
        writeln!(f, "  input: {:?}", self.input)?;
        for (name, output) in &self.outputs {
            writeln!(f, "  {name}: {output}")?;
        }
        write!(
            f,
            "  {} on {}",
            self.environment.tool, self.environment.platform
        )?;
        for (name, version) in &self.environment.components {
            write!(f, ", {name} {version}")?;
        }
        Ok(())
    }
}

/// Read a saved report back.
pub fn load_report<In: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> std::io::Result<DivergenceReport<In>> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
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
            "Bounds { max_dim: 8 }",
            format!("{label}/rank1/numeric/1e-6"),
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

    fn report() -> DivergenceReport<Vec<f32>> {
        DivergenceReport {
            seed: 4242,
            label: "matmul".to_string(),
            generator: "Bounds { max_rank: 4, max_dim: 8 }".to_string(),
            input: vec![1.0, 2.0],
            minimisation: MinimisationRecord {
                steps: 7,
                candidates_tried: 43,
                stopped: crate::minimize::StopReason::LocalMinimum,
                minimal: true,
            },
            outputs: vec![
                ("burn-ndarray".to_string(), "[3.0]".to_string()),
                ("burn-tch".to_string(), "[3.5]".to_string()),
            ],
            tolerance: crate::tolerance::Tolerance::new(1e-6, 1e-9),
            environment: Environment::detect().with("burn", "0.21.0"),
            summary: "1 of 1 elements differ".to_string(),
        }
    }

    /// **The property that makes a report worth writing.** What is saved can be read
    /// back identically — including the case itself, which is what lets someone
    /// reproduce the failure without running our generator at all.
    #[test]
    fn a_report_survives_a_round_trip() {
        let path = temp_path("report");
        let original = report();

        original.save(&path).unwrap();
        let loaded: DivergenceReport<Vec<f32>> = load_report(&path).unwrap();

        assert_eq!(loaded, original);
        std::fs::remove_file(&path).ok();
    }

    /// The case must be stored *in full*, not merely referenced by seed. A seed is
    /// meaningless outside the exact generator that produced it, so a report carrying
    /// only a seed would rot the moment an operation was added.
    #[test]
    fn the_case_itself_is_recorded_not_just_its_seed() {
        let json = serde_json::to_string(&report()).unwrap();

        assert!(json.contains("\"input\""), "{json}");
        assert!(
            json.contains("1.0"),
            "the values themselves must be present"
        );
    }

    /// A report must say which versions it applies to. A divergence is a claim about
    /// particular releases, and one that cannot say which is easy to dismiss.
    #[test]
    fn a_report_records_what_produced_it() {
        let report = report();

        assert!(report.environment.tool.contains("diff-fuzzer"));
        assert!(!report.environment.platform.is_empty());
        assert!(
            report
                .environment
                .components
                .iter()
                .any(|(name, _)| name == "burn")
        );
    }

    /// Whether shrinking *finished* must be visible, so a report cannot describe a case
    /// as minimal when the search merely ran out of budget.
    #[test]
    fn a_report_says_whether_minimisation_completed() {
        let unfinished = MinimisationRecord {
            steps: 200,
            candidates_tried: 900,
            stopped: crate::minimize::StopReason::StepBudget,
            minimal: false,
        };

        assert!(!unfinished.minimal);
        assert!(
            unfinished.stopped.to_string().contains("not minimal"),
            "{}",
            unfinished.stopped
        );
    }

    /// Filenames derive from the finding, not from the clock, so re-running a campaign
    /// overwrites the same file rather than accumulating copies of one divergence.
    #[test]
    fn report_filenames_are_stable() {
        assert_eq!(report().filename(), "matmul-4242.json");
        assert_eq!(report().filename(), report().filename());
    }

    /// **The property de-duplication exists for.** One problem reported once, however
    /// many inputs reach it.
    #[test]
    fn a_repeated_signature_is_reported_only_once() {
        let mut seen = Seen::new();

        assert!(seen.is_new("matmul/rank2/numeric/1e-6"));
        assert!(!seen.is_new("matmul/rank2/numeric/1e-6"));
        assert!(!seen.is_new("matmul/rank2/numeric/1e-6"));

        assert_eq!(seen.distinct(), 1);
        // Every occurrence is still counted: how *reachable* a problem is is worth
        // knowing, and discarding duplicates outright would lose it.
        assert_eq!(seen.total(), 3);
    }

    #[test]
    fn distinct_signatures_are_each_reported() {
        let mut seen = Seen::new();

        assert!(seen.is_new("exp/rank1/numeric/1e-7"));
        assert!(seen.is_new("matmul/rank2/undefined"));
        assert_eq!(seen.distinct(), 2);
    }

    /// Counts come out in a stable order, so a campaign's summary does not reshuffle
    /// between runs and become hard to compare.
    #[test]
    fn counts_are_reported_in_a_stable_order() {
        let mut seen = Seen::new();
        for signature in ["zeta", "alpha", "mu", "alpha"] {
            seen.is_new(signature);
        }

        let order: Vec<&str> = seen.counts().map(|(s, _)| s).collect();
        assert_eq!(order, vec!["alpha", "mu", "zeta"]);
    }

    /// A finding must record the configuration its seed depends on. Without it the seed
    /// identifies nothing, and a reader would have to be told the bounds out of band.
    #[test]
    fn a_finding_records_the_generator_configuration() {
        let recorded = finding(1, "matmul");
        assert!(!recorded.generator.is_empty());

        let path = temp_path("generator");
        let mut log = FindingsLog::open(&path).unwrap();
        log.append(&recorded).unwrap();

        let loaded = &read_findings(&path).unwrap()[0];
        assert_eq!(loaded.generator, recorded.generator);
        assert_eq!(loaded.signature, recorded.signature);

        std::fs::remove_file(&path).ok();
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
