//! What a finding is, on disk.
//!
//! Core's `DivergenceReport` cannot be used here: it requires a `tolerance`, on the
//! reasoning that a numeric divergence is unfalsifiable without the threshold it was judged
//! against. That reasoning is right and simply does not apply — this domain's comparison is
//! exact, so there is no threshold to state (`PENDING` 1.6).
//!
//! # What a report has to survive
//!
//! **The whole case, not the seed.** A seed reproduces a case only for the generator that
//! produced it; change the generator and the same seed means something else. The tensor
//! domain recorded 814 findings by seed and later found **810 could no longer be
//! reproduced** — not wrong, just unsupportable, which is worse than either.
//!
//! **The versions.** A finding is a claim about *these engines*. Swapping one expires it.
//!
//! **The rendered SQL.** Redundant with the case, and worth the duplication: it is what a
//! maintainer will read, and what they will paste into a shell.

use crate::ast::SqlCase;
use crate::render::Dialect;
use crate::signature::DisagreementKind;
use serde::{Deserialize, Serialize};

/// The versions a finding is a claim about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Environment {
    pub sqlite: String,
    pub duckdb: String,
    pub platform: String,
}

impl Environment {
    /// Ask each engine what it is, rather than reading it off a crate version.
    ///
    /// The two do not agree: the `duckdb` crate is `1.10505.0` while the engine reports
    /// `v1.5.5`. A maintainer wants the engine's answer.
    pub fn detect() -> Environment {
        let sqlite = rusqlite::Connection::open_in_memory()
            .and_then(|conn| conn.query_row("SELECT sqlite_version()", [], |row| row.get(0)))
            .unwrap_or_else(|_| "unknown".to_string());

        let duckdb = duckdb::Connection::open_in_memory()
            .and_then(|conn| conn.query_row("SELECT version()", [], |row| row.get(0)))
            .unwrap_or_else(|_| "unknown".to_string());

        Environment {
            sqlite,
            duckdb,
            platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        }
    }
}

/// One finding, complete enough to act on without this repository.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SqlDivergence {
    /// The de-duplication key. Recomputed by triage rather than trusted, since the rules
    /// that produce it have changed before and will again.
    pub signature: String,
    pub kind: DisagreementKind,
    /// Which engines disagreed. Beside the signature, never inside it.
    pub disagreeing: Vec<String>,
    /// The seed, as context. **Not** the means of reproduction.
    pub seed: u64,
    /// The generator's configuration, named in full, so a later pool can tell whether it
    /// was drawn from the same distribution.
    pub generator: String,
    /// The minimized case, whole.
    pub case: SqlCase,
    /// How much minimization achieved, and whether it finished.
    pub minimisation: Minimisation,
    /// The SQL a person will actually read.
    pub sql: Vec<String>,
    /// What each engine produced, rendered.
    pub outputs: Vec<(String, String)>,
    pub environment: Environment,
    pub summary: String,
}

/// What the shrinker managed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Minimisation {
    pub steps: usize,
    pub candidates_tried: usize,
    /// Whether the search reached a local minimum or ran out of budget.
    ///
    /// "Minimized to three rows" and "stopped at three rows with reductions untried" are
    /// different claims, and a report that cannot tell them apart overstates itself.
    pub complete: bool,
    pub complexity_before: (usize, usize),
    pub complexity_after: (usize, usize),
}

impl SqlDivergence {
    /// A stable filename, derived from content rather than from a counter or a timestamp,
    /// so re-finding the same problem overwrites rather than accumulating copies of it.
    pub fn filename(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in self.signature.bytes().chain(self.sql.join(";").bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{}-{hash:016x}.json", self.kind.as_str())
    }

    pub fn save(&self, directory: &str) -> std::io::Result<String> {
        std::fs::create_dir_all(directory)?;
        let path = format!("{directory}/{}", self.filename());
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(path)
    }

    pub fn load(path: &str) -> std::io::Result<SqlDivergence> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// The SQL as one pasteable script.
    pub fn script(&self) -> String {
        self.case
            .statements(Dialect::Sqlite)
            .iter()
            .map(|statement| format!("{statement};"))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// A **metamorphic** violation: one engine contradicting itself.
///
/// # Why this is not a [`SqlDivergence`]
///
/// A differential finding names **two** engines that disagreed, and its whole meaning is the
/// disagreement — which is why `SqlDivergence` carries `disagreeing: Vec<String>` and a
/// `DisagreementKind`. A metamorphic violation names **one** engine and needs no second opinion:
/// the query contradicted itself under a transform that preserves meaning by the definition of
/// SQL. Forcing it into the differential shape would put a single engine in a field named for a
/// disagreement and leave `kind` describing something that did not happen.
///
/// The two share [`Environment`] and the same content-derived filename discipline, because those
/// parts *are* the same problem.
///
/// # Why this type exists at all
///
/// Until S10.1 only TLP violations were serialised. NoREC, `HAVING` and — most importantly —
/// **index-invariance** printed a line to the log and wrote nothing. Index-invariance is the one
/// relation in this crate whose positive result needs no interpretation: an index changes *how*
/// an answer is reached, never *what* it is, so a violation is a bug with no legal-difference
/// argument available. **That was the relation least able to produce a filable report.**
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetamorphicViolation {
    /// Which relation failed — `tlp-rows`, `tlp-aggregate`, `tlp-grouped`, `tlp-having`,
    /// `norec`, `index-invariance`. Part of the filename, so two relations failing on one case
    /// produce two reports rather than one overwriting the other.
    pub relation: String,
    /// The single engine that contradicted itself. **One, never two** — that is the point of the
    /// technique, and the field is singular to keep it from being read as a comparison.
    pub engine: String,
    /// Context, not the means of reproduction: a seed only reproduces against the generator that
    /// produced it, which is why the whole case is carried below.
    pub seed: u64,
    pub generator: String,
    /// The case as generated, whole.
    pub case: SqlCase,
    /// Each variant that was run, labelled, with the SQL a person will read. The labels differ
    /// per relation — `whole`/`is_true`/`is_false`/`is_unknown` for TLP, `filtered`/`projected`
    /// for NoREC, `with_indexes`/`without_indexes` for index-invariance — so the report says
    /// what was compared rather than assuming the reader knows the relation.
    pub variants: Vec<(String, Vec<String>)>,
    /// What each variant produced, rendered, in the same order.
    pub outputs: Vec<(String, String)>,
    /// The difference itself. Named for the two sides of the relation rather than for engines,
    /// since there is only one engine here.
    pub only_in_whole: Vec<String>,
    pub only_in_partitions: Vec<String>,
    pub environment: Environment,
    pub summary: String,
}

impl MetamorphicViolation {
    /// A stable filename derived from content, so re-finding the same problem overwrites rather
    /// than accumulating copies — the same rule [`SqlDivergence::filename`] follows, and for the
    /// same reason: a campaign that finds one bug ten thousand times should leave one file.
    pub fn filename(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let statements = self
            .variants
            .iter()
            .flat_map(|(_, sql)| sql.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join(";");
        for byte in self.relation.bytes().chain(statements.bytes()) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{}-{}-{hash:016x}.json", self.relation, self.engine)
    }

    pub fn save(&self, directory: &str) -> std::io::Result<String> {
        std::fs::create_dir_all(directory)?;
        let path = format!("{directory}/{}", self.filename());
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(path)
    }

    pub fn load(path: &str) -> std::io::Result<MetamorphicViolation> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Every variant as one pasteable script, separated by a comment naming each.
    ///
    /// A metamorphic repro is **several** queries whose *relationship* is the bug, so a script
    /// that emitted only one of them would not reproduce anything.
    pub fn script(&self) -> String {
        self.variants
            .iter()
            .map(|(label, sql)| {
                let body = sql
                    .iter()
                    .map(|statement| format!("{statement};"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!("-- {label}\n{body}")
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::signature;

    /// Every relation's violation serialises, round-trips, and gets a **content-derived** name.
    ///
    /// The acceptance criterion for S10.1: before it, only TLP wrote a file. NoREC, `HAVING` and
    /// index-invariance printed a log line and nothing else — including the one relation whose
    /// positive result needs no interpretation at all.
    #[test]
    fn every_relation_can_write_a_report_that_round_trips() {
        let case = SqlCase::fixed_example();
        let directory = std::env::temp_dir().join("sql-adapter-violation-test");
        let directory = directory.to_string_lossy().to_string();
        let _ = std::fs::remove_dir_all(&directory);

        let mut written = Vec::new();
        for relation in [
            "tlp-rows",
            "tlp-aggregate",
            "tlp-grouped",
            "tlp-having",
            "norec",
            "index-invariance",
        ] {
            let violation = MetamorphicViolation {
                relation: relation.to_string(),
                engine: "duckdb".to_string(),
                seed: 7,
                generator: "test-generator".to_string(),
                case: case.clone(),
                variants: vec![("whole".to_string(), case.statements(Dialect::Sqlite))],
                outputs: vec![("whole".to_string(), "Rows([])".to_string())],
                only_in_whole: vec!["a row the whole had".to_string()],
                only_in_partitions: vec![],
                environment: Environment::detect(),
                summary: format!("{relation} violated"),
            };

            let path = violation.save(&directory).expect("save");
            let loaded = MetamorphicViolation::load(&path).expect("load");
            assert_eq!(loaded, violation, "{relation} did not round-trip");
            written.push(path);
        }

        // **Six relations, six distinct files.** The filename includes the relation, so two
        // relations failing on one case do not overwrite each other — which the previous
        // seed-keyed name would have done in reverse, writing a new file per seed for what was
        // the same bug.
        let unique: std::collections::HashSet<&String> = written.iter().collect();
        assert_eq!(unique.len(), 6, "relations collided: {written:?}");

        // And the same violation twice writes **one** file, so a campaign that finds one bug ten
        // thousand times leaves one report rather than burying it in copies.
        let before = std::fs::read_dir(&directory).unwrap().count();
        let repeat = MetamorphicViolation::load(&written[0]).unwrap();
        repeat.save(&directory).expect("save again");
        let after = std::fs::read_dir(&directory).unwrap().count();
        assert_eq!(
            before, after,
            "re-saving the same violation created a second file"
        );

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// A metamorphic repro is **several** queries whose relationship is the bug, so the script
    /// must carry all of them, labelled.
    #[test]
    fn the_script_names_every_variant() {
        let case = SqlCase::fixed_example();
        let violation = MetamorphicViolation {
            relation: "index-invariance".to_string(),
            engine: "sqlite".to_string(),
            seed: 1,
            generator: "g".to_string(),
            case: case.clone(),
            variants: vec![
                ("with_indexes".to_string(), vec!["SELECT 1".to_string()]),
                ("without_indexes".to_string(), vec!["SELECT 2".to_string()]),
            ],
            outputs: vec![],
            only_in_whole: vec![],
            only_in_partitions: vec![],
            environment: Environment::detect(),
            summary: "s".to_string(),
        };

        let script = violation.script();
        assert!(script.contains("-- with_indexes"), "{script}");
        assert!(script.contains("-- without_indexes"), "{script}");
        assert!(script.contains("SELECT 1;"), "{script}");
        assert!(script.contains("SELECT 2;"), "{script}");
    }

    fn example() -> SqlDivergence {
        let case = SqlCase::fixed_example();
        SqlDivergence {
            signature: signature(&case, DisagreementKind::RowContent),
            kind: DisagreementKind::RowContent,
            disagreeing: vec!["sqlite".to_string(), "duckdb".to_string()],
            seed: 42,
            generator: "sql-v1(test)".to_string(),
            sql: case.statements(Dialect::Sqlite),
            case,
            minimisation: Minimisation {
                steps: 3,
                candidates_tried: 17,
                complete: true,
                complexity_before: (12, 8),
                complexity_after: (4, 2),
            },
            outputs: vec![
                ("sqlite".to_string(), "[1, 'one']".to_string()),
                ("duckdb".to_string(), "[1, 'ONE']".to_string()),
            ],
            environment: Environment::detect(),
            summary: "row 0 column 1 differs".to_string(),
        }
    }

    #[test]
    fn a_report_survives_a_round_trip_through_disk() {
        let directory = std::env::temp_dir().join("sql-adapter-report-test");
        let directory = directory.to_string_lossy().to_string();
        let _ = std::fs::remove_dir_all(&directory);

        let report = example();
        let path = report.save(&directory).expect("save");
        let loaded = SqlDivergence::load(&path).expect("load");

        assert_eq!(report, loaded, "a saved finding must read back identically");
        std::fs::remove_dir_all(&directory).ok();
    }

    /// The property that makes a finding a *reproduction* rather than a story.
    #[test]
    fn a_loaded_report_still_runs_on_both_engines() {
        use diff_fuzzer_core::traits::Implementation;

        let directory = std::env::temp_dir().join("sql-adapter-repro-test");
        let directory = directory.to_string_lossy().to_string();
        let _ = std::fs::remove_dir_all(&directory);

        let path = example().save(&directory).expect("save");
        let loaded = SqlDivergence::load(&path).expect("load");

        // Re-run the case as it came off disk, with no reference to the generator or the
        // seed that produced it.
        crate::backends::SqliteImpl
            .run(&loaded.case)
            .expect("the saved case still runs on sqlite");
        crate::backends::DuckDbImpl
            .run(&loaded.case)
            .expect("and on duckdb");

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn the_filename_is_derived_from_content() {
        let first = example();
        let mut second = example();
        assert_eq!(
            first.filename(),
            second.filename(),
            "same content, same name"
        );

        second.seed = 999;
        assert_eq!(
            first.filename(),
            second.filename(),
            "the seed is context, not identity — re-finding a problem must overwrite"
        );

        second.signature.push_str("+extra");
        assert_ne!(
            first.filename(),
            second.filename(),
            "a different problem, a different file"
        );
    }

    #[test]
    fn the_environment_reports_what_the_engines_say() {
        let environment = Environment::detect();
        // Not asserting exact versions — they change. Asserting that detection worked at
        // all, since "unknown" in a finding is a finding nobody can act on.
        assert_ne!(environment.sqlite, "unknown");
        assert_ne!(environment.duckdb, "unknown");
        assert!(
            environment.duckdb.starts_with('v'),
            "{}",
            environment.duckdb
        );
    }

    #[test]
    fn the_script_is_pasteable() {
        let script = example().script();
        assert!(script.starts_with("CREATE TABLE"));
        assert!(
            script.contains(";\n"),
            "statements are separated for pasting"
        );
        assert!(script.ends_with(';'));
    }
}
