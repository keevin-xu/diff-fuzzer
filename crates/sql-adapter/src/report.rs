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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::signature;

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
