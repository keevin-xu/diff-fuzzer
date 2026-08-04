//! Cases that were judged and **agreed** — the counter-examples a trigger claim must survive.
//!
//! # Why a divergence-finding tool stores non-divergences
//!
//! A claim about *what triggers* a bug is only worth something if it separates cases that
//! diverge from cases that do not. Fitted to divergences alone, any rule can be found:
//! with 41 findings that all contain an overflowing product, "contains an overflowing
//! product" looks like an explanation — until you notice it also holds for cases that
//! agree perfectly. That is exactly what happened here, and only the non-diverging cases
//! exposed it.
//!
//! So negatives are not incidental. **They are the half of the evidence that makes the
//! other half falsifiable.**
//!
//! # Why they must be captured *during* a run
//!
//! They cannot be reconstructed afterwards. A finding records a seed, but a fuzzing
//! finding's seed is meaningless — libFuzzer's stream depends on a corpus that evolves as
//! it runs, and under `-fork=1` on child processes that no longer exist. Sampling as the
//! campaign runs is the only way to get negatives drawn from **the same distribution as
//! the findings**, which matters more than it sounds: scored against negatives from a
//! different generator, a search would happily learn *"which generator produced this
//! case"* instead of *"what triggers the bug"* — and would score well doing it.
//!
//! # What does *not* count
//!
//! Only a case the oracle judged **`Agree`**. A `Skipped` case — a backend refused it, or
//! both returned `NaN` so no arithmetic was compared — is not evidence that the case fails
//! to diverge. It is evidence that nothing was learned, and recording it as a negative
//! would quietly poison the set with cases that were never actually tested.

use crate::input::TensorOp;
use std::io;
use std::path::Path;

/// Write one non-diverging case into `directory`.
///
/// Named by a hash of the case, so a case sampled twice overwrites rather than
/// accumulating — the same content-derived naming the findings use, and for the same
/// reason: a directory that grows without bound stops being readable.
pub fn save_case(directory: impl AsRef<Path>, case: &TensorOp) -> io::Result<()> {
    let directory = directory.as_ref();
    std::fs::create_dir_all(directory)?;

    let path = directory.join(format!("neg-{}-{:x}.json", case.name(), digest(case)));
    std::fs::write(path, serde_json::to_string(case)?)
}

/// Write a batch of cases as a single file.
///
/// Used where the cases come from a deliberate experiment rather than from sampling, and
/// belong together — the probe cases in `findings/negatives/batched_probe.json` are one
/// experiment's output and are only meaningful read as a set.
pub fn save_batch(path: impl AsRef<Path>, cases: &[TensorOp]) -> io::Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(cases)?)
}

/// Load every negative at or below `directory`.
///
/// Accepts both shapes written above — a file holding one case, and a file holding an
/// array of them — because the two arrive from different sources and a caller should not
/// have to care which.
pub fn load(directory: impl AsRef<Path>) -> Vec<TensorOp> {
    let mut cases = Vec::new();
    let mut pending = vec![directory.as_ref().to_path_buf()];

    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };

        for path in entries.filter_map(Result::ok).map(|entry| entry.path()) {
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                // Named rather than silently skipped: a lost negative weakens every
                // predicate scored against this set, and does so invisibly.
                eprintln!("could not read {}", path.display());
                continue;
            };

            if let Ok(batch) = serde_json::from_str::<Vec<TensorOp>>(&text) {
                cases.extend(batch);
            } else if let Ok(one) = serde_json::from_str::<TensorOp>(&text) {
                cases.push(one);
            } else {
                eprintln!(
                    "could not parse {} as a case or a case list",
                    path.display()
                );
            }
        }
    }

    cases
}

/// A stable identifier for a case, used only for naming files.
fn digest(case: &TensorOp) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{case:?}").hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{TensorValue, UnaryOp};

    fn case(value: f32) -> TensorOp {
        TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![2], vec![value, value]))
    }

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("diff-fuzzer-neg-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn a_saved_case_loads_back_identically() {
        let dir = temp_dir("roundtrip");
        let original = case(1.5);
        save_case(&dir, &original).expect("writable");

        let loaded = load(&dir);
        assert_eq!(loaded.len(), 1);
        assert_eq!(format!("{:?}", loaded[0]), format!("{original:?}"));
    }

    /// Content-derived naming, so a campaign that meets the same case twice does not grow
    /// the directory. Without this a long run accumulates duplicates of whatever it sees
    /// most often — which is exactly the least interesting case.
    #[test]
    fn saving_the_same_case_twice_leaves_one_file() {
        let dir = temp_dir("dedup");
        save_case(&dir, &case(2.0)).expect("writable");
        save_case(&dir, &case(2.0)).expect("writable");

        assert_eq!(load(&dir).len(), 1);
    }

    #[test]
    fn different_cases_are_kept_apart() {
        let dir = temp_dir("distinct");
        save_case(&dir, &case(1.0)).expect("writable");
        save_case(&dir, &case(2.0)).expect("writable");

        assert_eq!(load(&dir).len(), 2);
    }

    /// Both file shapes are read by one call, since they arrive from different sources —
    /// sampled one at a time by a campaign, or written as a set by an experiment.
    #[test]
    fn a_batch_file_and_single_files_load_together() {
        let dir = temp_dir("mixed");
        save_case(&dir, &case(1.0)).expect("writable");
        save_batch(dir.join("probe.json"), &[case(2.0), case(3.0)]).expect("writable");

        assert_eq!(load(&dir).len(), 3);
    }

    #[test]
    fn loading_a_missing_directory_yields_nothing_rather_than_failing() {
        assert!(load(temp_dir("absent")).is_empty());
    }
}
