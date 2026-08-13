//! Surviving a crash that `catch_unwind` cannot catch.
//!
//! # The gap this fills
//!
//! `catching_panics` turns a Rust panic into a value. It does **nothing** for a C++ `abort()`,
//! a segfault, or a stack overflow — those end the process immediately, with no unwinding and
//! no chance to record anything. ONNX Runtime is C++ behind Rust bindings, so it is exactly the
//! participant that can die that way.
//!
//! A process death in the middle of a campaign loses two things: the run, and — far worse — **the
//! identity of the case that caused it.** The run is cheap to repeat. The case is not: without it
//! there is nothing to report, nothing to shrink, and no way to avoid hitting the same wall on
//! the next attempt.
//!
//! So the case is written to disk *before* it is executed, and the record is cleared when
//! execution returns. If the file still exists at startup, the case inside it is the one that
//! killed the previous run — and `05-MEASUREMENT-AND-CAMPAIGNS.md` notes that a case which kills
//! a runtime outright is itself a strong finding, not merely an obstacle.
//!
//! # Why this is not paranoia, and why it is also not proof
//!
//! Measured at N2: **zero aborts** across the census probes, the throughput sweeps and repeated
//! 500-seed runs. `PENDING` 1.4 recorded that as *not yet needed* rather than *never needed*,
//! precisely because every one of those cases carried ordinary values. N4 turned on `±inf`,
//! `NaN`, subnormals and `f32::MAX`. The re-examination that item asked for belongs here.
//!
//! # Why no `fsync`
//!
//! The write is flushed to the operating system but not to the disk. That is deliberate and
//! sufficient: a process that dies takes its own memory with it, but the bytes it already handed
//! to the kernel survive in the page cache and the file will contain them. Only losing the whole
//! machine would lose the record, and a campaign does not need to survive that. Paying for
//! `fsync` on every case would cost far more than the guard is worth.

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::case::OnnxCase;

/// What was in flight when the process died.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InFlight {
    /// The runtime that was executing. Names the suspect.
    pub runtime: String,
    /// The seed, for reproducing the surrounding run.
    pub seed: u64,
    /// The case itself — the artifact that matters.
    pub case: OnnxCase,
}

/// A file that names the case currently being executed.
///
/// Armed before execution, disarmed after. A file left armed is evidence.
#[derive(Debug)]
pub struct CrashSentinel {
    file: File,
    path: PathBuf,
}

impl CrashSentinel {
    /// Open the sentinel, **returning any case left in flight by a previous run**.
    ///
    /// Recovery and creation are one operation on purpose. Two separate calls would let a
    /// caller create the sentinel without ever checking it, and a guard nobody reads is worse
    /// than no guard: it costs the writes and delivers none of the evidence.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<(Self, Option<InFlight>)> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // A leftover record is read before the file is reopened for writing, since opening it
        // for writing is what destroys it.
        let recovered = match std::fs::read_to_string(&path) {
            Ok(text) if !text.trim().is_empty() => serde_json::from_str(&text).ok(),
            _ => None,
        };

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        Ok((Self { file, path }, recovered))
    }

    /// Record what is about to run. Call immediately before executing.
    ///
    /// The handle is kept open and rewritten in place rather than created and deleted per case:
    /// this runs on every execution, so it sits directly in the campaign's inner loop.
    pub fn arm(&mut self, runtime: &str, seed: u64, case: &OnnxCase) -> std::io::Result<()> {
        let record = InFlight {
            runtime: runtime.to_string(),
            seed,
            case: case.clone(),
        };
        let text = serde_json::to_string(&record).map_err(std::io::Error::other)?;
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(text.as_bytes())?;
        // Flushed to the kernel, not to the disk. See the module comment.
        self.file.flush()
    }

    /// Record that execution returned. Call immediately after, however it returned.
    ///
    /// A crash is not the only outcome worth clearing: a case that merely *rejected* must not be
    /// left looking like the one that killed the process.
    pub fn disarm(&mut self) -> std::io::Result<()> {
        self.file.set_len(0)?;
        self.file.flush()
    }

    /// Where the sentinel lives, for a report to point at.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::OpKind;
    use crate::validation::well_formed;

    fn temp(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("dfsent-{name}-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    /// **The property the module exists for.** A sentinel armed and never disarmed — which is
    /// what a process death looks like from the outside — hands the case back on the next open.
    #[test]
    fn an_armed_sentinel_survives_to_the_next_run() {
        let path = temp("survives");
        let case = well_formed(OpKind::Add, &[2, 3], 22);
        {
            let (mut sentinel, recovered) = CrashSentinel::open(&path).unwrap();
            assert!(
                recovered.is_none(),
                "a fresh sentinel has nothing to report"
            );
            sentinel.arm("onnxruntime", 4182, &case).unwrap();
            // Dropped without disarming: the process "died" here.
        }

        let (_sentinel, recovered) = CrashSentinel::open(&path).unwrap();
        let recovered = recovered.expect("the in-flight case must survive");
        assert_eq!(recovered.runtime, "onnxruntime");
        assert_eq!(recovered.seed, 4182);
        assert_eq!(recovered.case.op, OpKind::Add);
        let _ = std::fs::remove_file(&path);
    }

    /// And a case that completed must not be left looking like the culprit.
    #[test]
    fn a_disarmed_sentinel_reports_nothing() {
        let path = temp("disarmed");
        let case = well_formed(OpKind::Add, &[2], 22);
        {
            let (mut sentinel, _) = CrashSentinel::open(&path).unwrap();
            sentinel.arm("onnxruntime", 7, &case).unwrap();
            sentinel.disarm().unwrap();
        }
        let (_sentinel, recovered) = CrashSentinel::open(&path).unwrap();
        assert!(
            recovered.is_none(),
            "a completed case must not be reported as in flight"
        );
        let _ = std::fs::remove_file(&path);
    }

    /// Re-arming must fully replace the previous record, not leave a tail of it behind. A
    /// shorter case written over a longer one would otherwise deserialize as garbage — and the
    /// evidence would be lost in exactly the situation it is needed.
    #[test]
    fn rearming_replaces_rather_than_overwrites() {
        let path = temp("rearm");
        let long = well_formed(OpKind::Add, &[8, 8, 8], 22);
        let short = well_formed(OpKind::Abs, &[1], 22);
        {
            let (mut sentinel, _) = CrashSentinel::open(&path).unwrap();
            sentinel.arm("onnxruntime", 1, &long).unwrap();
            sentinel.arm("tract", 2, &short).unwrap();
        }
        let (_sentinel, recovered) = CrashSentinel::open(&path).unwrap();
        let recovered = recovered.expect("the second case must be readable");
        assert_eq!(recovered.runtime, "tract");
        assert_eq!(recovered.case.op, OpKind::Abs);
        let _ = std::fs::remove_file(&path);
    }

    /// A truncated record — a process killed mid-write — must not stop the next run from
    /// starting. Losing one record is bad; refusing to launch is worse.
    #[test]
    fn a_corrupt_record_does_not_block_startup() {
        let path = temp("corrupt");
        std::fs::write(&path, "{\"runtime\": \"onnxrunt").unwrap();
        let (_sentinel, recovered) = CrashSentinel::open(&path).unwrap();
        assert!(recovered.is_none());
        let _ = std::fs::remove_file(&path);
    }
}
