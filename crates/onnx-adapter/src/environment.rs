//! Which versions a finding applies to.
//!
//! A divergence is **version-specific**: "this runtime disagrees with the specification"
//! is a claim about particular releases, and a maintainer's first question is which ones.
//!
//! This domain has one component the other two do not, and it is the important one. The
//! **`onnx` Python package version is the specification revision under test.** When the
//! reference implementation says a runtime is wrong, it says so according to *that*
//! release of the spec's executable definition. A finding that does not name it is not
//! reproducible, because a later `onnx` may legitimately have changed the answer.
//!
//! Constants are a hazard — a bump elsewhere leaves them silently stale while every
//! report written afterwards names the wrong version and looks perfectly correct. So each
//! one is checked by a test against the file that actually determines it, and the build
//! fails on drift. That is the same discipline `tensor-adapter/src/environment.rs` uses
//! for `Cargo.lock`, extended here to `requirements.txt` and to the vendored schema.

use diff_fuzzer_core::Environment;

/// The `onnx` Python package supplying the reference implementation.
///
/// **This is the specification revision every conformance finding is judged against.**
/// Checked against `requirements.txt` by a test below.
pub const ONNX_PYTHON_VERSION: &str = "1.22.0";

/// The highest `ai.onnx` opset [`ONNX_PYTHON_VERSION`] knows about.
///
/// Measured 2026-08-12 with `onnx.defs.onnx_opset_version()`, not read from a changelog.
/// Recorded because it bounds what the reference can be asked about — an opset above this
/// is not a runtime disagreement, it is a question the judge cannot answer.
pub const MAX_OPSET: i64 = 27;

/// Rust bindings to ONNX Runtime.
///
/// A pre-release, deliberately — see `DECISIONS.md` N0.1. There is no stable `ort` 2.x,
/// and this workspace's own precedent otherwise forbids pre-releases.
pub const ORT_VERSION: &str = "2.0.0-rc.13";

/// The protobuf runtime the generated ONNX types are built on.
pub const PROST_VERSION: &str = "0.14.4";

/// The environment an ONNX finding was produced in.
pub fn environment() -> Environment {
    Environment::detect()
        .with("onnx (python, reference)", ONNX_PYTHON_VERSION)
        .with("prost", PROST_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn crate_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// The version `requirements.txt` pins for a package.
    fn pinned_version(package: &str) -> Option<String> {
        let text = std::fs::read_to_string(crate_root().join("requirements.txt")).ok()?;
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("{package}==")))
            .map(str::trim)
            .map(str::to_owned)
    }

    /// The constant naming the specification revision must match what is actually
    /// installed. If these drift, every finding names the wrong spec revision while
    /// looking correct — the precise failure this file exists to prevent.
    #[test]
    fn the_recorded_onnx_version_matches_the_pin() {
        let pinned = pinned_version("onnx")
            .expect("requirements.txt must pin onnx; it is the specification under test");

        assert_eq!(
            pinned, ONNX_PYTHON_VERSION,
            "ONNX_PYTHON_VERSION is stale. requirements.txt pins {pinned}. \
             This constant is the specification revision written into every finding."
        );
    }

    /// Prove the check above could fail, rather than passing because it looked in the
    /// wrong place and found nothing. A test that cannot fail is not evidence.
    #[test]
    fn the_pin_lookup_actually_reads_the_file() {
        assert!(
            pinned_version("onnx").is_some(),
            "the lookup found no onnx pin at all, so the version check above is vacuous"
        );
        assert!(
            pinned_version("a-package-that-is-not-installed").is_none(),
            "the lookup matched a package that is not pinned, so it matches anything"
        );
    }

    /// The vendored schema must be byte-identical to the one shipped by the pinned
    /// `onnx` package.
    ///
    /// This is the check that keeps the domain honest. We build models from
    /// `proto/onnx.proto` and the reference implementation judges them; if the two ever
    /// come from different releases, we would be testing conformance to a schema nobody
    /// is enforcing, and nothing else in the pipeline would notice.
    ///
    /// The virtual environment is **required**, not optional — the reference
    /// implementation is a participant in this domain, so a machine that cannot run it
    /// cannot run the domain. Failing loudly here is better than skipping quietly: a
    /// check that silently stops checking is this project's most repeated defect.
    #[test]
    fn the_vendored_schema_matches_the_installed_onnx() {
        let venv = crate_root().join("../../.venv-onnx");
        let installed = find_installed_proto(&venv).unwrap_or_else(|| {
            panic!(
                "no onnx.proto found under {}. The ONNX domain needs the reference \
                 implementation; create the environment with:\n  \
                 python3 -m venv .venv-onnx && \
                 ./.venv-onnx/bin/python -m pip install -r crates/onnx-adapter/requirements.txt",
                venv.display()
            )
        });

        let vendored = std::fs::read(crate_root().join("proto/onnx.proto"))
            .expect("the vendored schema must exist; the build script compiles it");
        let upstream = std::fs::read(&installed).expect("installed onnx.proto must be readable");

        assert!(
            vendored == upstream,
            "proto/onnx.proto has drifted from the schema shipped by the installed onnx \
             ({}). Re-vendor it:\n  cp {} {}",
            ONNX_PYTHON_VERSION,
            installed.display(),
            crate_root().join("proto/onnx.proto").display()
        );
    }

    /// Locate `site-packages/onnx/onnx.proto` without hardcoding a Python version.
    fn find_installed_proto(venv: &Path) -> Option<PathBuf> {
        let lib = venv.join("lib");
        for entry in std::fs::read_dir(lib).ok()? {
            let candidate = entry.ok()?.path().join("site-packages/onnx/onnx.proto");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        None
    }

    /// The environment record must actually name the specification revision. A report
    /// missing it cannot be acted on.
    #[test]
    fn the_environment_names_the_specification_revision() {
        let recorded = environment();
        let onnx = recorded
            .components
            .iter()
            .find(|(name, _)| name.starts_with("onnx"))
            .expect("the environment must record the onnx version");

        assert_eq!(onnx.1, ONNX_PYTHON_VERSION);
        assert!(!recorded.platform.is_empty());
    }
}
