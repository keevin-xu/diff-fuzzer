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

/// The **native** ONNX Runtime library `ort` links against.
///
/// This is the version a maintainer will care about — `ort` is only the binding. Not
/// verifiable from `Cargo.lock`, because it is not a Rust crate: `ort-sys`'s build script
/// downloads it. Confirm it by reading that script's output, which names the artifact it
/// fetched:
///
/// ```text
/// grep 'downloading from' target/*/build/ort-sys-*/output
/// ```
///
/// Observed 2026-08-12: `.../ms@1.28.0/aarch64-apple-darwin+coreml.tar.lzma2`.
///
/// The same hand-tied arrangement as `LIBTORCH_VERSION` in the tensor adapter, and the
/// same hazard: nothing fails if `ort-sys` starts fetching a different build, so this
/// constant must be re-checked whenever `ort` is bumped.
pub const ONNXRUNTIME_NATIVE_VERSION: &str = "1.28.0";

/// The primary target runtime.
pub const TRACT_VERSION: &str = "0.23.4";

/// The secondary target runtime. Present only under the `candle` cargo feature.
pub const CANDLE_ONNX_VERSION: &str = "0.11.0";

/// The protobuf runtime the generated ONNX types are built on.
pub const PROST_VERSION: &str = "0.14.4";

/// The environment an ONNX finding was produced in.
///
/// `candle` appears only when it was actually compiled in. Listing a participant that did
/// not run would overstate what a finding was checked against — and a campaign quietly
/// running three participants instead of four while reporting four is the kind of silent
/// narrowing `08-RISKS.md` §4 is about.
pub fn environment() -> Environment {
    let recorded = Environment::detect()
        .with("onnx (python, reference)", ONNX_PYTHON_VERSION)
        .with("onnxruntime (native)", ONNXRUNTIME_NATIVE_VERSION)
        .with("ort", ORT_VERSION)
        .with("tract-onnx", TRACT_VERSION)
        .with("prost", PROST_VERSION);

    #[cfg(feature = "candle")]
    let recorded = recorded.with("candle-onnx", CANDLE_ONNX_VERSION);

    recorded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn crate_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    /// The version `Cargo.lock` resolved for a crate.
    fn locked_version(crate_name: &str) -> Option<String> {
        let lockfile = crate_root().join("../../Cargo.lock");
        let text = std::fs::read_to_string(lockfile).ok()?;

        // Entries look like:
        //   [[package]]
        //   name = "ort"
        //   version = "2.0.0-rc.13"
        let needle = format!("name = \"{crate_name}\"\n");
        let start = text.find(&needle)? + needle.len();
        let line = text[start..].lines().next()?;
        Some(
            line.strip_prefix("version = \"")?
                .strip_suffix('"')?
                .to_owned(),
        )
    }

    /// Every recorded crate version must match what cargo actually resolved.
    ///
    /// A dependency bump would otherwise leave these constants stale, and every finding
    /// written afterwards would name the wrong version while looking perfectly correct.
    #[test]
    fn recorded_crate_versions_match_the_lockfile() {
        for (crate_name, recorded) in [
            ("ort", ORT_VERSION),
            ("tract-onnx", TRACT_VERSION),
            ("candle-onnx", CANDLE_ONNX_VERSION),
            ("prost", PROST_VERSION),
        ] {
            let locked = locked_version(crate_name)
                .unwrap_or_else(|| panic!("{crate_name} is not in Cargo.lock"));
            assert_eq!(
                locked, recorded,
                "the recorded {crate_name} version is stale; Cargo.lock says {locked}"
            );
        }
    }

    /// Prove the lookup above could fail rather than passing on an empty search.
    #[test]
    fn the_lockfile_lookup_actually_reads_the_file() {
        assert!(locked_version("ort").is_some());
        assert!(locked_version("a-crate-that-does-not-exist").is_none());
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
