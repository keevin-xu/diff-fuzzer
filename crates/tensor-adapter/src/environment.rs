//! Which versions a finding applies to.
//!
//! A divergence is **version-specific**. "These two backends disagree" is not a claim
//! about software in general — it is a claim about particular releases, and a
//! maintainer's first question will be which ones. A report that cannot answer is easy
//! to dismiss and impossible to act on.
//!
//! The versions below are constants, which makes them a hazard: a dependency bump would
//! leave them silently stale, and every report written afterwards would name the wrong
//! version while looking perfectly correct. That is exactly the sort of quiet wrongness
//! this project tries to design out — so a test checks them against `Cargo.lock`, and the
//! build fails when they drift.

use diff_fuzzer_core::Environment;

/// The framework under test.
pub const BURN_VERSION: &str = "0.21.0";

/// The Rust bindings the libtorch backend goes through.
pub const TCH_VERSION: &str = "0.22.0";

/// PyTorch's C++ library, downloaded by `torch-sys`'s build script.
///
/// Not verifiable from `Cargo.lock` — it is not a Rust crate — so this one is tied to
/// `torch-sys` by hand. Confirm it against
/// `target/*/build/torch-sys-*/out/libtorch/torch-*.dist-info` if it is ever in doubt.
pub const LIBTORCH_VERSION: &str = "2.9.0";

/// The environment a tensor finding was produced in.
pub fn environment() -> Environment {
    Environment::detect()
        .with("burn", BURN_VERSION)
        .with("tch", TCH_VERSION)
        .with("libtorch", LIBTORCH_VERSION)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the version `Cargo.lock` resolved for a crate.
    fn locked_version(crate_name: &str) -> Option<String> {
        let lockfile = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../Cargo.lock")
            .canonicalize()
            .ok()?;
        let text = std::fs::read_to_string(lockfile).ok()?;

        // Entries look like:
        //   [[package]]
        //   name = "burn"
        //   version = "0.21.0"
        let needle = format!("name = \"{crate_name}\"\n");
        let start = text.find(&needle)? + needle.len();
        let line = text[start..].lines().next()?;
        let version = line.strip_prefix("version = \"")?.strip_suffix('"')?;

        Some(version.to_string())
    }

    /// **The anti-drift check.** A constant recording a dependency's version is only as
    /// good as its last update, and a stale one would make every report name the wrong
    /// release while looking entirely correct. Upgrading a dependency now fails the
    /// build until the constant follows.
    #[test]
    fn recorded_versions_match_the_lockfile() {
        for (crate_name, recorded) in [("burn", BURN_VERSION), ("tch", TCH_VERSION)] {
            let locked = locked_version(crate_name)
                .unwrap_or_else(|| panic!("{crate_name} not found in Cargo.lock"));

            assert_eq!(
                locked, recorded,
                "{crate_name} is at {locked} but reports say {recorded}; \
                 update the constant in environment.rs"
            );
        }
    }

    #[test]
    fn the_environment_names_every_system_under_test() {
        let environment = environment();
        let names: Vec<&str> = environment
            .components
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();

        assert!(names.contains(&"burn"));
        assert!(names.contains(&"tch"));
        assert!(names.contains(&"libtorch"));
    }

    #[test]
    fn the_environment_records_the_platform_and_tool() {
        let environment = environment();

        assert!(environment.tool.contains("diff-fuzzer"));
        // Architecture and operating system, e.g. `aarch64-macos`.
        assert!(
            environment.platform.contains('-'),
            "{}",
            environment.platform
        );
    }
}
