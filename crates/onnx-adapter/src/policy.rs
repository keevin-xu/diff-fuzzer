//! The comparison policy, captured as data so a stored finding can name the rules it was judged
//! under.
//!
//! # Why a finding must record its policy
//!
//! `N7.8` asks that a stored finding be replayed **under the policy it recorded, never the current
//! one**. The reason is the same one that makes a stored case better than a stored seed: the
//! artifact must not silently change meaning underneath itself.
//!
//! A finding says "these runtimes disagreed". That is a claim *relative to a set of comparison
//! rules*. Loosen one rule — decide `+0.0` and `−0.0` agree after all — and the recorded finding
//! becomes a case that no longer diverges, with nothing to indicate that the tool changed rather
//! than the runtimes. Tighten one, and cases that were agreements become findings retroactively.
//!
//! # And why replaying under an old policy is impossible, so drift is reported instead
//!
//! The honest limit: the policy is **code**, not configuration. There is no way to execute the
//! comparison rules of six months ago from a record written six months ago — the code that
//! implemented them is gone.
//!
//! So this module does the only sound thing available. It records a fingerprint of the deciding
//! code, and replay **refuses to claim a verdict** when the fingerprint has moved, reporting
//! [`crate::repro::Replay::PolicyDrift`] instead. A replay that silently applied today's rules to
//! yesterday's finding would produce a confident answer to a question nobody asked.
//!
//! > **Detecting that the question changed is worth more than answering the wrong one.**

use crate::known::{CATALOG, Handling};

/// A fingerprint of every module that decides whether two results agree.
///
/// The sibling of `gen_shape::GENERATOR_FINGERPRINT`, and it carries the same known gap: a new
/// module that participates in the comparison must be added to this list by a human. Noted rather
/// than papered over.
///
/// Erring toward spurious mismatch is deliberate here too — a comment-only edit changes the hash
/// and forces a re-check that was not strictly needed. That costs a re-run and is visible; a
/// missed change silently reinterprets stored findings and is not.
pub const POLICY_FINGERPRINT: u32 = {
    let hash = fnv1a(include_bytes!("normalize.rs"), 0xcbf2_9ce4_8422_2325);
    let hash = fnv1a(include_bytes!("oracle.rs"), hash);
    let hash = fnv1a(include_bytes!("known.rs"), hash);
    let hash = fnv1a(include_bytes!("signature.rs"), hash);
    (hash ^ (hash >> 32)) as u32
};

/// FNV-1a as a `const fn`, so the hash is computed during compilation.
///
/// Duplicated from `gen_shape` rather than shared, because sharing it would mean one of the two
/// fingerprints hashing a file that contains the other's hash — and a fingerprint that changes
/// whenever an unrelated fingerprint changes reports drift that did not happen.
const fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
        index += 1;
    }
    hash
}

/// A human-readable statement of the rules in force, plus the fingerprint.
///
/// Stored in every finding. The prose is for a reader; the fingerprint is what decides whether a
/// replay is meaningful.
pub fn describe() -> String {
    let forgiven: Vec<&str> = CATALOG
        .iter()
        .filter(|e| e.handling == Handling::ForgivenByComparison)
        .map(|e| e.id)
        .collect();
    let declined: Vec<&str> = CATALOG
        .iter()
        .filter(|e| e.handling == Handling::DeclinedByGenerator)
        .map(|e| e.id)
        .collect();
    let excluded: Vec<&str> = CATALOG
        .iter()
        .filter(|e| matches!(e.handling, Handling::ExcludedByConfiguration { .. }))
        .map(|e| e.id)
        .collect();

    format!(
        "comparison=bit-exact tolerance=none nan-vs-nan=agree signed-zero=disagree \
         forgiven=[{}] declined=[{}] excluded=[{}] fingerprint={POLICY_FINGERPRINT:08x}",
        forgiven.join(","),
        declined.join(","),
        excluded.join(",")
    )
}

/// The fingerprint recorded inside a policy description, if it carries one.
///
/// Parsed back out rather than stored in a second field, so a description and its fingerprint
/// cannot disagree — there is only one place the value lives.
pub fn fingerprint_of(description: &str) -> Option<u32> {
    let token = description
        .split_whitespace()
        .find_map(|part| part.strip_prefix("fingerprint="))?;
    u32::from_str_radix(token, 16).ok()
}

/// Has the policy changed since this description was written?
pub fn has_drifted(description: &str) -> bool {
    fingerprint_of(description) != Some(POLICY_FINGERPRINT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fingerprint_is_set_and_stable_within_a_build() {
        assert_ne!(POLICY_FINGERPRINT, 0);
        assert_eq!(POLICY_FINGERPRINT, POLICY_FINGERPRINT);
    }

    /// The description must round-trip its own fingerprint, or drift detection compares against
    /// nothing and silently reports "no drift" forever.
    #[test]
    fn a_description_carries_a_readable_fingerprint() {
        let described = describe();
        assert_eq!(fingerprint_of(&described), Some(POLICY_FINGERPRINT));
        assert!(!has_drifted(&described));
    }

    /// **The property the module exists for.** A description written under different rules must
    /// be detected as drifted rather than quietly accepted.
    #[test]
    fn a_description_from_another_policy_is_detected() {
        assert!(has_drifted("comparison=bit-exact fingerprint=deadbeef"));
        assert!(
            has_drifted("comparison=bit-exact"),
            "a description with no fingerprint at all cannot be verified, so it must count as \
             drifted rather than as matching"
        );
    }

    /// The description must name the rules a reader would need to interpret a finding, not just
    /// carry an opaque hash — a fingerprint alone says "something changed" and nothing else.
    #[test]
    fn the_description_names_the_rules_in_force() {
        let described = describe();
        for expected in [
            "comparison=bit-exact",
            "tolerance=none",
            "nan-vs-nan=agree",
            "signed-zero=disagree",
            "nan-payload",
            "cast-out-of-range",
        ] {
            assert!(
                described.contains(expected),
                "{expected} missing from the policy description: {described}"
            );
        }
    }

    /// Every catalog entry must appear somewhere in the description. An entry the policy record
    /// omits is a rule a reader cannot know was in force.
    #[test]
    fn every_catalog_entry_is_named() {
        let described = describe();
        for entry in CATALOG {
            assert!(
                described.contains(entry.id),
                "catalog entry {:?} is missing from the policy description",
                entry.id
            );
        }
    }
}
