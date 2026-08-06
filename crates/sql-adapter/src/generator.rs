//! Producing cases to test.
//!
//! **This is a placeholder, and it is the largest thing still missing.** Real generation —
//! a schema, seeded rows, and a type-aware query built so that every case is valid by
//! construction — is S2, and it is the single biggest piece of domain knowledge in this
//! adapter. What lives here now returns one fixed case, which is enough to prove that a
//! case can travel through every seam.
//!
//! The trait is bound now rather than later for one reason: it makes the shape of the
//! eventual work visible. `generate` receives `&mut SeededRng` and nothing else, so every
//! choice a real generator makes must come from that seed — which is what makes a run
//! replayable.

use crate::ast::SqlCase;
use diff_fuzzer_core::rng::SeededRng;
use diff_fuzzer_core::traits::Generator;

/// Returns one fixed case, ignoring the seed.
///
/// Named for what it is. A type called `SqlGenerator` that generated nothing would be a
/// small lie that survives until someone trusts it.
#[derive(Debug, Clone, Copy, Default)]
pub struct FixedCaseGenerator;

impl Generator for FixedCaseGenerator {
    type In = SqlCase;

    /// The `rng` is deliberately unused, and the underscore says so at the call site.
    ///
    /// Not a stub to be filled in — S2 replaces this type entirely. Until then, every
    /// seed produces the same case, which makes the determinism test at S1.8 true but
    /// weak: it proves the *pipeline* is deterministic, not the generator, because there
    /// is no generator yet. Saying so is the point; a determinism test that passes because
    /// nothing varies would otherwise read as evidence it is not.
    fn generate(&self, _rng: &mut SeededRng) -> SqlCase {
        SqlCase::fixed_example()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_seed_produces_the_fixed_case_for_now() {
        let mut first = SeededRng::from_seed(1);
        let mut second = SeededRng::from_seed(999_999);

        assert_eq!(
            FixedCaseGenerator.generate(&mut first),
            FixedCaseGenerator.generate(&mut second)
        );
        assert_eq!(
            FixedCaseGenerator.generate(&mut first),
            SqlCase::fixed_example()
        );
    }
}
