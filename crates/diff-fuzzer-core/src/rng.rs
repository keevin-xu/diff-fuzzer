//! The single source of randomness for the whole project.
//!
//! Every random choice — which operation to test, what shape it has, what numbers go
//! in it — flows through [`SeededRng`]. Nothing anywhere may call a thread-local or
//! system RNG.
//!
//! The reason is that a finding is only worth anything if it can be reproduced. A
//! divergence we saw once and cannot show again is not a bug report; it is a rumour.
//! Recording one 64-bit seed has to be enough to replay a run exactly, and that only
//! holds if there is no second, unrecorded source of randomness anywhere.

use rand::rand_core::TryRng;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::convert::Infallible;

/// A deterministic random number generator that remembers the seed it came from.
///
/// Built on ChaCha8 rather than `rand`'s default `StdRng` deliberately: `StdRng` is
/// documented as free to switch algorithms in a future release, which would silently
/// change what every previously-saved seed produces. `ChaCha8Rng` guarantees the same
/// seed yields the same stream forever, so a seed recorded today still reproduces
/// years from now.
#[derive(Debug, Clone)]
pub struct SeededRng {
    /// Kept so it can be attached to any finding this run produces.
    seed: u64,
    inner: ChaCha8Rng,
}

impl SeededRng {
    /// Create a generator from an explicit seed.
    ///
    /// The same seed always produces the same sequence of values.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            seed,
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// The seed this generator was created from.
    ///
    /// Every divergence report carries this, because it is the whole of what someone
    /// else needs to reproduce the run.
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

// Implementing `TryRng` — the low-level "produce raw random bits, possibly failing"
// trait — is what makes `SeededRng` usable anywhere the `rand` ecosystem expects a
// generator. The three methods below are the entire obligation.
//
// `Infallible` is a type with no possible values, used here to say at the type level
// that this generator cannot fail: an algorithmic generator just computes its next
// number, unlike a hardware or OS source that might be unavailable. Declaring that
// gets the rest for free — `rand` blanket-implements the infallible `Rng` trait for
// anything whose `TryRng::Error` is `Infallible`, and blanket-implements the large
// `RngExt` convenience trait (`random_range`, `random`, sampling distributions) on
// top of that.
//
// The pattern is worth recognising, because it recurs throughout Rust: implement one
// small trait, inherit a large amount of functionality written against it. It is the
// same move this project's own traits make — a backend implements `run`, and gets to
// participate in the whole engine.
impl TryRng for SeededRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.inner.next_u32())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.inner.next_u64())
    }

    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.inner.fill_bytes(dst);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngExt;

    /// Draw a short sequence of numbers, so the tests below compare streams rather
    /// than single values (two different generators could coincide on one value).
    fn sample(seed: u64) -> Vec<u32> {
        let mut rng = SeededRng::from_seed(seed);
        (0..8).map(|_| rng.random_range(0..1000)).collect()
    }

    #[test]
    fn same_seed_produces_the_same_sequence() {
        assert_eq!(sample(42), sample(42));
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        assert_ne!(sample(42), sample(43));
    }

    #[test]
    fn seed_is_remembered() {
        assert_eq!(SeededRng::from_seed(12345).seed(), 12345);
    }

    /// Pins the actual numbers ChaCha8 produces for a known seed.
    ///
    /// This test exists to fail loudly if the underlying generator ever changes —
    /// swapping the algorithm, or a dependency update quietly altering it, would
    /// invalidate every seed recorded in every past finding. That is exactly the kind
    /// of silent break that is worth a guard.
    #[test]
    fn stream_is_stable_across_builds() {
        assert_eq!(sample(42), vec![224, 681, 146, 950, 772, 427, 344, 627]);
    }
}
