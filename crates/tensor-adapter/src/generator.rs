//! Producing tensor test cases.
//!
//! This one produces exactly one case, every time, ignoring the random generator
//! entirely. That is intentional at this stage: the goal right now is to prove a case
//! can flow all the way through the pipeline, and a fixed case makes every other part
//! easier to reason about while it is being built. Real generation — choosing an
//! operation, then arguments satisfying that operation's rules — comes next.

use crate::input::{Matrix, TensorOp};
use diff_fuzzer_core::{Generator, SeededRng};

/// Always produces the same small elementwise `add`.
///
/// Values are chosen to be exactly representable in `f32` and easy to add in your
/// head, so that a wrong result is obvious rather than something to be squinted at:
///
/// ```text
///   [1 2]     [10 20]     [11 22]
///   [3 4]  +  [30 40]  =  [33 44]
/// ```
#[derive(Debug, Default, Clone, Copy)]
pub struct FixedAddGenerator;

impl Generator for FixedAddGenerator {
    type In = TensorOp;

    /// The `_rng` parameter is unused, hence the leading underscore — without it the
    /// compiler warns about an unused variable. The parameter stays in the signature
    /// because it is part of the trait's contract, and because every later generator
    /// will need it.
    fn generate(&self, _rng: &mut SeededRng) -> TensorOp {
        TensorOp::add(
            Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]),
            Matrix::new(2, 2, vec![10.0, 20.0, 30.0, 40.0]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::OpKind;

    #[test]
    fn produces_the_documented_case() {
        let mut rng = SeededRng::from_seed(0);
        let case = FixedAddGenerator.generate(&mut rng);

        assert_eq!(case.op, OpKind::Add);
        assert_eq!(case.lhs.shape(), [2, 2]);
        assert_eq!(case.lhs.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(case.rhs.data(), &[10.0, 20.0, 30.0, 40.0]);
    }

    /// The real generator's central guarantee is that a seed determines the case
    /// exactly. This generator satisfies it trivially by ignoring the seed, but the
    /// test is written now so the guarantee is already under test when generation
    /// starts actually using randomness.
    #[test]
    fn same_seed_produces_the_same_case() {
        let generate_with = |seed| FixedAddGenerator.generate(&mut SeededRng::from_seed(seed));
        assert_eq!(generate_with(7), generate_with(7));
    }

    #[test]
    fn every_seed_produces_the_same_case_for_now() {
        let generate_with = |seed| FixedAddGenerator.generate(&mut SeededRng::from_seed(seed));
        assert_eq!(generate_with(1), generate_with(2));
    }
}
