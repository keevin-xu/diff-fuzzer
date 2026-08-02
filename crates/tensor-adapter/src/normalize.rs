//! Making two backends' results comparable.
//!
//! Now that a backend hands back `burn`'s own backend-independent container, this step
//! is mostly extraction: pull the shape and the numbers into a form that can be
//! compared and printed.
//!
//! It will not stay that small. Whenever two systems are compared, results that *mean*
//! the same thing routinely *look* different, and this is where that gets reconciled —
//! how NaN compares to NaN, what to do with differing precisions, which differences
//! are legal rather than interesting. Comparing before canonicalising is the standard
//! way a project like this drowns in differences that mean nothing.

use burn::tensor::TensorData;
use diff_fuzzer_core::{Agreement, ApproxEq, Normalizer, Tolerance, compare};

/// A tensor result in a form any backend's output can be converted to.
///
/// Shape is kept alongside the values because a flat list alone would make a 2x3 and a
/// 3x2 result indistinguishable — and two backends returning the same numbers in a
/// different shape is a real disagreement, in fact a louder one than a numeric
/// difference, since they disagree about the operation's meaning rather than its
/// arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalTensor {
    pub shape: Vec<usize>,
    /// Values in row-major order.
    pub values: Vec<f32>,
}

/// Converts backend results into [`CanonicalTensor`].
///
/// One normaliser serves every backend, because `TensorData` is already
/// backend-independent — the rank-and-backend-specific types were resolved at the
/// point of execution. Earlier this type had to be parameterised by backend; that it
/// no longer does is a small piece of evidence that the boundary is in the right
/// place.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorNormalizer;

impl Normalizer for TensorNormalizer {
    type Out = TensorData;
    type Canon = CanonicalTensor;

    /// Takes the result by value, because reading the numbers out consumes it — and
    /// consuming avoids copying a buffer that may be large.
    fn normalize(&self, out: Self::Out) -> CanonicalTensor {
        // `burn` keeps the shape in its own `Shape` type; a plain `Vec` is what we want
        // to hold on to, since a canonical result should not depend on burn's types.
        let shape = out.shape.to_vec();

        // This cannot fail today: every backend in use is instantiated with `f32`, so
        // the extraction always matches. Adding a backend with a different element
        // type is the change that would make this fallible, and would mean this trait
        // needs to return a `Result`.
        let values = out
            .to_vec::<f32>()
            .expect("backends are instantiated with f32 elements");

        CanonicalTensor { shape, values }
    }
}

/// How two tensor results are compared.
///
/// Shape is settled first and separately. Two results of different shapes do not differ
/// *by an amount* — they differ about what the operation produced, which no tolerance
/// should ever absorb however loose. Only once the shapes match does the question
/// become numeric.
impl ApproxEq for CanonicalTensor {
    fn approx_compare(&self, other: &Self, tolerance: Tolerance) -> Agreement {
        if self.shape != other.shape {
            return Agreement::Structural {
                reason: format!("shapes differ: {:?} vs {:?}", self.shape, other.shape),
            };
        }

        // Equal shapes imply equal element counts, since the shape determines them —
        // an invariant established when the value was constructed.
        let comparison = compare(&self.values, &other.values, tolerance);
        if comparison.agrees() {
            Agreement::Agree(comparison)
        } else {
            Agreement::Disagree(comparison)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{libtorch, ndarray};
    use crate::input::{BinaryOp, TensorOp, TensorValue};
    use diff_fuzzer_core::Implementation;

    fn case() -> TensorOp {
        TensorOp::binary(
            BinaryOp::Add,
            TensorValue::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            TensorValue::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]),
        )
    }

    #[test]
    fn normalizing_keeps_shape_and_values() {
        let out = ndarray().run(&case()).unwrap();
        let canon = TensorNormalizer.normalize(out);

        assert_eq!(canon.shape, vec![2, 2]);
        assert_eq!(canon.values, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// Two different backends, two different execution paths, one common form — and on
    /// this input, identical.
    ///
    /// Exact equality is used here and is the wrong tool in general. It works because
    /// these values are small integers, which `f32` represents precisely. Once inputs
    /// are generated rather than hand-picked, two correct backends will routinely
    /// differ in the final bits, and comparison has to move to a tolerance.
    #[test]
    fn both_backends_normalize_to_the_same_result() {
        let case = case();
        let from_cpu = TensorNormalizer.normalize(ndarray().run(&case).unwrap());
        let from_torch = TensorNormalizer.normalize(libtorch().run(&case).unwrap());

        assert_eq!(from_cpu, from_torch);
    }
}
