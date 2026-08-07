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
/// Three parts, and the first two exist to stop the third from being compared when it
/// should not be.
///
/// **Shape**, because a flat list alone would make a 2x3 and a 3x2 result
/// indistinguishable — and two backends returning the same numbers in a different shape
/// is a real disagreement, in fact a louder one than a numeric difference, since they
/// disagree about the operation's meaning rather than its arithmetic.
///
/// **Dtype**, recorded as the backend actually reported it, *before* any conversion.
/// Values are then converted to a single common precision so that comparison is
/// meaningful — but the original type is kept, because converting first and comparing
/// afterwards would silently absorb a genuine disagreement about what type the operation
/// returns. Normalising a difference away and never mentioning it is how a tool stops
/// noticing things.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalTensor {
    pub shape: Vec<usize>,
    /// The element type the backend produced, before conversion to the comparison
    /// precision.
    pub dtype: String,
    /// Values in row-major order, converted to `f32` — the common precision at which
    /// comparison happens.
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

        // Recorded before conversion, so a backend that returned a different element
        // type can still be caught saying so.
        let dtype = format!("{:?}", out.dtype);

        // Converted rather than extracted. Reading `f32` out of a result that is not
        // `f32` would fail, and previously did so by panicking — which turns a
        // *finding* (two backends disagreeing about the result type) into a crash of
        // the tool. Converting always succeeds, and the dtype recorded above is what
        // preserves the difference for the comparison to notice.
        let values = out
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("values were just converted to f32");

        CanonicalTensor {
            shape,
            dtype,
            values,
        }
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

        // Element type is settled before the numbers are, and for the same reason as
        // shape: two backends returning different types for the same operation disagree
        // about what it *produces*, not about the value it computed. The values have
        // already been converted to a common precision, so a numeric comparison here
        // would succeed and quietly hide it.
        if self.dtype != other.dtype {
            return Agreement::Structural {
                reason: format!("element types differ: {} vs {}", self.dtype, other.dtype),
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
    use crate::backends::{flex, libtorch};
    use crate::input::{BinaryOp, TensorOp, TensorValue};
    use diff_fuzzer_core::Implementation;

    fn case() -> TensorOp {
        TensorOp::binary(
            BinaryOp::Add,
            TensorValue::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            TensorValue::new(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]),
        )
    }

    /// **A backend getting the broadcast output shape wrong must read as a divergence, not be
    /// absorbed.** Broadcasting is the first feature where two backends could plausibly
    /// disagree about the *shape* of a result rather than its values — one stretching an axis
    /// and another failing to — so the phase confirms the existing structural rule covers it
    /// rather than assuming so.
    ///
    /// The widest tolerance in the codebase is used deliberately: no amount of numeric
    /// slack may ever excuse a shape difference.
    #[test]
    fn a_disagreement_about_the_broadcast_output_shape_is_structural() {
        // What `[3,1] + [3,4]` should produce, against a backend that failed to stretch.
        let stretched = CanonicalTensor {
            shape: vec![3, 4],
            dtype: "F32".to_string(),
            values: vec![0.0; 12],
        };
        let unstretched = CanonicalTensor {
            shape: vec![3, 1],
            dtype: "F32".to_string(),
            values: vec![0.0; 3],
        };

        let generous = Tolerance {
            rtol: 1.0,
            atol: 1e30,
        };
        match stretched.approx_compare(&unstretched, generous) {
            Agreement::Structural { reason } => {
                assert!(
                    reason.contains("shapes differ"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("a shape disagreement was absorbed as {other:?}"),
        }
    }

    #[test]
    fn normalizing_keeps_shape_dtype_and_values() {
        let out = flex().run(&case()).unwrap();
        let canon = TensorNormalizer.normalize(out);

        assert_eq!(canon.shape, vec![2, 2]);
        assert_eq!(canon.values, vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(canon.dtype, "F32");
    }

    /// Both backends must report the same element type. They are configured with the
    /// same one, so a difference here would be genuinely surprising — which is exactly
    /// why it is worth asserting rather than assuming.
    #[test]
    fn both_backends_report_the_same_element_type() {
        let case = case();
        let from_cpu = TensorNormalizer.normalize(flex().run(&case).unwrap());
        let from_torch = TensorNormalizer.normalize(libtorch().run(&case).unwrap());

        assert_eq!(from_cpu.dtype, from_torch.dtype);
    }

    /// A result of a different element type must not panic during normalisation.
    ///
    /// It used to: reading `f32` out of a non-`f32` result failed, which turned a
    /// *finding* — two backends disagreeing about the result type — into a crash of the
    /// tool. Conversion makes it a comparison instead.
    #[test]
    fn a_different_element_type_is_converted_rather_than_crashing() {
        let doubles = TensorData::new(vec![1.0f64, 2.0], vec![2]);
        let canon = TensorNormalizer.normalize(doubles);

        assert_eq!(canon.values, vec![1.0f32, 2.0]);
        assert_eq!(canon.dtype, "F64");
    }

    /// ...and having been converted, the difference is still reported, because the
    /// original type was recorded first. Normalising a difference away and never
    /// mentioning it is how a tool stops noticing things.
    #[test]
    fn an_element_type_difference_is_reported_as_structural() {
        let floats = TensorNormalizer.normalize(TensorData::new(vec![1.0f32, 2.0], vec![2]));
        let doubles = TensorNormalizer.normalize(TensorData::new(vec![1.0f64, 2.0], vec![2]));

        // The values are identical after conversion, so a purely numeric comparison
        // would report agreement.
        assert_eq!(floats.values, doubles.values);

        let outcome = floats.approx_compare(&doubles, Tolerance::new(1e30, 1e30));
        let Agreement::Structural { reason } = outcome else {
            panic!("an element-type difference was absorbed: {outcome:?}");
        };
        assert!(reason.contains("element types differ"), "{reason}");
    }

    /// Shape is checked before element type, so a case differing in both reports the
    /// more fundamental disagreement first.
    #[test]
    fn a_shape_difference_is_reported_before_an_element_type_difference() {
        let a = TensorNormalizer.normalize(TensorData::new(vec![1.0f32], vec![1]));
        let b = TensorNormalizer.normalize(TensorData::new(vec![1.0f64, 2.0], vec![2]));

        let Agreement::Structural { reason } = a.approx_compare(&b, Tolerance::EXACT) else {
            panic!("expected a structural difference");
        };
        assert!(reason.contains("shapes differ"), "{reason}");
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
        let from_cpu = TensorNormalizer.normalize(flex().run(&case).unwrap());
        let from_torch = TensorNormalizer.normalize(libtorch().run(&case).unwrap());

        assert_eq!(from_cpu, from_torch);
    }
}
