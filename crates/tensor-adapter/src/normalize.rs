//! Making two backends' results comparable.
//!
//! Each backend hands back its own tensor type — one wrapping a Rust array, the other
//! wrapping a pointer into libtorch's C++ heap. They are different Rust types, so they
//! cannot be compared at all until both are converted into a common form.
//!
//! This step is small here and will not stay small. Whenever two systems are compared,
//! results that *mean* the same thing routinely *look* different — orderings, layouts,
//! ways of spelling the same number — and comparing before canonicalising is the
//! standard way a project like this drowns in differences that mean nothing. With two
//! backends of one framework the gap is unusually narrow, which is exactly why that
//! pairing was chosen.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use diff_fuzzer_core::Normalizer;
use std::marker::PhantomData;

/// A tensor result in a form any backend's output can be converted to.
///
/// Shape is kept alongside the values because a flat list alone would make a 2x3 and
/// a 3x2 result indistinguishable — and two backends returning the same numbers in a
/// different shape is a real disagreement, in fact a louder one than a numeric
/// difference.
#[derive(Debug, Clone, PartialEq)]
pub struct CanonicalTensor {
    pub shape: [usize; 2],
    /// Values row by row.
    pub values: Vec<f32>,
}

/// Converts one backend's tensors into [`CanonicalTensor`].
///
/// `PhantomData<B>` is needed because `B` appears nowhere in the struct's fields, yet
/// the type still has to be tied to one backend — a normaliser for ndarray tensors
/// must not be usable on libtorch tensors. `PhantomData` says "this type is associated
/// with `B`" while occupying no memory at runtime.
#[derive(Debug, Clone, Copy)]
pub struct TensorNormalizer<B: Backend> {
    _backend: PhantomData<B>,
}

impl<B: Backend> TensorNormalizer<B> {
    pub fn new() -> Self {
        Self {
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Default for TensorNormalizer<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Normalizer for TensorNormalizer<B> {
    type Out = Tensor<B, 2>;
    type Canon = CanonicalTensor;

    /// Takes the tensor by value, because reading the numbers out of a backend's
    /// representation consumes it — and consuming avoids copying a buffer that may be
    /// large.
    fn normalize(&self, out: Self::Out) -> CanonicalTensor {
        // Read the shape before the tensor is consumed on the next line.
        let shape = out.dims();

        // This cannot fail: every backend in use is instantiated with `f32` as its
        // element type, so the extraction always matches. If a backend with a
        // different element type is ever added, this is the line that will need to
        // become fallible — and the trait would need to return a `Result` for it.
        let values = out
            .into_data()
            .to_vec::<f32>()
            .expect("backends are instantiated with f32 elements");

        CanonicalTensor { shape, values }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{libtorch, ndarray};
    use crate::generator::FixedAddGenerator;
    use diff_fuzzer_core::{Generator, Implementation, SeededRng};

    fn fixed_case() -> crate::input::TensorOp {
        FixedAddGenerator.generate(&mut SeededRng::from_seed(0))
    }

    #[test]
    fn normalizing_keeps_shape_and_values() {
        let out = ndarray().run(&fixed_case()).unwrap();
        let canon = TensorNormalizer::new().normalize(out);

        assert_eq!(canon.shape, [2, 2]);
        assert_eq!(canon.values, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// The first time both halves of the comparison meet: two different backends, two
    /// different native tensor types, one common form — and on this input, identical.
    ///
    /// Exact equality is used here and is the wrong tool in general. It works because
    /// these values are small integers, which `f32` represents precisely. Once inputs
    /// are generated rather than hand-picked, two correct backends will routinely
    /// differ in the final bits, and comparison has to move to a tolerance.
    #[test]
    fn both_backends_normalize_to_the_same_result() {
        let case = fixed_case();
        let from_cpu = TensorNormalizer::new().normalize(ndarray().run(&case).unwrap());
        let from_torch = TensorNormalizer::new().normalize(libtorch().run(&case).unwrap());

        assert_eq!(from_cpu, from_torch);
    }
}
