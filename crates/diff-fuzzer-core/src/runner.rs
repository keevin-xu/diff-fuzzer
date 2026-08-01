//! Gluing an implementation to the normaliser for its output.
//!
//! This exists to solve a concrete problem. Two backends produce two *different* Rust
//! types — `Tensor<NdArray, 2>` and `Tensor<LibTorch, 2>` — so they cannot be kept in
//! one list, and a driver cannot loop over them. But once each has been paired with
//! the normaliser that converts its output, both become "something that turns an input
//! into a `CanonicalTensor`", which is a single type the driver can hold many of.
//!
//! So the pairing is where the backend-specific types stop and uniformity begins. That
//! is also why the driver can be written once for any number of implementations rather
//! than hardcoding two, which is what makes adding a third backend later a small
//! change instead of a rewrite.

use crate::traits::{Implementation, Input, Normalizer, RunError};

/// Something that turns an input into a comparable result.
///
/// Deliberately narrow — one method that does the work, one that says who did it. That
/// narrowness is what lets the driver hold a mixed collection of these as trait objects
/// (`&dyn Runner<In = ..., Canon = ...>`), despite each concrete one having a different
/// underlying implementation and output type.
pub trait Runner {
    type In: Input;
    type Canon;

    /// How this system identifies itself in reports.
    fn name(&self) -> &str;

    /// Execute the input and convert the result into comparable form.
    fn run_and_normalize(&self, input: &Self::In) -> Result<Self::Canon, RunError>;
}

/// An [`Implementation`] paired with the [`Normalizer`] for its output.
///
/// The `where` clause on the trait implementation below is what enforces the pairing:
/// `N: Normalizer<Out = I::Out>` means the normaliser must accept exactly what this
/// implementation produces. Pairing a backend with the wrong normaliser is a compile
/// error, not a confusing result at runtime.
#[derive(Debug, Clone, Copy)]
pub struct NormalizedRunner<I, N> {
    implementation: I,
    normalizer: N,
}

impl<I, N> NormalizedRunner<I, N> {
    pub fn new(implementation: I, normalizer: N) -> Self {
        Self {
            implementation,
            normalizer,
        }
    }
}

impl<I, N> Runner for NormalizedRunner<I, N>
where
    I: Implementation,
    N: Normalizer<Out = I::Out>,
{
    type In = I::In;
    type Canon = N::Canon;

    fn name(&self) -> &str {
        self.implementation.name()
    }

    fn run_and_normalize(&self, input: &Self::In) -> Result<Self::Canon, RunError> {
        // `?` returns early if the implementation could not run this input, so a
        // failure propagates to the driver rather than being mistaken for a result.
        let output = self.implementation.run(input)?;
        Ok(self.normalizer.normalize(output))
    }
}
