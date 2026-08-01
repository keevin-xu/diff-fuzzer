//! What a tensor test case looks like.
//!
//! Deliberately narrow for now: a single operation on two rank-2 `f32` tensors. The
//! shape this eventually grows into — an enum with a variant per operation, each
//! carrying its own arguments — is sketched at the bottom of this file, but building
//! it before anything can execute would be guessing.
//!
//! One thing worth noticing: the tensor data lives here as a plain `Vec<f32>`, not as
//! a `burn` tensor. A test case has to be independent of any backend, because the
//! entire point is to hand the *same* case to several of them. A case that already
//! belonged to one backend could not be given to another.

use diff_fuzzer_core::Input;

/// A rank-2 tensor (a matrix) of `f32`, stored row-major.
///
/// Rank is in the name rather than the data because `burn` puts rank in the *type*
/// (`Tensor<B, 2>`), checked at compile time. That means rank cannot be an ordinary
/// runtime value that varies per test case — supporting several ranks later will mean
/// an enum with a variant per rank, not a `rank: usize` field.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    /// `rows * cols` values, row by row.
    data: Vec<f32>,
}

impl Matrix {
    /// Build a matrix, checking that the data matches the shape.
    ///
    /// # Panics
    ///
    /// If `data.len() != rows * cols`. This is an invariant every constructor path is
    /// responsible for upholding — a violation is a bug in the generator, not bad
    /// input from outside, so failing immediately and loudly is right. Everything
    /// downstream may then assume the shape is consistent.
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrix data length must equal rows * cols"
        );
        Self { rows, cols, data }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The values, row by row.
    ///
    /// Returns a borrowed slice rather than a copy of the `Vec`. The caller can read
    /// every element without anything being duplicated, and cannot modify or free it,
    /// because the matrix still owns it.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
}

/// Which operation a test case applies.
///
/// One variant today. It exists as an enum anyway because that is the shape the real
/// generator needs, and adding a variant to an existing enum is a smaller change than
/// introducing the enum later — every `match` on it will then tell us, at compile
/// time, exactly which places need to handle the new operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpKind {
    /// Elementwise addition of two identically-shaped tensors.
    Add,
}

/// One tensor test case: an operation and its arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorOp {
    pub op: OpKind,
    pub lhs: Matrix,
    pub rhs: Matrix,
}

impl TensorOp {
    /// Build an elementwise `add` case.
    ///
    /// # Panics
    ///
    /// If the two shapes differ. Elementwise addition requires identical shapes, and
    /// producing arguments that satisfy an operation's rules is the generator's job —
    /// so a mismatch here means the generator is wrong, and should fail loudly rather
    /// than be discovered later as a confusing backend error.
    pub fn add(lhs: Matrix, rhs: Matrix) -> Self {
        assert_eq!(
            lhs.shape(),
            rhs.shape(),
            "elementwise add requires identical shapes"
        );
        Self {
            op: OpKind::Add,
            lhs,
            rhs,
        }
    }
}

// Declaring this type a valid test case. The trait has no methods; it requires `Clone`
// (failures get repeatedly copied and modified while being shrunk) and `Debug` (a case
// that cannot be printed cannot be reported), both of which are derived above.
impl Input for TensorOp {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_reports_its_shape_and_data() {
        let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.shape(), [2, 3]);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        assert_eq!(m.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    #[should_panic(expected = "must equal rows * cols")]
    fn matrix_rejects_data_that_does_not_match_its_shape() {
        Matrix::new(2, 3, vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "identical shapes")]
    fn add_rejects_mismatched_shapes() {
        TensorOp::add(
            Matrix::new(2, 2, vec![0.0; 4]),
            Matrix::new(3, 3, vec![0.0; 9]),
        );
    }
}
