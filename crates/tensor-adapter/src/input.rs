//! What a tensor test case looks like.
//!
//! Two ideas shape everything here.
//!
//! **A case is backend-independent.** Values live in a plain `Vec<f32>`, never in a
//! `burn` tensor, because the whole point is handing the *same* case to several
//! backends — a case that already belonged to one could not be given to another.
//!
//! **A case cannot describe something invalid.** Each operation gets its own variant
//! carrying exactly the arguments it needs: a unary operation has no second argument
//! to mismatch, a reduction always has an axis. Combined with checked constructors,
//! the shapes of invalid inputs are largely unrepresentable rather than merely
//! unlikely — which matters because an input rejected as malformed tests nothing but
//! the validation code.

use diff_fuzzer_core::Input;

/// A tensor of `f32`: its shape, and its values in row-major order.
///
/// Rank is `shape.len()` — a runtime value here, deliberately. `burn` puts rank in the
/// *type* (`Tensor<B, 2>`), which would force rank to be fixed at compile time for
/// every case. Keeping it as data means one generator can produce vectors, matrices
/// and batched tensors alike; the cost is a single place where the runtime rank is
/// turned back into a type, which lives in the backend module.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorValue {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl TensorValue {
    /// Build a tensor value, checking that the data matches the shape.
    ///
    /// # Panics
    ///
    /// If `data.len()` differs from the product of `shape`, or if the shape is empty.
    /// Both mean the generator built something incoherent — a defect in this tool, not
    /// bad input from outside — so failing immediately is right, and everything
    /// downstream may then assume shape and data agree.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        assert!(
            !shape.is_empty(),
            "a tensor must have at least one dimension"
        );
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length must equal the product of the shape {shape:?}"
        );
        Self { shape, data }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total number of values.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Operations taking one tensor and returning one of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Abs,
    /// Can overflow to infinity for large arguments — deliberately kept, as numeric
    /// extremes are where implementations tend to part company.
    Exp,
    /// Undefined below zero. Arguments are generated non-negative for now; lifting
    /// that restriction is its own experiment, once there is a policy for what two
    /// backends both returning NaN should mean.
    Sqrt,
}

/// Operations taking two tensors of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// Divisors are generated away from zero for now, for the same reason as `Sqrt`.
    Div,
}

/// Operations collapsing one axis of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Summing many values is where floating-point addition's lack of associativity
    /// shows up: two backends adding in a different order get different last bits.
    /// Expect this operation to need a looser comparison than the others.
    Sum,
}

/// One tensor test case.
///
/// An enum rather than a struct with optional fields, so that each operation carries
/// precisely its own arguments and nothing else. Adding an operation means adding a
/// variant, after which the compiler names every place that has to handle it.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorOp {
    Unary {
        kind: UnaryOp,
        arg: TensorValue,
    },
    Binary {
        kind: BinaryOp,
        lhs: TensorValue,
        rhs: TensorValue,
    },
    Reduce {
        kind: ReduceOp,
        arg: TensorValue,
        /// Which axis collapses. Always less than the argument's rank.
        axis: usize,
    },
    /// Matrix multiplication, batched for ranks above two.
    Matmul {
        lhs: TensorValue,
        rhs: TensorValue,
    },
}

impl TensorOp {
    pub fn unary(kind: UnaryOp, arg: TensorValue) -> Self {
        TensorOp::Unary { kind, arg }
    }

    /// # Panics
    ///
    /// If the two shapes differ. Elementwise operations require identical shapes.
    pub fn binary(kind: BinaryOp, lhs: TensorValue, rhs: TensorValue) -> Self {
        assert_eq!(
            lhs.shape(),
            rhs.shape(),
            "elementwise {kind:?} requires identical shapes"
        );
        TensorOp::Binary { kind, lhs, rhs }
    }

    /// # Panics
    ///
    /// If `axis` is not a dimension of `arg`.
    pub fn reduce(kind: ReduceOp, arg: TensorValue, axis: usize) -> Self {
        assert!(
            axis < arg.rank(),
            "axis {axis} is out of range for a rank-{} tensor",
            arg.rank()
        );
        TensorOp::Reduce { kind, arg, axis }
    }

    /// # Panics
    ///
    /// If the operands are not both at least rank 2, if their inner dimensions do not
    /// agree, or if their batch dimensions differ. `[.., m, k]` times `[.., k, n]`
    /// gives `[.., m, n]`; the shared `k` is the constraint.
    pub fn matmul(lhs: TensorValue, rhs: TensorValue) -> Self {
        assert!(
            lhs.rank() >= 2 && rhs.rank() >= 2,
            "matmul needs rank 2 or more, got {} and {}",
            lhs.rank(),
            rhs.rank()
        );
        assert_eq!(
            lhs.rank(),
            rhs.rank(),
            "matmul operands must have equal rank"
        );
        let (ls, rs) = (lhs.shape(), rhs.shape());
        assert_eq!(
            ls[ls.len() - 1],
            rs[rs.len() - 2],
            "matmul inner dimensions must agree: {ls:?} times {rs:?}"
        );
        assert_eq!(
            ls[..ls.len() - 2],
            rs[..rs.len() - 2],
            "matmul batch dimensions must agree: {ls:?} times {rs:?}"
        );
        TensorOp::Matmul { lhs, rhs }
    }

    /// A short stable label, used in reports and for grouping findings.
    pub fn name(&self) -> &'static str {
        match self {
            TensorOp::Unary { kind, .. } => match kind {
                UnaryOp::Neg => "neg",
                UnaryOp::Abs => "abs",
                UnaryOp::Exp => "exp",
                UnaryOp::Sqrt => "sqrt",
            },
            TensorOp::Binary { kind, .. } => match kind {
                BinaryOp::Add => "add",
                BinaryOp::Sub => "sub",
                BinaryOp::Mul => "mul",
                BinaryOp::Div => "div",
            },
            TensorOp::Reduce { kind, .. } => match kind {
                ReduceOp::Sum => "sum",
            },
            TensorOp::Matmul { .. } => "matmul",
        }
    }

    /// The rank of this case's arguments, which decides how it gets executed.
    pub fn rank(&self) -> usize {
        match self {
            TensorOp::Unary { arg, .. } | TensorOp::Reduce { arg, .. } => arg.rank(),
            TensorOp::Binary { lhs, .. } | TensorOp::Matmul { lhs, .. } => lhs.rank(),
        }
    }
}

impl Input for TensorOp {}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(shape: &[usize]) -> TensorValue {
        let count = shape.iter().product();
        TensorValue::new(shape.to_vec(), vec![1.0; count])
    }

    #[test]
    fn a_value_reports_its_shape_rank_and_data() {
        let v = TensorValue::new(vec![2, 3], (0..6).map(|i| i as f32).collect());
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.rank(), 2);
        assert_eq!(v.len(), 6);
        assert_eq!(v.data()[5], 5.0);
    }

    #[test]
    #[should_panic(expected = "product of the shape")]
    fn a_value_rejects_data_that_does_not_match_its_shape() {
        TensorValue::new(vec![2, 3], vec![1.0, 2.0]);
    }

    #[test]
    fn operations_report_their_name_and_rank() {
        assert_eq!(TensorOp::unary(UnaryOp::Exp, value(&[4])).name(), "exp");
        assert_eq!(TensorOp::unary(UnaryOp::Exp, value(&[4])).rank(), 1);
        assert_eq!(
            TensorOp::reduce(ReduceOp::Sum, value(&[2, 3, 4]), 1).rank(),
            3
        );
    }

    #[test]
    #[should_panic(expected = "identical shapes")]
    fn elementwise_rejects_mismatched_shapes() {
        TensorOp::binary(BinaryOp::Add, value(&[2, 2]), value(&[3, 3]));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn reduce_rejects_an_axis_beyond_the_rank() {
        TensorOp::reduce(ReduceOp::Sum, value(&[2, 3]), 2);
    }

    #[test]
    #[should_panic(expected = "inner dimensions")]
    fn matmul_rejects_mismatched_inner_dimensions() {
        TensorOp::matmul(value(&[2, 3]), value(&[4, 5]));
    }

    #[test]
    #[should_panic(expected = "rank 2 or more")]
    fn matmul_rejects_vectors() {
        TensorOp::matmul(value(&[3]), value(&[3]));
    }

    #[test]
    fn matmul_accepts_agreeing_inner_dimensions_including_batched() {
        TensorOp::matmul(value(&[2, 3]), value(&[3, 4]));
        TensorOp::matmul(value(&[5, 2, 3]), value(&[5, 3, 4]));
    }

    #[test]
    #[should_panic(expected = "batch dimensions")]
    fn matmul_rejects_mismatched_batch_dimensions() {
        TensorOp::matmul(value(&[5, 2, 3]), value(&[6, 3, 4]));
    }
}
