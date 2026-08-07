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
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TensorValue {
    shape: Vec<usize>,
    #[serde(with = "non_finite_safe")]
    data: Vec<f32>,
}

/// Serialising `f32` values that JSON cannot represent.
///
/// **JSON has no `NaN` or infinity.** `serde_json` writes them as `null`, and reading a
/// `null` back into an `f32` fails — so a finding containing one is written to disk and is
/// then *unreadable*.
///
/// That is not hypothetical. It happened the first time a campaign generated non-finite
/// inputs (PHASE-7E): three findings were saved, none could be parsed, and triage reported
/// **"a campaign that found nothing"** — the reassuring message, for findings it had failed
/// to read. A long campaign would have produced hundreds of unreadable files and said
/// nothing was wrong.
///
/// So non-finite values are written as strings and read back from either form. Strings
/// rather than bit patterns because **a finding is read by people**: `"NaN"` in the JSON says
/// what it is, where `2143289344` does not.
mod non_finite_safe {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// A value on the wire: an ordinary number, or a name JSON can carry.
    #[derive(Serialize, Deserialize)]
    #[serde(untagged)]
    enum Value {
        Number(f32),
        Named(String),
    }

    pub fn serialize<S: Serializer>(data: &[f32], serializer: S) -> Result<S::Ok, S::Error> {
        let wire: Vec<Value> = data
            .iter()
            .map(|v| {
                if v.is_finite() {
                    Value::Number(*v)
                } else if v.is_nan() {
                    Value::Named("NaN".to_string())
                } else if *v > 0.0 {
                    Value::Named("inf".to_string())
                } else {
                    Value::Named("-inf".to_string())
                }
            })
            .collect();
        wire.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<f32>, D::Error> {
        let wire = Vec::<Value>::deserialize(deserializer)?;
        wire.into_iter()
            .map(|value| match value {
                Value::Number(v) => Ok(v),
                Value::Named(name) => match name.as_str() {
                    "NaN" => Ok(f32::NAN),
                    "inf" => Ok(f32::INFINITY),
                    "-inf" => Ok(f32::NEG_INFINITY),
                    other => Err(serde::de::Error::custom(format!(
                        "unrecognised value {other:?}"
                    ))),
                },
            })
            .collect()
    }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
    /// Undefined below zero, `-inf` at zero — and, unlike `exp`, **its error is worst near
    /// `x = 1`**, not at large arguments.
    ///
    /// The condition number of `log` is `1 / |ln x|`, which diverges as `x` approaches 1
    /// because `log(1) = 0` and a relative error against zero is unbounded. That is the
    /// opposite shape of bound from everything else in `POLICY.md`, and it is why `log`
    /// needed its own derivation rather than `exp`'s.
    Log,
}

/// Operations taking two tensors of the same shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    /// Divisors are generated away from zero for now, for the same reason as `Sqrt`.
    Div,
}

/// Operations collapsing one axis of a tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReduceOp {
    /// Summing many values is where floating-point addition's lack of associativity
    /// shows up: two backends adding in a different order get different last bits.
    /// Expect this operation to need a looser comparison than the others.
    Sum,
    /// A sum followed by a division. **Not the same bound as `Sum`**: the division adds a
    /// rounding of its own, and on Metal it is licensed 2.5 ULP rather than one.
    ///
    /// Also worth testing because backends may divide at different points — before summing,
    /// after, or fused into the accumulation — which changes both overflow behaviour and
    /// rounding.
    Mean,
    /// **No arithmetic at all**: the result is one of the inputs, returned unchanged.
    ///
    /// Which makes any disagreement a *semantic* one rather than a numeric one, and
    /// therefore impossible to excuse as precision. The interesting question is `NaN`:
    /// IEEE-754's `maxNum` ignores it and returns the other operand, and implementations
    /// genuinely differ — a hand-written SIMD kernel, libtorch, and a GPU reduction have no
    /// reason to agree on it. Signed zero is the other tie case.
    Max,
    /// The mirror of [`ReduceOp::Max`], and it can differ independently: an implementation
    /// may handle `NaN` one way in one and another way in the other.
    Min,
}

/// Operations normalising a tensor along one axis.
///
/// **Separate from [`UnaryOp`] even though the arity is the same**, because these carry a
/// dimension and unary operations do not. An optional `dim` on `Unary` would be meaningless
/// for `neg` and would invite exactly the bug where it is silently ignored; a separate
/// variant makes the compiler name every place that must handle it.
///
/// **This is where the interesting divergences are expected**, and the reason is that the
/// three backends implement it three different ways rather than sharing one kernel:
/// `burn-flex` hand-writes it, `burn-tch` delegates to libtorch, and `burn-wgpu` does not
/// override the default at all — composing five separate operations
/// (`max_dim`, `sub`, `exp`, `sum_dim`, `div`). Every operation tested before this one had
/// all three backends performing essentially the same arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ActivationOp {
    /// `exp(x_i - max) / sum_j exp(x_j - max)`, along one dimension.
    ///
    /// The max-subtraction is a stability measure, not part of the definition — which is
    /// itself a source of disagreement, since implementations may apply it differently or
    /// (in a fused kernel) at a different point in the computation.
    Softmax,
}

/// One tensor test case.
///
/// An enum rather than a struct with optional fields, so that each operation carries
/// precisely its own arguments and nothing else. Adding an operation means adding a
/// variant, after which the compiler names every place that has to handle it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// A normalisation along one axis. See [`ActivationOp`].
    Activation {
        kind: ActivationOp,
        arg: TensorValue,
        /// Which axis is normalised over. Always less than the argument's rank.
        dim: usize,
    },
}

impl TensorOp {
    pub fn unary(kind: UnaryOp, arg: TensorValue) -> Self {
        TensorOp::Unary { kind, arg }
    }

    /// # Panics
    ///
    /// If the two shapes do not combine. Since PHASE-7C elementwise operations **broadcast**,
    /// so the requirement is compatibility rather than identity: same rank, and each pair of
    /// extents equal or one of them `1`. See [`crate::ops::broadcast`].
    pub fn binary(kind: BinaryOp, lhs: TensorValue, rhs: TensorValue) -> Self {
        assert!(
            crate::ops::broadcast::compatible(lhs.shape(), rhs.shape()),
            "elementwise {kind:?} requires broadcast-compatible shapes, got {:?} and {:?}",
            lhs.shape(),
            rhs.shape()
        );
        TensorOp::Binary { kind, lhs, rhs }
    }

    /// # Panics
    ///
    /// If `dim` is not a dimension of `arg`.
    ///
    /// burn's `softmax` panics on an out-of-range dimension rather than returning an error,
    /// so the constraint is enforced here where a case is built, not discovered at run time.
    pub fn activation(kind: ActivationOp, arg: TensorValue, dim: usize) -> Self {
        assert!(
            dim < arg.rank(),
            "dim {dim} is out of range for a rank-{} tensor",
            arg.rank()
        );
        TensorOp::Activation { kind, arg, dim }
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
                UnaryOp::Log => "log",
            },
            TensorOp::Binary { kind, .. } => match kind {
                BinaryOp::Add => "add",
                BinaryOp::Sub => "sub",
                BinaryOp::Mul => "mul",
                BinaryOp::Div => "div",
            },
            TensorOp::Reduce { kind, .. } => match kind {
                ReduceOp::Sum => "sum",
                ReduceOp::Mean => "mean",
                ReduceOp::Max => "max",
                ReduceOp::Min => "min",
            },
            TensorOp::Matmul { .. } => "matmul",
            TensorOp::Activation { kind, .. } => match kind {
                ActivationOp::Softmax => "softmax",
            },
        }
    }

    /// The rank of this case's arguments, which decides how it gets executed.
    pub fn rank(&self) -> usize {
        match self {
            TensorOp::Unary { arg, .. }
            | TensorOp::Reduce { arg, .. }
            | TensorOp::Activation { arg, .. } => arg.rank(),
            TensorOp::Binary { lhs, .. } | TensorOp::Matmul { lhs, .. } => lhs.rank(),
        }
    }
}

impl Input for TensorOp {}

#[cfg(test)]
mod tests {
    /// **The regression that cost three findings.**
    ///
    /// JSON has no `NaN` or infinity, so `serde_json` wrote them as `null` and reading them
    /// back failed. Three real findings were saved to disk, none could be parsed, and triage
    /// announced "a campaign that found nothing". A long run would have lost hundreds
    /// silently.
    #[test]
    fn a_case_holding_non_finite_values_survives_a_json_round_trip() {
        let case = TensorOp::reduce(
            ReduceOp::Max,
            TensorValue::new(
                vec![2, 3],
                vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, 1.5],
            ),
            1,
        );

        let json = serde_json::to_string(&case).expect("serialises");
        let back: TensorOp = serde_json::from_str(&json).expect("must parse back");

        let TensorOp::Reduce { arg, .. } = &back else {
            unreachable!()
        };
        let values = arg.data();
        assert!(values[0].is_nan(), "NaN did not survive");
        assert_eq!(values[1], f32::INFINITY);
        assert_eq!(values[2], f32::NEG_INFINITY);
        // Signed zero must survive too: its sign is observable, and `0.0 == -0.0` would let a
        // broken round trip pass unnoticed.
        assert_eq!(values[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(values[4].to_bits(), (-0.0f32).to_bits());
        assert_eq!(values[5], 1.5);
    }

    /// Ordinary numbers stay ordinary in the file — a finding is read by people, and turning
    /// every value into a string to solve a problem three of them have would be a poor trade.
    #[test]
    fn finite_values_are_still_written_as_numbers() {
        let case = TensorOp::unary(UnaryOp::Neg, TensorValue::new(vec![2], vec![1.5, -2.5]));
        let json = serde_json::to_string(&case).expect("serialises");

        assert!(json.contains("1.5"), "{json}");
        assert!(
            !json.contains("\"1.5\""),
            "finite values should not be quoted: {json}"
        );
    }

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

    /// Incompatible extents — neither equal nor 1 — are still rejected. Broadcasting widened
    /// what is legal; it did not remove the constraint.
    #[test]
    #[should_panic(expected = "broadcast-compatible")]
    fn elementwise_rejects_incompatible_shapes() {
        TensorOp::binary(BinaryOp::Add, value(&[2, 2]), value(&[3, 3]));
    }

    /// Differing ranks are rejected too, for burn's reason rather than ours: `Tensor<B, D>`
    /// fixes the rank at compile time, so there is no way to express the NumPy form.
    #[test]
    #[should_panic(expected = "broadcast-compatible")]
    fn elementwise_rejects_differing_ranks() {
        TensorOp::binary(BinaryOp::Add, value(&[4]), value(&[3, 4]));
    }

    /// A stretched axis is now accepted where it previously panicked.
    #[test]
    fn elementwise_accepts_a_stretched_axis() {
        TensorOp::binary(BinaryOp::Add, value(&[3, 1]), value(&[3, 4]));
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
