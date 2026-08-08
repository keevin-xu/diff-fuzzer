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
    /// The error function — **no closed form, so every implementation picks its own
    /// approximation.**
    ///
    /// `burn-flex` delegates to `libm::erff`, a piecewise-rational approximation derived from
    /// fdlibm whose formula switches near `|x| = 0.84375`; `burn-tch` uses libtorch's own.
    /// **A switch point is a property of the input that selects between code paths** — the
    /// same shape as the tile remainder that explained the one bug filed upstream, and the
    /// reason this operation is worth more than its arithmetic suggests.
    ///
    /// Bounded in `[-1, 1]` and monotone, so it has none of `exp`'s overflow behaviour.
    Erf,
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
    /// A product across the axis.
    ///
    /// **Chosen because it is the closest sibling to the one clear bug found so far.** A
    /// parallel reduction seeds an accumulator with an identity — `1` here, where `max` seeds
    /// `-f32::MAX` — and `burn-wgpu` was found returning that sentinel instead of `-inf`.
    /// Whether the multiplicative identity is handled better is a question with the same
    /// shape and a different answer available.
    ///
    /// Its edges are also sharper than a sum's: `prod([inf, 0])` is `NaN`, overflow arrives
    /// far faster, and a single zero annihilates the whole axis.
    Prod,
}

/// Operations producing one output **per element**, scanning along an axis.
///
/// **Structurally unlike everything else tested.** A reduction collapses an axis to one
/// value; a scan produces a running result at every position. That difference is not
/// cosmetic — a sequential implementation keeps a running total, while a parallel one uses a
/// prefix-scan algorithm (Hillis–Steele or Blelloch) that **associates the additions
/// differently by construction**.
///
/// Floating-point addition is not associative, so two correct implementations of the same
/// scan can return different last bits *by design*. That makes this the strongest remaining
/// candidate for a **numeric** disagreement — the class 3.9 million cases of ordinary
/// arithmetic failed to produce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ScanOp {
    /// Running sum along the axis.
    CumSum,
    /// Running product along the axis.
    ///
    /// The same association question as [`ScanOp::CumSum`], plus one of its own: a running
    /// product reaches `inf` far faster than a running sum reaches anything, and **where it
    /// overflows depends on the order the multiplications are grouped in**. A tree scan can
    /// overflow at a position a sequential one does not, and vice versa.
    CumProd,
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

/// The non-tensor parameters of a 2-D convolution.
///
/// **A struct rather than four loose fields on the variant**, because these four travel
/// together everywhere — generation, shrinking, validity checking and the backend call all
/// want the whole set — and because they are the *only* part of a convolution that is not a
/// tensor.
///
/// **Why these four are the interesting part.** They do not change what a convolution
/// computes so much as *which code path computes it*. `burn-flex` selects among five
/// algorithms using exactly these values (`burn-flex/src/ops/conv.rs:1506`), and the one
/// upstream forward-pass bug this phase is modelled on — burn#4727 — triggered on
/// `groups > 1` together with `padding > 0`. So the generator's job is largely to move these
/// numbers across path boundaries.
///
/// `Copy` because it is four `usize`s and a pair of two-element arrays: cheap to duplicate,
/// and passing it by value avoids threading a borrow through the generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Conv2dParams {
    /// Step between successive windows, `[height, width]`. Never zero.
    pub stride: [usize; 2],
    /// Zeros added to each side of each spatial dimension, `[height, width]`.
    pub padding: [usize; 2],
    /// Spacing between kernel taps, `[height, width]`. Never zero; `1` means dense.
    pub dilation: [usize; 2],
    /// How many independent channel groups the convolution splits into.
    ///
    /// `1` is an ordinary convolution; `groups == in_channels == out_channels` is a depthwise
    /// convolution, which `burn-flex` gives its own kernel. Never zero, and it must divide
    /// both channel counts — see [`TensorOp::conv2d`].
    pub groups: usize,
}

impl Default for Conv2dParams {
    /// The identity-ish configuration: dense, unpadded, unit stride, one group.
    ///
    /// Note this is **not** `#[derive(Default)]`, which would give `0` for stride, dilation
    /// and groups — all three of which are invalid. A derived default here would have been a
    /// silent trap.
    fn default() -> Self {
        Conv2dParams {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
        }
    }
}

/// The spatial extent a convolution produces along one axis, or `None` if the window does not
/// fit even once.
///
/// `floor((in + 2*pad - dil*(k-1) - 1) / stride) + 1`, the standard formula. Returning
/// `Option` rather than panicking is deliberate: **generation and shrinking both need to ask
/// this question speculatively**, about a configuration they are considering and may reject.
/// A panicking version would force them to duplicate the arithmetic to avoid tripping it.
///
/// The subtraction is done in `i64` because `dil*(k-1) + 1` can exceed `in + 2*pad`, and the
/// same expression in `usize` would wrap to an enormous number rather than going negative —
/// which would report a valid output size for a window that does not fit.
pub fn conv2d_output_size(
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Option<usize> {
    let effective_kernel = dilation as i64 * (kernel as i64 - 1) + 1;
    let span = input as i64 + 2 * padding as i64 - effective_kernel;
    if span < 0 || stride == 0 {
        return None;
    }
    Some((span / stride as i64) as usize + 1)
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
    /// A running result along one axis, one output per element. See [`ScanOp`].
    Scan {
        kind: ScanOp,
        arg: TensorValue,
        /// Which axis is scanned along. Always less than the argument's rank.
        dim: usize,
    },
    /// A 2-D convolution — the first operation whose three backends run different
    /// *algorithms* rather than the same arithmetic in a different order.
    ///
    /// `burn-flex` uses tiled im2col + GEMM with five shape-selected fast paths, `burn-tch`
    /// hands off to libtorch, and `burn-wgpu` runs its own CubeCL kernel. See
    /// `planning/phases/PHASE-7G-convolution.md`.
    ///
    /// **The first variant with an optional operand.** `bias` is `Option` because a
    /// convolution genuinely may not have one, and that is itself a divergence surface: a
    /// backend that folds the bias into its accumulator rounds differently from one that adds
    /// it in a separate pass.
    Conv2d {
        /// `[batch, in_channels, height, width]`.
        input: TensorValue,
        /// `[out_channels, in_channels / groups, kernel_height, kernel_width]`.
        weight: TensorValue,
        /// `[out_channels]`, when present.
        bias: Option<TensorValue>,
        params: Conv2dParams,
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
    /// If `dim` is not a dimension of `arg`.
    pub fn scan(kind: ScanOp, arg: TensorValue, dim: usize) -> Self {
        assert!(
            dim < arg.rank(),
            "dim {dim} is out of range for a rank-{} tensor",
            arg.rank()
        );
        TensorOp::Scan { kind, arg, dim }
    }

    /// Builds a 2-D convolution, refusing anything dimensionally invalid.
    ///
    /// # Why every one of these is an assertion rather than a `Result`
    ///
    /// An invalid convolution is not a finding and not an error to recover from — it is a
    /// **generator bug**. burn panics on a malformed one, and under `cargo-fuzz` that panic
    /// is reported as a crash, which would bury real divergences under our own noise. So the
    /// constraints are enforced where a case is *built*, and step 7G.2's job is to prove the
    /// generator can never reach them.
    ///
    /// # Panics
    ///
    /// If any of the following does not hold:
    /// - `input` is rank 4 and `weight` is rank 4;
    /// - `groups`, every `stride` and every `dilation` are non-zero;
    /// - `in_channels` and `out_channels` are both divisible by `groups`;
    /// - `weight`'s second dimension equals `in_channels / groups`;
    /// - `bias`, if present, is rank 1 with `out_channels` elements;
    /// - the window fits at least once along both spatial axes.
    pub fn conv2d(
        input: TensorValue,
        weight: TensorValue,
        bias: Option<TensorValue>,
        params: Conv2dParams,
    ) -> Self {
        assert_eq!(
            input.rank(),
            4,
            "conv2d input must be [batch, in_channels, h, w], got {:?}",
            input.shape()
        );
        assert_eq!(
            weight.rank(),
            4,
            "conv2d weight must be [out_channels, in_channels/groups, kh, kw], got {:?}",
            weight.shape()
        );
        assert!(params.groups > 0, "groups must be non-zero");
        assert!(
            params.stride.iter().all(|s| *s > 0),
            "stride must be non-zero, got {:?}",
            params.stride
        );
        assert!(
            params.dilation.iter().all(|d| *d > 0),
            "dilation must be non-zero, got {:?}",
            params.dilation
        );

        let (in_channels, out_channels) = (input.shape()[1], weight.shape()[0]);
        assert_eq!(
            in_channels % params.groups,
            0,
            "in_channels {in_channels} is not divisible by groups {}",
            params.groups
        );
        assert_eq!(
            out_channels % params.groups,
            0,
            "out_channels {out_channels} is not divisible by groups {}",
            params.groups
        );
        assert_eq!(
            weight.shape()[1],
            in_channels / params.groups,
            "weight's input-channel dimension must be in_channels/groups = {}",
            in_channels / params.groups
        );

        if let Some(bias) = &bias {
            assert_eq!(
                bias.shape(),
                [out_channels],
                "bias must be [out_channels] = [{out_channels}]"
            );
        }

        // Both spatial axes, so a window that fits horizontally but not vertically is caught.
        for axis in 0..2 {
            assert!(
                conv2d_output_size(
                    input.shape()[2 + axis],
                    weight.shape()[2 + axis],
                    params.stride[axis],
                    params.padding[axis],
                    params.dilation[axis],
                )
                .is_some_and(|size| size > 0),
                "the kernel does not fit along spatial axis {axis}: input {:?}, kernel {:?}, \
                 {params:?}",
                input.shape(),
                weight.shape()
            );
        }

        TensorOp::Conv2d {
            input,
            weight,
            bias,
            params,
        }
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
                UnaryOp::Erf => "erf",
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
                ReduceOp::Prod => "prod",
            },
            TensorOp::Matmul { .. } => "matmul",
            TensorOp::Activation { kind, .. } => match kind {
                ActivationOp::Softmax => "softmax",
            },
            TensorOp::Scan { kind, .. } => match kind {
                ScanOp::CumSum => "cumsum",
                ScanOp::CumProd => "cumprod",
            },
            TensorOp::Conv2d { .. } => "conv2d",
        }
    }

    /// Every tensor this case carries, in argument order.
    ///
    /// **The second thing `conv2d` found four copies of** — `features::operands`,
    /// `negatives::operand_values`, `tolerance::has_subnormal_input` and
    /// `examples/triage.rs` each rebuilt this list. One of them used a fixed `[&_; 2]` array
    /// and had to write `[arg, arg]` for single-operand cases, which worked only because
    /// every caller happened to be asking a disjunctive question. A convolution with a bias
    /// has three operands and broke it.
    pub fn operands(&self) -> Vec<&TensorValue> {
        match self {
            TensorOp::Unary { arg, .. }
            | TensorOp::Reduce { arg, .. }
            | TensorOp::Activation { arg, .. }
            | TensorOp::Scan { arg, .. } => vec![arg],
            TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => vec![lhs, rhs],
            TensorOp::Conv2d {
                input,
                weight,
                bias,
                ..
            } => match bias {
                Some(bias) => vec![input, weight, bias],
                None => vec![input, weight],
            },
        }
    }

    /// How many values this case holds across all of its operands.
    ///
    /// **Centralised here after `conv2d` found four copies of it** — in `shrink`'s tests,
    /// `examples/campaign.rs`, `examples/triage_findings.rs` and `tests/repro.rs`. Each was a
    /// separate `match` that a new variant silently broke, and the fourth would have been
    /// found only by a failing build. One definition means the next operation updates one
    /// place.
    pub fn element_count(&self) -> usize {
        match self {
            TensorOp::Unary { arg, .. }
            | TensorOp::Reduce { arg, .. }
            | TensorOp::Activation { arg, .. }
            | TensorOp::Scan { arg, .. } => arg.len(),
            TensorOp::Binary { lhs, rhs, .. } | TensorOp::Matmul { lhs, rhs } => {
                lhs.len() + rhs.len()
            }
            TensorOp::Conv2d {
                input,
                weight,
                bias,
                ..
            } => input.len() + weight.len() + bias.as_ref().map_or(0, |b| b.len()),
        }
    }

    /// The rank of this case's arguments, which decides how it gets executed.
    pub fn rank(&self) -> usize {
        match self {
            TensorOp::Unary { arg, .. }
            | TensorOp::Reduce { arg, .. }
            | TensorOp::Activation { arg, .. }
            | TensorOp::Scan { arg, .. } => arg.rank(),
            TensorOp::Binary { lhs, .. } | TensorOp::Matmul { lhs, .. } => lhs.rank(),
            // Always 4. The constructor enforces it, so this reads the value rather than
            // asserting it again — a second copy of the rule is a second thing to get wrong.
            TensorOp::Conv2d { input, .. } => input.rank(),
        }
    }
}

impl Input for TensorOp {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tensor of the given shape filled with ones — shape is all these tests care about.
    fn ones(shape: &[usize]) -> TensorValue {
        TensorValue::new(shape.to_vec(), vec![1.0; shape.iter().product()])
    }

    /// `[batch=1, in=4, h=5, w=5]` convolved with `[out=2, in=4, kh=3, kw=3]`.
    fn plain() -> (TensorValue, TensorValue) {
        (ones(&[1, 4, 5, 5]), ones(&[2, 4, 3, 3]))
    }

    #[test]
    fn a_valid_convolution_constructs_and_reports_itself() {
        let (input, weight) = plain();
        let case = TensorOp::conv2d(input, weight, None, Conv2dParams::default());

        assert_eq!(case.name(), "conv2d");
        assert_eq!(case.rank(), 4, "a convolution is always rank 4");
    }

    /// **The default must be a valid convolution, not a zeroed struct.** A derived `Default`
    /// would give `stride: [0, 0]`, `dilation: [0, 0]` and `groups: 0`, all three invalid —
    /// so this test guards a hand-written impl that exists precisely to avoid that trap.
    #[test]
    fn the_default_parameters_describe_a_dense_unpadded_convolution() {
        let p = Conv2dParams::default();
        assert_eq!(
            (p.stride, p.padding, p.dilation, p.groups),
            ([1, 1], [0, 0], [1, 1], 1)
        );

        let (input, weight) = plain();
        TensorOp::conv2d(input, weight, None, p);
    }

    #[test]
    fn a_grouped_convolution_constructs_when_the_channels_divide() {
        // 4 input channels in 2 groups: each group sees 2, so the weight's second dim is 2.
        let case = TensorOp::conv2d(
            ones(&[1, 4, 5, 5]),
            ones(&[2, 2, 3, 3]),
            None,
            Conv2dParams {
                groups: 2,
                ..Default::default()
            },
        );
        assert_eq!(case.name(), "conv2d");
    }

    #[test]
    #[should_panic(expected = "not divisible by groups")]
    fn channels_that_do_not_divide_by_groups_are_rejected() {
        TensorOp::conv2d(
            ones(&[1, 3, 5, 5]),
            ones(&[2, 1, 3, 3]),
            None,
            Conv2dParams {
                groups: 2,
                ..Default::default()
            },
        );
    }

    /// The subtlest constraint, and the one a generator gets wrong most easily: the weight's
    /// second dimension is `in_channels / groups`, **not** `in_channels`.
    #[test]
    #[should_panic(expected = "in_channels/groups")]
    fn a_weight_sized_for_the_wrong_group_count_is_rejected() {
        TensorOp::conv2d(
            ones(&[1, 4, 5, 5]),
            ones(&[2, 4, 3, 3]),
            None,
            Conv2dParams {
                groups: 2,
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "does not fit along spatial axis")]
    fn a_kernel_larger_than_its_input_is_rejected() {
        TensorOp::conv2d(
            ones(&[1, 1, 3, 3]),
            ones(&[1, 1, 5, 5]),
            None,
            Conv2dParams::default(),
        );
    }

    /// Both spatial axes are checked, so a kernel that fits horizontally and not vertically
    /// is still caught. An earlier draft checked only one axis and would have passed this.
    #[test]
    #[should_panic(expected = "spatial axis 0")]
    fn a_kernel_that_fits_on_one_axis_only_is_still_rejected() {
        TensorOp::conv2d(
            ones(&[1, 1, 2, 9]),
            ones(&[1, 1, 5, 5]),
            None,
            Conv2dParams::default(),
        );
    }

    #[test]
    fn padding_can_make_an_otherwise_oversized_kernel_fit() {
        TensorOp::conv2d(
            ones(&[1, 1, 3, 3]),
            ones(&[1, 1, 5, 5]),
            None,
            Conv2dParams {
                padding: [1, 1],
                ..Default::default()
            },
        );
    }

    #[test]
    #[should_panic(expected = "bias must be")]
    fn a_bias_of_the_wrong_length_is_rejected() {
        let (input, weight) = plain();
        TensorOp::conv2d(input, weight, Some(ones(&[3])), Conv2dParams::default());
    }

    #[test]
    fn a_bias_matching_the_output_channels_is_accepted() {
        let (input, weight) = plain();
        TensorOp::conv2d(input, weight, Some(ones(&[2])), Conv2dParams::default());
    }

    #[test]
    #[should_panic(expected = "groups must be non-zero")]
    fn zero_groups_is_rejected() {
        let (input, weight) = plain();
        TensorOp::conv2d(
            input,
            weight,
            None,
            Conv2dParams {
                groups: 0,
                ..Default::default()
            },
        );
    }

    /// **The reason `conv2d_output_size` does its arithmetic in `i64`.** In `usize`, the
    /// subtraction `input + 2*padding - effective_kernel` wraps to an enormous number when the
    /// kernel is too big, and the function would report a valid size for a window that does
    /// not fit — turning a rejected case into a panic inside burn.
    #[test]
    fn the_output_size_is_none_when_the_window_does_not_fit() {
        assert_eq!(conv2d_output_size(3, 5, 1, 0, 1), None);
        assert_eq!(conv2d_output_size(5, 3, 1, 0, 1), Some(3));
        assert_eq!(conv2d_output_size(5, 3, 2, 0, 1), Some(2));
        assert_eq!(conv2d_output_size(5, 3, 1, 1, 1), Some(5));
        // Dilation widens the kernel: 3 taps spaced 2 apart span 5.
        assert_eq!(conv2d_output_size(5, 3, 1, 0, 2), Some(1));
        assert_eq!(conv2d_output_size(4, 3, 1, 0, 2), None);
    }
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
