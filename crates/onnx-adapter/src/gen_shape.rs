//! The world: what a generated case may look like, before any numbers are chosen.
//!
//! This module owns the **configuration** — which operators and element types are in play,
//! how large a case may be — and, in the next step, the shape generation itself. Values are
//! `gen_value.rs`'s job. The split is the analogue of the SQL domain's state-then-query
//! split: deciding the world before the numbers keeps constraint logic in one place and value
//! strategy in another, and lets the special-value rate vary independently of shape.
//!
//! # The rule that is easiest to get wrong
//!
//! **Enabling an axis must add cases, never remove them.** The SQL adapter learned this by
//! enabling joins and finding that a run reporting clean agreement had quietly stopped testing
//! ordering — joined queries are unordered, so every query became one. An axis that *displaces*
//! rather than *adds* turns a widened campaign into a narrower one while looking like progress.
//!
//! Every axis below is additive: it admits operators or element types that were previously
//! excluded, and never changes how an already-admitted case is built.
//!
//! # Bounding each knob does not bound the case
//!
//! `02-METHODOLOGY.md`: rank and dimension **multiply**, so caps of 4 and 64 permit 16.7
//! million elements. [`Bounds::element_budget`] bounds the *work*, which is the quantity that
//! actually needs bounding.
//!
//! The matching warning, from the same source: **a cap chosen because it looks free is still a
//! change to the distribution.** An element budget justified as "the old worst case, so it
//! costs nothing" once took a divergence rate from 9-in-2,000 to 0-in-2,000 — it had clamped
//! away exactly the shapes that diverge, while costing the full runtime anyway. The budget here
//! is therefore **measured on both sides** before it is trusted (N3.8), not assumed free.

use diff_fuzzer_core::axes::GenerationAxes;
use diff_fuzzer_core::rng::SeededRng;
use rand::RngExt;

use crate::attrs::Attrs;
use crate::case::{ElemType, OnnxCase, OpKind, TensorData, TensorValue};
use crate::gen_value;
use crate::ops::{self, Family, Tier};
use crate::validation::input_name;

/// A fingerprint of the code that decides what a case looks like.
///
/// # The half of drift a description cannot see
///
/// `GenerationAxes::description()` is derived from **declared** configuration, so it catches an
/// axis being flipped or a bound changed. It is blind to the *generation logic* changing while
/// every axis stays put — and the engine's own documentation cites the SQL adapter's
/// joins-versus-ordering fix as the motivating example: making joins probabilistic at 60%
/// rather than unconditional changed the distribution materially and touched no axis and no
/// scalar. Two corpora either side of it would have looked comparable.
///
/// So the source of every module that decides case content is hashed at **compile time** and
/// reported through `logic_version()`. Editing any of them changes the fingerprint, which
/// changes the description, which stops an old corpus being silently compared against a new one.
///
/// **What this still asks a human to remember:** a new module that decides case content must be
/// added to the list below. That is the one gap in the scheme, and it is noted rather than
/// papered over.
///
/// Erring toward *spurious* mismatch is deliberate: a comment-only edit changes the hash and
/// invalidates pools that are in fact still comparable. That costs a re-run and is visible. A
/// missed mismatch silently corrupts a measurement and is not.
pub const GENERATOR_FINGERPRINT: u32 = {
    let hash = fnv1a(include_bytes!("gen_shape.rs"), 0xcbf2_9ce4_8422_2325);
    let hash = fnv1a(include_bytes!("gen_value.rs"), hash);
    let hash = fnv1a(include_bytes!("ops.rs"), hash);
    let hash = fnv1a(include_bytes!("generator.rs"), hash);
    // Folded into 32 bits so the description stays short; collision risk is irrelevant for
    // detecting accidental drift.
    (hash ^ (hash >> 32)) as u32
};

/// FNV-1a, written as a `const fn` so the hash is computed during compilation.
///
/// A `while` loop rather than an iterator because iterators are not available in a `const`
/// context. The algorithm is chosen for being short enough to write here, not for quality —
/// it detects accidental edits, and nothing depends on it resisting an adversary.
const fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

/// What a generated case may contain.
///
/// Each boolean is an **axis** in the engine's sense: named, reported in the description
/// whether on or off, and additive. Each number is a **scalar** bound.
#[derive(Debug, Clone, PartialEq)]
pub struct Bounds {
    // ── which operators ───────────────────────────────────────────────────────────
    /// IEEE-754 elementwise float arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max`,
    /// `Abs`, `Neg`, `Sign`, `Sqrt`, `Floor`, `Ceil`, `Round`. Tier B — the densest
    /// special-value surface, and where both of this project's prior findings would have lived.
    pub float_elementwise: bool,
    /// `Equal`, `Greater`, `Less`. Tier A: discrete answers, and they return `Bool` whatever
    /// they were given.
    pub comparisons: bool,
    /// `And`, `Or`, `Xor`, `Not`. Tier A, and boolean-only by their type constraints.
    pub logical: bool,
    /// `Identity`, `Transpose`, `Concat`, `Gather`, `Where`, `Cast`. Tier A, shape-manipulating
    /// or selecting.
    ///
    /// **Excludes** the five operators that take a shape/axes/pads *input* — see
    /// [`Self::shape_input_operators`].
    pub structural: bool,
    /// `QuantizeLinear`, `DequantizeLinear`, `MatMulInteger`, `DynamicQuantizeLinear` — the
    /// Tier Q surface added at PHASE-N9.
    ///
    /// **A separate axis so the yield can be measured against the Tier A/B baseline.** N9.7
    /// requires reporting what quantization buys, and a rate without a baseline is not a
    /// measurement — the same rule that made `special_values` its own axis at N4.
    ///
    /// Off by default for exactly that reason: the baseline is the default configuration.
    pub quantized: bool,
    /// `Reshape`, `Squeeze`, `Unsqueeze`, `Slice`, `Pad` — the operators whose second input is
    /// an `I64` shape, axes, or pad vector.
    ///
    /// **A separate axis from [`Self::structural`]**, because their second input is
    /// configuration rather than data and is emitted as an initializer — see
    /// [`crate::case::InputRole`].
    ///
    /// It was held off by default while `PENDING` 1.10 was open: the census measured all five
    /// failing **0/5 on `tract`**, which types graphs statically at load and could not infer an
    /// output shape whose shape input only arrived at run time. Emitting those inputs as
    /// initializers fixed it — `tract` went from **0/5 to 5/5** on four of them and 4/5 on
    /// `Pad`, and its overall support rose from 97 to **121 of 126**. The axis is now on.
    pub shape_input_operators: bool,

    // ── which element types ───────────────────────────────────────────────────────
    /// 64-bit float alongside 32-bit. Doubles the float surface at no structural cost.
    pub float64: bool,
    /// `I32` and `I64`. Integer arithmetic is exactly determined — no rounding argument exists
    /// — which makes it the cleanest oracle available.
    pub integer_types: bool,
    /// `Bool`. Required by the logical operators and by `Where`'s condition.
    pub bool_type: bool,

    // ── what the values look like ─────────────────────────────────────────────────
    /// Inject `±inf`, `NaN`, `±0.0`, subnormals and type extremes at
    /// [`Self::special_value_rate`].
    ///
    /// **Off by default at N3**, so its effect on yield can be measured against a baseline
    /// rather than assumed. *A rate without a baseline is not a measurement.*
    pub special_values: bool,
    /// Rank-0 scalars and zero-length dimensions. Legal ONNX, and where implementations differ.
    pub degenerate_shapes: bool,
    /// Draw each case's opset from the operator's own span instead of pinning [`Self::opset`].
    ///
    /// # What this reaches that nothing else does
    ///
    /// The opset-22 corpus is **saturated**: N8 reached 40 signatures by ~50,000 seeds and
    /// 3,000,000 cases later yield 44. More seeds at one opset buy a bound, not information, and
    /// the roadmap's answer to that is to widen. Opset is the cheapest width left, because the
    /// valid span per operator already exists (`SPECS.md` §2.12) and costs nothing to compute.
    ///
    /// **Not the comparison `opset-invariance` makes.** That relation runs one runtime at two
    /// opsets and compares it against *itself*, so it is blind to any defect that does not vary
    /// with the version — which is all four problems this domain has found. Generating *at* opset
    /// 13 instead puts the **differential** oracle, which compares runtimes to each other, onto
    /// code paths it has never executed.
    ///
    /// Off by default, like [`Self::quantized`]: the default configuration is the baseline any
    /// yield from this axis has to be measured against.
    pub vary_opset: bool,

    // ── scalars ───────────────────────────────────────────────────────────────────
    pub max_rank: usize,
    pub max_dim: i64,
    /// The cap on **total elements per tensor** — the bound on *work*.
    ///
    /// Rank and dimension multiply, so bounding each separately does not bound the case.
    pub element_budget: usize,
    /// Fraction of elements drawn from the special-value pool, when
    /// [`Self::special_values`] is on.
    pub special_value_rate: f64,
    pub opset: i64,
}

impl Default for Bounds {
    /// The N3 starting configuration: **Tier A and Tier B elementwise, ordinary values**.
    ///
    /// Deliberately narrow. `08-RISKS.md` §10 names over-building before a finding as this
    /// project's documented main risk, and the counter-risk — over-narrow generation producing
    /// a confident zero — is addressed by widening *with a measurement on both sides*, not by
    /// starting wide.
    fn default() -> Self {
        Self {
            float_elementwise: true,
            comparisons: true,
            logical: true,
            structural: true,
            shape_input_operators: true,
            quantized: false,

            float64: true,
            integer_types: true,
            bool_type: true,

            // Off so N4 can measure what they buy against this baseline.
            special_values: false,
            degenerate_shapes: true,
            // Off by default: the default configuration is the baseline this axis is measured
            // against, exactly as `quantized` is.
            vary_opset: false,

            // These three are chosen together, and the test
            // `the_knobs_permit_more_than_the_budget_allows` is what keeps them honest.
            //
            // The first attempt had rank 3 and dimension 6, which permits 6³ = **216**
            // elements — *under* the 256 budget. The budget bounded nothing at all: it was
            // not merely "free", it was inert, and a bound that cannot bind is a bound
            // nobody should trust a measurement to. Rank 4 and dimension 8 permit 4,096,
            // so the budget now genuinely binds.
            //
            // **The budget's value is not yet justified.** 256 is a starting point, and
            // `02-METHODOLOGY.md` is explicit that a cap justified as costing nothing once
            // took a divergence rate from 9-in-2,000 to 0-in-2,000 — it had clamped away
            // exactly the shapes that diverge while costing the full runtime anyway. N3.8
            // measures on both sides of it before it is believed.
            max_rank: 4,
            max_dim: 8,
            element_budget: 256,
            special_value_rate: 0.25,
            opset: crate::model::DEFAULT_OPSET,
        }
    }
}

impl Bounds {
    /// The narrowest useful configuration: one operator family, one element type.
    ///
    /// N3.5 asks for "one axis first, end to end, before the table". This is that.
    pub fn one_axis() -> Self {
        Self {
            float_elementwise: true,
            comparisons: false,
            logical: false,
            structural: false,
            shape_input_operators: false,
            quantized: false,
            float64: false,
            integer_types: false,
            bool_type: false,
            special_values: false,
            degenerate_shapes: false,
            ..Self::default()
        }
    }

    /// The baseline control for measuring what special values buy.
    ///
    /// *A rate without a baseline is not a measurement* — a run rejecting everything at 0% is
    /// equally consistent with "the rules are wrong" and "this pair never disagrees".
    pub fn without_special_values(&self) -> Self {
        Self {
            special_values: false,
            ..self.clone()
        }
    }

    /// The same configuration with special values on.
    pub fn with_special_values(&self) -> Self {
        Self {
            special_values: true,
            ..self.clone()
        }
    }

    /// The same configuration with the quantized surface on.
    ///
    /// A constructor rather than a field flip at the call site, matching
    /// [`Self::with_special_values`]: a measurement comparing two configurations should differ
    /// in exactly one named thing, and a hand-edited struct literal cannot be trusted to.
    pub fn with_quantized(&self) -> Self {
        Self {
            quantized: true,
            ..self.clone()
        }
    }

    /// Draw each case's opset from the operator's span. See [`Self::vary_opset`].
    pub fn with_opsets(&self) -> Self {
        Self {
            vary_opset: true,
            ..self.clone()
        }
    }

    /// The element types this configuration permits.
    ///
    /// `F32` is unconditional: a configuration generating no element type at all would produce
    /// nothing, and an empty pool is an error rather than an empty pool.
    pub fn element_types(&self) -> Vec<ElemType> {
        let mut types = vec![ElemType::F32];
        if self.float64 {
            types.push(ElemType::F64);
        }
        if self.integer_types {
            types.push(ElemType::I32);
            types.push(ElemType::I64);
        }
        if self.bool_type {
            types.push(ElemType::Bool);
        }
        // **Tied to the quantized axis, not to `integer_types`.** `int8` and `uint8` exist in
        // this adapter only to be quantized into: no Tier A or Tier B operator accepts them, so
        // adding them to the general integer pool would produce nothing but `None` from
        // `build_case`.
        //
        // Missing this is why `DequantizeLinear` and `MatMulInteger` were generated **zero**
        // times on the first N9 measurement while appearing to be enabled — their type
        // constraints are `int8`/`uint8`, the pool never offered those types, and the operator
        // silently never came up. The yield table showed 0 cases rather than 0 findings, which
        // is the distinction `05-MEASUREMENT-AND-CAMPAIGNS.md` insists on and the reason that
        // table exists at all.
        if self.quantized {
            types.push(ElemType::I8);
            types.push(ElemType::U8);
        }
        types
    }

    /// The operators this configuration permits, at its opset.
    ///
    /// Filtered by three things in order: the axis that admits the operator's family, the
    /// operator's `since` version, and whether it has any buildable element type under the
    /// permitted set. The last matters — enabling `logical` without `bool_type` would
    /// otherwise admit `And` while nothing could construct a case for it.
    pub fn operators(&self) -> Vec<OpKind> {
        let permitted = self.element_types();
        OpKind::ALL
            .into_iter()
            .filter(|op| self.admits_family(*op))
            .filter(|op| self.opset >= ops::spec(*op).since)
            .filter(|op| {
                permitted
                    .iter()
                    .any(|elem| ops::probe(*op, *elem, self.opset).is_some())
            })
            .collect()
    }

    /// Whether the axis covering this operator's family is enabled.
    fn admits_family(&self, op: OpKind) -> bool {
        let spec = ops::spec(op);
        match spec.family {
            Family::Reshape | Family::Squeeze | Family::Unsqueeze | Family::Slice | Family::Pad => {
                self.shape_input_operators
            }
            Family::Comparison => self.comparisons,
            // The logical operators share `BinaryElementwise` with the arithmetic ones, so the
            // family alone cannot separate them — their *element type* does. Boolean-only
            // means logical; anything else is arithmetic.
            Family::BinaryElementwise | Family::UnaryElementwise => {
                if spec.data_types == [ElemType::Bool] {
                    self.logical
                } else if spec.tier == Tier::B {
                    self.float_elementwise
                } else {
                    self.structural
                }
            }
            Family::Quantize
            | Family::Dequantize
            | Family::MatMulInteger
            | Family::DynamicQuantize => self.quantized,
            Family::Select
            | Family::Cast
            | Family::Transpose
            | Family::Concat
            | Family::Gather
            | Family::Shape
            | Family::Size => self.structural,
        }
    }

    /// The largest tensor this configuration permits, as a check that the budget binds.
    ///
    /// `max_dim ^ max_rank` is what the *knobs* permit; the budget is what is actually allowed.
    /// Reporting both is how the gap stays visible.
    pub fn unbudgeted_worst_case(&self) -> u128 {
        (self.max_dim.max(1) as u128).pow(self.max_rank as u32)
    }
}

/// Every axis, **including the disabled ones**, in a fixed order.
///
/// Listing disabled axes is not decoration: a description mentioning only what is enabled
/// cannot distinguish "this axis is off" from "this axis did not exist yet", and the second is
/// what makes an old corpus silently incomparable with a new one.
impl GenerationAxes for Bounds {
    fn axes(&self) -> Vec<(&'static str, bool)> {
        vec![
            ("float-elementwise", self.float_elementwise),
            ("comparisons", self.comparisons),
            ("logical", self.logical),
            ("structural", self.structural),
            ("shape-input-ops", self.shape_input_operators),
            ("quantized", self.quantized),
            ("float64", self.float64),
            ("integer-types", self.integer_types),
            ("bool-type", self.bool_type),
            ("special-values", self.special_values),
            ("degenerate-shapes", self.degenerate_shapes),
            ("vary-opset", self.vary_opset),
        ]
    }

    fn scalars(&self) -> Vec<(&'static str, String)> {
        vec![
            ("max-rank", self.max_rank.to_string()),
            ("max-dim", self.max_dim.to_string()),
            ("element-budget", self.element_budget.to_string()),
            ("special-rate", format!("{:.2}", self.special_value_rate)),
            ("opset", self.opset.to_string()),
        ]
    }

    /// The generation-logic fingerprint — the half of drift the axes cannot see.
    ///
    /// Answering `Some` rather than taking the `None` default is deliberate: `None` claims that
    /// generation logic never changes in ways that matter. The SQL domain falsified that three
    /// times, and there is no reason to expect this one to be different.
    fn logic_version(&self) -> Option<String> {
        Some(format!("{GENERATOR_FINGERPRINT:08x}"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────
// Shape generation
// ─────────────────────────────────────────────────────────────────────────────────────

/// Build one valid case for `op`, at `elem`, with shapes drawn from `bounds`.
///
/// **Correct-by-construction.** Every branch below produces a case that satisfies the
/// operator's own rules — the right arity, the right types, shapes that make the operator
/// meaningful, and attributes in range. Nothing is generated and then filtered: a rejected
/// case tests the validator rather than the operator, and a case that is invalid *and*
/// crashes a runtime is our bug rather than theirs.
///
/// Returns `None` when the specification forbids the pair, which is a fact about ONNX rather
/// than about any runtime and must not be confused with one.
pub fn generate_case(
    op: OpKind,
    elem: ElemType,
    bounds: &Bounds,
    rng: &mut SeededRng,
) -> Option<OnnxCase> {
    if !ops::spec(op).data_types.contains(&elem) || bounds.opset < ops::spec(op).since {
        return None;
    }
    // **The opset, drawn or pinned.** Uniform over the operator's own span, whose ends are what
    // the census probes — see `ops::opset_span` and `PENDING` 2.6. `Round` has a single-point
    // span, so `random_range` on an inclusive range of one value is still valid.
    let opset = if bounds.vary_opset {
        let span = ops::opset_span(op);
        rng.random_range(*span.start()..=*span.end())
    } else {
        bounds.opset
    };

    let case = match ops::spec(op).family {
        Family::UnaryElementwise => {
            let dims = shape(bounds, rng);
            OnnxCase::new(op, opset, vec![tensor("a", &dims, elem, op, bounds, rng)])
        }

        // ── Quantization. `SPECS.md` §2q. Correct-by-construction throughout. ─────────
        //
        // Per-tensor granularity only: scale and zero-point are **scalars**. The spec allows
        // per-axis and blocked (§2q.1), and both are deliberately out of scope for now — each
        // adds a shape-agreement rule of its own, and the oracle strength is identical.
        Family::Quantize => {
            let dims = shape(bounds, rng);
            // The output type, chosen here because the zero-point carries it (§2q.1).
            let target = *pick(&ElemType::QUANTIZED, rng);
            // **The scale is chosen before the values, so the values can be aligned to it.**
            // Reversing this was the reason the rounding-mode probe did not exist: with the
            // values drawn first, there is nothing to be exactly halfway *between*.
            let scale = scale_tensor(bounds, rng);
            let scale_value = scale_of(&scale);
            OnnxCase::new(
                op,
                opset,
                vec![
                    quantize_input("a", &dims, elem, scale_value, bounds, rng),
                    scale,
                    zero_point_tensor("zero_point", target, rng),
                ],
            )
        }

        Family::Dequantize => {
            let dims = shape(bounds, rng);
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    scale_tensor(bounds, rng),
                    // "x_zero_point and x must have the same type" (§2q.2).
                    zero_point_tensor("zero_point", elem, rng),
                ],
            )
        }

        // `[m, k] x [k, n]` → `[m, n]`.
        //
        // **`k` is capped deliberately.** §2q.3 permits the `int32` accumulation to overflow,
        // which would make the answer undetermined — the sixth time this domain has met that
        // shape. Rather than declining the operator, the constraint is satisfied by
        // construction: with 8-bit inputs the largest product is 128 x 128 = 16,384, so `k`
        // terms sum to at most 16,384k, and any `k` below ~131,000 cannot overflow. The cap
        // here is far below that, so overflow is unreachable rather than merely unlikely.
        Family::MatMulInteger => {
            let m = bounded_dim(bounds, rng);
            let k = bounded_dim(bounds, rng);
            let n = bounded_dim(bounds, rng);
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &[m, k], elem, op, bounds, rng),
                    tensor("b", &[k, n], elem, op, bounds, rng),
                    // **Distinct names.** Both zero-points were called `zero_point`, which is a
                    // duplicate input name and an invalid model — our validator rejected all 880
                    // generated cases, so `MatMulInteger` silently contributed *nothing* while
                    // appearing enabled. Caught by the yield table, not by a test.
                    zero_point_tensor("a_zero_point", elem, rng),
                    zero_point_tensor("b_zero_point", elem, rng),
                ],
            )
        }

        // One input, no parameters: it derives its own (§2q.4).
        //
        // **Two constraints the formula needs and the specification never states.**
        //
        // `y_scale = (max(0, max(x)) - min(0, min(x))) / 255` is undefined in two ways here:
        //
        // - **A non-finite input** makes the numerator `inf` or `NaN`, and the answer with it.
        //   Measured: reference and ONNX Runtime produce `scale = NaN, zero_point = 255`;
        //   `tract` produces `scale = inf, zero_point = 0`. Neither is wrong, because ONNX does
        //   not say.
        // - **An empty input** has no `max` or `min` at all. `onnx.reference` *rejects* the
        //   model outright, which is the clearest possible statement that the case is not
        //   determined.
        //
        // Both are excluded by construction, the sixth and seventh time this domain has met
        // "ONNX is silent, so two runtimes differ". `SPECS.md` §2q.6.
        Family::DynamicQuantize => {
            let dims = non_empty_shape(bounds, rng);
            OnnxCase::new(
                op,
                opset,
                vec![dynamic_quantize_input("a", &dims, elem, bounds, rng)],
            )
        }

        Family::BinaryElementwise | Family::Comparison => {
            // One shape for both operands. Broadcasting changes the output shape and needs its
            // own rule and its own tests; admitting it here silently would leave `output_spec`
            // quietly wrong. It is a deliberate omission, not an oversight.
            let dims = shape(bounds, rng);
            let left = tensor("a", &dims, elem, op, bounds, rng);
            // **`Div`'s divisor avoids integer zeros whatever the value axis says.** Retrieved:
            // the ONNX `Div` page specifies truncating division for integers and never mentions
            // a zero divisor, so the answer is undetermined and the case must not be generated.
            // Two runtimes panic on it; the reference returns numpy's `0`. Floats keep their
            // zeros — division by zero there is IEEE-754 defined and is exactly the surface this
            // domain wants. `SPECS.md` §2.2b, `PENDING` 1.11.
            let mut left = left;
            let right = if op == OpKind::Div {
                let count = element_count(&dims) as usize;
                let divisor =
                    TensorValue::new("b", dims.clone(), gen_value::nonzero(elem, count, rng));
                // **`MIN / -1` is declined as a pair, not by banning `-1`.** The overflow is
                // undetermined (`SPECS.md` §2.11) but `-1` alone is an ordinary divisor, and
                // excluding it removed integer division by `-1` from the corpus entirely.
                // Shapes match here by construction, so the pairing is exact.
                gen_value::decline_min_over_negative_one(&mut left.data, &divisor.data);
                divisor
            } else {
                tensor("b", &dims, elem, op, bounds, rng)
            };
            OnnxCase::new(op, opset, vec![left, right])
        }

        Family::Select => {
            let dims = shape(bounds, rng);
            OnnxCase::new(
                op,
                opset,
                vec![
                    // The condition is genuinely boolean whatever the data type is.
                    tensor("a", &dims, ElemType::Bool, op, bounds, rng),
                    tensor("b", &dims, elem, op, bounds, rng),
                    tensor("c", &dims, elem, op, bounds, rng),
                ],
            )
        }

        Family::Cast => {
            let dims = shape(bounds, rng);
            // Cast to a *different* type, or the case exercises no conversion at all.
            let permitted = bounds.element_types();
            let targets: Vec<ElemType> = permitted.iter().copied().filter(|t| *t != elem).collect();
            let target = *targets.get(rng.random_range(0..targets.len().max(1)))?;
            // **A float value outside the target integer's range has no determined answer** —
            // the `Cast` reference says "fixed point: undefined if OOR", and `saturate` applies
            // only to float8. So when the target is an integer and the source is a float, the
            // values are drawn from a pool that stays in range *for that target*. `SPECS.md`
            // §2.5, and §2.5b for why "that target" is load-bearing: the pool was once a fixed
            // `int32`-sized one, which is out of range for `uint8` on every negative value.
            //
            // An *integer* source is deliberately not routed here: narrowing an integer is
            // specified to wrap (§2.5b), so those cases have a right answer and are compared.
            let count = element_count(&dims) as usize;
            let data = if !target.is_floating() && elem.is_floating() {
                gen_value::cast_safe(elem, target, count, bounds.special_value_rate, rng)
            } else if bounds.special_values {
                gen_value::with_specials(
                    elem,
                    count,
                    bounds.special_value_rate,
                    gen_value::undetermined_for(op),
                    rng,
                )
            } else {
                gen_value::ordinary(elem, count, rng)
            };
            OnnxCase::new(op, opset, vec![TensorValue::new("a", dims.clone(), data)])
                .with_attrs(Attrs::new().int("to", i64::from(target.wire())))
        }

        Family::Transpose => {
            let dims = shape_bounded(bounds, 1, bounds.degenerate_shapes, rng);
            // A genuine permutation: Fisher-Yates over the axis indices, so every ordering is
            // reachable and none is malformed by construction.
            let mut perm: Vec<i64> = (0..dims.len() as i64).collect();
            for index in (1..perm.len()).rev() {
                perm.swap(index, rng.random_range(0..=index));
            }
            OnnxCase::new(op, opset, vec![tensor("a", &dims, elem, op, bounds, rng)])
                .with_attrs(Attrs::new().ints("perm", perm))
        }

        Family::Concat => {
            let dims = shape_nonempty(bounds, 1, rng);
            let axis = rng.random_range(0..dims.len());
            // Every input shares the shape except along `axis`, where they may differ. Keeping
            // them equal there too would be valid but would never exercise the join.
            //
            // The extent along `axis` is capped by what the budget allows given the *other*
            // dimensions — replacing a budgeted dimension with an unbudgeted one is exactly
            // the bug that `shape_bounded` documents.
            let rest = element_count(&dims) / dims[axis].max(1);
            let ceiling = bounds
                .max_dim
                .min((bounds.element_budget as i64 / rest.max(1)).max(1));
            let count = rng.random_range(2..=3);
            let inputs = (0..count)
                .map(|index| {
                    let mut own = dims.clone();
                    own[axis] = rng.random_range(1..=ceiling);
                    tensor(&input_name(index), &own, elem, op, bounds, rng)
                })
                .collect();
            OnnxCase::new(op, opset, inputs).with_attrs(Attrs::new().int("axis", axis as i64))
        }

        Family::Gather => {
            // A non-empty shape from the start: the axis must have at least one element or no
            // index into it is valid. Requested, not repaired — see `shape_bounded`.
            let dims = shape_nonempty(bounds, 1, rng);
            let axis = rng.random_range(0..dims.len());
            let extent = dims[axis];
            // Indices are **data**: their values decide the answer, so they stay fed. Drawn
            // strictly inside the axis extent — an out-of-range index is undefined behaviour
            // in ONNX, and a case whose answer is undetermined is a false finding waiting to
            // be triaged.
            let index_count = rng.random_range(1..=3usize);
            let indices: Vec<i64> = (0..index_count)
                .map(|_| rng.random_range(0..extent))
                .collect();
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new("b", vec![index_count as i64], TensorData::I64(indices)),
                ],
            )
            .with_attrs(Attrs::new().int("axis", axis as i64))
        }

        Family::Reshape => {
            let dims = shape(bounds, rng);
            let total: i64 = dims.iter().product::<i64>().max(0);
            // A target whose element count matches exactly — the one rule `Reshape` has.
            let target = factorization(total, bounds, rng);
            // **`0` in a `Reshape` target does not mean "zero-length".** ONNX reads it as
            // "copy the corresponding dimension from the input" unless `allowzero=1`:
            //
            // > "A dimension could also be 0, in which case the actual dimension value is
            // > unchanged (i.e. taken from the input tensor)."
            //
            // The generator emits a literal `0` when the tensor is empty, so it must say so.
            // Without this, `[5,8,6,0] -> [0]` asks for an output of shape `[5]` — five
            // elements from a zero-element input — which is **invalid**, and 13% of generated
            // `Reshape` cases were exactly that. `SPECS.md` §2.4.
            let mut case = OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new(
                        "b",
                        vec![target.len() as i64],
                        TensorData::I64(target.clone()),
                    )
                    .as_initializer(),
                ],
            );
            if target.contains(&0) {
                case = case.with_attrs(Attrs::new().int("allowzero", 1));
            }
            case
        }

        Family::Squeeze => {
            // `Squeeze` may only remove a dimension of extent 1, so one is placed deliberately
            // rather than hoped for. Setting a dimension *down* to 1 only shrinks the tensor,
            // so it cannot breach the budget.
            let mut dims = shape_nonempty(bounds, 1, rng);
            let axis = rng.random_range(0..dims.len());
            dims[axis] = 1;
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new("b", vec![1], TensorData::I64(vec![axis as i64]))
                        .as_initializer(),
                ],
            )
        }

        Family::Unsqueeze => {
            let dims = shape(bounds, rng);
            // A new axis may be inserted anywhere from 0 to rank inclusive.
            let axis = rng.random_range(0..=dims.len());
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new("b", vec![1], TensorData::I64(vec![axis as i64]))
                        .as_initializer(),
                ],
            )
        }

        Family::Shape | Family::Size => {
            let dims = shape(bounds, rng);
            OnnxCase::new(op, opset, vec![tensor("a", &dims, elem, op, bounds, rng)])
        }

        Family::Slice => {
            let dims = shape_nonempty(bounds, 1, rng);
            // A half-open range inside the first axis. `start < end <= extent` keeps the result
            // non-empty and the answer determined.
            let start = rng.random_range(0..dims[0]);
            let end = rng.random_range(start + 1..=dims[0]);
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new("b", vec![1], TensorData::I64(vec![start])).as_initializer(),
                    TensorValue::new("c", vec![1], TensorData::I64(vec![end])).as_initializer(),
                ],
            )
        }

        Family::Pad => {
            let dims = shape_bounded(bounds, 1, bounds.degenerate_shapes, rng);
            // `pads` is [begin_0..begin_n, end_0..end_n] — two entries per dimension. Only
            // non-negative amounts: a negative pad *crops*, which is legal but is a different
            // operation and deserves its own deliberate coverage rather than arriving by luck.
            let mut pads: Vec<i64> = Vec::with_capacity(dims.len() * 2);
            for _ in 0..dims.len() * 2 {
                pads.push(rng.random_range(0..=2));
            }
            OnnxCase::new(
                op,
                opset,
                vec![
                    tensor("a", &dims, elem, op, bounds, rng),
                    TensorValue::new("b", vec![pads.len() as i64], TensorData::I64(pads))
                        .as_initializer(),
                ],
            )
            .with_attrs(Attrs::new().string("mode", "constant"))
        }
    };
    Some(case)
}

/// A shape within the bounds, respecting the **element budget**.
fn shape(bounds: &Bounds, rng: &mut SeededRng) -> Vec<i64> {
    shape_bounded(bounds, 0, bounds.degenerate_shapes, rng)
}

/// A shape with at least `min_rank` dimensions, **none of them zero**.
///
/// For operators that need a non-empty axis to be meaningful — `Gather` must have something to
/// index, `Slice` something to slice, `Concat` something to join.
fn shape_nonempty(bounds: &Bounds, min_rank: usize, rng: &mut SeededRng) -> Vec<i64> {
    shape_bounded(bounds, min_rank, false, rng)
}

/// The one place a shape is drawn, respecting the **element budget**.
///
/// # Why every caller goes through here
///
/// Rank and dimension multiply, so drawing each dimension independently from `1..=max_dim`
/// would blow the budget on most ranks. Each dimension is capped by what the budget has left
/// after the ones already chosen, which bounds the *work* rather than the knobs.
///
/// **`allow_zero` is a parameter rather than a repair afterwards, and that is the point.** The
/// first version generated a shape and then raised a zero dimension to 1 where an operator
/// needed a non-empty axis. That broke the budget silently: a zero dimension contributes
/// nothing to the running product, so `[0,8,8,8]` is within any budget — and raising the zero
/// to 1 turns it into 512 elements. A generated 300-element tensor against a 256 budget is
/// what caught it.
///
/// The general shape of that mistake is worth naming, because it is the one this project keeps
/// re-learning: **construct-then-repair defeats a constraint that construction established.**
/// The constraint has to be an input to construction.
fn shape_bounded(
    bounds: &Bounds,
    min_rank: usize,
    allow_zero: bool,
    rng: &mut SeededRng,
) -> Vec<i64> {
    let lowest = if bounds.degenerate_shapes {
        min_rank
    } else {
        min_rank.max(1)
    };
    let rank = rng.random_range(lowest..=bounds.max_rank.max(lowest));

    let mut dims = Vec::with_capacity(rank);
    let mut remaining = bounds.element_budget.max(1) as i64;
    for _ in 0..rank {
        // The largest this dimension may be without the running product exceeding the budget.
        let ceiling = bounds.max_dim.min(remaining.max(1));
        // Zero-length dimensions are legal ONNX and are where implementations differ, so they
        // are reachable — but only where the operator tolerates them and the axis is enabled.
        let low = if allow_zero { 0 } else { 1 };
        let dim = rng.random_range(low..=ceiling.max(low));
        dims.push(dim);
        // A zero dimension makes the tensor empty, so nothing further can exceed the budget.
        if dim == 0 {
            remaining = bounds.element_budget.max(1) as i64;
        } else {
            remaining = (remaining / dim).max(1);
        }
    }
    dims
}

/// How many elements a shape implies.
fn element_count(dims: &[i64]) -> i64 {
    dims.iter().product::<i64>().max(0)
}

/// A shape whose element count is exactly `total`, for `Reshape`.
///
/// Built by repeatedly splitting off a divisor, so the product is correct **by construction**
/// rather than by a check afterwards. A `Reshape` whose target does not match the input's
/// element count is invalid, and generating one would be our bug appearing as a runtime's.
fn factorization(total: i64, bounds: &Bounds, rng: &mut SeededRng) -> Vec<i64> {
    if total == 0 {
        // An empty tensor: any shape containing a zero has the same element count. Keep it
        // simple and honest rather than inventing an arbitrary rank.
        return vec![0];
    }
    let mut remaining = total;
    let mut dims = Vec::new();
    while remaining > 1 && dims.len() + 1 < bounds.max_rank.max(1) {
        // Divisors of what is left, so the running product stays exact.
        let divisors: Vec<i64> = (1..=remaining).filter(|d| remaining % d == 0).collect();
        let divisor = divisors[rng.random_range(0..divisors.len())];
        dims.push(divisor);
        remaining /= divisor;
    }
    dims.push(remaining);
    dims
}

/// A tensor of the given shape and type, filled according to `bounds`.
///
/// **One place decides how a data tensor is filled**, so the special-value rate and the
/// `NaN` exclusions apply everywhere rather than at whichever call site remembered them. That
/// is the same rule as `shape_bounded`: a property belongs in the construction path, not at the
/// sites where it was first needed.
/// Values for a quantizer's input, **some of them exactly halfway between two levels**.
///
/// # The rounding-mode probe
///
/// `PHASE-N9` names this explicitly: *"values exactly halfway between quantization levels (the
/// rounding-mode probe)"*. It is the single most discriminating input a quantizer can be given,
/// because half-to-even and half-away-from-zero agree on **every other value** and differ only
/// here (`SPECS.md` §2q.1).
///
/// **It was missing, and the cost was measurable.** F-008 — `tract` adding the zero-point before
/// rounding rather than after — is visible *only* on an exact tie, and was found by a uniformly
/// random value happening to land on one: about **1 in 57,000 cases**. Generating ties on purpose
/// turns a coincidence into a probe.
///
/// # Why ties are exact here, and would not otherwise be
///
/// A tie must survive `x / scale` in `f32`. For an arbitrary scale it does not: `(n + 0.5) * scale`
/// rounds, and dividing back gives something merely *near* `n + 0.5`, which rounds unambiguously
/// and probes nothing.
///
/// So a tie is only emitted when the scale is a **power of two**, where multiplication and division
/// are exact in binary floating point and `x / scale` recovers `n + 0.5` bit for bit. That is why
/// [`scale_tensor`] offers powers of two at all.
fn quantize_input(
    name: &str,
    dims: &[i64],
    elem: ElemType,
    scale: f32,
    bounds: &Bounds,
    rng: &mut SeededRng,
) -> TensorValue {
    let count = element_count(dims) as usize;
    let mut data = gen_value::ordinary(elem, count, rng);

    // Only meaningful when the values are floats and the scale is an exact power of two.
    if bounds.special_values
        && scale.is_finite()
        && scale > 0.0
        && is_power_of_two(scale)
        && let TensorData::F32(values) = &mut data
    {
        for value in values.iter_mut() {
            if rng.random_bool(bounds.special_value_rate) {
                // n + 0.5 for a small n, so the quantized result stays well inside the
                // 8-bit range whatever the zero-point turns out to be.
                let n = rng.random_range(-60..60) as f32;
                *value = (n + 0.5) * scale;
            }
        }
    }
    TensorValue::new(name, dims.to_vec(), data)
}

/// Values for `DynamicQuantizeLinear`, constructed so the **derived** scale is a power of two
/// and the values sit exactly on rounding ties.
///
/// # Why this needs constructing rather than sampling
///
/// `DynamicQuantizeLinear` takes no parameters — it derives `y_scale` from the data
/// (`SPECS.md` §2q.4):
///
/// ```text
/// y_scale = (max(0, max(x)) - min(0, min(x))) / 255
/// ```
///
/// So the rounding-mode probe cannot be aimed at a scale that was chosen; the scale has to be
/// *arranged* by choosing the data. Pick a power of two `s`, split `255 * s` across the minimum
/// and maximum, and the derived scale is exactly `s`:
///
/// ```text
/// min = -m*s,  max = (255-m)*s   =>   (max - min) / 255 = 255*s / 255 = s
/// ```
///
/// Every step is exact in `f32`: `255 * s` is exact for a power-of-two `s`, and dividing it by 255
/// returns exactly `s`. The remaining values are then `(n + 0.5) * s`, which divide back to exactly
/// `n + 0.5` — real ties, not near misses.
///
/// **This is what turns F-008 from a coincidence into a probe.** That defect is visible only on an
/// exact tie with an odd zero-point, and was found by a uniform draw landing on one at roughly 1 in
/// 57,000 cases.
fn dynamic_quantize_input(
    name: &str,
    dims: &[i64],
    elem: ElemType,
    bounds: &Bounds,
    rng: &mut SeededRng,
) -> TensorValue {
    let count = element_count(dims) as usize;
    if !bounds.special_values || elem != ElemType::F32 || count < 2 {
        // **A degenerate range divides by zero**, and a single element or an all-zero tensor is
        // the easiest way to reach one: `y_scale = (max(0,max) - min(0,min))/255` is zero exactly
        // when every value is zero. Measured: `tract` returns `scale = 0.0`, ONNX Runtime returns
        // `1.0`, and ONNX says nothing about dividing by the formula's zero result. `SPECS.md`
        // §2q.6. A non-zero value is forced so the range cannot collapse.
        let mut fallback = finite_tensor(name, dims, elem, rng);
        if let TensorData::F32(values) = &mut fallback.data
            && values.iter().all(|x| *x == 0.0)
            && let Some(first) = values.first_mut()
        {
            *first = 1.0;
        }
        return fallback;
    }

    const POWERS_OF_TWO: [f32; 6] = [0.015_625, 0.062_5, 0.25, 0.5, 1.0, 2.0];
    let scale = *pick(&POWERS_OF_TWO, rng);
    // How much of the 255-step range sits below zero. Kept off both ends so the derived
    // zero-point is interior, and **often odd** — the parity is half of what F-008 needs.
    let below = rng.random_range(1..255) as f32;

    let mut values = vec![-below * scale, (255.0 - below) * scale];
    for _ in 2..count {
        if rng.random_bool(bounds.special_value_rate.max(0.5)) {
            // An exact tie, kept inside the range the two extremes established.
            let n = rng.random_range(-(below as i64) + 1..(255.0 - below) as i64);
            values.push((n as f32 + 0.5) * scale);
        } else {
            values.push(rng.random_range(-below * scale..(255.0 - below) * scale));
        }
    }
    // The extremes must not sit predictably first, or every case shares a layout.
    let n = values.len();
    for i in (1..n).rev() {
        values.swap(i, rng.random_range(0..=i));
    }
    TensorValue::new(name, dims.to_vec(), TensorData::F32(values))
}

/// Whether an `f32` is an exact power of two — mantissa all zero and the exponent normal.
fn is_power_of_two(value: f32) -> bool {
    value > 0.0 && value.is_finite() && (value.to_bits() & 0x007f_ffff) == 0
}

/// Read a scale back out of the tensor it was built into.
fn scale_of(tensor: &TensorValue) -> f32 {
    match &tensor.data {
        TensorData::F32(v) => v.first().copied().unwrap_or(1.0),
        _ => 1.0,
    }
}

/// A tensor of **finite** values only.
///
/// The quantization operators divide by a scale and saturate into an 8-bit range; an infinity or
/// a `NaN` anywhere in the input makes the result undetermined, and ONNX states nothing about
/// either. Rather than feed them and forgive the disagreement afterwards, they are not generated
/// — the same choice made five times over in `known.rs`.
fn finite_tensor(name: &str, dims: &[i64], elem: ElemType, rng: &mut SeededRng) -> TensorValue {
    let count = element_count(dims) as usize;
    TensorValue::new(name, dims.to_vec(), gen_value::ordinary(elem, count, rng))
}

/// A shape with at least one element.
///
/// `DynamicQuantizeLinear` derives its scale from the input's `max` and `min`, which do not exist
/// for an empty tensor — and `onnx.reference` rejects such a model rather than defining it.
fn non_empty_shape(bounds: &Bounds, rng: &mut SeededRng) -> Vec<i64> {
    let mut dims = shape(bounds, rng);
    if dims.contains(&0) {
        for d in dims.iter_mut() {
            if *d == 0 {
                *d = 1;
            }
        }
    }
    if dims.is_empty() {
        dims.push(1);
    }
    dims
}

/// Uniformly pick one of a slice's elements.
fn pick<'a, T>(choices: &'a [T], rng: &mut SeededRng) -> &'a T {
    &choices[rng.random_range(0..choices.len())]
}

/// A **positive, finite** scale.
///
/// Positivity is not decoration: the quantization formula divides by it, so a zero or negative
/// scale makes the answer undetermined or infinite. `PHASE-N9` names generating invalid
/// quantization parameters — and then reading the rejection as a divergence — as one of its
/// three risks, so the parameters are made valid by construction rather than filtered afterwards.
///
/// The range spans four orders of magnitude so that both saturation (a tiny scale pushes values
/// past the boundary) and coarse quantization (a large scale collapses distinct inputs onto one
/// level) are reachable.
fn scale_tensor(bounds: &Bounds, rng: &mut SeededRng) -> TensorValue {
    // **Powers of two are offered deliberately, not decoratively.** They are the only scales for
    // which `(n + 0.5) * scale` divides back to exactly `n + 0.5` in `f32`, so they are what makes
    // the rounding-mode probe in `quantize_input` an exact tie rather than a near miss.
    const POWERS_OF_TWO: [f32; 8] = [
        0.003_906_25, // 2^-8
        0.015_625,    // 2^-6
        0.062_5,      // 2^-4
        0.25,         // 2^-2
        0.5,
        1.0,
        2.0,
        4.0,
    ];
    let scale = if bounds.special_values && rng.random_bool(0.5) {
        *pick(&POWERS_OF_TWO, rng)
    } else if bounds.special_values && rng.random_bool(bounds.special_value_rate) {
        *pick(&[1.0e-3_f32, 1.0e-2, 0.5, 1.0, 1.0e2], rng)
    } else {
        rng.random_range(0.01_f32..2.0)
    };
    TensorValue::new("scale", vec![], TensorData::F32(vec![scale]))
}

/// A zero-point **inside the target type's range**, which is what makes it valid.
///
/// The range comes from `ElemType::saturation_range`, itself quoted from `SPECS.md` §2q.1, so
/// there is one definition of what `int8` and `uint8` can hold rather than a literal here and
/// another in the census probe.
fn zero_point_tensor(name: &str, target: ElemType, rng: &mut SeededRng) -> TensorValue {
    let (low, high) = target
        .saturation_range()
        .expect("a zero-point's type is always a quantization target");
    let value = rng.random_range(low..=high);
    let payload = match target {
        ElemType::U8 => TensorData::U8(vec![value as u8]),
        _ => TensorData::I8(vec![value as i8]),
    };
    TensorValue::new(name, vec![], payload)
}

/// A single dimension within the configured bound.
fn bounded_dim(bounds: &Bounds, rng: &mut SeededRng) -> i64 {
    rng.random_range(1..=bounds.max_dim.max(1))
}

fn tensor(
    name: &str,
    dims: &[i64],
    elem: ElemType,
    op: OpKind,
    bounds: &Bounds,
    rng: &mut SeededRng,
) -> TensorValue {
    let count = element_count(dims) as usize;
    let data = if bounds.special_values {
        // Values the specification does not determine for this operator are never generated.
        // The rule lives in one place (`gen_value::undetermined_for`) so it cannot drift from
        // the catalog that documents it.
        let exclude = gen_value::undetermined_for(op);
        gen_value::with_specials(elem, count, bounds.special_value_rate, exclude, rng)
    } else {
        gen_value::ordinary(elem, count, rng)
    };
    TensorValue::new(name, dims.to_vec(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The opset axis must be additive**, like every other axis in this module.
    ///
    /// The rule the module comment opens with: *enabling an axis must add cases, never remove
    /// them.* The SQL adapter learned it by enabling joins and silently losing all ordering
    /// coverage. Here the risk is concrete — the axis changes which opset a case is built at, and
    /// an operator whose span is a single point must still be generated.
    ///
    /// So: every operator reachable with the axis off is still reachable with it on.
    #[test]
    fn varying_the_opset_removes_no_operator() {
        use crate::generator::OnnxGenerator;
        use diff_fuzzer_core::rng::SeededRng;
        use diff_fuzzer_core::traits::Generator;
        use std::collections::BTreeSet;

        let reachable = |bounds: Bounds| -> BTreeSet<&'static str> {
            let generator = OnnxGenerator::new(bounds);
            (0..6000u64)
                .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)))
                .map(|case| case.op.onnx_name())
                .collect()
        };

        let pinned = reachable(Bounds::default().with_special_values().with_quantized());
        let varied = reachable(
            Bounds::default()
                .with_special_values()
                .with_quantized()
                .with_opsets(),
        );

        let lost: Vec<&str> = pinned.difference(&varied).copied().collect();
        assert!(
            lost.is_empty(),
            "varying the opset removed operators from the corpus: {lost:?}"
        );
    }

    /// And it must actually vary — an axis that changes nothing is worse than no axis, because
    /// it appears in the description and in the fingerprint while buying nothing.
    #[test]
    fn varying_the_opset_actually_produces_older_opsets() {
        use crate::generator::OnnxGenerator;
        use diff_fuzzer_core::rng::SeededRng;
        use diff_fuzzer_core::traits::Generator;
        use std::collections::BTreeSet;

        let generator = OnnxGenerator::new(
            Bounds::default()
                .with_special_values()
                .with_quantized()
                .with_opsets(),
        );
        let opsets: BTreeSet<i64> = (0..4000u64)
            .map(|seed| generator.generate(&mut SeededRng::from_seed(seed)).opset)
            .collect();

        assert!(
            opsets.len() > 1,
            "the axis is on and every case is still at one opset: {opsets:?}"
        );
        assert!(
            opsets.iter().any(|o| *o < 22),
            "no case below opset 22, so no older code path is reached: {opsets:?}"
        );
        // Never outside a span: an opset below an operator's `since` is a different operator.
        assert!(
            opsets.iter().all(|o| (1..=22).contains(o)),
            "an opset outside the legal range was generated: {opsets:?}"
        );
    }

    /// The trait's own rule, checked here because it is the one most easily broken.
    #[test]
    fn the_description_names_every_axis_including_the_disabled_ones() {
        let described = Bounds::default().description();
        for (name, _) in Bounds::default().axes() {
            assert!(
                described.contains(&format!("{name}=")),
                "{name} missing from the description: {described}"
            );
        }
        // A disabled axis must still be named, or it cannot be told from one that did not
        // exist. `special-values` is the axis that is off in the default configuration; if it
        // is ever turned on by default, this assertion must move to whichever axis is not,
        // rather than being deleted — the property is what matters, not the example.
        assert!(
            described.contains("special-values=off"),
            "a disabled axis must appear in the description: {described}"
        );
        assert!(
            Bounds::default().axes().iter().any(|(_, on)| !on),
            "this test is vacuous unless some axis is actually off"
        );
    }

    #[test]
    fn the_description_carries_the_bounds_and_the_fingerprint() {
        let described = Bounds::default().description();
        for fragment in [
            "max-rank=",
            "max-dim=",
            "element-budget=",
            "opset=",
            "logic=",
        ] {
            assert!(
                described.contains(fragment),
                "{fragment} missing: {described}"
            );
        }
    }

    /// Two configurations that differ must not share an identity — that is the entire point.
    #[test]
    fn different_configurations_are_not_comparable() {
        let base = Bounds::default();

        let wider = Bounds {
            max_rank: base.max_rank + 1,
            ..base.clone()
        };
        assert!(
            !base.comparable_with(&wider),
            "a changed bound must change identity"
        );

        let with_specials = base.with_special_values();
        assert!(
            !base.comparable_with(&with_specials),
            "a flipped axis must change identity"
        );

        assert!(
            base.comparable_with(&Bounds::default()),
            "identical configs must compare"
        );
    }

    /// **Enabling an axis must add operators, never remove them.**
    ///
    /// Checked for every axis by flipping it on and asserting the operator set grows or stays
    /// the same — never shrinks. This is the SQL joins-versus-ordering lesson as an executable
    /// check, and it is the reason that lesson is in the engine's documentation at all.
    #[test]
    fn every_axis_is_additive() {
        let off = Bounds {
            float_elementwise: false,
            comparisons: false,
            logical: false,
            structural: false,
            shape_input_operators: false,
            quantized: false,
            float64: false,
            integer_types: false,
            bool_type: false,
            special_values: false,
            degenerate_shapes: false,
            ..Bounds::default()
        };
        let baseline: Vec<OpKind> = off.operators();

        let flips: Vec<(&str, Bounds)> = vec![
            (
                "float-elementwise",
                Bounds {
                    float_elementwise: true,
                    ..off.clone()
                },
            ),
            (
                "comparisons",
                Bounds {
                    comparisons: true,
                    ..off.clone()
                },
            ),
            (
                "logical",
                Bounds {
                    logical: true,
                    bool_type: true,
                    ..off.clone()
                },
            ),
            (
                "structural",
                Bounds {
                    structural: true,
                    ..off.clone()
                },
            ),
            (
                "shape-input-ops",
                Bounds {
                    shape_input_operators: true,
                    quantized: false,
                    ..off.clone()
                },
            ),
            (
                "float64",
                Bounds {
                    float64: true,
                    ..off.clone()
                },
            ),
            (
                "integer-types",
                Bounds {
                    integer_types: true,
                    ..off.clone()
                },
            ),
            (
                "bool-type",
                Bounds {
                    bool_type: true,
                    ..off.clone()
                },
            ),
        ];

        for (name, flipped) in flips {
            let widened = flipped.operators();
            for op in &baseline {
                assert!(
                    widened.contains(op),
                    "enabling {name} removed {op:?} — an axis must add cases, never displace them"
                );
            }
        }
    }

    /// An axis that admits operators must actually admit some, or it is decoration.
    #[test]
    fn each_operator_axis_admits_something() {
        let off = Bounds {
            float_elementwise: false,
            comparisons: false,
            logical: false,
            structural: false,
            shape_input_operators: false,
            quantized: false,
            ..Bounds::default()
        };
        assert!(
            off.operators().is_empty(),
            "with every operator axis off, nothing is generated"
        );

        for (name, bounds) in [
            (
                "float-elementwise",
                Bounds {
                    float_elementwise: true,
                    ..off.clone()
                },
            ),
            (
                "comparisons",
                Bounds {
                    comparisons: true,
                    ..off.clone()
                },
            ),
            (
                "logical",
                Bounds {
                    logical: true,
                    ..off.clone()
                },
            ),
            (
                "structural",
                Bounds {
                    structural: true,
                    ..off.clone()
                },
            ),
            (
                "shape-input-ops",
                Bounds {
                    shape_input_operators: true,
                    quantized: false,
                    ..off.clone()
                },
            ),
        ] {
            assert!(
                !bounds.operators().is_empty(),
                "the {name} axis admits no operator at all"
            );
        }
    }

    /// The five shape-input operators are generated, and their configuration input is an
    /// **initializer** rather than a fed input.
    ///
    /// The role is what makes them work: emitted as fed inputs they failed 0/5 on `tract`,
    /// which types graphs statically at load. Asserting the role here — not just that the
    /// operators are present — is the difference between testing the decision and testing that
    /// somebody flipped a flag.
    #[test]
    fn shape_input_operators_pass_their_configuration_as_initializers() {
        let bounds = Bounds::default();
        let generated = bounds.operators();

        for op in [
            OpKind::Reshape,
            OpKind::Squeeze,
            OpKind::Unsqueeze,
            OpKind::Slice,
            OpKind::Pad,
        ] {
            assert!(generated.contains(&op), "{op:?} should be generated");

            let case = ops::probe(op, ElemType::F32, bounds.opset).expect("probe");
            assert!(
                case.initializers().count() >= 1,
                "{op:?} must pass its shape/axes/pads input as an initializer"
            );
            // The data input stays fed, so nothing can be constant-folded to a literal — the
            // property the graph-inputs rule was protecting in the first place.
            assert!(
                case.fed_inputs().count() >= 1,
                "{op:?} must still feed its data input at execution"
            );
        }
    }

    /// `Gather`'s indices are **data**, not configuration: their values decide the answer. A
    /// test because the distinction is the whole content of the role split, and getting it
    /// backwards would silently stop testing index selection.
    #[test]
    fn gather_indices_stay_a_fed_input() {
        let case = ops::probe(OpKind::Gather, ElemType::F32, 22).expect("probe");
        assert_eq!(
            case.initializers().count(),
            0,
            "Gather reads its indices as data; making them constant would fold the selection away"
        );
    }

    /// An operator must not be admitted when nothing can build a case for it. Enabling the
    /// logical operators without booleans would otherwise list `And` while no probe exists.
    #[test]
    fn an_operator_with_no_buildable_type_is_not_admitted() {
        let logical_without_bool = Bounds {
            float_elementwise: false,
            comparisons: false,
            structural: false,
            shape_input_operators: false,
            quantized: false,
            logical: true,
            bool_type: false,
            ..Bounds::default()
        };
        assert!(
            logical_without_bool.operators().is_empty(),
            "the logical operators need booleans; admitting them without is a case nothing can build"
        );
    }

    /// The default configuration must actually produce a useful surface.
    #[test]
    fn the_default_configuration_is_substantial() {
        let bounds = Bounds::default();
        let ops = bounds.operators();
        assert!(ops.len() >= 20, "only {} operators by default", ops.len());
        assert!(
            ops.iter()
                .filter(|o| ops::spec(**o).value_dependent)
                .count()
                >= 8,
            "the default must keep the value-dependent surface the N2 bar was set on"
        );
        assert_eq!(bounds.element_types().len(), 5);
    }

    #[test]
    fn the_one_axis_preset_is_actually_narrow() {
        let bounds = Bounds::one_axis();
        assert_eq!(bounds.element_types(), vec![ElemType::F32]);
        let ops = bounds.operators();
        assert!(!ops.is_empty());
        assert!(
            ops.iter().all(|o| ops::spec(*o).tier == Tier::B),
            "the one-axis preset should be Tier B float arithmetic only"
        );
    }

    /// Bounding each knob does not bound the case: the knobs permit far more than the budget.
    /// The test states the gap rather than asserting it away.
    #[test]
    fn the_knobs_permit_more_than_the_budget_allows() {
        let bounds = Bounds::default();
        assert!(
            bounds.unbudgeted_worst_case() > bounds.element_budget as u128,
            "if the knobs cannot exceed the budget, the budget is not doing anything"
        );
    }

    /// The fingerprint must be stable within a build and non-trivial.
    #[test]
    fn the_fingerprint_is_stable_and_set() {
        assert_ne!(GENERATOR_FINGERPRINT, 0);
        assert_eq!(GENERATOR_FINGERPRINT, GENERATOR_FINGERPRINT);
        let described = Bounds::default().description();
        assert!(described.contains(&format!("logic={GENERATOR_FINGERPRINT:08x}")));
    }
}
