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

use crate::case::{ElemType, OpKind};
use crate::ops::{self, Family, Tier};

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

            float64: true,
            integer_types: true,
            bool_type: true,

            // Off so N4 can measure what they buy against this baseline.
            special_values: false,
            degenerate_shapes: true,

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
            ("float64", self.float64),
            ("integer-types", self.integer_types),
            ("bool-type", self.bool_type),
            ("special-values", self.special_values),
            ("degenerate-shapes", self.degenerate_shapes),
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

#[cfg(test)]
mod tests {
    use super::*;

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
