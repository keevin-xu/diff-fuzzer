//! The catalog of differences that are **legal** — and how each one is kept from becoming noise.
//!
//! # An entry without a citation is not an entry
//!
//! `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §5 is explicit about this, and the type enforces it:
//! [`LegalDifference::citation`] is not an `Option`. A rule that forgives a difference cannot be
//! justified by recall, because forgiving is the direction that **hides defects silently and
//! permanently** — as opposed to over-tightening, which produces noise and corrects itself.
//!
//! # The shape of this catalog is the phase's real finding
//!
//! The starter list in §5 assumed a catalog of things the *oracle* forgives. That is how the SQL
//! domain's tolerance work went, and how the tensor domain's `known.rs` works.
//!
//! This domain ended up somewhere else. Of six candidate legal differences, **one** is forgiven by
//! the comparison. Three are never generated at all, and two cannot arise in the pinned
//! configuration. The catalog is mostly a record of differences the oracle **never sees**.
//!
//! That is not an accident, and it is the better arrangement:
//!
//! > **Declining to generate a case is sound whether or not you understood it. Forgiving a
//! > difference requires having been right.**
//!
//! An over-broad forgiveness rule eats real bugs and looks exactly like success. An over-broad
//! refusal to generate costs coverage, which shows up as a measurable hole rather than as a
//! confident zero. So where a case's answer is undetermined, this domain removes the case rather
//! than teaching the oracle to excuse the answer.
//!
//! # What keeps this file honest
//!
//! A catalog can rot in two ways: an entry can lose the behaviour it describes, and an entry can
//! lose the document it cites. Both are silent. The tests at the bottom close both — the
//! citations are checked **against `SPECS.md` itself, read at test time**, so renumbering a
//! section fails the build rather than orphaning an entry.

use crate::case::{ElemType, OpKind};

/// Where a claim was established, and how strongly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// The specification, vendor documentation, or the operator reference itself.
    Primary,
    /// A peer-reviewed or otherwise credible source that quotes a standard we could not obtain.
    ///
    /// Recorded distinctly because "we read the standard" and "we read someone describing the
    /// standard" must not look the same in confident prose.
    Secondary,
    /// Established by running the implementations, not by reading anything.
    ///
    /// **Never sufficient on its own to forgive a difference.** Three backends agreeing is
    /// evidence about behaviour and says nothing about what is required.
    Measured,
}

/// The evidence behind an entry. Mandatory.
#[derive(Debug, Clone, Copy)]
pub struct Citation {
    /// The section of `crates/onnx-adapter/SPECS.md` holding the retrieval and its date.
    pub specs_section: &'static str,
    /// The source itself, so a reader need not go through `SPECS.md` to check it.
    pub url: &'static str,
    pub kind: SourceKind,
}

/// How a legal difference is prevented from becoming a false finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Handling {
    /// The comparison treats the two results as agreeing, or as unjudged.
    ///
    /// The only handling that requires having been **right**, and therefore the only one that
    /// needs a specification behind it rather than merely evidence.
    ForgivenByComparison,
    /// The generator never produces the case, so the oracle never has to judge it.
    ///
    /// Sound either way: if the reasoning was wrong, the cost is coverage, which is visible.
    DeclinedByGenerator,
    /// Cannot arise while the configuration holds — and what would make it live again.
    ExcludedByConfiguration { becomes_live_if: &'static str },
}

/// One catalogued legal difference.
#[derive(Debug, Clone, Copy)]
pub struct LegalDifference {
    /// Stable identifier, for referring to an entry from a report.
    pub id: &'static str,
    /// The operators it applies to. Empty means "not operator-specific".
    pub operators: &'static [OpKind],
    /// The element types it applies to. Empty means "all".
    pub elem_types: &'static [ElemType],
    /// What may legitimately differ.
    pub what: &'static str,
    pub handling: Handling,
    /// **Not optional.** See the module comment.
    pub citation: Citation,
    /// Why the handling is the right one, in a sentence a reviewer can disagree with.
    pub note: &'static str,
}

/// The catalog.
pub const CATALOG: &[LegalDifference] = &[
    LegalDifference {
        id: "nan-payload",
        operators: &[],
        elem_types: &[ElemType::F32, ElemType::F64],
        what: "which NaN payload and sign bit propagate through an operation",
        handling: Handling::ForgivenByComparison,
        citation: Citation {
            specs_section: "5.3",
            url: "https://en.wikipedia.org/wiki/NaN",
            kind: SourceKind::Secondary,
        },
        note: "The one entry the comparison forgives, and the only one that needed a citation at \
               all. IEEE 754-2019 is paywalled and was not retrieved; a technical paper on NaN \
               propagation was fetched and failed to parse. Measured over 6,000 cases: 3,018 \
               NaN-vs-NaN element pairs compared, 0 with differing bits — so the rule has never \
               yet decided a case, and if the primary text contradicts it no verdict changes.",
    },
    LegalDifference {
        id: "integer-division-by-zero",
        operators: &[OpKind::Div],
        elem_types: &[ElemType::I32, ElemType::I64],
        what: "the result of an integer division by zero",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.2b",
            url: "https://onnx.ai/onnx/operators/onnx__Div.html",
            kind: SourceKind::Primary,
        },
        note: "The Div page specifies truncating integer division and never mentions a zero \
               divisor. tract and candle panic; the reference returns numpy's 0. Retrieving the \
               page is what turned a would-be finding into a generator rule.",
    },
    LegalDifference {
        id: "max-min-nan",
        operators: &[OpKind::Max, OpKind::Min],
        elem_types: &[ElemType::F32, ElemType::F64],
        what: "which operand wins when one of them is NaN",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.2c",
            url: "https://onnx.ai/onnx/operators/onnx__Max.html",
            kind: SourceKind::Primary,
        },
        note: "NaN is not mentioned in any version of the Max page, and unlike Add/Sub/Mul/Div/\
               Sqrt these are not IEEE-754 basic operations — IEEE's own maxNum/minNum semantics \
               changed between its 2008 and 2019 revisions. Neither document pins it down.",
    },
    LegalDifference {
        id: "integer-division-overflow",
        operators: &[OpKind::Div],
        elem_types: &[ElemType::I32, ElemType::I64],
        what: "the result of MIN / -1, whose true value is not representable",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.11",
            url: "https://onnx.ai/onnx/operators/onnx__Div.html",
            kind: SourceKind::Primary,
        },
        note: "The Div page specifies truncating division and says nothing about overflow. \
               onnx.reference and ONNX Runtime wrap to MIN; tract panics with \"attempt to divide \
               with overflow\". Declined by keeping -1 out of integer divisors, which makes the \
               pair unreachable. The panic is separately preserved as candidate finding F-006 — \
               declining a case and reporting a behaviour are not in conflict.",
    },
    LegalDifference {
        id: "max-min-signed-zero",
        operators: &[OpKind::Max, OpKind::Min],
        elem_types: &[ElemType::F32, ElemType::F64],
        what: "which zero is returned when the operands are +0.0 and -0.0",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.9",
            url: "https://onnx.ai/onnx/operators/onnx__Min.html",
            kind: SourceKind::Primary,
        },
        note: "The Min page mentions neither signed zero nor NaN in any of its five versions, and \
               these are not IEEE-754 basic operations. Found by the N6.6 funnel: reference, ONNX \
               Runtime and tract return -0.0 while candle returns +0.0 — three against one, which \
               looks exactly like a defect and is not one.",
    },
    LegalDifference {
        id: "sign-of-nan",
        operators: &[OpKind::Sign],
        elem_types: &[ElemType::F32, ElemType::F64],
        what: "the result of Sign(NaN)",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.10",
            url: "https://onnx.ai/onnx/operators/onnx__Sign.html",
            kind: SourceKind::Primary,
        },
        note: "The page specifies > 0, < 0 and == 0 only. NaN satisfies none of them and is never \
               mentioned. Reference, ONNX Runtime and tract return NaN; candle returns 0.0, which \
               is a defensible reading of the same text. Sign(-0.0) is NOT excluded — -0.0 == 0 \
               is true, so that answer is specified.",
    },
    LegalDifference {
        id: "quantize-non-finite-input",
        operators: &[OpKind::QuantizeLinear, OpKind::DynamicQuantizeLinear],
        elem_types: &[ElemType::F32],
        what: "the quantized result of an infinite or NaN input",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2q.6",
            url: "https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html",
            kind: SourceKind::Primary,
        },
        note: "The derived scale becomes inf or NaN. Measured: reference and ONNX Runtime give \
               scale NaN with zero-point 255; tract gives scale inf with zero-point 0. Nothing in \
               the specification chooses between them.",
    },
    LegalDifference {
        id: "dynamic-quantize-empty-input",
        operators: &[OpKind::DynamicQuantizeLinear],
        elem_types: &[ElemType::F32],
        what: "the scale derived from a tensor with no elements",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2q.6",
            url: "https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html",
            kind: SourceKind::Primary,
        },
        note: "max(x) and min(x) do not exist for an empty tensor. onnx.reference rejects the \
               model outright, which is the strongest available signal that the case is not \
               determined.",
    },
    LegalDifference {
        id: "cast-out-of-range",
        operators: &[OpKind::Cast],
        elem_types: &[ElemType::F32, ElemType::F64],
        what: "the result of casting a float outside the target integer's range",
        handling: Handling::DeclinedByGenerator,
        citation: Citation {
            specs_section: "2.5",
            url: "https://onnx.ai/onnx/operators/onnx__Cast.html",
            kind: SourceKind::Primary,
        },
        note: "\"fixed point: undefined if OOR\", and the saturate attribute applies only to \
               float8 conversion. tract saturates at int32 bounds and ONNX Runtime at int64 — 17 \
               divergences in 6,000 cases, every one of them ours.",
    },
    LegalDifference {
        id: "ort-optimization-level",
        operators: &[],
        elem_types: &[],
        what: "numeric results changed by ONNX Runtime's Extended and Layout optimizations",
        handling: Handling::ExcludedByConfiguration {
            becomes_live_if: "ONNX Runtime is run above GraphOptimizationLevel::Disable",
        },
        citation: Citation {
            specs_section: "3.1",
            url: "https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html",
            kind: SourceKind::Primary,
        },
        note: "ONNX Runtime documents Basic optimizations as semantics-preserving and does not \
               make the claim for the higher levels — an absence, not a denial, and recorded as \
               such. Its own docs acknowledge GELU Approximation changing results (F1 87.05 vs \
               87.03). ort's default is Level3, so the default is the one setting a conformance \
               comparison must not use.",
    },
    LegalDifference {
        id: "semantic-opset-change",
        operators: &[],
        elem_types: &[],
        what: "behaviour that genuinely changed between opset versions",
        handling: Handling::ExcludedByConfiguration {
            becomes_live_if: "cases are generated at more than one opset (PENDING 2.6, N10)",
        },
        citation: Citation {
            specs_section: "2.8",
            url: "https://onnx.ai/onnx/operators/onnx__Add.html",
            kind: SourceKind::Primary,
        },
        note: "Add changed semantically at opset 7 (broadcast attributes replaced by Numpy-style \
               broadcasting); Sqrt never changed semantically across 1, 6 and 13. Every case runs \
               at a fixed opset 22, so no opset difference can currently arise. The rule for N10: \
               a version bump is not evidence of a semantic change — read the history.",
    },
];

/// Look an entry up by its identifier.
pub fn entry(id: &str) -> Option<&'static LegalDifference> {
    CATALOG.iter().find(|e| e.id == id)
}

/// Entries that apply to an operator.
pub fn for_operator(op: OpKind) -> Vec<&'static LegalDifference> {
    CATALOG
        .iter()
        .filter(|e| e.operators.is_empty() || e.operators.contains(&op))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{TensorData, TensorValue};
    use crate::normalize::{Agreement, OnnxNormalizer, compare};
    use crate::outcome::OnnxOutcome;
    use diff_fuzzer_core::Normalizer;

    fn specs_text() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("SPECS.md");
        std::fs::read_to_string(path).expect("SPECS.md must be readable")
    }

    /// **The registry guard (N6.4), half one: the citation must still exist.**
    ///
    /// A catalog entry pointing at a `SPECS.md` section that has been renumbered or deleted is
    /// orphaned, and nothing about it looks wrong — the entry still reads as cited. So the
    /// section headings are read out of the document itself at test time.
    #[test]
    fn every_entry_cites_a_section_that_exists() {
        let specs = specs_text();
        for entry in CATALOG {
            let heading = format!("### {}", entry.citation.specs_section);
            assert!(
                specs.contains(&heading),
                "entry {:?} cites SPECS.md section {} — no such heading exists",
                entry.id,
                entry.citation.specs_section
            );
        }
    }

    /// And the cited URL must appear in that document too, so `SPECS.md` and the catalog cannot
    /// drift into citing different things for the same claim.
    #[test]
    fn every_cited_url_appears_in_specs() {
        let specs = specs_text();
        for entry in CATALOG {
            assert!(
                specs.contains(entry.citation.url),
                "entry {:?} cites {} — that URL does not appear in SPECS.md",
                entry.id,
                entry.citation.url
            );
        }
    }

    /// **The registry guard, half two: the behaviour must still be there.**
    ///
    /// `nan-payload` is the only entry claiming the comparison forgives something. If the NaN
    /// rule is ever removed from `normalize.rs`, this entry becomes a lie that still reads as
    /// documentation. Rebuilt from a real comparison rather than asserted.
    #[test]
    fn the_forgiven_entry_matches_what_the_comparison_actually_does() {
        let canon = |bits: u32| {
            OnnxNormalizer.normalize(OnnxOutcome::Ok(vec![TensorValue::f32(
                "out",
                vec![1],
                vec![f32::from_bits(bits)],
            )]))
        };
        // Two NaNs with different payloads.
        assert_eq!(
            compare(&canon(0x7fc0_0000), &canon(0x7fc0_1234)),
            Agreement::ByLicense,
            "the nan-payload entry claims the comparison forgives this, and it does not"
        );
        // And the forgiveness must not extend to a NaN against a number.
        assert_eq!(
            compare(&canon(0x7fc0_0000), &canon(0x3f80_0000)),
            Agreement::No,
            "forgiveness leaked beyond what the entry describes"
        );

        let entry = entry("nan-payload").expect("the entry must exist");
        assert_eq!(entry.handling, Handling::ForgivenByComparison);
    }

    /// **Only one entry may be forgiven by the comparison**, and it must be the cited one.
    ///
    /// This is the invariant the module comment argues for: forgiving requires having been
    /// right, so each new instance of it is a decision someone must make deliberately. A test
    /// that fails when the count changes is what makes that deliberate.
    #[test]
    fn forgiveness_is_rationed() {
        let forgiven: Vec<&str> = CATALOG
            .iter()
            .filter(|e| e.handling == Handling::ForgivenByComparison)
            .map(|e| e.id)
            .collect();
        assert_eq!(
            forgiven,
            vec!["nan-payload"],
            "a new forgiveness rule was added — it needs a specification, not just a citation"
        );
    }

    /// A forgiveness rule may never rest on measurement alone. Three backends agreeing is
    /// evidence about behaviour and says nothing about what the standard requires.
    #[test]
    fn nothing_is_forgiven_on_measurement_alone() {
        for entry in CATALOG {
            if entry.handling == Handling::ForgivenByComparison {
                assert_ne!(
                    entry.citation.kind,
                    SourceKind::Measured,
                    "entry {:?} forgives a difference on measured evidence",
                    entry.id
                );
            }
        }
    }

    /// Entries excluded by configuration must say what would make them live again, or the
    /// exclusion is indistinguishable from having forgotten about them.
    #[test]
    fn excluded_entries_state_their_trigger() {
        for entry in CATALOG {
            if let Handling::ExcludedByConfiguration { becomes_live_if } = entry.handling {
                assert!(
                    !becomes_live_if.trim().is_empty(),
                    "entry {:?} is excluded by configuration but names no trigger",
                    entry.id
                );
            }
        }
    }

    /// Identifiers are how a report refers to an entry, so they must be unique.
    #[test]
    fn identifiers_are_unique() {
        let mut ids: Vec<&str> = CATALOG.iter().map(|e| e.id).collect();
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "duplicate catalog identifiers");
    }

    /// The generator must actually decline what the catalog says it declines. Three entries rest
    /// on that claim, and it is the load-bearing half of the "decline, do not forgive" design.
    #[test]
    fn declined_entries_are_actually_declined() {
        use crate::gen_shape::Bounds;
        use crate::generator::OnnxGenerator;
        use diff_fuzzer_core::rng::SeededRng;
        use diff_fuzzer_core::traits::Generator;

        let generator = OnnxGenerator::new(Bounds::default().with_special_values());
        let mut divisor_zeros = 0;
        let mut divisor_minus_ones = 0;
        let mut maxmin_nans = 0;
        let mut maxmin_negative_zeros = 0;
        let mut sign_nans = 0;

        for seed in 0..3000u64 {
            let case = generator.generate(&mut SeededRng::from_seed(seed));
            match case.op {
                OpKind::Div if !crate::ops::data_elem_type(&case).is_floating() => {
                    if let Some(divisor) = case.inputs.get(1) {
                        divisor_zeros += match &divisor.data {
                            TensorData::I32(v) => v.iter().filter(|x| **x == 0).count(),
                            TensorData::I64(v) => v.iter().filter(|x| **x == 0).count(),
                            _ => 0,
                        };
                        // `-1` too: paired with a `MIN` dividend it overflows, and ONNX does not
                        // say what that produces.
                        divisor_minus_ones += match &divisor.data {
                            TensorData::I32(v) => v.iter().filter(|x| **x == -1).count(),
                            TensorData::I64(v) => v.iter().filter(|x| **x == -1).count(),
                            _ => 0,
                        };
                    }
                }
                OpKind::Max | OpKind::Min => {
                    for input in &case.inputs {
                        // Signed zero is checked on the **bit pattern**: `-0.0 == 0.0` is true,
                        // so a value comparison here would find nothing however broken the
                        // generator was.
                        match &input.data {
                            TensorData::F32(v) => {
                                maxmin_nans += v.iter().filter(|x| x.is_nan()).count();
                                maxmin_negative_zeros += v
                                    .iter()
                                    .filter(|x| x.to_bits() == (-0.0f32).to_bits())
                                    .count();
                            }
                            TensorData::F64(v) => {
                                maxmin_nans += v.iter().filter(|x| x.is_nan()).count();
                                maxmin_negative_zeros += v
                                    .iter()
                                    .filter(|x| x.to_bits() == (-0.0f64).to_bits())
                                    .count();
                            }
                            _ => {}
                        }
                    }
                }
                OpKind::Sign => {
                    for input in &case.inputs {
                        sign_nans += match &input.data {
                            TensorData::F32(v) => v.iter().filter(|x| x.is_nan()).count(),
                            TensorData::F64(v) => v.iter().filter(|x| x.is_nan()).count(),
                            _ => 0,
                        };
                    }
                }
                _ => {}
            }
        }
        assert_eq!(divisor_zeros, 0, "integer-division-by-zero was generated");
        assert_eq!(
            divisor_minus_ones, 0,
            "integer-division-overflow was reachable: a -1 divisor can pair with a MIN dividend"
        );
        assert_eq!(maxmin_nans, 0, "max-min-nan was generated");
        assert_eq!(
            maxmin_negative_zeros, 0,
            "max-min-signed-zero was generated"
        );
        assert_eq!(sign_nans, 0, "sign-of-nan was generated");
    }
}
