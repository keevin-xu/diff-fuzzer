//! Is this case well-formed?
//!
//! # Why this exists before the generator
//!
//! The generator is **correct-by-construction**: it refuses to emit a bad shape rather
//! than producing one and filtering it later. So in normal operation this module should
//! never reject anything, which raises a fair question — why write it at all?
//!
//! Three reasons, each earned:
//!
//! 1. **A crash is only a finding if the model is valid.** Our own malformed model
//!    crashing a runtime is our bug, not theirs. This is the first of the two gates in
//!    front of every crash report; the second is the reference implementation accepting
//!    the model.
//! 2. **Shrinking proposes cases nobody designed.** The minimizer drops dimensions and
//!    values to find a smaller reproduction, and a non-local reduction can easily produce
//!    something illegal. `validate` is the gate that makes those proposals safe to make
//!    without reasoning about each one individually.
//! 3. **It is the check that catches the generator regressing.** N3's validity stress test
//!    runs thousands of generated cases through here; a generator bug shows up as a
//!    rejection rather than as a mysterious flood of divergences three phases later. Both
//!    prior domains produced campaigns of hundreds of findings that were all their own —
//!    one SQL sweep produced 825 from invalid queries.
//!
//! # What is *not* checked here
//!
//! Whether the answer is **determined**. A case can be perfectly well-formed and still
//! have an answer the specification does not pin down, and that is a generator
//! responsibility, not a validation one — refusing to *emit* such a case is sound, while
//! detecting it after the fact generally is not.

use crate::case::{ElemType, OnnxCase, OpKind, TensorData, TensorValue};

/// Why a case is not well-formed.
///
/// A typed enum rather than a string, so a test can assert *which* rule fired. A test
/// asserting only "it was rejected" passes when the wrong rule fires, which is how a
/// validator drifts away from what its tests claim to check.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Invalid {
    #[error("{op} takes {expected} input(s), got {actual}")]
    WrongArity {
        op: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error(
        "input {name} has dimension {dim} at position {position}; dimensions cannot be negative"
    )]
    NegativeDimension {
        name: String,
        position: usize,
        dim: i64,
    },

    #[error("input {name} declares shape {dims:?} ({expected} elements) but carries {actual}")]
    ShapeDataMismatch {
        name: String,
        dims: Vec<i64>,
        expected: usize,
        actual: usize,
    },

    #[error("{op} requires all inputs to share a shape; got {first:?} and {second:?}")]
    ShapeMismatch {
        op: &'static str,
        first: Vec<i64>,
        second: Vec<i64>,
    },

    #[error("{op} requires all inputs to share an element type")]
    ElemTypeMismatch { op: &'static str },

    #[error("input names must be unique and non-empty; {name:?} is not")]
    BadInputName { name: String },

    #[error("opset {opset} is out of the supported range {min}..={max}")]
    OpsetOutOfRange { opset: i64, min: i64, max: i64 },

    #[error("attribute names must be unique and non-empty; {name:?} is not")]
    BadAttributeName { name: String },
}

/// The opset range this adapter is willing to build models for.
///
/// The lower bound is where the operators in [`OpKind`] have the semantics assumed here;
/// the upper bound is what the pinned `onnx` release knows about, since the reference
/// cannot adjudicate an opset it has never heard of. Both are **provisional pending the
/// N2 census**, which measures what the runtimes actually accept.
pub const MIN_OPSET: i64 = 7;
/// See [`MIN_OPSET`]. Equal to `environment::MAX_OPSET`, checked by a test so the two
/// cannot drift.
pub const MAX_OPSET: i64 = 27;

/// Check a case against every rule. Returns all violations, not just the first.
///
/// Returning **all** of them is deliberate: fixing one and rediscovering the next is how a
/// generator bug takes four iterations to characterise instead of one, and the SQL domain
/// spent real time on exactly that (three stacked barriers, each hiding the next).
pub fn validate(case: &OnnxCase) -> Vec<Invalid> {
    let mut problems = Vec::new();
    let op_name = case.op.onnx_name();

    if case.opset < MIN_OPSET || case.opset > MAX_OPSET {
        problems.push(Invalid::OpsetOutOfRange {
            opset: case.opset,
            min: MIN_OPSET,
            max: MAX_OPSET,
        });
    }

    let expected_arity = case.op.arity();
    if case.inputs.len() != expected_arity {
        problems.push(Invalid::WrongArity {
            op: op_name,
            expected: expected_arity,
            actual: case.inputs.len(),
        });
    }

    let mut seen_names: Vec<&str> = Vec::new();
    for input in &case.inputs {
        // Every ONNX value is referenced by name; a duplicate or empty name makes the
        // graph ambiguous or unlinkable.
        if input.name.is_empty() || seen_names.contains(&input.name.as_str()) {
            problems.push(Invalid::BadInputName {
                name: input.name.clone(),
            });
        }
        seen_names.push(&input.name);

        for (position, dim) in input.dims.iter().enumerate() {
            if *dim < 0 {
                problems.push(Invalid::NegativeDimension {
                    name: input.name.clone(),
                    position,
                    dim: *dim,
                });
            }
        }

        // Only meaningful once the shape itself is sane, or the expected count is nonsense.
        if input.dims.iter().all(|d| *d >= 0) {
            let expected = input.element_count();
            if input.data.len() != expected {
                problems.push(Invalid::ShapeDataMismatch {
                    name: input.name.clone(),
                    dims: input.dims.clone(),
                    expected,
                    actual: input.data.len(),
                });
            }
        }
    }

    // Attribute names must be unique. ONNX looks an attribute up by name, so a duplicate
    // makes the node ambiguous — and which one wins is not something the specification
    // pins down, which makes it exactly the kind of case the generator must never emit.
    let mut seen_attrs: Vec<&str> = Vec::new();
    for (name, _) in case.attrs.iter() {
        if name.is_empty() || seen_attrs.contains(&name) {
            problems.push(Invalid::BadAttributeName {
                name: name.to_owned(),
            });
        }
        seen_attrs.push(name);
    }

    // Shape and type agreement across inputs.
    //
    // Every operator here is elementwise over identically-shaped inputs. ONNX would permit
    // broadcasting for `Add`/`Sub`/`Mul`, and that is a **deliberate N3 decision rather
    // than an N1 omission**: broadcasting changes the output shape, so it needs its own
    // shape rule and its own tests, and adding it silently here would leave `output_dims`
    // quietly wrong.
    if let Some(first) = case.inputs.first() {
        for other in case.inputs.iter().skip(1) {
            if other.dims != first.dims {
                problems.push(Invalid::ShapeMismatch {
                    op: op_name,
                    first: first.dims.clone(),
                    second: other.dims.clone(),
                });
            }
            if other.elem_type() != first.elem_type() {
                problems.push(Invalid::ElemTypeMismatch { op: op_name });
            }
        }
    }

    problems
}

/// Convenience: is this case well-formed?
pub fn is_valid(case: &OnnxCase) -> bool {
    validate(case).is_empty()
}

/// Build a well-formed case for one operator, for tests and for the trivial N1 generator.
///
/// Lives beside `validate` on purpose: the function that constructs a valid case and the
/// function that judges validity should be read together, so a rule added to one is
/// obviously missing from the other.
pub fn well_formed(op: OpKind, dims: &[i64], opset: i64) -> OnnxCase {
    well_formed_typed(op, dims, opset, ElemType::F32)
}

/// Build a well-formed case for one operator at a given element type.
///
/// Values are distinct per input so a case cannot pass by symmetry — if a runtime swapped
/// its operands, `Sub` would notice and `Add` would not.
pub fn well_formed_typed(op: OpKind, dims: &[i64], opset: i64, elem: ElemType) -> OnnxCase {
    let count = dims.iter().product::<i64>().max(0) as usize;
    let inputs = (0..op.arity())
        .map(|index| {
            let base = (index as i64 + 1) * 10;
            let data = match elem {
                ElemType::F32 => {
                    TensorData::F32((0..count).map(|i| (base + i as i64) as f32).collect())
                }
                ElemType::F64 => {
                    TensorData::F64((0..count).map(|i| (base + i as i64) as f64).collect())
                }
                ElemType::I32 => {
                    TensorData::I32((0..count).map(|i| (base + i as i64) as i32).collect())
                }
                ElemType::I64 => TensorData::I64((0..count).map(|i| base + i as i64).collect()),
                // Alternating rather than constant, so a runtime that returned a fixed
                // value would be caught.
                ElemType::Bool => {
                    TensorData::Bool((0..count).map(|i| (i + index) % 2 == 0).collect())
                }
            };
            TensorValue::new(&input_name(index), dims.to_vec(), data)
        })
        .collect();
    OnnxCase::new(op, opset, inputs)
}

/// The conventional name for the input at `index`: `a`, `b`, `c`, …
pub fn input_name(index: usize) -> String {
    // 26 inputs is far more than any operator here takes; beyond that, fall back to a
    // numbered form rather than wrapping around and producing a duplicate name.
    if index < 26 {
        ((b'a' + index as u8) as char).to_string()
    } else {
        format!("in{index}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::ElemType;

    const OPSET: i64 = 22;

    /// Every operator must be constructible in a well-formed way. Iterates `ALL`, so a new
    /// operator is covered automatically rather than when someone remembers.
    #[test]
    fn well_formed_cases_validate_for_every_operator() {
        for op in OpKind::ALL {
            for dims in [vec![], vec![1], vec![2, 3], vec![2, 3, 4], vec![0, 3]] {
                let case = well_formed(op, &dims, OPSET);
                assert_eq!(
                    validate(&case),
                    vec![],
                    "{op:?} with shape {dims:?} should be valid"
                );
            }
        }
    }

    /// A validator that accepts everything passes every "is it valid?" test. Each rule
    /// below is therefore paired with a case that must trip *that specific rule*.
    #[test]
    fn wrong_arity_is_caught() {
        let mut case = well_formed(OpKind::Add, &[2], OPSET);
        case.inputs.pop();
        assert!(validate(&case).iter().any(|p| matches!(
            p,
            Invalid::WrongArity {
                expected: 2,
                actual: 1,
                ..
            }
        )));
    }

    #[test]
    fn a_negative_dimension_is_caught() {
        let mut case = well_formed(OpKind::Identity, &[2], OPSET);
        case.inputs[0].dims = vec![-1];
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::NegativeDimension { dim: -1, .. }))
        );
    }

    #[test]
    fn data_not_matching_the_declared_shape_is_caught() {
        let mut case = well_formed(OpKind::Identity, &[4], OPSET);
        // Drop a value without touching `dims`, so the declared shape and the stored data
        // disagree — the situation this rule exists to catch.
        let TensorData::F32(values) = &mut case.inputs[0].data else {
            unreachable!("well_formed builds f32 tensors");
        };
        values.pop();
        assert!(validate(&case).iter().any(|p| matches!(
            p,
            Invalid::ShapeDataMismatch {
                expected: 4,
                actual: 3,
                ..
            }
        )));
    }

    #[test]
    fn mismatched_input_shapes_are_caught() {
        let mut case = well_formed(OpKind::Add, &[2, 3], OPSET);
        case.inputs[1] = TensorValue::f32("b", vec![3, 2], vec![0.0; 6]);
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::ShapeMismatch { .. }))
        );
    }

    #[test]
    fn duplicate_input_names_are_caught() {
        let mut case = well_formed(OpKind::Add, &[2], OPSET);
        case.inputs[1].name = case.inputs[0].name.clone();
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::BadInputName { .. }))
        );
    }

    #[test]
    fn duplicate_attribute_names_are_caught() {
        let case = well_formed(OpKind::Identity, &[2], OPSET)
            .with_attrs(crate::attrs::Attrs::new().int("axis", 0).int("axis", 1));
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::BadAttributeName { .. })),
            "ONNX looks attributes up by name; a duplicate makes the node ambiguous"
        );
    }

    #[test]
    fn an_empty_attribute_name_is_caught() {
        let case = well_formed(OpKind::Identity, &[2], OPSET)
            .with_attrs(crate::attrs::Attrs::new().int("", 0));
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::BadAttributeName { .. }))
        );
    }

    /// Attributes are optional: a case with none must still be valid, or every elementwise
    /// operator would be rejected.
    #[test]
    fn a_case_with_no_attributes_is_valid() {
        for op in OpKind::ALL {
            let case = well_formed(op, &[2], OPSET);
            assert!(case.attrs.is_empty());
            assert_eq!(validate(&case), vec![]);
        }
    }

    #[test]
    fn an_out_of_range_opset_is_caught() {
        for opset in [MIN_OPSET - 1, MAX_OPSET + 1, 0, -1, 9999] {
            let case = well_formed(OpKind::Add, &[2], opset);
            assert!(
                validate(&case)
                    .iter()
                    .any(|p| matches!(p, Invalid::OpsetOutOfRange { .. })),
                "opset {opset} should be rejected"
            );
        }
    }

    /// All violations are reported, not just the first. Fixing one and rediscovering the
    /// next is how a generator bug takes four iterations to characterise instead of one.
    #[test]
    fn every_violation_is_reported_not_just_the_first() {
        let case = OnnxCase::new(
            OpKind::Add,
            9999,
            vec![TensorValue::f32("a", vec![-1], vec![])],
        );
        let problems = validate(&case);

        assert!(
            problems.len() >= 3,
            "expected several problems, got {problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| matches!(p, Invalid::OpsetOutOfRange { .. }))
        );
        assert!(
            problems
                .iter()
                .any(|p| matches!(p, Invalid::WrongArity { .. }))
        );
        assert!(
            problems
                .iter()
                .any(|p| matches!(p, Invalid::NegativeDimension { .. }))
        );
    }

    /// The opset ceiling must match what the pinned `onnx` actually supports. Two
    /// constants that must agree, in different modules, is exactly how a limit drifts.
    #[test]
    fn the_opset_ceiling_matches_the_recorded_specification() {
        assert_eq!(
            MAX_OPSET,
            crate::environment::MAX_OPSET,
            "the validator's ceiling and the recorded spec version have drifted"
        );
    }

    #[test]
    fn input_names_are_unique() {
        let names: Vec<String> = (0..30).map(input_name).collect();
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(names.len(), unique.len(), "input_name produced a collision");
    }

    /// Every element type must be constructible in a well-formed way, or the type exists
    /// in the enum while nothing can generate it — a variant nothing tests.
    #[test]
    fn well_formed_cases_validate_for_every_element_type() {
        for elem in ElemType::ALL {
            for op in OpKind::ALL {
                let case = well_formed_typed(op, &[2, 3], OPSET, elem);
                assert_eq!(
                    validate(&case),
                    vec![],
                    "{op:?} at {elem:?} should be valid"
                );
                assert_eq!(case.inputs[0].elem_type(), elem);
            }
        }
    }

    /// Now reachable with a real case: this rule previously had no way to fire, because
    /// only one element type existed.
    #[test]
    fn mismatched_element_types_are_caught() {
        let mut case = well_formed(OpKind::Add, &[2], OPSET);
        case.inputs[1] = TensorValue::new("b", vec![2], TensorData::I64(vec![1, 2]));
        assert!(
            validate(&case)
                .iter()
                .any(|p| matches!(p, Invalid::ElemTypeMismatch { .. })),
            "an f32 and an i64 input to the same elementwise operator must be rejected"
        );
    }
}
