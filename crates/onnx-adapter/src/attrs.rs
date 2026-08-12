//! Node attributes — an operator's static parameters.
//!
//! # Attributes are not inputs
//!
//! An ONNX node has two kinds of parameter, and confusing them is a common way to build a
//! model that is subtly wrong:
//!
//! - **inputs** are tensors, named, flowing through the graph, supplied at execution time;
//! - **attributes** are constants baked into the node — `axis`, `perm`, `to`, `keepdims`.
//!
//! Which is which is **per operator and per opset**, and it changes. `Squeeze` took its
//! `axes` as an *attribute* through opset 12 and as an *input* from opset 13. Getting that
//! wrong produces a model the runtime rejects, which — if it were not caught — would be
//! read as a capability gap rather than as our bug. That per-opset knowledge belongs in the
//! operator catalog; this module is only the representation.
//!
//! # Why a `Vec` of pairs rather than a map
//!
//! Two reasons, both about determinism.
//!
//! Serialization must be **byte-identical for the same case**, because that is the claim
//! the whole comparison rests on. A `HashMap` iterates in an unspecified order, so the same
//! case could produce different bytes on two runs — and a difference in bytes is a
//! difference in what each runtime was asked to compute.
//!
//! And attribute counts are tiny — one or two per node, never more than a handful — so the
//! lookup cost a map would save does not exist.

use serde::{Deserialize, Serialize};

use crate::pb::{AttributeProto, attribute_proto};

/// One attribute's value.
///
/// Only the kinds the Tier A and Tier B operator surface needs. ONNX also allows tensor,
/// graph, sparse-tensor and type-proto attributes; those belong to operators this domain
/// does not build (`If`, `Loop`, `Constant`), and adding a variant that nothing generates
/// would be a feature nothing tests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    /// `axis`, `to`, `keepdims` — a single integer. Also how ONNX encodes booleans.
    Int(i64),
    /// `perm`, `axes`, `starts`, `ends` — a list of integers.
    Ints(Vec<i64>),
    /// `alpha`, `beta`, `value` — a single float.
    ///
    /// Serialized as a bit pattern for the same reason tensor values are: a float
    /// attribute could legitimately be an infinity, and JSON would write it as `null`.
    Float(#[serde(with = "crate::case::f32_bits_scalar")] f32),
    /// `mode` — an enumerated string such as `"constant"` or `"reflect"`.
    String(String),
}

impl AttrValue {
    /// The ONNX `AttributeType` tag this value carries.
    ///
    /// The tag is a separate field from the payload in `AttributeProto`, and a runtime
    /// reads the tag to decide which payload field to look at. Setting the payload without
    /// the tag produces an attribute every runtime silently ignores — a failure that looks
    /// like the operator ignoring its parameter.
    fn attribute_type(&self) -> attribute_proto::AttributeType {
        match self {
            AttrValue::Int(_) => attribute_proto::AttributeType::Int,
            AttrValue::Ints(_) => attribute_proto::AttributeType::Ints,
            AttrValue::Float(_) => attribute_proto::AttributeType::Float,
            AttrValue::String(_) => attribute_proto::AttributeType::String,
        }
    }

    /// A short label for reports and signatures.
    pub fn kind(&self) -> &'static str {
        match self {
            AttrValue::Int(_) => "int",
            AttrValue::Ints(_) => "ints",
            AttrValue::Float(_) => "float",
            AttrValue::String(_) => "string",
        }
    }
}

/// A node's attributes, in a fixed order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Attrs(Vec<(String, AttrValue)>);

impl Attrs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an attribute, returning `self` so calls chain.
    ///
    /// Deliberately **not** deduplicating. A duplicate attribute name is an invalid model,
    /// and silently dropping one here would hide a generator bug that `validate` is meant
    /// to catch — the same "fix it where it is noticed" mistake that let one property be
    /// re-fixed at four separate sites in an earlier domain.
    #[must_use]
    pub fn with(mut self, name: &str, value: AttrValue) -> Self {
        self.0.push((name.to_owned(), value));
        self
    }

    #[must_use]
    pub fn int(self, name: &str, value: i64) -> Self {
        self.with(name, AttrValue::Int(value))
    }

    #[must_use]
    pub fn ints(self, name: &str, values: Vec<i64>) -> Self {
        self.with(name, AttrValue::Ints(values))
    }

    #[must_use]
    pub fn float(self, name: &str, value: f32) -> Self {
        self.with(name, AttrValue::Float(value))
    }

    #[must_use]
    pub fn string(self, name: &str, value: &str) -> Self {
        self.with(name, AttrValue::String(value.to_owned()))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Look one up by name. `None` if absent.
    pub fn get(&self, name: &str) -> Option<&AttrValue> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(name, _)| name.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &AttrValue)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }

    /// Convert to the protobuf form, preserving order.
    pub fn to_protos(&self) -> Vec<AttributeProto> {
        self.0
            .iter()
            .map(|(name, value)| {
                let mut proto = AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(value.attribute_type() as i32),
                    ..Default::default()
                };
                match value {
                    AttrValue::Int(v) => proto.i = Some(*v),
                    AttrValue::Ints(v) => proto.ints = v.clone(),
                    AttrValue::Float(v) => proto.f = Some(*v),
                    // ONNX string attributes are raw bytes, not UTF-8-validated text.
                    AttrValue::String(v) => proto.s = Some(v.clone().into_bytes()),
                }
                proto
            })
            .collect()
    }

    /// A stable one-line rendering, for a finding's signature and for reports.
    pub fn describe(&self) -> String {
        if self.is_empty() {
            return "-".to_string();
        }
        self.0
            .iter()
            .map(|(name, value)| match value {
                AttrValue::Int(v) => format!("{name}={v}"),
                AttrValue::Ints(v) => format!("{name}={v:?}"),
                // Bits alongside the value, consistent with how tensors are rendered: a
                // report showing `0` for both `+0.0` and `-0.0` hides a real difference.
                AttrValue::Float(v) => format!("{name}={v}#{:08x}", v.to_bits()),
                AttrValue::String(v) => format!("{name}={v:?}"),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attributes_keep_the_order_they_were_added() {
        let attrs = Attrs::new()
            .int("axis", 1)
            .ints("perm", vec![1, 0])
            .int("to", 7);
        assert_eq!(
            attrs.names().collect::<Vec<_>>(),
            vec!["axis", "perm", "to"]
        );
        assert_eq!(
            attrs
                .to_protos()
                .iter()
                .filter_map(|p| p.name.clone())
                .collect::<Vec<_>>(),
            vec!["axis", "perm", "to"]
        );
    }

    /// The property the whole comparison rests on: the same case must produce the same
    /// bytes. A map-backed representation could not guarantee this.
    #[test]
    fn conversion_is_deterministic() {
        let build = || {
            Attrs::new()
                .int("axis", 1)
                .ints("axes", vec![0, 2])
                .float("alpha", 0.5)
                .string("mode", "constant")
        };
        assert_eq!(build().to_protos(), build().to_protos());
    }

    /// Every variant must set **both** the type tag and its payload. An attribute with a
    /// payload but no tag is silently ignored by runtimes, which reads as the operator
    /// ignoring its parameter rather than as a malformed model.
    #[test]
    fn every_variant_sets_its_tag_and_payload() {
        let attrs = Attrs::new()
            .int("i", 7)
            .ints("is", vec![1, 2])
            .float("f", 1.5)
            .string("s", "constant");
        let protos = attrs.to_protos();

        use attribute_proto::AttributeType as Ty;
        assert_eq!(protos[0].r#type, Some(Ty::Int as i32));
        assert_eq!(protos[0].i, Some(7));

        assert_eq!(protos[1].r#type, Some(Ty::Ints as i32));
        assert_eq!(protos[1].ints, vec![1, 2]);

        assert_eq!(protos[2].r#type, Some(Ty::Float as i32));
        assert_eq!(protos[2].f, Some(1.5));

        assert_eq!(protos[3].r#type, Some(Ty::String as i32));
        assert_eq!(protos[3].s.as_deref(), Some(b"constant".as_slice()));
    }

    /// A float attribute could legitimately be an infinity, and JSON writes that as `null`.
    /// Same hazard as tensor values, same fix.
    #[test]
    fn a_float_attribute_survives_json() {
        for value in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN, -0.0, 1.5] {
            let attrs = Attrs::new().float("alpha", value);
            let restored: Attrs =
                serde_json::from_str(&serde_json::to_string(&attrs).unwrap()).unwrap();

            let Some(AttrValue::Float(round_tripped)) = restored.get("alpha") else {
                panic!("alpha did not survive as a float");
            };
            assert_eq!(
                round_tripped.to_bits(),
                value.to_bits(),
                "{value} changed crossing JSON"
            );
        }
    }

    #[test]
    fn lookup_finds_what_was_added_and_nothing_else() {
        let attrs = Attrs::new().int("axis", 3);
        assert_eq!(attrs.get("axis"), Some(&AttrValue::Int(3)));
        assert_eq!(attrs.get("perm"), None);
        assert_eq!(attrs.len(), 1);
        assert!(!attrs.is_empty());
        assert!(Attrs::new().is_empty());
    }

    /// Duplicates are preserved rather than silently collapsed, so `validate` can report
    /// them. Dropping one here would hide a generator bug.
    #[test]
    fn duplicates_are_preserved_for_the_validator_to_catch() {
        let attrs = Attrs::new().int("axis", 1).int("axis", 2);
        assert_eq!(attrs.len(), 2);
        assert_eq!(attrs.to_protos().len(), 2);
    }

    #[test]
    fn the_description_is_stable_and_shows_signed_zero() {
        assert_eq!(Attrs::new().describe(), "-");
        assert_eq!(
            Attrs::new()
                .int("axis", 1)
                .ints("perm", vec![1, 0])
                .describe(),
            "axis=1 perm=[1, 0]"
        );
        assert_ne!(
            Attrs::new().float("a", 0.0).describe(),
            Attrs::new().float("a", -0.0).describe(),
            "signed zero must be visible in an attribute description too"
        );
    }
}
