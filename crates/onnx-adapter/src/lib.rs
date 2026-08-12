//! The ONNX adapter — the project's third domain.
//!
//! # What this domain tests
//!
//! ONNX is an interchange format for machine-learning models. A model is a graph of
//! **operators** (`Add`, `Reshape`, `MatMul`, …), each specified in a published operator
//! reference with type constraints, attributes, and written semantics. Many independent
//! systems load and execute those models — ONNX Runtime, `tract`, `candle-onnx` — and all
//! of them claim to implement the same document. That shared claim is what makes them
//! comparable at all: where nobody promises agreement, every difference is legal and
//! there is no oracle.
//!
//! This adapter builds a model with **exactly one node**, runs it on several runtimes,
//! and compares. One node keeps a case small enough to minimize, to serialize into a bug
//! report, and to reason about.
//!
//! # What makes this domain different from the other two
//!
//! One participant is not a peer. `onnx.reference` is the specification's own executable
//! definition, so a mismatch against it is a **conformance violation** with a name
//! attached, rather than a peer disagreement where you cannot say who is wrong. Neither
//! the tensor domain nor the SQL domain had that.
//!
//! # Layout
//!
//! Read the modules in dependency order; each depends only on the ones above it.
//!
//! | module | what it answers |
//! |---|---|
//! | [`case`] | what a test case *is* |
//! | [`attrs`] | an operator's static parameters |
//! | [`ops`] | per-operator arity, types, output shape, and probe models |
//! | [`census`] | which runtime supports which operator, **measured** |
//! | [`capability`] | what each runtime *claims*, so a crash can be told from a gap |
//! | [`outcome`] | what a runtime gives back, **including its failures, as values** |
//! | [`validation`] | is this case well-formed? |
//! | [`model`] | case → protobuf bytes |
//! | [`runtimes`], [`reference`] | bytes → each participant → outcome |
//! | [`normalize`] | outcome → comparable form |
//! | [`oracle`] | comparable forms → verdict |
//! | [`testing`] | implementations that are **wrong on purpose** |
//! | [`generator`] | seed → case |
//! | [`environment`] | the versions a finding applies to |
//! | [`pb`] | the ONNX protobuf types, generated at build time |

pub mod attrs;
pub mod capability;
pub mod case;
pub mod census;
pub mod environment;
pub mod generator;
pub mod model;
pub mod normalize;
pub mod ops;
pub mod oracle;
pub mod outcome;
pub mod reference;
pub mod runtimes;
pub mod testing;
pub mod validation;

/// The ONNX protobuf types — `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, and
/// the rest of the schema.
///
/// These are **generated during the build** by `build.rs`, from the copy of the official
/// schema in `proto/onnx.proto`, and written into cargo's `OUT_DIR`. `include!` pastes
/// that generated file in here as if it had been typed at this spot.
///
/// Generated code is not committed, which is why there is no `pb.rs` to read: to see the
/// types, build the crate and open
/// `target/debug/build/onnx-adapter-*/out/onnx.rs`, or read `proto/onnx.proto`, which is
/// the source of truth for all of it.
///
/// A note on the shape of these types, because it surprises people coming from proto3:
/// `onnx.proto` is written in **proto2**, where every scalar field is optional. That is
/// why almost everything below is an `Option<T>` — `ModelProto::ir_version` is
/// `Option<i64>`, not `i64`. It is verbose, but it is honest: the format genuinely does
/// distinguish "this field was not set" from "this field was set to zero", and several
/// ONNX fields depend on that distinction.
// Lints are switched off for this module alone, because none of it is ours to fix: the doc
// comments are ONNX's own prose from `onnx.proto`, copied through by prost, and their
// formatting is a property of the upstream schema. Treating a clippy warning as a TODO is the
// project's rule (`CLAUDE.md` §3) — but a TODO that can only be resolved by editing someone
// else's specification is not one, and silencing it here keeps the *rest* of the crate at
// `-D warnings` where the rule does apply.
#[allow(
    clippy::doc_markdown,
    clippy::doc_overindented_list_items,
    clippy::derive_partial_eq_without_eq
)]
pub mod pb {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

/// Where this domain's findings are written.
///
/// Domain-scoped, matching the convention the other two adapters follow, so a campaign in
/// one domain can never overwrite another's evidence.
pub const FINDINGS_ROOT: &str = "findings/onnx";

/// # Maintenance rule: adding an operator or element type
///
/// **Adding a variant to `OpKind` or `ElemType` must be the same commit that:**
///
/// 1. extends `ops::spec` with its tier, family, `since` version and accepted types —
///    **retrieved from the schema registry, never recalled** — and records the retrieval in
///    `SPECS.md`;
/// 2. extends `ops::output_spec` and `ops::arity_range` if it joins no existing family;
/// 3. extends `ops::probe` so the census can measure it;
/// 4. **regenerates `findings/onnx/census.json`**, because a capability matrix that predates
///    the operator silently reports it as unsupported everywhere;
/// 5. extends the reference runner's dtype table, for a new element type.
///
/// This is a review rule, not a technical one, and it exists because `08-RISKS.md` §4 records
/// **ten** checks in earlier domains that silently narrowed while what they measured grew.
/// None announced itself. The mechanism was always the same: a check enumerated what existed
/// when it was written.
///
/// Two things make the compiler carry part of the burden — the exhaustive `match` in
/// `ops::spec` and the one over `TensorData` in every runtime both fail to compile until a new
/// variant is handled everywhere. **Steps 4 and 5 are not compiler-enforced**, and step 4 is
/// the one that fails silently.
///
/// Where sampled non-findings are written.
///
/// Negatives are not spare output. A predicate that has survived zero negatives has
/// survived nothing, and would otherwise score identically to one that survived
/// everything — so the cases that *did not* diverge are kept as evidence in their own
/// right.
pub const NEGATIVES_ROOT: &str = "findings/onnx/negatives";
