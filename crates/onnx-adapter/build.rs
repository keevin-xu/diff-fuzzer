//! Compiles `proto/onnx.proto` into Rust types at build time.
//!
//! # Why the schema is compiled here rather than depended on
//!
//! An ONNX model is a protobuf message, so before this domain can test anything it needs
//! Rust types for `ModelProto`, `GraphProto`, `NodeProto` and friends. Two crates publish
//! those already, and both were measured against the canonical schema on 2026-08-12:
//!
//! | source                     | `TensorProto` data types | protobuf runtime |
//! |----------------------------|--------------------------|------------------|
//! | `onnx.proto` (onnx 1.22.0) | **27**, IR version 13    | —                |
//! | `onnx-protobuf` 0.2.3      | 23                       | rust-protobuf 3.4|
//! | `onnx-pb` 0.1.4            | 17                       | **prost 0.6**    |
//!
//! Both lag, and the lag is not cosmetic: the missing types are `FLOAT4E2M1`,
//! `FLOAT8E8M0`, `UINT2`, `INT2` — the quantized surface this domain plans to test at
//! PHASE-N9, which `01-DOMAIN-RESEARCH.md` §6.2 calls the best remaining shot at
//! correctness bugs. Depending on a crate that cannot express those types would quietly
//! delete a phase from the roadmap.
//!
//! Compiling the schema ourselves also buys the property the domain actually rests on:
//! **the types we build models with are generated from the same `onnx.proto` that ships
//! with the `onnx` package acting as ground truth.** The reference implementation's
//! version is the specification revision every finding is judged against, so the schema
//! and the judge should not be able to drift apart. `proto/onnx.proto` is a verbatim copy
//! from the pinned `onnx` package, and `tests/proto_matches_reference.rs` fails if the
//! two ever diverge.
//!
//! # Why `protox` and not `protoc`
//!
//! `prost-build` normally shells out to the `protoc` binary, which is not installed on
//! this machine. `protox` is a protobuf compiler written in Rust, so the crate builds
//! anywhere cargo runs. Depending on a system binary whose version nobody records is the
//! same reproducibility hazard the workspace already avoids by bundling SQLite and DuckDB
//! from source rather than linking whatever the machine happens to have.

fn main() {
    // `protox::compile` parses the schema and produces the `FileDescriptorSet` that
    // `protoc` would otherwise emit; `prost_build` turns that into Rust.
    let descriptors = protox::compile(["proto/onnx.proto"], ["proto"])
        .expect("proto/onnx.proto should compile; it is a verbatim copy of a released schema");

    prost_build::Config::new()
        .compile_fds(descriptors)
        .expect("generating Rust types from the ONNX schema should succeed");

    // Without this, editing the schema would not trigger a rebuild and the generated
    // types would silently be the previous revision's.
    println!("cargo:rerun-if-changed=proto/onnx.proto");
}
