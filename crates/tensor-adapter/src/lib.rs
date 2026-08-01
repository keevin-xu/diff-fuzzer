//! # tensor-adapter
//!
//! The **DL/tensor adapter**: the per-software-type half of the project.
//!
//! Everything domain-specific about testing tensor libraries lives here — what a
//! valid tensor operation looks like, how to execute one on a given `burn` backend,
//! and how to canonicalize the resulting tensor for comparison. The engine in
//! `diff-fuzzer-core` drives all of it without knowing any of it.
//!
//! The differential is between two backends of the *same* framework (`burn`), which
//! means one generated op runs on both through an identical API — the design choice
//! that keeps false positives low. See `planning/05-TARGETS-AND-ORACLES.md`.
//!
//! ## Status
//!
//! Stub only (PHASE-0, step 0.5). The modules below are created in later phases:
//!
//! - `ops/`      — one module per operation, each encoding its own constraints  (PHASE-2)
//! - `generator` — correct-by-construction op generation from a seed            (PHASE-2)
//! - `backends`  — `impl Implementation` for each burn backend                  (PHASE-1→7)
//! - `normalize` — tensor output canonicalization                               (PHASE-1→4)
