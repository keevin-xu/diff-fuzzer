//! # diff-fuzzer-core
//!
//! The reusable, **target-agnostic** engine for differential testing.
//!
//! This crate knows nothing about tensors, SQL, or any other kind of software under
//! test. It only knows the trait seams defined in [`traits`] (PHASE-1): given some
//! way to *generate* an input, *run* it on several implementations, and *normalize*
//! the results, it can drive the loop, ask an oracle whether the results diverged,
//! shrink any divergence to a minimal case, and report it.
//!
//! Domain knowledge lives in per-target adapter crates instead — `tensor-adapter`
//! is the first one. See `planning/03-ARCHITECTURE.md`.
//!
//! ## Status
//!
//! Stub only (PHASE-0, step 0.5). The modules below are created in later phases:
//!
//! - `traits`    — Input, Implementation, Generator, Normalizer, Oracle, Verdict  (PHASE-1)
//! - `rng`       — SeededRng, the single source of randomness                     (PHASE-1)
//! - `tolerance` — allclose-style comparison + TolerancePolicy                    (PHASE-3/4)
//! - `oracle`    — DifferentialOracle (and, much later, a metamorphic one)        (PHASE-1→3)
//! - `minimize`  — ddmin-style shrinking                                          (PHASE-5)
//! - `report`    — DivergenceReport + emitter                                     (PHASE-5)
