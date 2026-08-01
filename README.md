# diff-fuzzer — Planning Package

> to be updated

A **differential testing + fuzzing framework**, written in **Rust**, whose first target is **deep-learning / tensor libraries**. This folder currently contains the *complete plan* for the project. No implementation code exists yet — the next step is for a fresh Claude Code instance to build it, phase by phase, following these documents.

## What this project is (one paragraph)

We generate structured, valid tensor operations (e.g. `matmul`, `softmax`, `conv`), run each one through **two different backends of the same DL framework** (Rust's `burn`: CPU vs. libtorch, later CPU vs. GPU), and **compare the numerical outputs within a tolerance**. When two backends disagree on the same operation, at least one has a bug. This is *differential testing*: it finds bugs without needing to know the "correct" answer. The framework is built with a **shared, reusable core** and **thin per-target adapters**, so that later we can add new oracles (metamorphic) and new software types (SQL engines) without rewriting the engine.

## Background Information

- **Language:** Rust, throughout the whole project.
- **First software type:** deep-learning / tensor libraries.
- **First oracle:** differential (metamorphic added later — designed for from day one).
- **First implementation pair (Route A):** `burn` framework, two backends — `burn-ndarray` (CPU) as one side, `burn-tch` (libtorch) as the other; a third backend (`burn-wgpu`, Metal GPU) added in a later phase for a CPU-vs-GPU differential.
- **Future (documented, not built yet):** metamorphic oracles (autodiff vs. numerical gradient), and a second software type (SQL engines) as a second adapter on the same core.