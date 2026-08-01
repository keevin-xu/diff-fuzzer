# diff-fuzzer

> to be updated

A **differential testing + fuzzing framework**, written in **Rust**, whose first target is **deep-learning / tensor libraries**.

## Status

The pipeline runs end to end on a single hardcoded operation: a seed produces a test case, the case runs on two `burn` backends (pure-Rust CPU and libtorch), both results are converted into a comparable form, and an oracle returns a verdict — reproducibly, with the seed attached to every log line and every finding.

A deliberately faulty backend is kept in the codebase and the test suite fails if the tool does not catch it. Without that, "no divergences found" would be indistinguishable from a comparison that had quietly stopped working.

```
$ cargo run -p tensor-adapter --example differential
seed 0   -> agree
seed 7 replayed identically: true

with a deliberately faulty backend:
divergence: burn-ndarray+fault(0.5) disagreed with burn-ndarray
  burn-ndarray:            values: [11.0, 22.0, 33.0, 44.0]
  burn-ndarray+fault(0.5): values: [11.5, 22.0, 33.0, 44.0]
```

Known limitations, all scheduled: results are compared for **exact equality** (wrong for floating point — two correct backends routinely differ in the last bits), `NaN` is currently treated as disagreeing with itself, and the generator produces **one** fixed case rather than varied valid operations.

**Build:** `cargo test` — libtorch is downloaded automatically by the build, so there is nothing to install.

## What this project is

We generate structured, valid tensor operations (e.g. `matmul`, `softmax`, `conv`), run each one through **two different backends of the same DL framework** (Rust's `burn`: CPU vs. libtorch, later CPU vs. GPU), and **compare the numerical outputs within a tolerance**. When two backends disagree on the same operation, at least one has a bug. This is *differential testing*: it finds bugs without needing to know the "correct" answer. The framework is built with a **shared, reusable core** and **thin per-target adapters**, so that later we can add new oracles (metamorphic) and new software types (SQL engines) without rewriting the engine.

## Background Information

- **Language:** Rust, throughout the whole project.
- **First software type:** deep-learning / tensor libraries.
- **First oracle:** differential (metamorphic added later — designed for from day one).
- **First implementation pair (Route A):** `burn` framework, two backends — `burn-ndarray` (CPU) as one side, `burn-tch` (libtorch) as the other; a third backend (`burn-wgpu`, Metal GPU) added in a later phase for a CPU-vs-GPU differential.
- **Future (documented, not built yet):** metamorphic oracles (autodiff vs. numerical gradient), and a second software type (SQL engines) as a second adapter on the same core.