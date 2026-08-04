# diff-fuzzer

A **differential testing + fuzzing framework**, written in **Rust**, whose first target is **deep-learning / tensor libraries**.

## Status

Generated tensor operations run on **three `burn` backends** — `burn-ndarray` (pure-Rust
CPU), `burn-tch` (libtorch/CPU), and `burn-wgpu` (GPU, via Metal) — and their results are
compared within a tolerance derived per operation *and per backend pair*. Generation is
coverage-guided via `cargo-fuzz`; divergences are automatically shrunk to a small
reproduction, de-duplicated, and written to disk with the seed and library versions needed
to replay them.

A deliberately faulty backend is kept in the codebase and the test suite fails if the tool
does not catch it. Without that, "no divergences found" would be indistinguishable from a
comparison that had quietly stopped working.

### What it found

**One issue filed upstream:** [tracel-ai/burn#5284](https://github.com/tracel-ai/burn/issues/5284)
— `matmul` returns `inf` on one backend and `NaN` on another when intermediate products
overflow `f32`.

Following it up produced a sharper result than the report. The number of disagreeing output
elements is exactly

```
(m mod 4) * (n mod 8)
```

which predicted every shape tested. libtorch's GEMM fuses its multiply-add inside a 4×8
micro-kernel and does not in the cleanup path handling the trailing corner — so for a
`14 × 27` output it returns **`inf` for 372 elements and `NaN` for 6, within a single
call**. That is a library disagreeing with *itself*, which needs no second backend to
observe.

Scale so far: a four-hour campaign at ~3,500 executions/second (**50,491,268 cases**), plus
seeded campaigns. All findings collapse to two distinct problems.

### How the numbers are justified

Every threshold is **derived from a specification, then checked against measurement** —
never tuned until the output looked clean. Operations the standard requires to be correctly
rounded are held to bit-for-bit equality; `exp` is bounded by its condition number;
summation and matrix multiplication get an allowance computed from each case's own shapes
and values.

The GPU's bounds come from the *Metal Shading Language Specification* §8, quoted with
retrieval dates in `crates/tensor-adapter/SPECS.md`: division `≤ 2.5 ULP`, `sqrt` composed
from `rsqrt ≤ 2 ULP`, subnormal flushing bounded by `f32::MIN_POSITIVE`. Each measured
error sits **inside** its derived bound with margin.

Where a specification permits a difference no tolerance can express — Metal may flush a
subnormal *input*, and `sqrt(1.4e-45)` is `3.7e-23` on a CPU and `0` on the GPU — the case
is recorded as **unjudged rather than passing**. Cases where nothing numeric could be
compared are likewise reported as unjudged. See `crates/tensor-adapter/POLICY.md` for the
full statement, including what the comparison is blind to.

### Known limitations, all recorded rather than glossed

- **A GPU campaign judges far less than its case count suggests.** Subnormals are injected
  deliberately, and any case containing one is unjudged against the GPU — 1,150 of 2,000 in
  a recent run. The skip column says so.
- **Grouping is by symptom, not cause.** Findings sharing a signature are not proven to
  share a bug; the mechanism for grouping by *trigger* is designed but unbuilt.
- **No broadcasting**, and seven planned operations are unimplemented.
- **One `SPECS.md` claim is still uncited** — the IEEE-754 correctly-rounded requirement
  that the strictest tolerance tier rests on. Listed explicitly in that file's §5.

**Build:** `cargo test` — libtorch is downloaded automatically by the build, so there is nothing to install.

## What this project is

We generate structured, valid tensor operations, run each one through **three backends of the same DL framework** (Rust's `burn`: pure-Rust CPU, libtorch, and Metal GPU), and **compare the numerical outputs within a tolerance**. With three implementations the oracle can say *which one is the outlier* rather than only that two disagree. When two backends disagree on the same operation, at least one has a bug. This is *differential testing*: it finds bugs without needing to know the "correct" answer. The framework is built with a **shared, reusable core** and **thin per-target adapters**, so that later we can add new oracles (metamorphic) and new software types (SQL engines) without rewriting the engine.

## Background Information

- **Language:** Rust, throughout the whole project.
- **First software type:** deep-learning / tensor libraries.
- **First oracle:** differential (metamorphic added later — designed for from day one).
- **Implementations (Route A):** the `burn` framework across three backends — `burn-ndarray` (pure-Rust CPU), `burn-tch` (libtorch/CPU), and `burn-wgpu` (Metal GPU). Adding the third cost **4 lines of production code and no changes to the engine**.
- **Future (documented, not built yet):** metamorphic oracles (autodiff vs. numerical gradient), and a second software type (SQL engines) as a second adapter on the same core.