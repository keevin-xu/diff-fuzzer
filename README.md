# diff-fuzzer

A **differential testing + fuzzing framework**, written in **Rust**, whose first target is **deep-learning / tensor libraries**.

## Status

Generated tensor operations run on **three `burn` backends** — `burn-flex` (pure-Rust CPU),
`burn-tch` (libtorch/CPU), and `burn-wgpu` (GPU, via Metal) — and their results are
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

Those campaigns ran against `burn-ndarray`, which `burn` has since stopped listing among its
first-party CPU backends; it was replaced by `burn-flex`. **810 of the 814 recorded findings
no longer reproduce** on the new pair — a fact the tool reports about *itself*, since a
recorded problem that can no longer be produced is a claim the repository can no longer
support.

### Grouping findings by cause, not by symptom

Two findings sharing a *signature* look alike; that is not evidence they share a bug. So the
tool also proposes a **predicate**: a claim about the *input*, of the form
`overflow_product ∧ magnitude_ratio_extreme`, built from 17 boolean properties of a case.

The difference is that a signature can only describe the past, while a predicate makes a
falsifiable claim about inputs nobody has run. It is tested that way: enumerate all 6,018
rules over at most three properties, discard any that fires on a case that did **not**
diverge, then generate fresh cases the rule matches and measure how often *those* diverge.

Run against the recorded findings, it proposed three rules and **rejected all three**, and
left **763 of 814 findings unexplained** — no rule over the current vocabulary separates them
from passing cases. That gap is the report's most useful output, and it is written down
rather than dropped. Reaching the filed bug's real trigger would need a property the 17
cannot express: whether *one output element* falls outside both tile boundaries at once.

The rates are only readable because the report also measures a **baseline** — the divergence
rate for cases drawn with no rule at all. On the current backend pair that is **7 in 4,000**
under wide bounds and **0 in 4,000** under default ones, which is both the first evidence
that `flex` disagrees with libtorch and the reason a rejected candidate is reported with its
lift above baseline (9.9×) rather than as a bare failure.

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
- **No predicate has been ratified.** The trigger-grouping machinery is built and every
  candidate it has produced so far failed validation. It is a mechanism that makes
  falsifiable claims, not a validated classifier, and the sample is far too small to be one.
- **The negative pool contains no near-misses.** Those are cases one edit away from
  diverging, and they are the only negatives strong enough to make a surviving rule mean
  much. Every candidate report says so on its own face.
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
- **Implementations (Route A):** the `burn` framework across three backends — `burn-flex` (pure-Rust CPU), `burn-tch` (libtorch/CPU), and `burn-wgpu` (Metal GPU). Adding the third cost **4 lines of production code and no changes to the engine**. *Replacing* one CPU backend with another later touched 25 files — all of them the adapter, its examples, and the fuzz target, mostly mechanical renames — and **still not one line of the engine**, which is the claim the split exists to support.
- **Future (documented, not built yet):** metamorphic oracles (autodiff vs. numerical gradient), and a second software type (SQL engines) as a second adapter on the same core.