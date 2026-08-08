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
the two 2026-08-07 campaigns below and several seeded runs.

Those campaigns ran against `burn-ndarray`, which `burn` has since stopped listing among its
first-party CPU backends; it was replaced by `burn-flex`. **810 of the 814 recorded findings
no longer reproduce** on the new pair — a fact the tool reports about *itself*, since a
recorded problem that can no longer be produced is a claim the repository can no longer
support.

### What three independent implementations actually disagree about

Two campaigns on 2026-08-07, fifteen operations across `burn-flex`, `burn-tch` and
`burn-wgpu`:

| | |
|---|---|
| full operation set, 6 h | 4.2 M cases, **1,834 findings**, 8 signatures — all `max`/`min` |
| eight arithmetic operations, 2.5 h | **3.9 M cases, 0 findings**, 12,709 non-diverging cases recorded |

**Every divergence this project has found is structural**, not numeric — a disagreement about
what the operation produced, rather than about how precisely:

- `matmul` returning `inf` against `NaN` when intermediate products overflow (filed as
  burn#5284)
- `max([1, NaN, 3])` — both CPU backends propagate the `NaN`, the GPU ignores it and returns
  `3.0`. IEEE-754 defines *both* conventions, so this is recorded as an observation rather
  than adjudicated
- `max([-inf, -inf])` returning **`-3.4028235e38`** on the GPU — exactly `-f32::MAX`, the
  sentinel a parallel reduction seeds its accumulator with. No convention permits this, and it
  corrupts silently: an infinity becomes a large finite number and downstream arithmetic
  carries on

**The 3.9 million clean cases are the other half of that claim**, and they only count because
the tolerances were audited first: `softmax` and `exp` had been carrying bounds so loose that
65% and 81% of their cases *could not fail*, while being reported as agreement. Fixing that —
and making the oracle report an unusable bound as **unjudged** rather than as a pass — is what
turned "we found nothing there" into evidence.

### Grouping findings by cause, not by symptom

Two findings sharing a *signature* look alike; that is not evidence they share a bug. So the
tool also proposes a **predicate**: a claim about the *input*, of the form
`overflow_product ∧ magnitude_ratio_extreme`, built from 28 boolean properties of a case.

The difference is that a signature can only describe the past, while a predicate makes a
falsifiable claim about inputs nobody has run. It is tested that way: enumerate all 27,776
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
- **`matmul` cases with extreme values are reported as unjudged, not passed.** Products of
  `1e30` genuinely cancel to anything, so no comparison distinguishes a defect from
  arithmetic. 96% of `matmul` cases fall here — correct rather than fixed, and now visible
  rather than counted as agreement.
- **`conv2d` is implemented and has found nothing.** It was added because it is the one
  operation whose three backends run genuinely different *algorithms* — `burn-flex` selects
  among five shape-dependent code paths, `burn-tch` delegates to libtorch, `burn-wgpu` runs
  its own kernel — and because burn has repeatedly shipped bugs in exactly those paths
  (#4727 fixed a missing channel offset triggered by `groups > 1` **and** `padding > 0`).
  A 42-configuration sweep of every path boundary agrees, with the bound 32x wider than the
  worst observed error; so does a campaign over ordinary values. **The two defects the work
  did surface were both in this tool**, not in `burn`: a tolerance that fell below one
  subnormal step, and a domain restriction that never excluded the values it claimed to.
- **Nine operations remain unimplemented**, including pooling and attention. Pooling looks
  unpromising — its tracker shows feature requests rather than correctness bugs, and it has
  no accumulation to disagree about. **The backward pass is the more interesting gap**: most
  of burn's historical convolution bugs are gradient bugs, which a forward-only differential
  oracle cannot see at all.
- **`reshape` is deferred for an architectural reason**: it is the first operation whose
  output rank differs from its input rank, which the rank-dispatch cannot express without
  quadrupling its instantiations.
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