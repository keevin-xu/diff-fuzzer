# SPECS — what the standards and implementations actually promise

**Companion to `POLICY.md`.** That file states the *decisions*; this one states the
*evidence* they rest on. Every numeric threshold in the policy should point at an entry
here, or be marked as measured rather than derived.

**Scope: the tensor domain.** A second domain adapter brings its own `SPECS.md`; §1 stops
applying entirely, since SQL engines make no promises about IEEE-754. See
`../../planning/09-ADOPTING-A-NEW-DOMAIN.md`.

---

## How to read an entry

Each carries: **the claim · where it comes from · retrieved when · quoted or paraphrased**.

The retrieval date is not bureaucracy. **A tolerance derived from a specification is a claim
about that revision of it** — if WGSL loosens a bound in 2027, a policy line citing the 2026
text is not wrong so much as stale, and the date is what makes the difference visible.

**§5 is the most important section.** It holds claims this project currently *relies on*
and has **not** verified against a source. Keeping them separate from the cited ones is the
whole point: otherwise confident prose makes recalled figures and read figures look
identical, which is exactly how a wrong number survives review.

---

## 1. The arithmetic standard — applies to every backend

### 1.1 Correctly-rounded operations

> **Claim:** IEEE-754 requires `+`, `−`, `×`, `÷` and `sqrt` to be *correctly rounded* —
> the result is the exact mathematical answer rounded once to the nearest representable
> value, so two conforming implementations must agree **bit-for-bit**.

- **Source:** IEEE 754-2019, §5.4.1 (arithmetic operations) and §5.4.2. **Not retrieved.**
- **Status:** widely taught and consistent with everything measured here — five of seven
  such operations are bit-identical across three independent backends (§4.2) — but **cited
  from memory, not from the text.** Formally belongs in §5 until read.
- **What it buys:** the entire `OpClass::CorrectlyRounded` tier, which demands exact
  equality. This is the single most load-bearing uncited claim in the project.

### 1.2 `exp` is *not* in that set

> **Claim:** transcendental functions including `exp` carry no correctly-rounded
> requirement; implementations may differ.

- **Source:** IEEE 754-2019 §9.2 lists them as *recommended*, not required operations.
  **Not retrieved.**
- **Status:** same as 1.1. Corroborated by measurement: `exp` differs by up to 4 ULP
  between CPU and GPU, and by 1 ULP between `flex` and `tch`.
- **But note §2b.3:** `flex` and `ndarray` agree on `exp` to **0 ULP**, which for an
  unrequired operation is evidence they share an implementation rather than evidence of
  conformance. Measurements between non-independent backends corroborate nothing.

### 1.3 Fused multiply-add rounds once

> **Claim:** `fma(a, b, c)` computes `a×b + c` with a **single** rounding, so the
> intermediate product is never rounded — and therefore never overflows to `±inf` on its
> own.

- **Source:** IEEE 754-2019 §5.4.1. **Not retrieved.**
- **Status:** **behaviourally confirmed here.** This is the mechanism behind
  [burn#5284](https://github.com/tracel-ai/burn/issues/5284): `matrixmultiply` fuses and
  returns `inf`, libtorch's corner-cleanup path does not and returns `NaN`. Whatever the
  text says, the *observable* difference is established.

---

## 2. `burn-ndarray` / `matrixmultiply`

### 2.1 Matmul uses a NEON fused multiply-add on aarch64

> **Claim:** `burn-ndarray`'s matmul reaches `ndarray::linalg::general_mat_mul` →
> `matrixmultiply::sgemm` → `kernel_target_neon`, which accumulates with
> `vfmaq_laneq_f32` — a fused multiply-add.

- **Source:** **read directly from the vendored crate source**, `matrixmultiply` 0.3.11.
- **Status:** **verified.** The strongest-evidenced entry in this file.

### 2.2 No documented numerical contract

> **Claim:** neither `burn-ndarray` nor `matrixmultiply` documents an accuracy guarantee,
> an accumulation order, or a subnormal policy.

- **Source:** searched both crates' documentation during PHASE-6 triage.
- **Status:** verified as an *absence*, which is weaker evidence than a presence — a
  guarantee could exist somewhere unread. Treated as "unspecified", which is why
  burn#5284 was filed as a **question** rather than a bug report.

---

## 2b. `burn-flex`

### 2b.1 Applicable specification: IEEE-754, same as any CPU backend

`flex` executes on the CPU, so the arithmetic standard in §1 applies unchanged. **No new
document to retrieve** — unlike `wgpu`, where Metal's own accuracy table had to be found.

### 2b.2 Measured conformance — 2026-08-05, step 7A.4

`examples/gpu_numerics.rs`, 200 random cases per operation, reported in ULPs:

| operation | vs `ndarray` | vs `tch` |
|---|---|---|
| `add` `sub` `mul` `div` `sqrt` `neg` `abs` | **0 ULP, 200/200 exact** | **0 ULP, 200/200 exact** |
| `exp` | **0 ULP** | 1 ULP |

**`flex` conforms to §1.1 on every correctly-rounded operation**, against both established
CPU backends. `Tolerance::EXACT` therefore carries over with no policy change — verified
rather than assumed, since "one CPU backend is like another" is exactly the sort of
assumption this file exists to stop.

### 2b.3 A caution: `flex` and `ndarray` are not fully independent for `exp`

`exp` carries **no** correctly-rounded requirement (§1.2), and the measurement shows it:
`flex` and `tch` differ by 1 ULP, which is two independent approximations landing on
adjacent representable numbers.

`flex` and `ndarray` agree to **0 ULP on all 200 cases** — which for an operation nobody is
required to round correctly is not conformance, it is **evidence of a shared
implementation**, most likely both deferring to Rust's `f32::exp`.

> **Two backends sharing an implementation are not independent, and a differential between
> them is weaker than it appears.** This is `POLICY.md` §9's "both implementations wrong the
> same way" blind spot, in a concrete and measurable form.

Consequence: a `flex`-vs-`ndarray` pair would add little for `exp`. It matters less than it
might, because PHASE-7A replaces one with the other rather than running both — but it is
worth knowing before any future pair is added on the assumption that more backends means
more independence.

### 2b.4 No documented numerical contract

Searched: `burn-flex` publishes no accuracy guarantee, accumulation order, or subnormal
policy. Same status as §2.2 and §3.2 — verified as an *absence*, which is weaker evidence
than a presence.

---

## 3. `burn-tch` / libtorch

### 3.1 GEMM fuses in the micro-kernel and not in the trailing corner

> **Claim:** libtorch's `f32` GEMM fuses its multiply-add inside a **4×8 micro-kernel** and
> does **not** in the cleanup path handling the trailing corner. The number of output
> elements that consequently disagree with a uniformly-fusing implementation is
> `(m mod 4) × (n mod 8)`.

- **Source:** **measured**, `examples/batched_probe.rs`, 2026-08-04. The formula predicted
  every shape tested with no exceptions: `16×32`, `17×32`, `16×33`, `8×16`, `12×24` → 0;
  `17×33` → 1; `14×27` → 6; `1×1` → 1.
- **Status:** the *effect* is verified and predictive. The *explanation* — that the tile is
  4×8 and that the cleanup path does not fuse — is **inference from the pattern**;
  libtorch's kernel source has not been read.
- **Consequence:** libtorch is internally inconsistent within one `matmul` call, returning
  `inf` for 372 elements and `NaN` for 6 in the `14×27` case.

### 3.2 No documented numerical contract

Same status as 2.2.

---

## 4. `burn-wgpu` / WGSL / Metal

### 4.1 Accuracy requirements — **RETRIEVED 2026-08-04**

- **Source:** *Metal Shading Language Specification*, version dated **2026-06-04**, §8
  "Numerical Compliance", Tables 8.1 and 8.2 — pages 368–371.
  https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- **How:** the PDF is 14 MB and defeated direct fetching; downloaded and text-extracted
  locally. **Quoted verbatim below**, not paraphrased.
- **Why Metal and not WGSL:** `wgpu` compiles WGSL → Metal Shading Language via `naga` on
  this machine, so Metal's table is the *operative* contract for what the hardware actually
  does. WGSL's own table remains unretrieved (five failed routes, recorded below) and would
  be the right source for a **portable** bound across devices.

#### The decisive detail: fast math is the default

> "Table 8.2 describes the minimum accuracy of single-precision floating-point arithmetic
> operations given as ULP values **with fast math enabled (which is the default unless you
> specify `-fno-fast-math` as a compiler option)**."

So Table 8.2 governs, not 8.1.

#### Table 8.2 — single precision, fast math enabled (verbatim rows)

| Math function | Minimum accuracy (ULP values) |
|---|---|
| `x + y` | Correctly rounded |
| `x - y` | Correctly rounded |
| `x * y` | Correctly rounded |
| `1.0 / x` | `<= 1 ulp` for x in the domain of 2⁻¹²⁶ to 2¹²⁶ |
| **`x / y`** | **`<= 2.5 ulp`** for y in the domain of 2⁻¹²⁶ to 2¹²⁶ |
| **`exp(x)`** | **`<= 3 + floor(fabs(2 * x)) ulp`** |
| `rsqrt` | `<= 2 ulp` |
| **`sqrt(x)`** | **"Implemented as `x * rsqrt(x)` with special cases handled correctly"** |
| `fma` | Correctly rounded |
| `fabs`, `fmax`, `fmin` | `0 ulp` |

#### §8.1 — denormalized numbers (verbatim)

> "Denormalized single-precision, half-precision, or brain floating-point numbers passed as
> input to or produced as the output of single-precision, half-precision, or brain
> floating-point arithmetic operations **may be flushed to zero**."

**Subnormal flushing is explicitly permitted.** §4.3's measurement is conforming behaviour,
not a defect.

#### Also relevant (verbatim)

> "the Metal compiler, in fast math mode … may do various optimization like reassociate
> floating-point operations that may dramatically change results in floating-point.
> Reassociation may change or ignore the sign of zero, allow optimizations to assume the
> arguments and result are not NaN or +/-INF, inhibit or create underflow or overflow…"

This licenses reassociation — which is what makes accumulation order unspecified, and is the
same family of permission behind burn#5284 on the CPU side.

> "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is
> **undefined**."

**Significant and uncomfortable:** with fast math on, Metal makes *no* guarantee about
`NaN`/`inf` handling at all. Any GPU finding whose signature is `undefined` is therefore
unspecified behaviour rather than a defect — see `POLICY.md`.

#### The measurement discriminates between the two tables

Table 8.1 (fast math **off**) lists `x / y` and `sqrt` as *correctly rounded*, which would
mean 0 ULP. §4.2 measured 1 and 2 ULP. **So the measurement is itself evidence that fast
math is enabled**, independently of the spec's statement that it is the default.

#### WGSL's own table — still unretrieved

Kept because a *portable* bound (across devices, not just this Mac) would come from WGSL,
not Metal. Five routes failed on 2026-08-04:

1. `https://www.w3.org/TR/WGSL/#floating-point-accuracy` — truncates during §3.
2. `https://www.w3.org/TR/WGSL/#floating-point-evaluation` — same.
3. `https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#spirvenv-precision-operation` — **HTTP 403**.
4. `https://raw.githubusercontent.com/gpuweb/gpuweb/main/wgsl/index.bs` — truncates before §15.
5. Two web searches — unrelated patents and the OpenCL numerical-compliance document, which
   is a *different* standard and must not be substituted.

- **Partial finding:** the WebGPU CTS (`src/webgpu/util/floating_point.ts`) builds addition's
  interval with `correctlyRoundedIntervalWithUnboundedPrecisionForAddition` — weak evidence
  that WGSL also requires correctly-rounded addition, consistent with Metal.

### 4.2 Measured behaviour — CPU versus GPU

**Measurement, not specification.** Recorded here because it is what any derived bound must
be checked against, and because §4.1 is missing.

Source: `examples/gpu_numerics.rs`, 2026-08-04, 200 random cases per operation, reported in
ULPs.

| operation | worst | exact on | reading |
|---|---|---|---|
| `add` `sub` `mul` `neg` `abs` | **0 ULP** | **200/200** | conforms to §1.1 |
| `div` | 1 ULP | 0/200 | never exact |
| `sqrt` | 2 ULP | 0/200 | never exact |
| `exp` | 4 ULP | 0/200 | not required to conform (§1.2) |

**The GPU is not broadly loose — it is precisely loose.** Five of seven operations in the
correctly-rounded tier are bit-identical, so the policy must name `div` and `sqrt` rather
than relaxing the whole class.

### 4.3 Subnormals are flushed to zero

- **Measured**, same source: `1e-45` → `±0.0`, `MIN_POSITIVE/2` → `±0.0`, while
  `MIN_POSITIVE` (the smallest *normal*) survives.
- **Not a rounding difference.** A subnormal becoming zero is a **relative error of 1.0**,
  which no `rtol` absorbs. Only an `atol` at the subnormal scale can, and that is a
  different knob from the one `div`/`sqrt` need.
- **Specification status: unknown** — pending §4.1.

### 4.3a Derived bounds, and the margin over measurement

**Derivation first, measurement second** — the order the method requires. Each bound comes
from §4.1's table; the measured column is a *check*, never the source.

| operation | §4.1 permits | derived bound | §4.2 measured | margin |
|---|---|---|---|---|
| `add` `sub` `mul` | correctly rounded | **`EXACT`** | 0 ULP | exact, as required |
| `neg` `abs` | `0 ulp` (`fabs`) | **`EXACT`** | 0 ULP | exact |
| `div` | `<= 2.5 ulp` | **`rtol = 2.5 ε`** ≈ `2.98e-7` | 1 ULP ≈ `1.19e-7` | **2.5x** |
| `sqrt` | `x * rsqrt(x)`, `rsqrt <= 2 ulp` | **`rtol = 3 ε`** ≈ `3.58e-7` | 2 ULP ≈ `2.38e-7` | **1.5x** |
| `exp` | `<= 3 + floor(2|x|) ulp` | scales with `|x|` — the existing condition-number policy already covers this | 4 ULP at `|x| <= 5` (13 permitted) | **3.2x** |

**`sqrt`'s derivation, since it is the only composed one.** The spec does not give `sqrt` a
ULP figure; it says `sqrt(x)` is *implemented as* `x * rsqrt(x)`, and gives `rsqrt <= 2 ulp`.
The multiply is correctly rounded, contributing at most a further half ULP, so **3 ULP** is
the bound the composition permits. Measurement lands at 2.

**Nothing here was chosen to fit the data.** `2.5` and `3 + floor(2|x|)` are the
specification's own numbers; `3` for `sqrt` follows from composing two of them. Had the
measurement exceeded any derived bound, that would be a finding about the GPU — which is
precisely the value of deriving first.

### 4.4 Reduction determinism

- **`sum_dim()` (axis reduction) is deterministic** — one distinct value over 10 runs at
  256, 4,096 and 65,536 elements. **This is the only reduction the generator emits.**
- **`sum()` (full reduction to a scalar) is not** — two distinct values, 1 ULP apart.
- **Measured**, `examples/gpu_numerics.rs` and `examples/wgpu_check.rs`, 2026-08-04.
- An earlier claim that "GPU reductions are non-deterministic" was **retracted**: true of a
  kernel this project never calls.

---

## 5. Relied upon, NOT verified

**Read this section as the project's honest liability list.** Everything here is currently
believed and load-bearing.

| claim | where it is used | risk if wrong |
|---|---|---|
| **§1.1** IEEE-754 requires `+ − × ÷ sqrt` correctly rounded | the entire `CorrectlyRounded` tier | **high** — it is why `add` must match bit-for-bit. Measurement across three independent backends corroborates it strongly, but that is evidence of *behaviour*, not of the *requirement* |
| **§1.2** `exp` is not required | the `Approximated` tier exists at all | moderate — if `exp` were required correctly-rounded, the derived bound would be hiding real bugs |
| **§1.3** FMA rounds once | the burn#5284 mechanism story | low — the behaviour is directly observed |
| ~~**§4.1** any WGSL ULP figure~~ | ~~nothing yet~~ | ✅ **resolved 2026-08-04** — Metal's table retrieved and quoted verbatim. WGSL's own table is still unretrieved and would be needed for a *portable* bound across devices, but Metal is the operative contract on this machine |
| **§4.1 fast math is enabled in `wgpu`'s Metal output** | the choice of Table 8.2 over 8.1 | **moderate** — the spec says fast math is the default, and §4.2's measurement independently agrees (1 and 2 ULP where Table 8.1 would require 0). Not confirmed by reading `naga`'s compiler flags |

**Also relied upon, and not derivable from any specification:** that `NaN` vs `NaN` counts
as *agreement*. `NaN` does not indicate whether an operation was mathematically defined — it
conflates "no answer exists" with "an answer existed and precision destroyed it" — so the
rule rests on the narrower claim that both implementations *behaved alike*. Corrected
2026-08-05, after the original justification turned out to be false for burn#5284 itself.
See `POLICY.md` §5.

**The discipline this file exists to enforce:** a number may enter `POLICY.md` only if it
cites §1–§4, or is explicitly labelled as measured. A number justified by "I recall the
figure is 2.5" belongs in §5, or nowhere.

---

## 6. Holding a bound needs no citation; loosening one does

**The asymmetry that decides how much a missing specification actually blocks.**

A tolerance is an allowance for disagreement. Every unit of it is sensitivity given away.
So the two directions carry completely different burdens of proof:

| | requires | if wrong |
|---|---|---|
| **Holding or tightening** a bound | nothing beyond evidence it is *achievable* | **false positives** — noisy, visible, self-correcting |
| **Loosening** a bound | a specification saying the difference is permitted | **hidden bugs** — silent, invisible, permanent |

A false positive announces itself. A tolerance wide enough to swallow a real defect
announces nothing, ever. **That asymmetry is why a missing citation blocks relaxation and
not retention** — and it is the same reasoning behind `signature.rs` erring finer rather
than coarser.

### What this unblocks, concretely

Given §4.1 is unretrieved, the GPU work divides cleanly:

- **`add` `sub` `mul` `neg` `abs` — decided, no citation needed.** Hold `Tolerance::EXACT`
  even for GPU pairs. §4.2 measured them bit-identical on 200/200 cases, so exactness is
  demonstrably *achievable* on this hardware; holding it is the strict direction and costs
  only false positives if a future device is looser. **No code changes.**
- **`div` `sqrt` — blocked.** Any change loosens, and there is no source permitting it.
- **Subnormal flushing — blocked.** Absorbing it needs an `atol` at the subnormal scale,
  which is also a relaxation.

**The interim position for the blocked operations is to leave them strict and let the
divergences be reported**, grouped by signature and marked as a pending class. That is
honest: they *do* differ, and whether that is legal is exactly the open question. Absorbing
them now would answer that question by assumption, in the direction that hides things.
