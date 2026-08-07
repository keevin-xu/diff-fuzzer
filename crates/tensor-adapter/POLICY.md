# Comparison Policy


**Evidence lives in `SPECS.md`**, beside this file. This document states *decisions*; that
one states what the standards and implementations actually promise, with sources and
retrieval dates. **Its §5 lists claims relied on here that have not been verified against a
source** — including, uncomfortably, the IEEE-754 correctly-rounded requirement that the
whole `CorrectlyRounded` tier rests on. Read it before trusting a threshold.

*The definitive statement of how diff-fuzzer decides whether two implementations
disagree — every threshold, where it comes from, and what would invalidate it.*

This is the document to read before trusting a finding, and the one to challenge if a
finding looks wrong. Everything here was derived from how floating-point arithmetic
works and then checked against measurement. **Nothing here was tuned until the output
looked clean.**

---

## 1. The problem, and the trap

Two implementations of the same arithmetic will not produce identical results. Floating
point rounds at every step, addition is not associative, and functions like `exp` are not
required to be exact. So "did they disagree?" cannot mean "are the bits different" — that
question flags correct code as broken, measurably 11.5% of the time on this project's own
operations.

It has to mean "did they differ by more than they legitimately could?" And that question
comes with a trap worth stating plainly:

> **There is an easy way to make this tool look like it works: raise the thresholds until
> the complaints stop.** Disagreements fall to zero, the output looks clean, and the
> fuzzer has quietly become a program that prints "no bugs found" regardless of input.
> Nothing observable distinguishes that from a healthy tool.

The opposite failure is real too — thresholds too tight flag every correct summation, you
stop trusting your own tool, and real signal drowns in noise.

So a threshold is only defensible if it comes from an **argument**, not from the data it
is judged against.

---

## 2. The decision procedure

Every threshold in this document was produced this way, in this order:

1. **Measure first.** Run the operations, record the *distribution* of error — median,
   99th percentile, worst case — per operation. (`examples/error_distribution.rs`)
2. **Derive from the arithmetic.** Work out what floating-point behaviour permits, from
   IEEE-754 guarantees, standard error bounds, and condition numbers.
3. **Check the derivation covers the measurement, with margin.** If it does not, either
   the derivation is wrong or something is happening that is not rounding — both worth
   stopping for.
4. **Never fit to the data.** A threshold set just above the largest observed error has
   no argument behind it, no margin for cases not yet generated, and would absorb a real
   bug that happened to be smaller than noise already seen.

Steps 3 and 4 are enforced by tests, not discipline:
`the_derived_bound_covers_the_measured_worst_case_for_sum`,
`..._for_matmul`, and `the_exp_bound_covers_the_measured_worst_case_at_large_arguments`
each assert the derived bound both **covers** the worst measured error **and stays within
a stated factor of it**. Too loose fails the build as surely as too tight.

---

## 3. The comparison rule

Two values agree when:

```
|a - b|  <=  atol  +  rtol * max(|a|, |b|)
             └floor┘   └── proportion ────┘
```

**Both components are required.** `rtol` alone becomes impossibly strict near zero, where
`1e-7` against `2e-7` is a 50% relative error and pure noise. `atol` alone becomes far too
permissive at scale, where a floor of `1e-8` on values near a million would let a backend
be wrong by 0.5 and pass. Each covers the other's blind spot.

**The `max` is deliberate and departs from `numpy.allclose`**, which scales by the second
argument alone and is therefore *asymmetric* — `close(a,b)` and `close(b,a)` can disagree.
That is defensible for numpy, which compares a result against a known-good reference; the
reference sets the meaningful scale. **This project has no reference.** Two backends are
compared, and which is passed first is an accident of list ordering. A verdict that
flipped when someone reordered a list would be indefensible in a bug report.

Error magnitudes are computed in `f64` even though the values are `f32`, because the
difference between two nearly-equal `f32` values loses precision if computed in `f32` —
a poor property for the number a bug report quotes.

---

## 4. Tolerance by operation class

Three classes, for three genuinely different reasons. Classes are named by *reason*
rather than by shape, because the reason is what a new operation must be classified by.

### 4.1 `CorrectlyRounded` — exact equality

**Operations:** `add`, `sub`, `mul`, `div`, `sqrt`, `neg`, `abs`
**Tolerance:** `rtol = 0`, `atol = 0`

**IEEE-754 *requires* `+`, `-`, `*`, `/` and `sqrt` to be correctly rounded** — the result
must be the representable number nearest the true answer, and there is exactly one such
number. Any two conforming implementations must therefore agree bit-for-bit. `neg` and
`abs` only touch the sign bit.

**Measured:** zero error across 14,000 cases, and still zero at 100× magnitude. The
standard being obeyed, not luck.

Holding these to exact equality is not strictness for its own sake — a difference here
would be a genuine standards violation, and any slack could only hide it.

### 4.2 `Approximated` — scaled by the condition number

**Operations:** `exp`
**Tolerance:** `rtol = 2 · (1 + |x|max) · ε`, `atol = 0`  (ε = `f32::EPSILON` = 1.19e-7)

`exp` is conspicuously **not** in IEEE-754's correctly-rounded set — doing so is
expensive, so each library picks its own approximation.

The bound comes from `exp`'s **condition number**, which is `|x|`: how much a relative
perturbation of the input is magnified in the output, since `exp(x+δ) = exp(x)·e^δ`.
Implementations reduce the argument before approximating (`x = k·ln2 + r`), and the error
in that reduction grows with `|x|`. One rounding step plus the condition-number term
gives `(1 + |x|)·ε`; **doubled** because two implementations may sit on opposite sides of
the true value.

**Measured:** worst error `1.192e-7` at `|x| ≤ 10` (exactly one ULP — the signature of
ordinary rounding), and `1.633e-4` at `|x| ≤ 1000` against a predicted `2.384e-4`.

**This replaced a fixed `2ε`, which failed instructively** — see §9.

### 4.3 `Accumulating` — derived per case

**Operations:** `sum`, `matmul`
**Tolerance, computed from each case's own shapes and values:**

```
rtol = 2 · n · ε
atol = 2 · n · ε · (n · largest_term)
```

where `n` is the number of terms summed — the reduced axis length for `sum`, the shared
inner dimension for `matmul` — and `largest_term` is the largest magnitude involved (for
`matmul`, the product of the two operands' largest magnitudes).

Floating-point addition is not associative, so a different summation order gives a
different answer and neither is wrong. The standard bound for summing `n` terms is
`|computed − exact| ≤ n · ε · Σ|xᵢ|`, and `Σ|xᵢ| ≤ n · largest_term`. Doubled, again, for
two implementations erring in opposite directions.

**Computed per case, not from a global worst case**, so a small input is held to a tighter
standard than a large one. A reduction over 2 elements gets roughly a sixteenth the
allowance of one over 8.

**Measured:** worst absolute error `7.63e-6` (`sum`) and `3.05e-5` (`matmul`), against
derived bounds an order of magnitude larger. **Validated at scale:** at 8× depth and 100×
magnitude these produced **zero** false positives, because the bound scales with the
arithmetic.

**Note the absolute term is doing real work here.** `sum`'s *relative* error reaches
`1.23e-3` while its absolute error stays at `7.63e-6` — that is cancellation, where terms
sum near zero and a tiny absolute error becomes an enormous relative one. The error did
not grow; the denominator shrank.

---

## 4a. GPU pairs — what is decided and what is blocked

Added at PHASE-7 step 7.4, when a third backend on genuinely different hardware first made
"a tolerance is a property of a *pair*" more than a theoretical point.

**Nothing in this policy currently varies by backend pair.** That is a decision, not an
oversight, and it rests on `SPECS.md` §6: *holding a bound needs only evidence it is
achievable; loosening one needs a specification permitting the difference.*

| operation | GPU behaviour (measured) | policy |
|---|---|---|
| `add` `sub` `mul` `neg` `abs` | **0 ULP, exact on 200/200** | **`EXACT` unchanged.** Exactness is demonstrably achievable on this hardware |
| `div` | 1 ULP, never exact | **`EXACT` unchanged — and therefore reported.** See below |
| `sqrt` | 2 ULP, never exact | same |
| subnormals | **flushed to zero** | not absorbed. A subnormal becoming zero is a *relative* error of 1.0; only an `atol` at that scale could absorb it, and that is a relaxation |
| `exp` | 4 ULP | already `Approximated`; the derived bound covers it |

**Updated 2026-08-04 — the specification was retrieved and the bounds are now derived.**
The Metal Shading Language Specification §8 (`SPECS.md` §4.1, quoted verbatim) permits, with
fast math enabled — *which it states is the default*:

| operation | permitted | derived bound | measured | margin |
|---|---|---|---|---|
| `add` `sub` `mul` `neg` `abs` | correctly rounded / `0 ulp` | **`EXACT`** | 0 ULP | exact |
| `div` | `<= 2.5 ulp` | `rtol = 2.5 ε` | 1 ULP | 2.5x |
| `sqrt` | `x * rsqrt(x)`, `rsqrt <= 2 ulp` | `rtol = 3 ε` | 2 ULP | 1.5x |
| subnormals | *"may be flushed to zero"* | **legal** | flushed | — |
| `exp` | `<= 3 + floor(2\|x\|) ulp` | existing condition-number bound covers it | 4 ULP (13 permitted) | 3.2x |

**Every measurement sits inside a bound derived from the specification, not fitted to it.**
Had one exceeded its bound, that would have been a finding about the GPU — which is the
entire point of deriving first.

**One consequence worth stating plainly.** The same specification says that with fast math
enabled, *"the behavior of handling NaN or INF (as inputs or outputs) is **undefined**"*. So
a GPU divergence whose signature is `undefined` is unspecified behaviour rather than a
defect — the GPU equivalent of the question burn#5284 asks about the CPU backends, except
here the specification answers it explicitly.

---

## 5. Undefined and infinite values

Four cases, decided explicitly rather than left to `==`:

| Case | Verdict | Reasoning |
|---|---|---|
| `NaN` vs `NaN` | **agree** | both produced no usable value, and the comparison cannot tell *why* — see below. Deliberately not `==`, which reports `NaN != NaN` |
| `NaN` vs number | **disagree** | they differ about whether an answer *exists* — more fundamental than differing about its value |
| same infinity | **agree** | both overflowed the same way |
| opposite infinities, or infinity vs finite | **disagree** | not a difference of degree; and `inf - inf` is `NaN`, so the arithmetic could not judge it anyway |

`NaN` against infinity classifies as "one undefined" — the `NaN` rule is checked first, on
purpose, since "not a number" is a stronger statement than "out of range".

**Correction, 2026-08-05: `NaN` does not mean "the operation was undefined".** This table
previously justified the first row as *"both asked something with no answer, both said so"*.
That reasoning is **wrong for the case this project cares most about**. `NaN` conflates two
different situations:

| | example | mathematically defined? |
|---|---|---|
| genuinely no answer | `sqrt(-1)`, `0/0` | **no** — `NaN` is correct |
| an answer exists and precision destroyed it | `(1e30 × 1e30) + (-1e30 × 1e30)` | **yes — the answer is exactly `0`** |

The second is [burn#5284](https://github.com/tracel-ai/burn/issues/5284). Nothing about it
is undefined; the `NaN` appears only because intermediates overflowed to `±inf` and
`inf + (-inf)` is `NaN`.

**The rule survives; its justification does not.** Two backends both returning `NaN` did
*behave the same way*, which is the only thing a differential comparison can establish — it
has no access to the exact answer and so cannot distinguish "no answer exists" from "an
answer was lost". The honest statement is: **agreement here records that the implementations
matched, not that the result was correct.**

Which is precisely why such a case is recorded as `Skipped` rather than `Agree` (§7). The
mechanism was right for a reason that was not written down.

**No tolerance, however enormous, absorbs a special *disagreement*; no tolerance, however
strict, breaks a special *agreement*.** Both directions are tested.

**Vacuous agreements are counted.** Two of these four cases agree *without any arithmetic
being compared*. A result that is entirely `NaN` on both sides passes while having told us
nothing — an exclusion wearing the costume of agreement. Such cases are reported as
`Skipped`, not `Agree` (§7).

---

## 5a. Broadcasting — a reasoned no-change

**Added PHASE-7C, 2026-08-06.** Elementwise operands may now differ in shape, combining by
stretching an axis of extent 1. **No tolerance changes, and this section exists so that a
future reader can tell "we considered this" from "nobody looked".**

### Why no bound moves

**A stretched axis is re-read, not re-computed.** When `[3,1]` combines with `[3,4]`, the
left operand's single column is *loaded four times*; the same stored `f32` enters four
additions. No arithmetic occurs that would not have occurred with an explicitly materialised
`[3,4]` operand holding four copies of that value, and copying a float introduces no rounding.

So for every operation class in §4, the per-case bound is computed from the **result** shape
and the operand values exactly as before, and broadcasting does not enter the derivation.

This holds whichever way a backend implements it:

- **Stride-0 iteration** (reading the same address repeatedly) performs the identical
  multiply-adds in the identical order.
- **Materialising a temporary** copies the value first, and a copy is exact.

Either way the arithmetic is unchanged, so the bound derived for it is unchanged.

### What broadcasting *can* change, and where that is handled

A backend can disagree about the **output shape** — one stretching an axis where another
does not. That is not a tolerance question at any width: it is a structural difference,
handled by §6 and never absorbed. Confirmed by test rather than assumed
(`a_disagreement_about_the_broadcast_output_shape_is_structural`), against the widest
tolerance in the codebase.

### The one thing to watch

An operand of extent 1 is stretched across the whole result, so **a single extreme value now
influences every output element** rather than one. That does not loosen any bound — the
per-element derivation is unchanged — but it does mean a case containing one `1e30` can
overflow the entire result rather than a corner of it. That is a change in *how often* the
existing overflow classes fire, not in what they permit, and the `broadcast_whole_operand`
feature exists so the search can say so if it turns out to matter.

---

## 6. Structural differences

Checked **before** any numeric comparison, and never absorbed by a tolerance however
loose:

1. **Shape** — two backends returning the same numbers in a different shape disagree
   about what the operation *means*, not about arithmetic.
2. **Element type** — recorded *before* values are converted to the common comparison
   precision. Converting first and comparing after would make an `f32` and an `f64`
   result compare equal, silently absorbing a genuine disagreement.

Shape is checked before element type, so a case differing in both reports the more
fundamental disagreement.

**Precision contract:** both backends are instantiated with `f32`; comparison happens at
`f32`; error arithmetic is `f64`.

---

## 7. When a case is not judged

A skip is a decision to have *no opinion*, and every one carries its reason. **Silent
exclusions are how a fuzzer hides real bugs from its own operator** — a skip that leaves
no trace is indistinguishable from a pass.

| Reason | Meaning |
|---|---|
| `TooFewResults` | fewer than two results survived; nothing to compare |
| `CouldNotRun` | an implementation declined the input — not evidence of being wrong |
| `NothingComparable` | every element was settled by a special-value rule; no arithmetic examined |
| `KnownLegal` | behaviour genuinely unspecified. Unused so far; GPU reductions are its first expected user |

A **partly** undefined result still agrees, because the defined part genuinely was
checked. Only a wholly vacuous comparison is skipped.

---

## 8. What is deliberately *not* filtered

**∇Fuzz's neighbour sampling for non-differentiable points.** That technique exists for a
*metamorphic gradient* oracle, where `(f(x+h) − f(x)) / h` is genuinely meaningless at a
kink, so a difference there is an artifact of the method. **A differential oracle
comparing forward values has no such problem** — both backends receive bit-identical
inputs, so there is no perturbation for a kink to amplify. Importing the machinery would
be cargo-culting a fix for a problem this oracle does not have. It becomes relevant when
the metamorphic oracle arrives.

Boundary cases are instead *tested directly* (`tests/boundaries.rs`), because a random
generator will never produce them — the chance of drawing exactly `0.0` from a continuous
range is nil.

---

## 9. Known blind spots

Stated because an invisible blind spot is worse than a known one.

**Signed zero.** `0.0 == -0.0` is true, so two backends disagreeing about the *sign* of a
zero are reported as agreeing. The values are numerically equal and nothing downstream
divides by them, so the policy is defensible — but the comparison cannot see it. A test
inspects the sign bits *outside* the comparison and asserts what was measured, so a future
divergence fails the build even though the oracle would not notice.

**`NaN` versus `inf` is judged a disagreement — and this was CONFIRMED UPSTREAM on 2026-08-04.** §5
classifies `NaN` against `inf` as `OneUndefined`, a real divergence. There is a reasonable
opposing reading: both mean "this computation left the representable range," so a human
reading a log would call them the same signal, and treating them as agreement would
collapse an entire class of findings (including `burn-001`) into silence.

The policy takes the other side because **the program consuming the value distinguishes
them**, even though a human skimming a log may not. Measured on both backends, from the
`burn-001` case (`examples/overflow_downstream.rs`):

| downstream operation | ndarray (`inf`) | tch (`NaN`) |
|---|---|---|
| `.clamp(0.0, 1.0)` | `1.0` | `NaN` |
| `relu` | `inf` | `NaN` |
| `.recip()` | `0.0` | `NaN` |
| `sigmoid` | `1.0` | `NaN` |
| `.greater_elem(0.0)` | `true` | `false` |

The `recip` and `sigmoid` rows are the sharp ones: on one backend the overflow is
**laundered into an ordinary finite value** with nothing left to indicate a problem, while
on the other it stays `NaN` and is visible. That is the opposite of equivalent — it is the
difference between a silent wrong answer and a loud one. `inf` is an ordered value with
arithmetic; `NaN` is the absence of one, and it is contagious.

**These numbers were measured, not reasoned.** An earlier draft asserted that `clamp` maps
`NaN` to `0.0` — it does not; Rust's `f32::max` swallows `NaN` but `f32::clamp` propagates
it, and the two had been conflated. The corrected measurement supports the conclusion more
strongly than the guess did, which is luck, not vindication.

The cost of this choice is accepted: symmetric-overflow cases are reported that some would
call legal. The cost of the *opposite* choice was judged worse — merging them would also
hide every future case where one backend `NaN`s and another `inf`s for reasons that are not
symmetric overflow, and it would hide them invisibly. That is the coarse-signature failure
mode: it looks exactly like success.

**Answered.** A burn maintainer, on [#5284](https://github.com/tracel-ai/burn/issues/5284):
*"I don't think inf / NaN should be interchangeable, it's a divergence that then propagates
non-uniformly through downstream ops."* Their stated reason is **the same one measured
here independently** — the `clamp`/`recip`/`sigmoid` table above. The same reply notes burn
has *no explicit numerical-agreement contract across backends*, which is why this policy had
to reason it out from first principles rather than cite one.

**So this is no longer a blind spot.** It stays in §9 as a record of how it was decided
before the answer arrived, and of the fact that a judgment call was made where a
specification would have been better.

**Both implementations wrong the same way.** The structural blind spot of *all*
differential testing. Two backends of one framework share code above the backend split, so
a bug there makes both wrong identically and they agree. This is what a metamorphic oracle
addresses, and why one is designed for.

**`exp` headroom at small arguments.** The derived bound is roughly 20× the worst error
observed at `|x| ≤ 10`, so sensitivity to a genuine `exp` bug there is reduced by about an
order of magnitude. Deliberate: the model bounds what is *permissible* for a function the
standard does not require to be correctly rounded, while measurement shows what these two
libraries *happen* to do today. Tightening to the latter would fit the threshold to an
implementation detail of the current versions.

**The lesson that produced this policy's structure.** The `exp` tolerance was originally a
fixed `2ε`, derived from data measured at `|x| ≤ 10`, where it held perfectly across
20,000 cases. Run at `|x| ≤ 1000` it produced **235 false positives**, because it did not
scale with the quantity that actually drives the error.

> **Fixed thresholds inherit the scope of the evidence they were derived from.**

The accumulating class never had this problem, because it was computed per case from the
start. The fix was not a bigger number — it was depending on the right variable.

---

## 10. Evidence

Measured on `burn-ndarray` vs `burn-tch` (libtorch 2.9.0), `f32`.

**Error distribution, 20,000 cases, default bounds** (`|x| ≤ 10`, dims ≤ 8):

| op | median rel. | p99 | max rel. | max abs. |
|---|---|---|---|---|
| `add` `sub` `mul` `div` `neg` `abs` `sqrt` | 0 | 0 | **0** | **0** |
| `exp` | 8.88e-8 | 1.18e-7 | 1.19e-7 | 1.95e-3 |
| `matmul` | 0 | 1.37e-6 | 3.87e-5 | 3.05e-5 |
| `sum` | 0 | 8.25e-6 | 1.23e-3 | 7.63e-6 |

**Campaign results under this policy:**

| Configuration | Cases | Diverged | Skipped |
|---|---|---|---|
| Default bounds | 1,000,000 | **0** | 0 |
| Wide bounds (dims ≤ 64, \|x\| ≤ 1000) | 10,000 | **0** | 5 |
| Special values, restricted domains | 200,000 | **0** | 8 |
| Special values, **domains unrestricted** | 200,000 | **0** | 807 |

The last row is the strongest: under deliberately adversarial input — zeros, subnormals,
overflow-scale values, `sqrt` of negatives, division by zero — **no false positives**, and
every unjudged case declined with a stated reason.

**A clean run only means something because the detector is independently proven to
work.** `testing.rs` keeps a backend that is wrong by a known amount, and the test suite
fails if the tool does not catch it. Without that, "no divergences found" would be
indistinguishable from a comparison that had quietly stopped working.

### Reproducing these numbers

```bash
cargo run --release -p tensor-adapter --example error_distribution        # §10 table
cargo run --release -p tensor-adapter --example error_distribution wide
cargo run --release -p tensor-adapter --example campaign 1000000          # default
cargo run --release -p tensor-adapter --example campaign 10000 wide
cargo run --release -p tensor-adapter --example campaign 200000 open      # unrestricted
cargo run --release -p tensor-adapter --example triage findings/campaign-wide.jsonl wide
```

---

## 11. What would invalidate this policy

The `exp` incident generalises: **a threshold is only valid over the conditions it was
derived and checked against.** Each of the following requires re-deriving and re-measuring
before findings can be trusted again.

| Change | What must be revisited |
|---|---|
| **Widening `Bounds`** (magnitude, dimensions) | Any threshold not computed per case. Re-run `error_distribution` at the new bounds. |
| **A new operation** | Classify it — correctly rounded, approximated, or accumulating? If none fit, it needs a new class and a new derivation, not the nearest existing one. |
| **A new element type** (`f64`, `bf16`) | Every ε in this document. `bf16` in particular has ~3 decimal digits, so all bounds shift by orders of magnitude. |
| **A GPU backend** | Fused multiply-add changes rounding; atomics make some reductions genuinely non-deterministic. Expect the first real use of `KnownLegal`. |
| **A cross-framework pair** | Normalisation assumptions, not just tolerances — different APIs mean different output conventions. |
| **A `burn` or libtorch upgrade** | Re-measure. The measured errors describe *these versions*; a kernel rewrite can move them. |

**The general rule:** if a threshold is a constant rather than a function of the case, ask
what evidence produced it and whether that evidence still applies.
