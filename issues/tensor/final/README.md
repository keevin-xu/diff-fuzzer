# Final — copy-paste ready

Exactly what goes into the GitHub issue form, and nothing else. No checklists, no status
headers, no notes to ourselves — everything in these files is intended to be read by a
maintainer.

## Layout

Two files per issue, matching the two fields GitHub asks for:

```
<project>-<NNN>-title.txt   -> the Title field    (select all, copy, paste)
<project>-<NNN>-body.md     -> the Body field     (select all, copy, paste)
```

Split rather than one file with a separator, so each is a clean select-all with nothing to
trim afterwards.

## Relationship to `../`

The parent directory holds the **working draft**: the same text plus the triage checklist,
what was deliberately left out, what is inference versus verified, and what to do
depending on the answer. That is for us. **This directory is for them.**

A file appearing here means the draft's checklist is complete and Kevin has reviewed it.
Anything still uncertain stays in the parent.

## Contents

| Files | Subject | Working draft |
|---|---|---|
| `burn-001-*` | `matmul` disagrees when intermediate products overflow `f32` | `../burn-001-matmul-overflow.md` — **FILED** as [#5284](https://github.com/tracel-ai/burn/issues/5284) |
| `burn-002-*` | `max`/`min` on cubecl backends return `±f32::MAX` instead of `±inf` | `../burn-002-reduce-infinity-sentinel.md` |
| `burn-003-*` | `conv2d` padded positions become `NaN` on `burn-tch` only, with a non-finite weight | `../burn-003-conv2d-padding-nonfinite.md` |

## Not here, and why

Every divergence class this project has recorded on the current backends, and what became of it.
Two of the five are **legal**, and saying so is the point of the table — a fileable finding is
one that survives this column.

| class | instances | disposition |
|---|---|---|
| `matmul` `inf` vs `NaN` | 6 | **filed** as #5284. Re-verified 2026-08-10: still diverges, but `flex` now agrees with libtorch and `wgpu` is the outlier — the roles reversed after the backend swap |
| `max`/`min` return `±f32::MAX` | 196 | **fileable** — `burn-002`. `-inf` is representable and is the correct answer; no convention permits a finite sentinel |
| `max`/`min` `NaN` ordering | 1,639 | **legal, not filed.** IEEE-754 defines *both* conventions — `maxNum` ignores `NaN`, `maximum` propagates it. CPUs take one, the GPU the other. Recorded as an observation |
| `cumprod` saturating intermediate | 46 | **legal, not filed.** A cumulative product has no specified association, and PyTorch documents that backends may differ between finite and non-finite results (`SPECS.md` §3.3). The oracle now declines these rather than reporting them |
| `conv2d` padding with a non-finite weight | 23 | **candidate** — `burn-003`. Drafted as a question: the specification is silent (`SPECS.md` §3.4), and it needs a non-finite weight to reach |

**The 837 findings under `findings/tensor/runs/archive/` are excluded entirely.** They predate
the `ndarray`→`flex` swap, and 810 of 814 no longer reproduce — a recorded finding that cannot be
produced is not a claim this repository can support.

## Reproduction is re-checked, not trusted

Every candidate above was re-run from scratch by `examples/verify_findings.rs` before drafting,
using **only `burn`'s public API**. Two reasons this is not ceremony:

- **Reproduction decays.** The backend swap invalidated 810 findings at a stroke. A report filed
  from a stale recording is worse than no report.
- **A repro that imports our crate is not a repro.** A maintainer will not install a fuzzer to
  confirm a bug, and should not have to.

`burn-003` also carries a **control** — the identical case with a finite weight, which must
agree — so the report can say what the finding is *not* about.

## After filing

Record the URL and date in the parent draft, and set its status to `FILED`. Leave these
files unchanged as a record of exactly what was sent.
