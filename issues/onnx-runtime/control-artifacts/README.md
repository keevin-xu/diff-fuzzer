# Control artifacts — 533 drafts of faults we injected on purpose

**Nothing in this folder is a finding. Do not read these looking for bugs, and never file one.**

## What they are

Every `DRAFT-*.md` here was written automatically by the **control** arm of the N9 campaign
(`--control`), in a single burst on **2026-08-13 at 16:31**.

A control run injects a deliberate fault — `WrongValues` — into every operator it runs, so that
the campaign can demonstrate its detector is capable of firing. That is its whole purpose: a
divergence rate means nothing without one. The N9 control reported **94% of substantive cases
diverging across 581 signatures**, against roughly 2.7% and 44 for the real run, and that contrast
is what makes the real number a measurement rather than an assertion.

The consequence nobody had thought through: those 581 signatures match no entry in `problems.rs`,
because no real problem produced them. The campaign treats an unmatched signature as *possibly
novel* and writes a draft so the evidence is not lost — correct behaviour for a real run, and
exactly wrong for a control, where an unexplained signature is **ours by construction**.

So `Abs`, `Add`, `Xor`, `Not`, `Floor`, `Ceil`, `Sqrt`, `MatMulInteger` and the rest all acquired
drafts describing arithmetic we broke ourselves.

## Why they were moved rather than deleted

They are the evidence that the writer misfired, and the count is the measure of it: the directory
holding this project's eight hand-written finding reports was **533 of 541 files noise**. Deleting
that would leave the fix looking tidier than the mistake was.

They are also regenerable — re-running the control reproduces them — so nothing here is unique.
Moved, not destroyed, at Kevin's decision.

## The two defects, and what changed

1. **The control wrote drafts at all.** Drafting is now suppressed under `--control`.
2. **A draft did not say which run produced it.** Every file here ends its provenance line as
   *"during run recorded in `findings/onnx/logs/`"* — naming no run. So a control artifact and a
   real divergence were **indistinguishable by reading the file**. The draft header now carries
   the run name and the oracle kind.

The second is the one worth remembering. It is the same failure as the campaign line reporting
`tract 53,402` and `onnxruntime 53,402` — a number that counted *participation* and read as
*attribution*. Both are cases of an artifact that is accurate about what it measured and silent
about what that means, which is the form in which a wrong reading survives review.

## Where the real reports are

`../FINDING-00N-*.md` — eight, hand-written, each carrying its triage ladder.
`../final/` — four, copy-paste ready for their upstream trackers.
