# Issue drafts

Upstream reports, written here for review **before** anything is posted. Filing is
Kevin's — reports go on his account under his name, so nothing leaves this directory
without his say-so.

## Templates

`templates/` holds each project's issue template, captured verbatim. **Match the
structure the maintainers expect** — a report that ignores their template reads as
someone who did not look.

Where a section does not apply (burn's asks for Screenshots and Smartphone details, which
mean nothing for a numerical library), write a short "Not applicable" rather than deleting
the heading or leaving the placeholder comment in. A deleted section looks like the
template was ignored; an unedited placeholder looks careless.

## Two tiers

| Directory | Contents | Audience |
|---|---|---|
| `./` | **Working drafts** — the text *plus* the triage checklist, what was deliberately omitted, what is inference versus verified, and what to do depending on the reply | us |
| `./final/` | **Copy-paste ready** — exactly what goes in the GitHub form, split into a title file and a body file. Nothing meta | them |

A draft reaches `final/` only once its checklist is complete and Kevin has reviewed it.

## Convention

| | |
|---|---|
| **Filename** | `<project>-<NNN>-<slug>.md`, numbered in the order drafted |
| **Format** | follow `templates/<project>-*.md` if the project has one |
| **Status** | stated at the top of each draft: `DRAFT`, `READY`, `FILED`, or `WITHDRAWN` |
| **Filed** | add the issue URL and the date to the draft; never delete it |

## Before anything is filed

From the triage ladder — each rung cleared before the next is worth asking:

1. **Does it reproduce** from the recorded case, on a clean run?
2. **Is it our tool's fault** — comparison, normalisation, a stray source of randomness?
3. **Is it floating-point noise**, within what the arithmetic predicts?
4. **Is it a legal difference** the specification permits?
5. **Has it been reported already?** Search their tracker; link related issues.

Only then draft. And when the answer to (4) is genuinely unclear, **ask rather than
assert** — a wrong assertion costs credibility, a well-posed question costs nothing.

## Drafts

### `tensor/`

| # | Subject | Status |
|---|---|---|
| 001 | `burn` matmul: backends disagree when intermediate products overflow | **FILED** 2026-08-03 — [burn#5284](https://github.com/tracel-ai/burn/issues/5284) |

### `onnx-runtime/`

| # | Subject | Status |
|---|---|---|
| 001 | `tract`: `Sign(0) = 1` for integer tensors | **DO NOT FILE** — fixed on `main` by [tract#2533](https://github.com/sonos/tract/pull/2533), merged three weeks after our pinned release |
| 002 | `Reshape` of a zero-size tensor: `tract` and candle reject what the reference and ONNX Runtime accept | **READY as two reports** — `final/tract-002-*` and `final/candle-001-*`. Different bugs: tract fails to analyse; candle appears to ignore `allowzero=1` |
| 003 | candle fails on rank-0 scalars | **DRAFT** — likely close without filing; candle's coverage is openly incomplete |
| 004 | ONNX Runtime: `Where` returns `+0.0` for a `-0.0` selected from `X` | **READY** — `final/onnxruntime-001-*` |
| 005 | `tract`: `Sign(-0.0)` returns `-0.0` instead of `0` | **READY** — `final/tract-001-*` |

**Every reproduction printed in `onnx-runtime/final/` is asserted by
`crates/onnx-adapter/tests/published_reproductions.rs`.** A report is a claim made to somebody
else, and it goes stale silently — bump a runtime and the file on disk still reads as true. The
test fails first, so a stale report is corrected or withdrawn before it is sent.
