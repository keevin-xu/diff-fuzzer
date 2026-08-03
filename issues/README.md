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

| # | Subject | Status |
|---|---|---|
| 001 | `burn` matmul: backends disagree when intermediate products overflow | **FILED** 2026-08-03 — [burn#5284](https://github.com/tracel-ai/burn/issues/5284) |
