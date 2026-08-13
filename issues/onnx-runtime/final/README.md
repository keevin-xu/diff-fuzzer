# Final — copy-paste ready

Exactly what goes into the issue form, and nothing else. No status headers, no checklists, no
notes to ourselves. The working drafts in `../` hold all of that.

The shared convention is `issues/final` as described in `../../tensor/final/README.md`; this
directory follows it.

## Contents

| Files | Project | Subject | Working draft |
|---|---|---|---|
| `tract-001-*` | [sonos/tract](https://github.com/sonos/tract) | `Sign(-0.0)` returns `-0.0` instead of `0` | `../FINDING-005-tract-sign-negative-zero.md` |
| `onnxruntime-001-*` | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | `Where` loses the sign of `-0.0` selected from `X` | `../FINDING-004-onnxruntime-where-signed-zero.md` |
| `tract-002-*` | [sonos/tract](https://github.com/sonos/tract) | `Reshape` of a zero-size tensor fails to load | `../FINDING-002-reshape-empty-tensor.md` |
| `candle-001-*` | [huggingface/candle](https://github.com/huggingface/candle) | `Reshape` appears to ignore `allowzero=1` | `../FINDING-007-candle-reshape-allowzero.md` |

## Not here, and why

| Draft | Why it is not final |
|---|---|
| F-001 — `tract` `Sign(0) = 1` for integers | **Fixed upstream** by tract#2533, merged three weeks after our pinned release. Correct finding, wrong thing to send. |
| F-003 — candle at rank 0 | candle's operator coverage is openly incomplete; telling its maintainers so is not worth their time. |

## Form notes

**tract** has no issue template — a plain body.

**F-002 became two reports**, not one: `tract` and candle fail the same model for different reasons, and one issue describing both would ask each maintainer to read about the other's runtime.

**Both F-002 reports are written as questions**, not assertions — the specification does not address zero-size tensors directly, so the honest form is "the reference and ONNX Runtime both execute this; should it load?"

**ONNX Runtime** uses a YAML issue form (`08-general.yml`) with required dropdowns. The body file
is written to be pasted into *Describe the issue*; the remaining fields are short and are listed
at the top of the body file as a comment block to transcribe, since they are dropdowns rather than
free text.
