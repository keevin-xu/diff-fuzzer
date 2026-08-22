# diff-fuzzer

A **differential testing + fuzzing framework**, written in **Rust**.

### Reported upstream

Across two domains — `burn`'s tensor backends, and four ONNX runtimes compared on single-node
models — **six reports filed and four fixes merged**, two of them written by other people in
response to the report.

| report | project | subject | outcome |
|---|---|---|---|
| `burn-001` | tracel-ai/burn | `matmul` returns `inf` against `NaN` when products overflow | [#5284](https://github.com/tracel-ai/burn/issues/5284) filed |
| `burn-003` | tracel-ai/burn | `conv2d` padded positions become `NaN` with a non-finite weight | [#5385](https://github.com/tracel-ai/burn/issues/5385) **fixed**, by another contributor |
| `tract-001` | sonos/tract | `Sign(-0.0)` returns `-0.0` where ONNX specifies `0` | [#2670](https://github.com/sonos/tract/issues/2670) → [#2671](https://github.com/sonos/tract/pull/2671) **merged** |
| `tract-003` | sonos/tract | `DynamicQuantizeLinear` rounds ties away from zero | [#2672](https://github.com/sonos/tract/pull/2672) **merged** |
| `candle-001` | huggingface/candle | `Reshape` ignores `allowzero=1`, and infers `-1` against the wrong volume | [#3907](https://github.com/huggingface/candle/issues/3907) → [#3908](https://github.com/huggingface/candle/pull/3908) open |
| `onnxruntime-001` | microsoft/onnxruntime | `Where` returns `+0.0` for a selected `-0.0`, on both branches | [#32191](https://github.com/microsoft/onnxruntime/issues/32191) → [#32192](https://github.com/microsoft/onnxruntime/pull/32192) open |

**Two of the four fixes were written by other people.** `burn-003` was filed as a question,
labelled `bug` by a maintainer, and closed by someone else's pull request, *Fix non-finite padding
in Flex conv fast paths*. The report did the work of establishing that the divergence was real,
which mechanism produced it on each backend, and that a control with a finite weight agreed; the
fix followed from that without needing us.

**And a fix merged that this project did not write.** The `tract-001` pull request carried a
comment that the neighbouring quantized `Sign` path looked wrong at zero too — flagged
explicitly as *untested*, and kept out of the diff. A maintainer fixed exactly that in
[#2673](https://github.com/sonos/tract/pull/2673), with a quantized regression test. Asking
rather than bundling an unverified guess is what turned it into a separate, better-tested
change than either of us would have written alone.

## Background Information

- **Language:** Rust, throughout the whole project.
- **First software type:** deep-learning / tensor libraries.
- **First oracle:** differential (metamorphic added later — designed for from day one).
- **Implementations (Route A):** the `burn` framework across three backends — `burn-flex` (pure-Rust CPU), `burn-tch` (libtorch/CPU), and `burn-wgpu` (Metal GPU). Adding the third cost **4 lines of production code and no changes to the engine**. *Replacing* one CPU backend with another later touched 25 files — all of them the adapter, its examples, and the fuzz target, mostly mechanical renames — and **still not one line of the engine**, which is the claim the split exists to support.
- **Future (documented, not built yet):** metamorphic oracles (autodiff vs. numerical gradient), and a second software type (SQL engines) as a second adapter on the same core.