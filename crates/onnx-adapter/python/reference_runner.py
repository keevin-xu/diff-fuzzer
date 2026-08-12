"""Run ONNX models on `onnx.reference`, the specification's own executable definition.

This is the ground-truth participant. It is a separate process because the reference is
Python and the rest of the harness is Rust, and because a subprocess boundary is the
thinnest possible coupling — no PyO3, no shared address space.

# The wire format, and why it is not JSON

Values cross this boundary as **raw little-endian bytes**, never as text.

JSON cannot carry this domain's subject matter. It has no literal for `NaN` or `±inf`
(Python's `json` module emits non-standard ones that `serde_json` rejects), and it cannot
distinguish `+0.0` from `-0.0` after a round trip. Those three values are precisely what
the domain exists to test: both of this project's prior real findings were special-value
bugs, and signed zero is a documented blind spot carried over from the tensor domain.
A text encoding would silently destroy the evidence before the oracle ever saw it.

Raw bytes preserve the exact bit pattern, including NaN payloads.

## Framing

All integers are unsigned little-endian unless stated. The runner reads requests in a loop
until stdin closes, so one process can serve many cases — which is what makes the
"reference as a per-case participant" design affordable (see PENDING 1.2).

    request:
        u32          model length
        bytes        the serialized ModelProto
        u32          number of inputs
        per input:
            u32      name length
            bytes    name, UTF-8
            u32      rank
            i64*rank dimensions (signed)
            u32      payload length in bytes
            bytes    values, little-endian, dtype implied by the graph's declaration

    response:
        u8           0 = produced outputs, 1 = failed
        if 0:
            u32      number of outputs
            per output: same layout as an input
        if 1:
            u32      message length
            bytes    message, UTF-8

A failure is reported as a *value* on this channel, not as a non-zero exit code. An
implementation's error is a legitimate outcome, and the harness needs to compare it
against what the other runtimes said rather than treat it as the runner falling over.
"""

import struct
import sys
import traceback

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator

# numpy warns on overflow, underflow, division by zero, and invalid operations. In this
# domain those are **the subject matter, not errors**: a multiply overflowing to `inf` is
# exactly the shape of bug this project has already found once (burn#5284), and the
# reference producing `inf` is a legitimate answer to compare against.
#
# Left on, they also flood stderr — a 2,000-case throughput run emitted six distinct
# warnings repeatedly, and a real campaign would bury anything worth reading. Silencing
# them changes no value numpy computes; it only stops it commenting.
np.seterr(all="ignore")


def _read_exactly(stream, count):
    """Read exactly `count` bytes, or return None at a clean end of stream."""
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_u32(stream):
    raw = _read_exactly(stream, 4)
    return None if raw is None else struct.unpack("<I", raw)[0]


def _write_u32(stream, value):
    stream.write(struct.pack("<I", value))


def _write_tensor(stream, name, array):
    encoded = name.encode("utf-8")
    _write_u32(stream, len(encoded))
    stream.write(encoded)
    _write_u32(stream, array.ndim)
    for dim in array.shape:
        stream.write(struct.pack("<q", dim))
    # `tobytes` on a C-contiguous array preserves the exact in-memory bit pattern, which
    # is the whole point of this format.
    payload = np.ascontiguousarray(array).tobytes()
    _write_u32(stream, len(payload))
    stream.write(payload)


# ONNX TensorProto data type -> numpy dtype. Only the types the harness currently builds.
# Deliberately not exhaustive: a type that appears here but is never generated is a type
# nothing tests, and a type generated but missing here fails loudly rather than silently
# decoding as something else.
_DTYPES = {
    1: np.float32,   # FLOAT
    6: np.int32,     # INT32
    7: np.int64,     # INT64
    9: np.bool_,     # BOOL
    11: np.float64,  # DOUBLE
}


def _input_dtypes(model):
    """The numpy dtype each graph input declares, keyed by name."""
    dtypes = {}
    for value_info in model.graph.input:
        elem_type = value_info.type.tensor_type.elem_type
        if elem_type not in _DTYPES:
            raise ValueError(
                f"input {value_info.name!r} has element type {elem_type}, which this "
                f"runner does not decode. Add it to _DTYPES deliberately."
            )
        dtypes[value_info.name] = _DTYPES[elem_type]
    return dtypes


def _handle(model_bytes, raw_inputs):
    """Run one case. Returns a list of (name, ndarray)."""
    model = onnx.load_model_from_string(model_bytes)

    # The validity gate. `06-ORACLES-AND-LEGAL-DIFFERENCES.md` §2: a crash is only a
    # finding if the model is valid, and the reference accepting the model is the
    # practical definition of validity. Checking here means an invalid model is reported
    # as our error rather than becoming a divergence against some runtime.
    onnx.checker.check_model(model)

    dtypes = _input_dtypes(model)
    feeds = {}
    for name, dims, payload in raw_inputs:
        if name not in dtypes:
            raise ValueError(f"input {name!r} is not declared by the graph")
        array = np.frombuffer(payload, dtype=dtypes[name]).reshape(dims)
        feeds[name] = array

    evaluator = ReferenceEvaluator(model)
    outputs = evaluator.run(None, feeds)
    names = [o.name for o in model.graph.output]
    return list(zip(names, outputs))


def main():
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        model_length = _read_u32(stdin)
        if model_length is None:
            return  # clean end of stream: the harness closed the pipe
        model_bytes = _read_exactly(stdin, model_length)

        raw_inputs = []
        for _ in range(_read_u32(stdin)):
            name = _read_exactly(stdin, _read_u32(stdin)).decode("utf-8")
            rank = _read_u32(stdin)
            dims = [struct.unpack("<q", _read_exactly(stdin, 8))[0] for _ in range(rank)]
            payload = _read_exactly(stdin, _read_u32(stdin))
            raw_inputs.append((name, dims, payload))

        try:
            results = _handle(model_bytes, raw_inputs)
        except Exception:
            # Reported as a value, not as a crash of this process: the harness must be
            # able to compare "the reference rejected this" against what the runtimes
            # said. The full traceback goes across because the message is the most useful
            # part of a report.
            message = traceback.format_exc().encode("utf-8")
            stdout.write(b"\x01")
            _write_u32(stdout, len(message))
            stdout.write(message)
        else:
            stdout.write(b"\x00")
            _write_u32(stdout, len(results))
            for name, array in results:
                _write_tensor(stdout, name, array)

        # Without this the harness blocks forever waiting on a buffered reply.
        stdout.flush()


if __name__ == "__main__":
    main()
