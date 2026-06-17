"""TensorFlow-free ONNX artifact builder (the reproducible-here serve graph).

The canonical retrain path trains a Keras LSTM and exports it via ``tf2onnx``
(:func:`lstmforecast.models.lstm.export_onnx`). That path needs the heavy
``[train]`` extra (TensorFlow), which the serve container never ships.

This module builds the SAME ``(N, look_back, n_features) -> (N, 1)`` next-day
return graph DIRECTLY with the ``onnx`` builder (an ``onnxruntime``-adjacent, much
lighter dependency), so the committed artifact is reproducible WITHOUT TensorFlow
and the honest-NULL story stays intact: the graph is a small, seeded LSTM whose
recurrent weights are initialized but untrained-on-signal, so on random-walk data
it cannot beat persistence — exactly the documented result. ``onnx`` is imported
LAZILY inside the function so importing this module (and ``lstmforecast``) never
pulls it in.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lstmforecast._rng import make_rng

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path

    from lstmforecast.models.lstm import LstmConfig

#: ONNX standard input name (matches the tf2onnx export signature).
INPUT_NAME = "sequence"
#: ONNX standard output name (matches the Keras ``return_hat`` head).
OUTPUT_NAME = "return_hat"


def build_native_lstm_onnx(
    config: LstmConfig,
    path: str | Path,
    *,
    seed: int = 7,
    weight_scale: float = 0.1,
) -> str:
    """Build a small, seeded LSTM-shaped ONNX graph WITHOUT TensorFlow.

    Constructs a single-layer LSTM (``hidden_size = config.units``) followed by a
    linear head mapping the last hidden state to one next-day return, using the
    ``onnx`` builder API directly. Weights are drawn from a seeded RNG and scaled
    small, so the graph is a valid, runnable LSTM forecaster whose output on a
    random walk carries no predictive signal — the honest NULL holds.

    The exported input signature ``(None, look_back, n_features)`` and the
    ``sequence`` / ``return_hat`` tensor names match the canonical tf2onnx export,
    so :class:`lstmforecast.models.onnx_runtime.OnnxForecaster` serves either
    artifact identically.

    LAZY IMPORT: ``numpy`` and ``onnx`` are imported inside this function; the
    serve container never needs ``onnx`` (only ``onnxruntime``).

    Parameters
    ----------
    config:
        The architecture/size bundle (``units``, ``look_back``, ``n_features``).
    path:
        Destination ``.onnx`` filepath.
    seed:
        Seed for the weight RNG (reproducible artifact).
    weight_scale:
        Standard deviation of the initialized weights (kept small).

    Returns
    -------
    str
        The path the ONNX artifact was written to.
    """
    import os

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    units = int(config.units)
    look_back = int(config.look_back)
    n_features = int(config.n_features)

    gen = make_rng(seed)

    # ONNX LSTM weight layout (opset 17): gates ordered i, o, f, c.
    #   W: (num_directions=1, 4*hidden, input_size)
    #   R: (num_directions=1, 4*hidden, hidden_size)
    #   B: (num_directions=1, 8*hidden)  [Wb | Rb]
    w = (gen.standard_normal((1, 4 * units, n_features)) * weight_scale).astype("float32")
    r = (gen.standard_normal((1, 4 * units, units)) * weight_scale).astype("float32")
    b = np.zeros((1, 8 * units), dtype="float32")
    # Linear head: last hidden state (units) -> single next-day return.
    w_dense = (gen.standard_normal((units, 1)) * weight_scale).astype("float32")
    b_dense = np.zeros((1,), dtype="float32")

    inp = helper.make_tensor_value_info(
        INPUT_NAME, TensorProto.FLOAT, [None, look_back, n_features]
    )
    out = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [None, 1])

    initializers = [
        numpy_helper.from_array(w, "lstm_W"),
        numpy_helper.from_array(r, "lstm_R"),
        numpy_helper.from_array(b, "lstm_B"),
        numpy_helper.from_array(w_dense, "dense_W"),
        numpy_helper.from_array(b_dense, "dense_b"),
        numpy_helper.from_array(np.array([0], dtype="int64"), "axis0"),
    ]
    nodes = [
        # (batch, seq, feat) -> (seq, batch, feat): ONNX LSTM wants time-major.
        helper.make_node("Transpose", [INPUT_NAME], ["x_time_major"], perm=[1, 0, 2]),
        helper.make_node(
            "LSTM",
            ["x_time_major", "lstm_W", "lstm_R", "lstm_B"],
            ["lstm_Y", "lstm_Yh", "lstm_Yc"],
            hidden_size=units,
        ),
        # Y_h: (num_directions=1, batch, hidden) -> drop the direction axis.
        helper.make_node("Squeeze", ["lstm_Yh", "axis0"], ["h_last"]),
        helper.make_node("MatMul", ["h_last", "dense_W"], ["dense_proj"]),
        helper.make_node("Add", ["dense_proj", "dense_b"], [OUTPUT_NAME]),
    ]
    graph = helper.make_graph(
        nodes,
        f"lstm_{config.architecture}_native",
        [inp],
        [out],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)

    out_path = os.fspath(path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    onnx.save(model, out_path)
    return out_path
