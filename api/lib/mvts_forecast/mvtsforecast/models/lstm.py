"""Small multivariate LSTM (torch, [train] only) — lazy, never imported at load.

A compact LSTM encoder over the ``(look_back, n_features)`` window that emits a
single next-step RETURN forecast for the target channel. ``torch`` is imported
LAZILY inside the builder/trainer (the ``[train]`` extra), so importing this
module pulls in NO torch and has no side effects — the import-purity guard the
Stock-Price-Forecast footgun violated. The trained graph is exported to ONNX and
served, torch-free, via :mod:`mvtsforecast.models.onnx_runtime`.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class LstmConfig:
    """Immutable hyperparameters for the small multivariate LSTM.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels (basket size, minus the dropped target column).
    hidden_size:
        LSTM hidden width.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Inter-layer dropout probability.
    use_revin:
        Whether the encoder consumes RevIN-normalized windows.
    """

    look_back: int = 60
    n_features: int = 2
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    use_revin: bool = True

    def __post_init__(self) -> None:
        """Validate that sizes/rates are in range.

        Raises
        ------
        ValidationError
            If ``look_back``, ``n_features``, ``hidden_size``, or ``num_layers``
            is ``< 1``, or ``dropout`` is outside ``[0, 1)``.
        """
        if min(self.look_back, self.n_features, self.hidden_size, self.num_layers) < 1:
            raise ValidationError(f"LstmConfig: sizes must be >= 1, got {self!r}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(f"LstmConfig: dropout must be in [0, 1), got {self.dropout}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_lstm(config: LstmConfig) -> Any:
    """Build (but do not train) the small LSTM ``torch.nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The returned module
    maps a ``(batch, look_back, n_features)`` tensor to a ``(batch, 1)`` next-step
    return forecast. Returned as ``Any`` so the package never references a torch
    type at module scope.

    Parameters
    ----------
    config:
        The :class:`LstmConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If any config field is out of range.
    """
    import torch
    from torch import nn

    class _LstmForecaster(nn.Module):
        """LSTM encoder + linear head -> one next-step return per window."""

        def __init__(self, cfg: LstmConfig) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=cfg.n_features,
                hidden_size=cfg.hidden_size,
                num_layers=cfg.num_layers,
                batch_first=True,
                # torch forbids dropout with a single layer (it would be a no-op).
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(cfg.hidden_size, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, look_back, n_features). Read out the LAST hidden state.
            output, _ = self.lstm(x)
            last = output[:, -1, :]
            forecast: torch.Tensor = self.head(last)
            return forecast

    return _LstmForecaster(config)


def train_lstm(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: LstmConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the small LSTM on the (pre-scaled) train-fold windows.

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]`` extra).
    Training is OFFLINE — never invoked on the request path. The fitted module is
    later exported to ONNX for the torch-free serve path.

    Parameters
    ----------
    x_train:
        Train-fold sequence tensor ``(n_train, look_back, n_features)``.
    y_train:
        Train-fold next-step target returns ``(n_train,)``.
    config:
        The model hyperparameters.
    epochs, lr:
        Optimization budget and learning rate.
    seed:
        Seeds torch's RNG for a reproducible fit.

    Returns
    -------
    Any
        The trained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If torch is not installed.
    ValidationError
        If the tensors are mis-shaped or empty.
    """
    import torch

    x_arr, y_arr = _validate_train_tensors(x_train, y_train, config)

    torch.manual_seed(int(seed))
    model = build_lstm(config)
    _fit_module(model, x_arr, y_arr, epochs=epochs, lr=lr)
    return model


def _validate_train_tensors(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: Any,
) -> tuple[FloatArray, FloatArray]:
    """Coerce/validate the train tensors against ``config`` (shared by all models).

    ``config`` is any frozen model config exposing ``look_back`` / ``n_features``
    (``LstmConfig`` / ``PatchTSTConfig`` / ``TransformerVSConfig``).
    """
    import numpy as np

    x_arr = np.asarray(x_train, dtype=np.float64)
    y_arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
    if x_arr.ndim != 3:
        raise ValidationError(f"train: x_train must be a 3-D tensor, got ndim={x_arr.ndim}.")
    if x_arr.shape[0] == 0:
        raise ValidationError("train: x_train must be non-empty.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValidationError(
            f"train: x_train has {x_arr.shape[0]} samples but y_train has "
            f"{y_arr.shape[0]}; they must match."
        )
    if (x_arr.shape[1], x_arr.shape[2]) != (config.look_back, config.n_features):
        raise ValidationError(
            f"train: x_train trailing dims {(x_arr.shape[1], x_arr.shape[2])} do not "
            f"match config (look_back={config.look_back}, n_features={config.n_features})."
        )
    return x_arr, y_arr


def _fit_module(
    model: Any,
    x_arr: FloatArray,
    y_arr: FloatArray,
    *,
    epochs: int,
    lr: float,
) -> None:
    """Full-batch MSE fit (shared by all three deep models; offline-only)."""
    import torch
    from torch import nn

    if epochs < 1:
        raise ValidationError(f"train: epochs must be >= 1, got {epochs}.")
    if lr <= 0.0:
        raise ValidationError(f"train: lr must be > 0, got {lr}.")

    x_t = torch.as_tensor(x_arr, dtype=torch.float32)
    y_t = torch.as_tensor(y_arr, dtype=torch.float32).reshape(-1, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        prediction = model(x_t)
        loss = loss_fn(prediction, y_t)
        loss.backward()
        optimizer.step()
    model.eval()
