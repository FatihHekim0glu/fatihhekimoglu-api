"""Interpretable variable-selection transformer (torch, [train] only) — lazy.

A simplified, interpretable transformer in the spirit of the Temporal Fusion
Transformer (Lim et al., 2021): a VARIABLE-SELECTION layer learns per-feature
gating weights (so the relative importance of each basket channel is readable),
followed by a small self-attention encoder over the look-back window and a
single next-step RETURN read-out for the target channel. RevIN instance-norm
(input-window-only) sits in front.

The variable-selection weights are exposed for interpretation, but the honest
headline is unchanged: on noisy daily returns this model does NOT reliably beat
naive. ``torch`` is imported LAZILY inside the builder/trainer (the ``[train]``
extra), so importing this module pulls in NO torch and has no side effects.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class TransformerVSConfig:
    """Immutable hyperparameters for the interpretable variable-selection transformer.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels gated by the variable-selection layer.
    d_model:
        Transformer embedding width.
    n_heads:
        Number of self-attention heads.
    n_layers:
        Number of transformer encoder blocks.
    dropout:
        Dropout probability.
    use_revin:
        Whether RevIN instance-norm (input-window-only) is applied first.
    """

    look_back: int = 60
    n_features: int = 2
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    dropout: float = 0.0
    use_revin: bool = True

    def __post_init__(self) -> None:
        """Validate sizes and the head/d_model divisibility.

        Raises
        ------
        ValidationError
            If any size is ``< 1``, ``d_model`` is not divisible by ``n_heads``,
            or ``dropout`` is outside ``[0, 1)``.
        """
        sizes = (self.look_back, self.n_features, self.d_model, self.n_heads, self.n_layers)
        if min(sizes) < 1:
            raise ValidationError(f"TransformerVSConfig: sizes must be >= 1, got {self!r}.")
        if self.d_model % self.n_heads != 0:
            raise ValidationError(
                f"TransformerVSConfig: d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(
                f"TransformerVSConfig: dropout must be in [0, 1), got {self.dropout}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_transformer_vs(config: TransformerVSConfig) -> Any:
    """Build (but do not train) the variable-selection transformer ``nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The module gates the
    ``(batch, look_back, n_features)`` input per-feature (the interpretable
    variable-selection weights), encodes the gated sequence with a small
    self-attention stack, and emits ``(batch, 1)``.

    The returned module exposes a ``variable_weights`` buffer (the softmaxed,
    learnable per-feature importances) so the relative channel contribution is
    readable after training.

    Parameters
    ----------
    config:
        The :class:`TransformerVSConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module`` exposing its variable-selection weights.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If ``d_model`` is not divisible by ``n_heads`` or a field is out of range.
    """
    import torch
    from torch import nn

    class _TransformerVS(nn.Module):
        """Per-feature variable selection + self-attention encoder + linear head."""

        def __init__(self, cfg: TransformerVSConfig) -> None:
            super().__init__()
            self.n_features = cfg.n_features
            # Learnable per-feature logits -> softmax gate (the interpretable
            # variable-selection weights). One scalar importance per input channel.
            self.var_logits = nn.Parameter(torch.zeros(cfg.n_features))
            # Project the gated (scalar-per-timestep) sequence into d_model tokens.
            self.value_embed = nn.Linear(1, cfg.d_model)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.look_back, cfg.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 2,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            self.head = nn.Linear(cfg.d_model, 1)

        def variable_weights(self) -> torch.Tensor:
            """Return the softmaxed per-feature importance weights (sum to 1)."""
            weights: torch.Tensor = torch.softmax(self.var_logits, dim=0)
            return weights

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, look_back, n_features). Gate features then collapse to a
            # scalar per timestep via the (normalized) variable-selection weights.
            gate = torch.softmax(self.var_logits, dim=0)
            gated = (x * gate).sum(dim=-1, keepdim=True)  # (batch, look_back, 1)
            tokens = self.value_embed(gated) + self.pos_embed
            encoded = self.encoder(tokens)
            # Read out the last position's encoded state.
            last = encoded[:, -1, :]
            forecast: torch.Tensor = self.head(last)
            return forecast

    return _TransformerVS(config)


def train_transformer_vs(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: TransformerVSConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the variable-selection transformer on the (pre-scaled) train windows.

    LAZY IMPORT: ``torch`` is imported inside this function (the ``[train]`` extra).
    Training is OFFLINE — never on the request path. The fitted module is exported
    to ONNX for the torch-free serve path.

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

    from mvtsforecast.models.lstm import _fit_module, _validate_train_tensors

    x_arr, y_arr = _validate_train_tensors(x_train, y_train, config)

    torch.manual_seed(int(seed))
    model = build_transformer_vs(config)
    _fit_module(model, x_arr, y_arr, epochs=epochs, lr=lr)
    return model
