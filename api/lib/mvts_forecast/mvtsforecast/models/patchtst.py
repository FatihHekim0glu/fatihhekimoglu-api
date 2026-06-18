"""PatchTST-style encoder (torch, [train] only) — patching + channel-independence.

A compact reimplementation of the PatchTST idea (Nie et al., 2023): split each
channel's look-back window into PATCHES, embed them, run a small CHANNEL-INDEPENDENT
transformer encoder (every channel shares the same weights and is processed
separately), and read out a single next-step RETURN forecast for the target
channel. RevIN instance-normalization (from the input window only) sits in front.

``torch`` is imported LAZILY inside the builder/trainer (the ``[train]`` extra), so
importing this module pulls in NO torch and has no side effects. The trained graph
is exported to ONNX and served torch-free.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mvtsforecast._exceptions import ValidationError
from mvtsforecast._typing import FloatArray, SequenceTensor


@dataclass(frozen=True, slots=True)
class PatchTSTConfig:
    """Immutable hyperparameters for the small PatchTST-style encoder.

    Attributes
    ----------
    look_back:
        Input window length (default 60).
    n_features:
        Number of input channels (processed channel-independently).
    patch_len:
        Length of each patch along the time axis.
    stride:
        Stride between consecutive patches.
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
    patch_len: int = 16
    stride: int = 8
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    use_revin: bool = True

    def __post_init__(self) -> None:
        """Validate sizes, patch geometry, and the head/d_model divisibility.

        Raises
        ------
        ValidationError
            If any size is ``< 1``, ``patch_len > look_back``,
            ``d_model`` is not divisible by ``n_heads``, or ``dropout`` is
            outside ``[0, 1)``.
        """
        sizes = (
            self.look_back,
            self.n_features,
            self.patch_len,
            self.stride,
            self.d_model,
            self.n_heads,
            self.n_layers,
        )
        if min(sizes) < 1:
            raise ValidationError(f"PatchTSTConfig: sizes must be >= 1, got {self!r}.")
        if self.patch_len > self.look_back:
            raise ValidationError(
                f"PatchTSTConfig: patch_len ({self.patch_len}) must be <= look_back "
                f"({self.look_back})."
            )
        if self.d_model % self.n_heads != 0:
            raise ValidationError(
                f"PatchTSTConfig: d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValidationError(f"PatchTSTConfig: dropout must be in [0, 1), got {self.dropout}.")

    @property
    def n_patches(self) -> int:
        """Number of patches a ``look_back`` window yields under this geometry."""
        return (self.look_back - self.patch_len) // self.stride + 1

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def build_patchtst(config: PatchTSTConfig) -> Any:
    """Build (but do not train) the PatchTST-style ``torch.nn.Module``.

    LAZY IMPORT: ``torch`` is imported inside this function. The module patches the
    ``(batch, look_back, n_features)`` input channel-independently, embeds and
    encodes the patches with a small transformer, and emits ``(batch, 1)``.

    Parameters
    ----------
    config:
        The :class:`PatchTSTConfig` hyperparameters.

    Returns
    -------
    Any
        An untrained ``torch.nn.Module``.

    Raises
    ------
    ImportError
        If the ``[train]`` extra (torch) is not installed.
    ValidationError
        If ``patch_len``/``stride`` are inconsistent with ``look_back`` or
        ``d_model`` is not divisible by ``n_heads``.
    """
    import torch
    from torch import nn

    class _PatchTST(nn.Module):
        """Patching + channel-independent transformer encoder + linear head."""

        def __init__(self, cfg: PatchTSTConfig) -> None:
            super().__init__()
            self.patch_len = cfg.patch_len
            self.stride = cfg.stride
            self.n_features = cfg.n_features
            self.n_patches = cfg.n_patches

            self.patch_embed = nn.Linear(cfg.patch_len, cfg.d_model)
            # Learned positional embedding over the patch axis.
            self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_model * 2,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
            # Flatten the per-channel encoded patches, then project all channels to 1.
            self.head = nn.Linear(self.n_patches * cfg.d_model * cfg.n_features, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, look_back, n_features) -> (batch, n_features, look_back).
            batch = x.shape[0]
            x = x.transpose(1, 2)
            # Channel-independent patching via unfold over the time axis:
            #   (batch, n_features, n_patches, patch_len).
            patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
            # Fold channels into the batch so every channel shares encoder weights.
            patches = patches.reshape(batch * self.n_features, self.n_patches, self.patch_len)
            tokens = self.patch_embed(patches) + self.pos_embed
            encoded = self.encoder(tokens)
            # Back to (batch, n_features * n_patches * d_model) for the joint head.
            flat = encoded.reshape(batch, -1)
            forecast: torch.Tensor = self.head(flat)
            return forecast

    return _PatchTST(config)


def train_patchtst(
    x_train: SequenceTensor,
    y_train: FloatArray,
    config: PatchTSTConfig,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 7,
) -> Any:
    """Train the PatchTST-style encoder on the (pre-scaled) train-fold windows.

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
    model = build_patchtst(config)
    _fit_module(model, x_arr, y_arr, epochs=epochs, lr=lr)
    return model
