"""RMT-signal eigenvector embedding of the correlation matrix.

Embeds assets into a low-dimensional Euclidean space spanned by the
*signal* eigenvectors of the correlation matrix — those whose eigenvalues exceed
the Marchenko-Pastur upper edge — excluding the market-mode eigenvector. K-means
(see :mod:`stockclusters.clustering.kmeans`) then runs on this embedding rather
than on the raw distances.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError
from stockclusters._typing import MatrixLike
from stockclusters._validation import ensure_dataframe

__all__ = ["rmt_signal_embedding"]


def rmt_signal_embedding(
    corr: MatrixLike,
    *,
    n_obs: int,
    drop_market_mode: bool = True,
    n_components: int | None = None,
) -> pd.DataFrame:
    r"""Embed assets using the RMT-signal eigenvectors of the correlation matrix.

    Eigendecomposes ``corr`` and keeps the eigenvectors whose eigenvalues exceed
    the Marchenko-Pastur upper edge :math:`(1 + \sqrt{q})^2` (the signal subspace),
    optionally dropping the top "market mode" eigenvector. Each asset is embedded
    at its (eigenvalue-scaled) loadings on the retained eigenvectors.

    Parameters
    ----------
    corr:
        An ``N x N`` correlation matrix.
    n_obs:
        The number of in-sample observations ``T`` (sets the MP edge via
        ``q = N / T``).
    drop_market_mode:
        If ``True``, exclude the largest-eigenvalue (market) eigenvector so the
        embedding captures sector/industry structure rather than the common mode.
    n_components:
        Optional explicit cap on the number of embedding dimensions; when ``None``
        all RMT-signal eigenvectors (post market-mode drop) are used.

    Returns
    -------
    pandas.DataFrame
        An ``N x d`` embedding, rows indexed by asset ticker, columns the retained
        signal components.

    Raises
    ------
    ValidationError
        If ``corr`` is not square or ``n_obs`` is non-positive.
    """
    frame = ensure_dataframe(corr, name="corr")
    n_rows, n_cols = frame.shape
    if n_rows != n_cols:
        raise ValidationError(f"corr must be square, got shape {frame.shape}.")
    if not frame.index.equals(frame.columns):
        frame.index = frame.columns
    if int(n_obs) <= 0:
        raise ValidationError(f"n_obs must be positive, got {n_obs}.")

    n = n_rows
    assets = list(frame.columns.astype(str))
    values = frame.to_numpy(dtype=np.float64)
    symmetric = 0.5 * (values + values.T)

    # Eigendecompose; eigh returns ascending eigenvalues. Reverse to descending so
    # index 0 is the market mode (largest eigenvalue).
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Marchenko-Pastur upper edge; eigenvalues above it carry signal. Convention:
    # q = N / T (matches stockclusters.estimators.rmt).
    q = float(n) / float(n_obs)
    lambda_plus = (1.0 + np.sqrt(q)) ** 2

    signal_mask = eigvals > lambda_plus
    signal_idx = np.flatnonzero(signal_mask)

    # Optionally drop the top "market mode" (the largest signal eigenvector) so the
    # embedding captures sector/industry structure rather than the common move.
    if drop_market_mode and signal_idx.size > 0:
        signal_idx = signal_idx[1:]

    # Fallback: if no signal eigenvectors survive (e.g. pure-noise input), keep a
    # single component (post-market-mode) so K-means still has a valid space.
    if signal_idx.size == 0:
        fallback = 1 if drop_market_mode else 0
        signal_idx = np.array([fallback], dtype=int)

    if n_components is not None:
        if int(n_components) <= 0:
            raise ValidationError(f"n_components must be positive, got {n_components}.")
        signal_idx = signal_idx[: int(n_components)]

    # Embed each asset at its eigenvalue-scaled loadings on the retained signal
    # eigenvectors: coordinate = sqrt(lambda) * eigenvector component.
    retained_vecs = eigvecs[:, signal_idx]
    retained_vals = np.clip(eigvals[signal_idx], 0.0, None)
    embedding = retained_vecs * np.sqrt(retained_vals)

    columns = [f"e{i}" for i in range(embedding.shape[1])]
    return pd.DataFrame(embedding, index=assets, columns=columns)
