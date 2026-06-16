"""Marchenko-Pastur RMT denoising for the clustering pipeline.

Thin re-export of the reused, tested implementation in
:mod:`stockclusters.estimators.rmt`. RMT denoising runs BEFORE clustering so that
eigenvalues below the Marchenko-Pastur upper edge :math:`(1 + \\sqrt{q})^2` (pure
noise) do not distort the correlation structure the dendrogram is built on. The
denoise-on/off ablation is reported honestly in the README validation table.

Importing this module has no side effects.
"""

from __future__ import annotations

from stockclusters.estimators.rmt import marchenko_pastur_clip

__all__ = ["marchenko_pastur_clip"]
