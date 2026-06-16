"""Cross-window cluster-label alignment.

Cluster integer labels are arbitrary, so comparing labelings across windows
requires matching cluster ids by membership overlap. This module aligns labels via
the Hungarian algorithm on a max-Jaccard cost matrix, and records the birth/death
of clusters when ``k`` changes between windows.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockclusters._exceptions import ValidationError

__all__ = ["align_labels", "births_and_deaths"]


def _jaccard_assignment(
    reference: pd.Series,
    target: pd.Series,
) -> tuple[list[int], list[int], dict[int, int]]:
    """Solve the max-Jaccard assignment of target clusters onto reference clusters.

    Returns the (sorted) reference cluster ids, target cluster ids, and the mapping
    ``{target_id: reference_id}`` produced by the Hungarian algorithm maximizing
    total Jaccard overlap on the shared assets.
    """
    from scipy.optimize import linear_sum_assignment

    common = reference.index.intersection(target.index)
    if len(common) < 1:
        raise ValidationError(
            "label alignment requires at least one common asset between the two windows."
        )
    common = common.sort_values()
    ref = reference.reindex(common).astype(int)
    tgt = target.reindex(common).astype(int)

    ref_ids = sorted(int(x) for x in pd.unique(ref))
    tgt_ids = sorted(int(x) for x in pd.unique(tgt))

    # Build a (target x reference) Jaccard-overlap matrix on the shared assets.
    ref_sets = {r: set(common[ref.to_numpy() == r]) for r in ref_ids}
    tgt_sets = {t: set(common[tgt.to_numpy() == t]) for t in tgt_ids}

    jaccard = np.zeros((len(tgt_ids), len(ref_ids)), dtype="float64")
    for i, t in enumerate(tgt_ids):
        ts = tgt_sets[t]
        for j, r in enumerate(ref_ids):
            rs = ref_sets[r]
            union = ts | rs
            jaccard[i, j] = len(ts & rs) / len(union) if union else 0.0

    # Hungarian maximizes overlap -> minimize the negated cost.
    row_idx, col_idx = linear_sum_assignment(-jaccard)
    mapping: dict[int, int] = {}
    for i, j in zip(row_idx, col_idx, strict=True):
        # Only honour assignments with strictly positive overlap; a zero-overlap
        # pairing is spurious and should be treated as a fresh (born) cluster.
        if jaccard[i, j] > 0.0:
            mapping[tgt_ids[i]] = ref_ids[j]
    return ref_ids, tgt_ids, mapping


def align_labels(
    reference: pd.Series,
    target: pd.Series,
) -> pd.Series:
    r"""Relabel ``target`` clusters to best match ``reference`` (max-Jaccard).

    Builds a cluster-vs-cluster Jaccard-overlap matrix on the shared assets and
    solves the assignment (Hungarian) that maximizes total overlap, then relabels
    ``target`` accordingly. Unmatched target clusters (when ``k`` grew) receive
    fresh ids beyond the reference range.

    Parameters
    ----------
    reference:
        The reference labeling (e.g. the earlier window), indexed by asset.
    target:
        The labeling to align to ``reference``, indexed by asset.

    Returns
    -------
    pandas.Series
        ``target`` relabeled to align with ``reference``, indexed by asset.

    Raises
    ------
    ValidationError
        If the two labelings share fewer than one common asset.
    """
    ref_ids, _tgt_ids, mapping = _jaccard_assignment(reference, target)

    # Fresh ids for unmatched (born) target clusters start beyond the reference
    # range so they cannot collide with any aligned id.
    next_id = (max(ref_ids) + 1) if ref_ids else 0
    for raw in sorted(set(target.astype(int).to_numpy())):
        t = int(raw)
        if t not in mapping:
            mapping[t] = next_id
            next_id += 1

    aligned = target.astype(int).map(mapping)
    return pd.Series(aligned.to_numpy(), index=target.index, name=target.name)


def births_and_deaths(
    reference: pd.Series,
    target: pd.Series,
) -> dict[str, list[int]]:
    r"""Identify clusters that appear or disappear between two windows.

    After alignment, returns the cluster ids present in ``target`` but not matched
    in ``reference`` (births) and those present in ``reference`` but absent in
    ``target`` (deaths).

    Parameters
    ----------
    reference:
        The earlier-window labeling, indexed by asset.
    target:
        The later-window labeling, indexed by asset.

    Returns
    -------
    dict
        ``{"births": [...], "deaths": [...]}`` with cluster ids.

    Raises
    ------
    ValidationError
        If the two labelings share no common assets.
    """
    ref_ids, tgt_ids, mapping = _jaccard_assignment(reference, target)

    # Reference clusters that no target cluster mapped onto have "died".
    matched_ref = set(mapping.values())
    deaths = sorted(r for r in ref_ids if r not in matched_ref)
    # Target clusters with no reference match are "births".
    births = sorted(t for t in tgt_ids if t not in mapping)
    return {"births": births, "deaths": deaths}
