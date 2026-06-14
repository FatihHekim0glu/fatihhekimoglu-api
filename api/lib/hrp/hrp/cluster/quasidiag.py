"""Quasi-diagonalization: recover the dendrogram leaf order.

Stage 2 of HRP. ``getQuasiDiag`` walks the SciPy linkage matrix and recovers the
order in which the original assets (leaves) appear in the dendrogram. Reordering
the covariance matrix by this leaf order places the largest covariances along the
diagonal (a "quasi-diagonal" matrix), which is what lets recursive bisection
allocate sensibly.

Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np

from hrp._exceptions import ValidationError

# quantcore-candidate: new code (HRP stage); parity oracle = PyPortfolioOpt /
# mlfinlab getQuasiDiag (dev-only), reconciling tie-breaking.


def get_quasi_diag(link: np.ndarray) -> list[int]:
    r"""Recover the dendrogram leaf order from a SciPy linkage matrix.

    Implements de Prado's ``getQuasiDiag``: starting from the final merge (the
    last row of ``link``), recursively replace each cluster id ``>= N`` with the
    pair of clusters it was formed from, until only original leaf indices
    (``0 <= i < N``) remain. The resulting left-to-right ordering is the
    dendrogram leaf order.

    The output is a permutation of ``range(N)`` (a valid bijection — asserted in
    the property suite), and reordering a covariance matrix by it yields a
    symmetric, quasi-diagonal matrix.

    TIE-BREAKING NOTE: the recovered order must match the parity oracles
    (PyPortfolioOpt and a second reference) bit-for-bit; tie-breaking in equal-
    distance merges is reconciled against those references rather than left
    implementation-defined.

    Parameters
    ----------
    link:
        An ``(N - 1) x 4`` SciPy linkage matrix (output of
        :func:`hrp.cluster.linkage.linkage_matrix`).

    Returns
    -------
    list[int]
        The leaf order: a length-``N`` permutation of ``0 .. N-1`` giving the
        positions of the original assets along the dendrogram.

    Raises
    ------
    ValidationError
        If ``link`` is not a valid ``(N - 1) x 4`` linkage matrix.
    """
    link = np.asarray(link)
    if link.ndim != 2 or link.shape[1] != 4:
        raise ValidationError(
            f"link must be an (N-1) x 4 linkage matrix, got shape {link.shape}."
        )

    n_merges = link.shape[0]
    if n_merges == 0:
        raise ValidationError("link must record at least one merge.")

    # Number of original leaves N = (rows in linkage) + 1.
    n_leaves = n_merges + 1
    # Integer view of the two merged-cluster id columns (columns 0 and 1).
    pairs = link[:, 0:2].astype(np.int64)

    # de Prado's getQuasiDiag. Seed the order with the two clusters joined at the
    # final merge, then repeatedly expand every id >= N (a merged cluster) into
    # the pair it was formed from, preserving left/right order. Cluster id
    # (N + k) corresponds to row k of `link`.
    order = pairs[-1].tolist()  # the two children of the last (root) merge

    while max(order) >= n_leaves:
        new_order: list[int] = []
        for item in order:
            if item < n_leaves:
                # Already an original leaf — keep it.
                new_order.append(int(item))
            else:
                row = item - n_leaves
                if row < 0 or row >= n_merges:
                    raise ValidationError(
                        f"link references cluster id {item} with no defining merge."
                    )
                left, right = pairs[row]
                new_order.append(int(left))
                new_order.append(int(right))
        order = new_order

    # The result must be a permutation of range(N) — a valid bijection.
    if sorted(order) != list(range(n_leaves)):
        raise ValidationError(
            "link does not yield a valid leaf-order permutation of range(N); "
            "it may be malformed or not a proper SciPy linkage matrix."
        )

    return order
