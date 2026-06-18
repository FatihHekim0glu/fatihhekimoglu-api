"""Seeded, stratified, group-aware train/val/test split (the locked test set).

The split is the project's central leakage guard. It is:

* **grouped** by normalized-sentence hash, so near-duplicate sentences never
  straddle two folds (the real PhraseBank leakage risk);
* **stratified** by class, so each fold preserves the (neutral-heavy) prior;
* **seeded**, so the same ``(dataset, seed)`` always yields the same split and
  the test set is locked once and never re-shuffled.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from finbert_sentiment._constants import (
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    DEFAULT_VAL_FRACTION,
    N_CLASSES,
)
from finbert_sentiment._exceptions import InsufficientDataError, ValidationError
from finbert_sentiment._rng import make_rng

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Immutable train/val/test index partition over a deduplicated dataset.

    The three index tuples are disjoint and together cover every row exactly
    once. Indices reference positions in the deduplicated
    :class:`~finbert_sentiment.data.load.LabelledDataset`.

    Attributes
    ----------
    train:
        Row indices assigned to the training fold.
    val:
        Row indices assigned to the validation fold.
    test:
        Row indices assigned to the locked test fold.
    seed:
        The seed that produced this split (for provenance).
    """

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        """Validate that the three folds are disjoint and non-overlapping."""
        train_set = set(self.train)
        val_set = set(self.val)
        test_set = set(self.test)
        if len(train_set) != len(self.train):
            raise ValidationError("train fold contains duplicate indices.")
        if len(val_set) != len(self.val):
            raise ValidationError("val fold contains duplicate indices.")
        if len(test_set) != len(self.test):
            raise ValidationError("test fold contains duplicate indices.")
        if train_set & val_set:
            raise ValidationError("train and val folds overlap.")
        if train_set & test_set:
            raise ValidationError("train and test folds overlap.")
        if val_set & test_set:
            raise ValidationError("val and test folds overlap.")

    @property
    def sizes(self) -> dict[str, int]:
        """Return ``{"train": n, "val": n, "test": n}`` fold sizes."""
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this split."""
        return asdict(self)


def _stratum_of(group_label_counts: np.ndarray) -> int:
    """Return the stratum (dominant class index) for a group's label counts.

    Ties break toward the lowest class index for determinism. After dedup each
    group is a single sentence, so this is simply that sentence's label; the
    general multi-row form keeps the algorithm correct if dedup is skipped.
    """
    return int(np.argmax(group_label_counts))


def stratified_group_split(
    labels: Sequence[int],
    group_hashes: Sequence[str],
    *,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> SplitIndices:
    """Produce a seeded, class-stratified, group-disjoint train/val/test split.

    Whole groups (all rows sharing a normalized-sentence hash) are assigned to a
    single fold; the assignment is stratified so each fold approximately
    preserves the per-class prior, and seeded so the result is reproducible. The
    test set produced here is treated as LOCKED.

    Parameters
    ----------
    labels:
        Integer class indices, one per row.
    group_hashes:
        The normalized-sentence hash for each row (the group key), parallel to
        ``labels``.
    test_fraction:
        Target fraction of rows in the test fold.
    val_fraction:
        Target fraction of rows in the validation fold (of the whole dataset).
    seed:
        Master seed for the assignment.

    Returns
    -------
    SplitIndices
        The disjoint, group-aware, stratified partition.

    Raises
    ------
    ValidationError
        If ``labels`` and ``group_hashes`` are misaligned or the fractions are
        not in ``(0, 1)`` with ``test_fraction + val_fraction < 1``.
    InsufficientDataError
        If there are too few groups/examples to place every class in every fold.
    """
    label_arr = np.asarray(list(labels), dtype=np.int64)
    if label_arr.ndim != 1:
        raise ValidationError("labels must be 1-D.")
    n_rows = int(label_arr.shape[0])
    if n_rows == 0:
        raise ValidationError("labels must be non-empty.")
    if len(group_hashes) != n_rows:
        raise ValidationError(
            f"labels/group_hashes misaligned: {n_rows} labels vs {len(group_hashes)} group hashes."
        )
    if not 0.0 < test_fraction < 1.0:
        raise ValidationError(f"test_fraction must be in (0, 1), got {test_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValidationError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if test_fraction + val_fraction >= 1.0:
        raise ValidationError(
            f"test_fraction + val_fraction must be < 1, got {test_fraction + val_fraction}."
        )
    if not (label_arr.min() >= 0 and label_arr.max() < N_CLASSES):
        raise ValidationError(f"labels must lie in [0, {N_CLASSES}).")

    # Build groups: hash -> (member row indices, per-class counts).
    members: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.int64))
    for row, (digest, lab) in enumerate(zip(group_hashes, label_arr.tolist(), strict=True)):
        members[digest].append(row)
        counts[digest][lab] += 1

    group_keys = list(members.keys())
    n_groups = len(group_keys)
    # Each class needs at least one group in each of train/val/test (3 folds).
    min_groups_per_class = 3
    stratum_group_count = np.zeros(N_CLASSES, dtype=np.int64)
    for key in group_keys:
        stratum_group_count[_stratum_of(counts[key])] += 1
    for cls in range(N_CLASSES):
        if int(stratum_group_count[cls]) < min_groups_per_class:
            raise InsufficientDataError(
                f"class {cls} has only {int(stratum_group_count[cls])} group(s); "
                f"need >= {min_groups_per_class} to place it in train/val/test."
            )

    # Bucket group keys by stratum and shuffle each stratum with the seeded RNG.
    rng = make_rng(seed)
    by_stratum: dict[int, list[str]] = {cls: [] for cls in range(N_CLASSES)}
    for key in group_keys:
        by_stratum[_stratum_of(counts[key])].append(key)
    for cls in range(N_CLASSES):
        bucket = by_stratum[cls]
        # Sort for input-order independence, then seeded-permute for the split.
        bucket.sort()
        perm = rng.permutation(len(bucket))
        by_stratum[cls] = [bucket[i] for i in perm]

    # Greedy per-stratum quota assignment by ROW count so folds preserve the
    # per-class prior even when groups vary in size.
    assignment: dict[str, str] = {}
    for cls in range(N_CLASSES):
        bucket = by_stratum[cls]
        stratum_rows = sum(int(counts[k].sum()) for k in bucket)
        test_target = stratum_rows * test_fraction
        val_target = stratum_rows * val_fraction
        # Guarantee >=1 group per fold for this stratum (we verified >=3 groups).
        test_keys = [bucket[0]]
        val_keys = [bucket[1]]
        train_keys = [bucket[2]]
        test_rows = int(counts[bucket[0]].sum())
        val_rows = int(counts[bucket[1]].sum())
        for key in bucket[3:]:
            group_rows = int(counts[key].sum())
            # Fill test, then val, then train, each up to its row target.
            if test_rows < test_target:
                test_keys.append(key)
                test_rows += group_rows
            elif val_rows < val_target:
                val_keys.append(key)
                val_rows += group_rows
            else:
                train_keys.append(key)
        for key in test_keys:
            assignment[key] = "test"
        for key in val_keys:
            assignment[key] = "val"
        for key in train_keys:
            assignment[key] = "train"

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    fold_lists = {"train": train_idx, "val": val_idx, "test": test_idx}
    for key in group_keys:
        fold_lists[assignment[key]].extend(members[key])

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()

    split = SplitIndices(
        train=tuple(train_idx),
        val=tuple(val_idx),
        test=tuple(test_idx),
        seed=seed,
    )
    # Post-condition: every class present in every fold (stratification held).
    for fold_name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if len(idxs) == 0:  # pragma: no cover - guaranteed non-empty by the >=3-groups pre-check
            raise InsufficientDataError(f"{fold_name} fold is empty after the split.")
        fold_labels = label_arr[np.asarray(idxs, dtype=np.int64)]
        present = set(np.unique(fold_labels).tolist())
        missing = set(range(N_CLASSES)) - present
        if missing:  # pragma: no cover - guaranteed populated by the >=3-groups pre-check
            raise InsufficientDataError(
                f"{fold_name} fold is missing class(es) {sorted(missing)}; "
                f"too few groups to stratify."
            )
    _ = n_groups  # documented: n_groups available for callers/manifest provenance
    return split


def assert_no_group_overlap(split: SplitIndices, group_hashes: Sequence[str]) -> None:
    """Assert that no normalized-sentence hash appears in more than one fold.

    The executable form of the headline leakage guarantee. Raises if any group
    hash is shared across train/val/test (which would let a near-duplicate
    sentence leak from train into test).

    Parameters
    ----------
    split:
        The split to check.
    group_hashes:
        The per-row group hashes the split was built from.

    Raises
    ------
    ValidationError
        If any group hash straddles two or more folds.
    """
    hashes = list(group_hashes)
    n_rows = len(hashes)
    fold_of: dict[str, str] = {}
    for fold_name, idxs in (("train", split.train), ("val", split.val), ("test", split.test)):
        for idx in idxs:
            if not 0 <= idx < n_rows:
                raise ValidationError(
                    f"split index {idx} is out of range for {n_rows} group hashes."
                )
            digest = hashes[idx]
            prior = fold_of.get(digest)
            if prior is not None and prior != fold_name:
                raise ValidationError(
                    f"group hash {digest!r} straddles folds {prior!r} and "
                    f"{fold_name!r}: near-duplicate leakage."
                )
            fold_of[digest] = fold_name
