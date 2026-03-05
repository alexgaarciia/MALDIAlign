import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def subsample_dataset_stratified(data, labels, meta, n_samples, ood=False):
    """
    Stratified subsampling by species for finetuning experiments.

    If ood=False:
        Returns only the finetuning subset.

    If ood=True:
        Returns finetuning subset + remaining samples as test set.
    """

    n_total = len(data)
    n_samples = min(n_samples, n_total)

    # -------------------------------------------------
    # CASE 1: n_samples == 0
    # -------------------------------------------------
    if n_samples == 0:
        finetuning_dict = {
            "data": np.empty((0, data.shape[1])),
            "label": np.empty((0,), dtype=labels.dtype),
            "meta": meta.iloc[[]].reset_index(drop=True),
            "idx": np.array([], dtype=int)
        }

        if not ood:
            return {"finetuning": finetuning_dict}

        test_dict = {
            "data": data,
            "label": labels,
            "meta": meta.reset_index(drop=True),
            "idx": np.arange(n_total)
        }

        return {
            "finetuning": finetuning_dict,
            "test": test_dict
        }

    # -------------------------------------------------
    # CASE 2: n_samples == n_total
    # -------------------------------------------------
    if n_samples == n_total:
        finetuning_dict = {
            "data": data,
            "label": labels,
            "meta": meta.reset_index(drop=True),
            "idx": np.arange(n_total)
        }

        if not ood:
            return {"finetuning": finetuning_dict}

        test_dict = {
            "data": np.empty((0, data.shape[1])),
            "label": np.empty((0,), dtype=labels.dtype),
            "meta": meta.iloc[[]].reset_index(drop=True),
            "idx": np.array([], dtype=int)
        }

        return {
            "finetuning": finetuning_dict,
            "test": test_dict
        }

    # -------------------------------------------------
    # NORMAL CASE: 0 < n_samples < n_total
    # -------------------------------------------------
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_samples,
        random_state=42
    )

    idx_rest, idx_finetuning = next(splitter.split(data, labels))
    idx_rest, idx_finetuning = np.sort(idx_rest), np.sort(idx_finetuning)

    finetuning_dict = {
        "data": data[idx_finetuning],
        "label": labels[idx_finetuning],
        "meta": meta.iloc[idx_finetuning].reset_index(drop=True),
        "idx": idx_finetuning
    }

    if not ood:
        return {"finetuning": finetuning_dict}

    test_dict = {
        "data": data[idx_rest],
        "label": labels[idx_rest],
        "meta": meta.iloc[idx_rest].reset_index(drop=True),
        "idx": idx_rest
    }

    return {
        "finetuning": finetuning_dict,
        "test": test_dict
    }
