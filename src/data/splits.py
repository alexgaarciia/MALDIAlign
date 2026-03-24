import pickle
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def subsample_dataset_stratified(data, labels, meta, n_samples, ood=False):
    """
    Perform stratified subsampling of a dataset based on class labels (e.g., species),
    preserving the original label distribution.

    This function is mainly intended for finetuning experiments where only a subset
    of the data is used, optionally simulating an out-of-distribution (OOD) setting.

    Parameters
    ----------
    data : np.ndarray
        Input features of shape (n_samples, n_features).

    labels : np.ndarray
        Array of labels used for stratification (e.g., species IDs).

    meta : pd.DataFrame
        Metadata associated with each sample. Must be index-aligned with `data`.

    n_samples : int
        Number of samples to include in the finetuning subset.

    ood : bool, default=False
        If True, the remaining samples (not selected for finetuning) are returned
        as a test set. If False, only the finetuning subset is returned.

    Returns
    -------
    dict
        Dictionary with the following structure:

        If ood=False:
            {
                "finetuning": {
                    "data": np.ndarray,
                    "label": np.ndarray,
                    "meta": pd.DataFrame,
                    "idx": np.ndarray
                }
            }

        If ood=True:
            {
                "finetuning": {...},
                "test": {...}
            }
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


def create_and_save_domain_splits(labels, meta, experiment_dir, val_size=0.1, test_size=0.2, seed=42):
    """
    Create stratified train/validation/test splits independently for each domain
    (e.g., hospital), and save them to disk.

    This function ensures that:
    - Each domain is split independently
    - Label distribution is preserved within each split (stratification)
    - A global split is also constructed by concatenating domain-specific splits

    Parameters
    ----------
    labels : np.ndarray
        Array of labels used for stratification (e.g., species IDs).

    meta : pd.DataFrame
        Metadata containing at least a "hospital" column defining domains.

    experiment_dir : pathlib.Path
        Directory where the split file will be saved.

    val_size : float, default=0.1
        Proportion of the total data to allocate to validation.

    test_size : float, default=0.2
        Proportion of the total data to allocate to test.

    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with:
        - splits_per_domain: dict of splits per domain
        - global: concatenated splits across all domains
    """
    splits = {}
    global_train, global_val, global_test = [], [], []

    for domain in meta["hospital"].unique():
        idx = np.where(meta["hospital"] == domain)[0]

        idx_temp, idx_test = train_test_split(
            idx,
            test_size=test_size,
            random_state=seed,
            stratify=labels[idx]
        )

        relative_val_size = val_size / (1.0 - test_size)
        
        idx_train, idx_val = train_test_split(
            idx_temp,
            test_size=relative_val_size,
            random_state=seed,
            stratify=labels[idx_temp]
        )

        print(f"\n[{domain}]")
        print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")
        print(f"Unique species in test from domain {domain}: {len(np.unique(labels[idx_test]))}")

        splits[domain] = {
            "train_idx": idx_train,
            "val_idx": idx_val,
            "test_idx": idx_test
        }

        global_train.append(idx_train)
        global_val.append(idx_val)
        global_test.append(idx_test)

    split_dict = {
        "splits_per_domain": splits,
        "global": {
            "train_idx": np.concatenate(global_train),
            "val_idx": np.concatenate(global_val),
            "test_idx": np.concatenate(global_test)
        }
    }

    save_path = experiment_dir / "data_splits.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(split_dict, f)

    print(f"\nSplits saved en: {save_path}\n")
    return split_dict
