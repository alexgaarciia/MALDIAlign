import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.config import load_config


def verify_data_path(data_dir):
    """
    Check whether a given path exists and print the result.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Path to check.

    Returns
    -------
    None
        This function does not return anything. It only prints
        whether the path exists or not.
    """

    if Path(data_dir).exists():
        print(f"Path exists: {data_dir}")
    else:
        print(f"Path does not exist: {data_dir}")
    

def load_pkl(pkl_file):
    """
    Load and deserialize a Python object from a pickle file.

    This function provides a safe and readable interface for loading
    `.pkl` files. It performs basic validation, including checking 
    for file existence and extension consistency, and raises informative
    errors if deserialization fails.

    Parameters
    ----------
    pkl_file : str or pathlib.Path
        Path to the pickle file to load.

    Returns
    -------
    object
        The Python object stored inside the pickle file.
    """

    pkl_path = Path(pkl_file)

    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    if pkl_path.suffix != ".pkl":
        print("The input file does not have a .pkl extension")
    
    try:
        with open(pkl_path, "rb") as pkl:
            return pickle.load(pkl)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling {pkl_path}: {e}")


def load_driams(driams_pkl, filter=None):
    """
    Load the DRIAMS dataset from a PKL file, optionally filtering by hospital.

    Parameters
    ----------
    driams_pkl : str or pathlib.Path
        Path to the PKL file containing the DRIAMS dataset.
    filter : list of str or None, optional
        List of hospital identifiers to keep (e.g. ["DRIAMS_A", "DRIAMS_D"]).
        If None, the full dataset is returned.

    Returns
    -------
    dict
        If filter is None:
            {
                "data": np.ndarray,
                "label": np.ndarray,
                "meta": pd.DataFrame
            }

        If filter is provided:
            Keys are hospital names and values are dictionaries with:
                - "data": np.ndarray
                - "label": np.ndarray
                - "meta": pd.DataFrame
    """

    driams = load_pkl(driams_pkl)
    data, label, meta = driams["data"], driams["label"], pd.DataFrame.from_records(list(driams["meta"]))

    if filter is None:
        return {
            "data": data,
            "label": label,
            "meta": meta 
        }
    
    result = {}
    for hosp in filter:
        mask = np.where(meta["hospital"].values == hosp)[0]
        subdata, sublabel, submeta = data[mask], label[mask], meta.iloc[mask]

        result[hosp] = {
            "data": subdata,
            "label": sublabel,
            "meta": submeta
        }
        
    return result


def load_marisma(marisma_pkl):
    """
    Load the MARISMA dataset from a PKL file.

    Parameters
    ----------
    marisma_pkl : str or pathlib.Path
        Path to the PKL file containing the MARISMA dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "MARISMA".
    """

    marisma = load_pkl(marisma_pkl)
    data, label, meta = marisma["data"], marisma["label"], pd.DataFrame.from_records(list(marisma["meta"]))

    meta.insert(
        loc=0,
        column="hospital",
        value="MARISMA"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }


def load_msumg(msumg_pkl):
    """
    Load the MS-UMG dataset from a PKL file.

    Only samples grown on standard agar are retained.

    Parameters
    ----------
    msumg_pkl : str or pathlib.Path
        Path to the PKL file containing the MS-UMG dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "MS-UMG".
    """

    msumg = load_pkl(msumg_pkl)
    data, label, meta = msumg["data"], msumg["label"], pd.DataFrame.from_records(list(msumg["meta"]))

    mask_agar = meta["agar_type"] == "agar"
    data = data[mask_agar.values]
    label = label[mask_agar.values]
    meta = meta.loc[mask_agar].reset_index(drop=True)

    meta.insert(
        loc=0,
        column="hospital",
        value="MS-UMG"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }


def load_rki(rki_pkl):
    """
    Load the RKI dataset from a PKL file.

    Parameters
    ----------
    rki_pkl : str or pathlib.Path
        Path to the PKL file containing the RKI dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "RKI".
    """

    rki = load_pkl(rki_pkl)
    data, label, meta = rki["data"], rki["label"], pd.DataFrame.from_records(list(rki["meta"]))

    meta.insert(
        loc=0,
        column="hospital",
        value="RKI"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }


def map_domains(meta):
    """
    Map each hospital to a unique integer domain ID.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame containing a column 'hospital'.

    Returns
    -------
    domain_ids : np.ndarray
        Array of integers representing the domain of each sample.
    """

    unique_hosp = np.unique(meta["hospital"])
    domain_map = {h: i for i, h in enumerate(unique_hosp)}
    domain_ids = meta["hospital"].map(domain_map).values

    return domain_ids


def construct_dataloaders(X_train_tensor, X_val_tensor, X_all_tensor, domain_train_tensor, domain_val_tensor, domain_all_tensor, batch_size, species_train_tensor=None, species_val_tensor=None, species_all_tensor=None):
    """
    Construct PyTorch DataLoaders for training, validation and full datasets.

    Optionally includes species labels for multi-task or conditional models.

    Parameters
    ----------
    X_train_tensor : torch.Tensor
        Training feature tensor (float32).
    X_val_tensor : torch.Tensor
        Validation feature tensor (float32).
    X_all_tensor : torch.Tensor
        Feature tensor containing all samples.
    domain_train_tensor : torch.Tensor
        Domain IDs for training samples (long).
    domain_val_tensor : torch.Tensor
        Domain IDs for validation samples (long).
    domain_all_tensor : torch.Tensor
        Domain IDs for all samples (long).
    batch_size : int
        Batch size for training and validation loaders.
    species_train_tensor : torch.Tensor or None, optional
        Species labels for training samples.
    species_val_tensor : torch.Tensor or None, optional
        Species labels for validation samples.
    species_all_tensor : torch.Tensor or None, optional
        Species labels for all samples.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    all_loader : torch.utils.data.DataLoader
        DataLoader for the full dataset.
    """

    

    if species_train_tensor is None:
        train_dataset = TensorDataset(X_train_tensor, domain_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor, domain_val_tensor)
        all_dataset   = TensorDataset(X_all_tensor, domain_all_tensor)
    else:
        train_dataset = TensorDataset(X_train_tensor, domain_train_tensor, species_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor, domain_val_tensor, species_val_tensor)
        all_dataset   = TensorDataset(X_all_tensor, domain_all_tensor, species_all_tensor)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    all_loader  = DataLoader(
        all_dataset, 
        batch_size=256, 
        shuffle=False)

    return train_loader, val_loader, all_loader


def prepare_data(domains, normalization="row_minmax", test_size=0.2, seed=42, batch_size=64, use_species_weight=False, classification=False, finetuning=False, splits_idx_path=None):
    """
    Load, preprocess and split MALDI-TOF datasets across multiple domains.

    This function supports two modes:

    1) Standard training mode (finetuning=False)
       - All samples from the selected domains are used.
       - A stratified train/validation split is performed by species.

    2) Finetuning base mode (finetuning=True)
       - Requires a precomputed splits_idx.pkl file.
       - Only samples labeled as "base" are used.
       - Train/validation split is performed within the base subset.

    Parameters
    ----------
    domains : list[str]
        List of domain identifiers (e.g. ["DRIAMS_A", "MARISMA"]).
    normalization : str
        Normalization strategy. Currently supports:
        - "row_minmax" : per-spectrum min-max scaling.
    test_size : float
        Fraction of data used for validation split.
    seed : int
        Random seed for reproducibility.
    batch_size : int
        Batch size for DataLoaders.
    use_species_weight : bool
        Whether to compute inverse-frequency species weights.
    classification : bool
        If True, returns NumPy arrays instead of DataLoaders.
    finetuning : bool
        Whether to use only "base" samples from a predefined split.
    splits_idx_path : str or Path, optional
        Path to splits_idx.pkl (required if finetuning=True).

    Returns
    -------
    dict
        Dictionary containing:
        - data_final : np.ndarray (original, non-normalized data)
        - label_final : np.ndarray (original labels)
        - meta_final : pd.DataFrame
        - train_loader : DataLoader
        - val_loader : DataLoader
        - all_loader : DataLoader
        - input_dim : int
        - species_weights : torch.Tensor or None
    """

    cfg = load_config()

    if finetuning:
        if splits_idx_path is None:
            raise ValueError("finetuning=True requires splits_idx_path")

        with open(splits_idx_path, "rb") as f:
            splits_idx = pickle.load(f)

    data_list, label_list, meta_list = [], [], []

    for d in domains:
        if d.startswith("DRIAMS_"):
            driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
            center = load_driams(driams_pkl, filter=[d])[d]
        elif d == "MARISMA":
            center = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
        elif d == "MS-UMG":
            center = load_msumg(cfg["data"]["MSUMG_PKL"])
        elif d == "RKI":
            center = load_rki(cfg["data"]["RKI_PKL"])
        else:
            raise ValueError(f"Unknown domain: {d}")

        if finetuning:
            idx_base = splits_idx[d]["base"]
            data_list.append(center["data"][idx_base])
            label_list.append(center["label"][idx_base])
            meta_list.append(center["meta"].iloc[idx_base])

        else:
            data_list.append(center["data"])
            label_list.append(center["label"])
            meta_list.append(center["meta"])

    data_final = np.vstack(data_list)
    label_final = np.concatenate(label_list)
    meta_final = pd.concat(meta_list, ignore_index=True)

    if "year" in meta_final.columns:
        meta_final["year"] = (
            meta_final["year"]
            .astype(str)
            .replace("nan", "Unknown")
        )
    else:
        meta_final["year"] = "Unknown"

    # Normalize data
    if normalization == "row_minmax":
        data_norm = (data_final - data_final.min(axis=1, keepdims=True)) / (
            data_final.max(axis=1, keepdims=True) - data_final.min(axis=1, keepdims=True) + 1e-8)
    else:
        data_norm = data_final

    # Encode species labels 
    unique_species, label_indices = np.unique(label_final, return_inverse=True)

    if classification:
        return {
            "X": data_norm,
            "y": label_indices,
            "meta": meta_final,
            "class_names": unique_species
        }

    # Construct dataloaders
    domain_ids = map_domains(meta_final)
    X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(
        data_norm,
        label_indices, 
        domain_ids,
        test_size=test_size,
        random_state=seed,
        stratify=label_indices
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_all_tensor = torch.tensor(data_norm, dtype=torch.float32)

    domain_train_tensor = torch.tensor(domain_train, dtype=torch.long)
    domain_val_tensor = torch.tensor(domain_val, dtype=torch.long)
    domain_all_tensor = torch.tensor(domain_ids, dtype=torch.long)

    species_train_tensor = torch.tensor(y_train, dtype=torch.long)
    species_val_tensor   = torch.tensor(y_val, dtype=torch.long)
    species_all_tensor   = torch.tensor(label_indices, dtype=torch.long)

    train_loader, val_loader, all_loader = construct_dataloaders(
        X_train_tensor, X_val_tensor, X_all_tensor,
        domain_train_tensor, domain_val_tensor, domain_all_tensor,
        batch_size=batch_size,
        species_train_tensor=species_train_tensor,
        species_val_tensor=species_val_tensor,
        species_all_tensor=species_all_tensor
    )

    species_weights = None
    if use_species_weight:
        counts = np.bincount(label_indices)
        inv_freq = 1.0 / counts
        normalized_weights = inv_freq / inv_freq.sum()
        species_weights = torch.tensor(normalized_weights, dtype=torch.float32)

    return {
        "data_final": data_final,
        "label_final": label_final,
        "meta_final": meta_final,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "all_loader": all_loader,
        "input_dim": data_final.shape[1],
        "species_weights": species_weights
    }


def subsample_dataset_stratified(data, labels, meta, n_samples):
    """
    Perform stratified subsampling of a dataset by species.

    This function selects `n_samples` instances while preserving
    the class distribution (species) using StratifiedShuffleSplit.

    The selected subset can be used as anchor samples for finetuning,
    while the remaining samples form the base set.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Species labels.
    meta : pd.DataFrame
        Metadata corresponding to the samples.
    n_samples : int
        Number of samples to select.

    Returns
    -------
    dict
        {
            "selected": {
                "data": np.ndarray,
                "label": np.ndarray,
                "meta": pd.DataFrame,
                "idx": np.ndarray
            },
            "rest": {
                "data": np.ndarray,
                "label": np.ndarray,
                "meta": pd.DataFrame,
                "idx": np.ndarray
            }
        }
    """

    n_total = len(data)
    n_samples = min(n_samples, n_total)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=n_samples,
        random_state=42
    )

    _, idx_sel = next(splitter.split(data, labels))
    idx_sel = np.sort(idx_sel)
    idx_rest = np.setdiff1d(np.arange(n_total), idx_sel)

    return {
        "selected": {
            "data": data[idx_sel],
            "label": labels[idx_sel],
            "meta": meta.iloc[idx_sel].reset_index(drop=True),
            "idx": idx_sel
        },
        "rest": {
            "data": data[idx_rest],
            "label": labels[idx_rest],
            "meta": meta.iloc[idx_rest].reset_index(drop=True),
            "idx": idx_rest
        }
    }
