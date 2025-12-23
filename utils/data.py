import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.config import load_config


def verify_data_path(data_dir):
    """
    Check whether a given path exists.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
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
    Load the DRIAMS dataset from a PKL file. Optionally filter by a list of hospitals.

    Parameters
    ----------
    driams_pkl : str
        Path to the PKL file containing the DRIAMS dataset.
    filter : list[str] or None, optional
        If provided, only the specified hospitals are returned.
        Example: ["DRIAMS_A", "DRIAMS_D"].
        If None, the entire dataset is returned.

    Returns
    -------
    If filter is None:
        data : np.ndarray
        label : np.ndarray
        meta : pd.DataFrame

    If filter is a list:
        result : dict
            Keys are hospital names.
            Each value is a dict with:
                - "data": np.ndarray
                - "label": np.ndarray
                - "meta": pd.DataFrame
    """

    driams = load_pkl(driams_pkl)
    data, label, meta = driams["data"], driams["label"], pd.DataFrame.from_records(list(driams["meta"]))

    if filter is None:
        return data, label, meta
    
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


def scale_data(X_train, X_val, prescaler=None):
    """
    Fit or load a StandardScaler and transform train and validation sets.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_val : np.ndarray
        Validation features.
    prescaler_path : str or None
        Path to an existing scaler (.pkl). If provided, the scaler is loaded
        and ONLY transform() is applied (no fitting).

    Returns
    -------
    scaler : StandardScaler
        The fitted or loaded scaler.
    X_train_scaled : np.ndarray
        Scaled training data.
    X_val_scaled : np.ndarray
        Scaled validation data.
    """
    
    if prescaler is not None:
        with open(prescaler, "rb") as file:
            scaler = pickle.load(file)
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

    return scaler, X_train_scaled, X_val_scaled


def construct_dataloaders(X_train_tensor, X_val_tensor, X_all_tensor, domain_train_tensor, domain_val_tensor, domain_all_tensor, batch_size, species_train_tensor=None, species_val_tensor=None, species_all_tensor=None):
    """
    Build PyTorch DataLoaders for train and validation sets.

    Parameters
    ----------
    X_train_tensor : torch.Tensor
        Training data already converted to a float32 tensor.
    X_val_tensor : torch.Tensor
        Validation data already converted to a float32 tensor.
    domain_train_tensor : torch.Tensor
        Domain IDs for the training samples (long tensor).
    domain_val_tensor : torch.Tensor
        Domain IDs for the validation samples (long tensor).
    batch_size : int
        Batch size for both DataLoaders.

    Returns
    -------
    train_loader : DataLoader
        DataLoader containing (X_train_tensor, domain_train_tensor).
    val_loader : DataLoader
        DataLoader containing (X_val_tensor, domain_val_tensor).
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


def prepare_data(domains=["DRIAMS_A", "DRIAMS_D"], normalization="row_minmax", test_size=0.2, seed=42, batch_size=64, use_species_weight=False, classification=False):
    """
    Load, preprocess, and split the DRIAMS dataset for either deep-learning
    models or classical classifiers.

    This function performs the following steps:
    - Loads the DRIAMS dataset from a pickle file defined in the project config.
    - Optionally filters samples by a list of hospital domains.
    - Concatenates data, labels, and metadata across selected domains.
    - Applies row-wise normalization to each spectrum.
    - Splits the dataset into training and validation sets with stratification
      by species labels.

    Depending on the value of `classification`, the function returns data in
    different formats:

    * If `classification=False` (default):
        The function prepares PyTorch DataLoaders suitable for training and
        evaluating deep-learning models (e.g. VAEs, cVAEs).

    * If `classification=True`:
        The function returns NumPy arrays suitable for training classical
        machine-learning classifiers (e.g. Random Forests) using scikit-learn.

    Parameters
    ----------
    domains : list of str, optional
        List of hospital identifiers to include (e.g., ["DRIAMS_A", "DRIAMS_D"]).
        If provided, only samples from these domains are used.
    normalization : str, optional
        Normalization strategy applied to the spectra.
        Currently supported:
        - "row_minmax": min-max normalization applied independently to each sample.
    test_size : float, optional
        Fraction of the dataset used for validation.
    seed : int, optional
        Random seed used for reproducible train/validation splitting.
    batch_size : int, optional
        Batch size used for the training and validation DataLoaders (deep-learning
        mode only).
    use_species_weight : bool, optional
        Whether to compute inverse-frequency species weights (deep-learning mode
        only).
    classification : bool, optional
        If True, return NumPy arrays for classical classifiers.
        If False, return PyTorch DataLoaders for deep-learning models.

    Returns
    -------
    dict
        If classification=False:
            {
                "data_final": np.ndarray,
                "label_final": np.ndarray,
                "meta_final": pd.DataFrame,
                "train_loader": torch.utils.data.DataLoader,
                "val_loader": torch.utils.data.DataLoader,
                "all_loader": torch.utils.data.DataLoader,
                "input_dim": int,
                "species_weights": torch.Tensor or None
            }

        If classification=True:
            {
                "X": np.ndarray,
                    Normalized feature matrix for all samples.
                "y": np.ndarray,
                    Integer-encoded species labels.
                "meta": pd.DataFrame,
                    Metadata for all samples (e.g. hospital domain).
                "class_names": np.ndarray,
                    Array mapping label indices to species names.
            }
    """

    # Load DRIAMS pickle file
    cfg = load_config()
    driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL2"]

    # Filter driams
    driams = load_driams(driams_pkl, filter=domains)
    data, label, meta = [], [], []
    if domains:
        for d in domains:
            data.append(driams[d]["data"])
            label.append(driams[d]["label"])
            meta.append(driams[d]["meta"])
        data_final = np.vstack(data)
        label_final = np.concatenate(label)
        meta_final  = pd.concat(meta, ignore_index=True)
    else:
        data_final, label_final, meta_final = driams

    # Normalize data
    if normalization == "row_minmax":
        data_norm = (data_final - data_final.min(axis=1, keepdims=True)) / (
            data_final.max(axis=1, keepdims=True) - data_final.min(axis=1, keepdims=True) + 1e-8
        )
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
