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


def construct_dataloaders(X_train_tensor, X_val_tensor, X_all_tensor, domain_train_tensor, domain_val_tensor, domain_all_tensor, batch_size):
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
    
    train_dataset = TensorDataset(X_train_tensor, domain_train_tensor)
    val_dataset   = TensorDataset(X_val_tensor, domain_val_tensor)
    all_dataset = TensorDataset(X_all_tensor, domain_all_tensor)

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

def prepare_data(domains=["DRIAMS_A", "DRIAMS_D"], normalization="row_minmax", test_size=0.2, seed=42, batch_size=64):
    """
    Load, preprocess, and split the DRIAMS dataset, returning PyTorch DataLoaders
    and metadata required for model training and evaluation.

    This function:
    - Loads the DRIAMS dataset from a pickle file defined in the project config.
    - Optionally filters samples by a list of hospital domains.
    - Concatenates data, labels, and metadata across selected domains.
    - Applies row-wise normalization to each spectrum.
    - Splits the dataset into training and validation sets with stratification
      by class labels.
    - Constructs PyTorch DataLoaders for training, validation, and full-dataset
      evaluation.

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
        Batch size used for the training and validation DataLoaders.

    Returns
    -------
    dict
        Dictionary containing:
        - "data_final" : np.ndarray
            Concatenated (unnormalized) spectral data.
        - "label_final" : np.ndarray
            Class labels corresponding to each sample.
        - "meta_final" : pd.DataFrame
            Metadata associated with each sample (e.g., hospital domain).
        - "train_loader" : torch.utils.data.DataLoader
            DataLoader for the training split.
        - "val_loader" : torch.utils.data.DataLoader
            DataLoader for the validation split.
        - "all_loader" : torch.utils.data.DataLoader
            DataLoader over the full dataset (no shuffling), typically used for
            evaluation or visualization.
        - "input_dim" : int
            Dimensionality of the input spectra (number of features per sample).
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
        data_final = driams[0]
        label_final = driams[1]
        meta_final = driams[2]

    # Normalize data
    if normalization == "row_minmax":
        data_norm = (data_final - data_final.min(axis=1, keepdims=True)) / (data_final.max(axis=1, keepdims=True) - data_final.min(axis=1, keepdims=True) + 1e-8)

    # Construct dataloaders
    domain_ids = map_domains(meta_final)
    X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(
        data_norm, 
        label_final, 
        domain_ids, 
        test_size=test_size, 
        random_state=seed, 
        stratify=label_final)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_all_tensor = torch.tensor(data_norm, dtype=torch.float32)

    domain_train_tensor = torch.tensor(domain_train, dtype=torch.long)
    domain_val_tensor = torch.tensor(domain_val, dtype=torch.long)
    domain_all_tensor = torch.tensor(domain_ids, dtype=torch.long)

    train_loader, val_loader, all_loader = construct_dataloaders(X_train_tensor, 
                                                                 X_val_tensor, 
                                                                 X_all_tensor, 
                                                                 domain_train_tensor, 
                                                                 domain_val_tensor, 
                                                                 domain_all_tensor, 
                                                                 batch_size=batch_size)

    return {
        "data_final": data_final,
        "label_final": label_final,
        "meta_final": meta_final,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "all_loader": all_loader,
        "input_dim": data_final.shape[1],
    }
