import pickle
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


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
