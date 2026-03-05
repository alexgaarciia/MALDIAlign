import numpy as np
import pandas as pd

from src.config.loader import load_config
from src.data.io import load_pkl
from src.data.datasets import load_driams, load_marisma, load_msumg, load_rki



def prepare_data(domains, pkl_path=None, normalization="row_minmax", test_size=0.2, seed=42, batch_size=64, use_species_weight=False, classification=False):
    """
    Load, preprocess and split MALDI-TOF datasets across multiple domains.

    All samples from the selected domains are loaded and concatenated.
    A stratified train/validation split is performed by species.

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

    Returns
    -------
    dict
        Dictionary containing:
        - data_final : np.ndarray (original data)
        - label_final : np.ndarray (original labels)
        - meta_final : pd.DataFrame
        - train_loader : DataLoader
        - val_loader : DataLoader
        - all_loader : DataLoader
        - input_dim : int
        - species_weights : torch.Tensor or None
    """

    dataset_pkl = pkl_path
    print("dataset_pkl:", dataset_pkl)
    print("domains:", domains)

    data_list, label_list, meta_list, amr_list = [], [], [], []

    if dataset_pkl is not None:

        dataset = load_pkl(dataset_pkl)

        data = dataset["data"]
        label = dataset["label"]
        meta = dataset["meta"]

        # ---------------------------------
        # Filter hospitals if requested
        # ---------------------------------
        if domains is not None:
            mask = meta["hospital"].isin(domains)
            data = data[mask.values]
            label = label[mask.values]
            meta = meta.loc[mask].reset_index(drop=True)

            if "amr" in dataset:
                amr = dataset["amr"][mask.values]
            else:
                amr = None
        else:
            amr = dataset.get("amr", None)

        data_list.append(data)
        label_list.append(label)
        meta_list.append(meta)

        if amr is not None:
            amr_list.append(amr)
    else:
        cfg = load_config()
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

            data_list.append(center["data"])
            label_list.append(center["label"])
            meta_list.append(center["meta"])

            if "amr" in center:
                amr_list.append(center["amr"])

    data_final = np.vstack(data_list)
    label_final = np.concatenate(label_list)
    meta_final = pd.concat(meta_list, ignore_index=True)

    if len(amr_list) > 0:
        amr_final = np.concatenate(amr_list)
    else:
        amr_final = None

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

    if amr_final is not None:

        X_train, X_val, y_train, y_val, domain_train, domain_val, amr_train, amr_val = train_test_split(
            data_norm,
            label_indices,
            domain_ids,
            amr_final,
            test_size=test_size,
            random_state=seed,
            stratify=label_indices if len(np.unique(label_indices)) > 1 else None
        )

    else:

        X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(
            data_norm,
            label_indices,
            domain_ids,
            test_size=test_size,
            random_state=seed,
            stratify=label_indices if len(np.unique(label_indices)) > 1 else None
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

    if amr_final is not None:

        amr_train_tensor = torch.tensor(amr_train, dtype=torch.float32)
        amr_val_tensor   = torch.tensor(amr_val, dtype=torch.float32)
        amr_all_tensor   = torch.tensor(amr_final, dtype=torch.float32)

    else:

        amr_train_tensor = None
        amr_val_tensor = None
        amr_all_tensor = None

    train_loader, val_loader, all_loader = construct_dataloaders(
        X_train_tensor, X_val_tensor, X_all_tensor,
        domain_train_tensor, domain_val_tensor, domain_all_tensor,
        batch_size=batch_size,
        species_train_tensor=species_train_tensor,
        species_val_tensor=species_val_tensor,
        species_all_tensor=species_all_tensor,
        amr_train_tensor=amr_train_tensor,
        amr_val_tensor=amr_val_tensor,
        amr_all_tensor=amr_all_tensor
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
