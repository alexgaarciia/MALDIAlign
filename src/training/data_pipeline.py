import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_msumg, load_rki
from src.data.io import load_pkl
from src.dataloaders.builders import construct_dataloaders
from src.data.preprocessing import map_domains, row_minmax_normalize
from src.data.splits import create_and_save_domain_splits


def prepare_data(domains, experiment_dir, pkl_path=None, species_list=None, antibiotics_filter=None, normalization="row_minmax", test_size=0.2, seed=42, batch_size=64, use_species_weight=False, classification=False):
    """
    Load, preprocess and split MALDI-TOF datasets across multiple domains.

    Parameters
    ----------
    domains : list[str]
        List of ALL domain identifiers to load.
    """

    cfg = load_config()

    dataset_pkl = pkl_path
    print("dataset_pkl:", dataset_pkl)
    print("domains:", domains)

    data_list, label_list, meta_list, amr_list = [], [], [], []
    antibiotics_names = ["AMR"]

    if dataset_pkl is not None:
        dataset = load_pkl(dataset_pkl)
        data = dataset["data"]
        label = dataset["label"]
        raw_meta = dataset["meta"]

        if isinstance(raw_meta, (list, np.ndarray)):
            meta = pd.DataFrame(list(raw_meta))
        else:
            meta = pd.DataFrame(raw_meta)

        antibiotics_names = dataset.get("antibiotics", ["AMR"])

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
        for d in domains:
            if d.startswith("DRIAMS_"):
                driams_pkl = cfg["data"]["DRIAMS_FULL"]
                center = load_driams(driams_pkl, filter=[d])[d]
            elif d == "MARISMA":
                center = load_marisma(cfg["data"]["MARISMa_FULL"])
            elif d == "MS-UMG":
                center = load_msumg(cfg["data"]["MSUMG_FULL"])
            elif d == "RKI":
                center = load_rki(cfg["data"]["RKI_FULL"])
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

    # ===============================
    # FILTER MS-UMG CHROM AGAR
    # ===============================
    if "agar" in meta_final.columns:
        chrom_mask = (meta_final["hospital"] == "MS-UMG") & (meta_final["agar"] == "chrom")
        n_chrom = chrom_mask.sum()
        if n_chrom > 0:
            keep = ~chrom_mask.values
            data_final = data_final[keep]
            label_final = label_final[keep]
            meta_final = meta_final.loc[keep].reset_index(drop=True)

            if len(amr_list) > 0:
                amr_concat = np.concatenate(amr_list)
                amr_list = [amr_concat[keep]]

            print(f"Removed {n_chrom} MS-UMG chrom-agar samples. Remaining: {len(data_final)}")

    # ===============================
    # FILTER ANTIBIOTICS BY NAME
    # ===============================
    if antibiotics_filter is not None and len(amr_list) > 0:
        keep_idx = []
        keep_names = []
        for name in antibiotics_filter:
            if name in antibiotics_names:
                keep_idx.append(antibiotics_names.index(name))
                keep_names.append(name)
            else:
                print(f"WARNING: antibiotic '{name}' not found in dataset. Available: {antibiotics_names}")

        if len(keep_idx) == 0:
            raise ValueError(f"None of the requested antibiotics found!")

        amr_concat = np.concatenate(amr_list) if len(amr_list) > 1 else amr_list[0]
        amr_filtered = amr_concat[:, keep_idx]
        amr_list = [amr_filtered]
        antibiotics_names = keep_names

        print(f"Filtered antibiotics: {len(keep_names)} selected")
        print(f"  Selected: {keep_names}")

    # ===============================
    # FILTERING BY SPECIES
    # ===============================
    if species_list is not None:
        print(f"Filtering data for {len(species_list)} species...")
        mask_sp = np.isin(label_final, species_list)

        data_final = data_final[mask_sp]
        label_final = label_final[mask_sp]
        meta_final = meta_final.iloc[mask_sp].reset_index(drop=True)

        if len(amr_list) > 0:
            amr_final_temp = np.concatenate(amr_list)
            amr_list = [amr_final_temp[mask_sp]]

        if len(data_final) == 0:
            raise ValueError("No samples left after filtering species!")

    # ===============================
    # AMR HANDLING
    # ===============================
    if len(amr_list) > 0:
        amr_final = np.concatenate(amr_list)
        if amr_final.ndim == 1:
            mask_valid = ~np.isnan(amr_final)
        else:
            mask_valid = ~np.all(np.isnan(amr_final), axis=1)

        data_final = data_final[mask_valid]
        label_final = label_final[mask_valid]
        meta_final = meta_final.iloc[mask_valid].reset_index(drop=True)
        amr_final = amr_final[mask_valid]
    else:
        amr_final = None

    if "year" in meta_final.columns:
        meta_final["year"] = meta_final["year"].astype(str).replace("nan", "Unknown")
    else:
        meta_final["year"] = "Unknown"

    # ===============================
    # PRINT AMR STATISTICS
    # ===============================
    if amr_final is not None:
        print("\n" + "="*70)
        print("AMR LABEL STATISTICS")
        print("="*70)
        
        # Global per antibiotic
        print(f"\n{'Antibiotic':<30} {'N_obs':>8} {'N_R':>8} {'N_S':>8} {'%R':>8} {'%NaN':>8}")
        print("-" * 70)
        
        for j, ab in enumerate(antibiotics_names):
            col = amr_final[:, j]
            mask = ~np.isnan(col)
            n_total = len(col)
            n_obs   = mask.sum()
            n_r     = (col[mask] == 1).sum()
            n_s     = (col[mask] == 0).sum()
            pct_r   = 100 * n_r / n_obs if n_obs > 0 else float("nan")
            pct_nan = 100 * (n_total - n_obs) / n_total
            print(f"{ab:<30} {n_obs:>8} {n_r:>8} {n_s:>8} {pct_r:>7.1f}% {pct_nan:>7.1f}%")
        
        # Per hospital per antibiotic
        print(f"\n── Per hospital ──")
        for hospital in meta_final["hospital"].unique():
            mask_h = (meta_final["hospital"] == hospital).values
            print(f"\n  {hospital} (n={mask_h.sum()})")
            print(f"  {'Antibiotic':<28} {'N_obs':>6} {'N_R':>6} {'N_S':>6} {'%R':>7} {'%NaN':>7}")
            print("  " + "-" * 60)
            for j, ab in enumerate(antibiotics_names):
                col = amr_final[mask_h, j]
                mask_obs = ~np.isnan(col)
                n_total = len(col)
                n_obs   = mask_obs.sum()
                n_r     = (col[mask_obs] == 1).sum()
                n_s     = (col[mask_obs] == 0).sum()
                pct_r   = 100 * n_r / n_obs if n_obs > 0 else float("nan")
                pct_nan = 100 * (n_total - n_obs) / n_total
                print(f"  {ab:<28} {n_obs:>6} {n_r:>6} {n_s:>6} {pct_r:>6.1f}% {pct_nan:>6.1f}%")
        
        print("\n" + "="*70)

    # Normalize
    if normalization == "row_minmax":
        data_norm = row_minmax_normalize(data_final)
    else:
        data_norm = data_final

    # Encode species
    unique_species, label_indices = np.unique(label_final, return_inverse=True)

    if classification:
        return {"X": data_norm, "y": label_indices, "meta": meta_final, "class_names": unique_species}

    domain_ids = map_domains(meta_final)
    domain_map = None
    num_domains = int(np.unique(domain_ids).shape[0])

    # ===============================
    # SPLITTING LOGIC
    # ===============================
    split_dict = create_and_save_domain_splits(
        labels=label_indices, meta=meta_final,
        experiment_dir=experiment_dir,
        val_size=0.1, test_size=0.2, seed=seed,
    )
    train_idx = split_dict["global"]["train_idx"]
    val_idx = split_dict["global"]["val_idx"]

    # ===============================
    # PREPARE TRAIN AMR
    # ===============================
    test_idx = split_dict["global"]["test_idx"]
    train_val_idx = np.concatenate([train_idx, val_idx])
    amr_train_final = amr_final[train_idx].copy() if amr_final is not None else None

    # ===============================
    # TENSORS
    # ===============================
    X_train_tensor = torch.tensor(data_norm[train_idx], dtype=torch.float32)
    X_val_tensor = torch.tensor(data_norm[val_idx], dtype=torch.float32)
    X_test_tensor = torch.tensor(data_norm[test_idx], dtype=torch.float32)
    X_all_tensor = torch.tensor(data_norm[train_val_idx], dtype=torch.float32)

    domain_train_tensor = torch.tensor(domain_ids[train_idx], dtype=torch.long)
    domain_val_tensor = torch.tensor(domain_ids[val_idx], dtype=torch.long)
    domain_test_tensor = torch.tensor(domain_ids[test_idx], dtype=torch.long)
    domain_all_tensor = torch.tensor(domain_ids[train_val_idx], dtype=torch.long)

    species_train_tensor = torch.tensor(label_indices[train_idx], dtype=torch.long)
    species_val_tensor = torch.tensor(label_indices[val_idx], dtype=torch.long)
    species_test_tensor = torch.tensor(label_indices[test_idx], dtype=torch.long)
    species_all_tensor = torch.tensor(label_indices[train_val_idx], dtype=torch.long)

    amr_all = None
    if amr_final is not None:
        amr_all = amr_final[train_val_idx]
        amr_train_tensor = torch.tensor(amr_train_final, dtype=torch.float32)
        amr_val_tensor = torch.tensor(amr_final[val_idx], dtype=torch.float32)
        amr_test_tensor = torch.tensor(amr_final[test_idx], dtype=torch.float32)
        amr_all_tensor = torch.tensor(amr_all, dtype=torch.float32)

        if amr_train_tensor.ndim == 1:
            n_pos = (amr_train_tensor == 1).sum().item()
            n_neg = (amr_train_tensor == 0).sum().item()
            pos_weight = torch.tensor([min(n_neg / max(n_pos, 1), 20)], dtype=torch.float32)
        else:
            valid_mask = ~torch.isnan(amr_train_tensor)
            n_pos = ((amr_train_tensor == 1) & valid_mask).sum(dim=0)
            n_neg = ((amr_train_tensor == 0) & valid_mask).sum(dim=0)
            pos_weight = n_neg / torch.clamp(n_pos, min=1)
            pos_weight = torch.clamp(pos_weight, max=20)
    else:
        amr_train_tensor = None
        amr_val_tensor = None
        amr_test_tensor = None
        amr_all_tensor = None
        pos_weight = None

    train_loader, val_loader, all_loader = construct_dataloaders(
        X_train_tensor, X_val_tensor, X_all_tensor,
        domain_train_tensor, domain_val_tensor, domain_all_tensor,
        batch_size=batch_size,
        species_train_tensor=species_train_tensor,
        species_val_tensor=species_val_tensor,
        species_all_tensor=species_all_tensor,
        amr_train_tensor=amr_train_tensor,
        amr_val_tensor=amr_val_tensor,
        amr_all_tensor=amr_all_tensor,
    )

    if amr_test_tensor is None:
        test_dataset = TensorDataset(X_test_tensor, domain_test_tensor, species_test_tensor)
    else:
        test_dataset = TensorDataset(X_test_tensor, domain_test_tensor, species_test_tensor, amr_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    species_weights = None
    if use_species_weight:
        y_train = label_indices[train_idx]
        counts = np.bincount(y_train, minlength=len(unique_species))
        counts[counts == 0] = 1
        inv_freq = 1.0 / counts
        normalized_weights = inv_freq / inv_freq.sum()
        species_weights = torch.tensor(normalized_weights, dtype=torch.float32)

    # Safe format for evaluation (train+val)
    safe_data_final = data_final[train_val_idx]
    safe_label_final = label_final[train_val_idx]
    safe_meta_final = meta_final.iloc[train_val_idx].reset_index(drop=True)

    # Test set
    test_data_final = data_final[test_idx]
    test_data_norm = data_norm[test_idx]
    test_label_final = label_final[test_idx]
    test_meta_final = meta_final.iloc[test_idx].reset_index(drop=True)
    amr_test = amr_final[test_idx] if amr_final is not None else None

    if amr_final is not None:
        if len(amr_final.shape) > 1 and amr_final.shape[1] == len(antibiotics_names):
            for i, atb_name in enumerate(antibiotics_names):
                safe_meta_final[atb_name] = amr_all[:, i]
                test_meta_final[atb_name] = amr_test[:, i]
        else:
            atb_name = antibiotics_names[0] if len(antibiotics_names) > 0 else "AMR"
            safe_meta_final[atb_name] = amr_all
            test_meta_final[atb_name] = amr_test
    else:
        antibiotics_names = None

    return {
        "data_final": safe_data_final,
        "label_final": safe_label_final,
        "meta_final": safe_meta_final,
        "test_data_final": test_data_final,
        "test_data_norm": test_data_norm,
        "test_label_final": test_label_final,
        "test_meta_final": test_meta_final,
        "amr_test": amr_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "all_loader": all_loader,
        "test_loader": test_loader,
        "input_dim": safe_data_final.shape[1],
        "species_weights": species_weights,
        "pos_weight": pos_weight,
        "species_names": unique_species,
        "antibiotics": antibiotics_names,
        "num_domains": num_domains,
        "domain_map": domain_map,
        "test_species_encoded": label_indices[test_idx]
    }
