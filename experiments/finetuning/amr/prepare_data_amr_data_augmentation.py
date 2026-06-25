############################################################
# PATH CONFIGURATION
############################################################
import os
import sys
from pathlib import Path

PROJECT_NAME = "MALDIAlign"
cwd = Path().resolve()
target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break
if target is not None and target != cwd:
    os.chdir(target)
    sys.path.append(str(target))

print("Working directory:", os.getcwd())


############################################################
# IMPORTS
############################################################
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.utils import resample

from src.data.io import load_pkl
from src.data.splits import subsample_dataset_stratified


############################################################
# CONFIGURATION
############################################################
SOURCE_DOMAINS   = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TARGET_DOMAIN    = "MS-UMG"

GRID_PREV = [0]
GRID_NEW = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000]
N_PARTITIONS = 10

BASE_OUTPUT = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data")

SPECIES_CONFIG = {
    # "Klebsiella_Pneumoniae": {
    #     "antibiotics": ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    #     "dataset_path":   "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
    #     "source_splits": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_223514/data_splits.pkl"),
    #     "target_pool_size": 2000,
    # },
    "Escherichia_Coli": {
        "antibiotics": ["Ampicillin", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "dataset_path":   "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "source_splits": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_223855/data_splits.pkl"),
        "target_pool_size": 5000,
    },
    # "Staphylococcus_Aureus": {
    #     "antibiotics": ['Oxacillin', 'Clindamycin', 'Erythromycin'],
    #     "dataset_path":   "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
    #     "source_splits": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_224204/data_splits.pkl"),
    #     "target_pool_size": 5000,
    # },
    # "Enterococcus_Faecium": {
    #     "antibiotics": ["Vancomycin"],
    #     "dataset_path":   "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
    #     "source_splits": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260528_234515/data_splits.pkl"),
    #     "target_pool_size": 1500,
    # },
    # "Pseudomonas_Aeruginosa": {
    #     "antibiotics": ["Meropenem", "Amikacin"],
    #     "dataset_path":   "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
    #     "source_splits": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_224359/data_splits.pkl"),
    #     "target_pool_size": 2500,
    # },
}


############################################################
# MAIN LOOP OVER SPECIES
############################################################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for species, cfg in SPECIES_CONFIG.items():
    antibiotics  = cfg["antibiotics"]
    splits_path  = cfg["source_splits"]
    dataset_path = cfg["dataset_path"]

    print(f"\n{'#'*70}")
    print(f"  SPECIES: {species}")
    print(f"  Antibiotics: {antibiotics}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'#'*70}")

    # ----------------------------------------------------------
    # 1. Load dataset for this species
    # ----------------------------------------------------------
    dataset     = load_pkl(dataset_path)
    data_all    = dataset["data"]
    amr_all     = dataset["amr"]
    ab_list_raw = list(dataset["antibiotics"])
    labels_all  = dataset["label"]
    raw_meta    = dataset["meta"]
    meta_all    = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) \
                  else pd.DataFrame(raw_meta)

    print(f"Total samples in dataset: {len(data_all)}")

    # ----------------------------------------------------------
    # 2. Source dataset filtered to species
    # ----------------------------------------------------------
    src_mask   = meta_all["hospital"].isin(SOURCE_DOMAINS)
    data_src   = data_all[src_mask.values]
    amr_src    = amr_all[src_mask.values]
    labels_src = labels_all[src_mask.values]
    meta_src   = meta_all.loc[src_mask].reset_index(drop=True)

    sp_mask    = (labels_src == species)
    data_src   = data_src[sp_mask]
    amr_src    = amr_src[sp_mask]
    labels_src = labels_src[sp_mask]
    meta_src   = meta_src.loc[sp_mask].reset_index(drop=True)

    print(f"\nSource ({species}): {len(data_src)} samples")
    for dom in SOURCE_DOMAINS:
        n = (meta_src["hospital"] == dom).sum()
        print(f"  {dom}: {n}")

    # ----------------------------------------------------------
    # 3. Load and validate source splits
    # ----------------------------------------------------------
    with open(splits_path, "rb") as f:
        source_splits_raw = pickle.load(f)
    src_splits = source_splits_raw["splits_per_domain"]

    for dom in SOURCE_DOMAINS:
        if dom in src_splits:
            tr = src_splits[dom]["train_idx"]
            if len(tr) > 0:
                assert tr.max() < len(data_src), \
                    f"{dom} train_idx max ({tr.max()}) >= dataset size ({len(data_src)})"
            print(f"  {dom}: train={len(tr)} ✓")

    # ----------------------------------------------------------
    # 4. Target domain MS-UMG filtered to species
    # ----------------------------------------------------------
    dom_mask = (meta_all["hospital"] == TARGET_DOMAIN).values
    data_t   = data_all[dom_mask]
    amr_t    = amr_all[dom_mask]
    labels_t = labels_all[dom_mask]
    meta_t   = meta_all.loc[dom_mask].reset_index(drop=True)

    # Chrom-agar filter
    if "agar" in meta_t.columns:
        chrom = (meta_t["agar"] == "chrom").values
        keep  = ~chrom
        data_t   = data_t[keep]
        amr_t    = amr_t[keep]
        labels_t = labels_t[keep]
        meta_t   = meta_t.loc[keep].reset_index(drop=True)
        print(f"\nMS-UMG: removed {chrom.sum()} chrom-agar samples")

    sp_mask  = (labels_t == species)
    data_t   = data_t[sp_mask]
    amr_t    = amr_t[sp_mask]
    labels_t = labels_t[sp_mask]
    meta_t   = meta_t.loc[sp_mask].reset_index(drop=True)

    keep_idx       = [ab_list_raw.index(n) for n in antibiotics if n in ab_list_raw]
    ab_list        = [n for n in antibiotics if n in ab_list_raw]
    amr_t_filtered = amr_t[:, keep_idx]

    print(f"\nMS-UMG ({species}): {len(data_t)} samples")
    for j, ab_name in enumerate(ab_list):
        y     = amr_t_filtered[:, j]
        valid = ~np.isnan(y)
        n_r   = int((y[valid] == 1).sum())
        n_s   = int((y[valid] == 0).sum())
        if (n_r + n_s) > 0:
            print(f"  {ab_name}: S={n_s}, R={n_r}, prev={100*n_r/(n_r+n_s):.1f}%")

    if len(data_t) == 0:
        print(f"No data for {species} in MS-UMG. Skipping.")
        continue

    # ----------------------------------------------------------
    # 5. Generate partitions
    # ----------------------------------------------------------
    pool_size = min(cfg["target_pool_size"], len(data_t))
    species_tag = species.replace("_", "")

    for i_part in range(N_PARTITIONS):
        # Divide MS-UMG in samples for finetuning vs test
        pool_split = subsample_dataset_stratified(
            data=data_t,
            labels=labels_t,
            meta=meta_t,
            n_samples=pool_size,
            global_indices=np.arange(len(data_t)),
            ood=True,
            random_state=i_part,
        )

        ft_pool_idx = pool_split["finetuning"]["idx"]
        test_idx = pool_split["test"]["idx"]
        # ft_pool_shuffled = resample(ft_pool_idx, stratify=labels_t[ft_pool_idx], replace=False, n_samples=len(ft_pool_idx),random_state=i_part)

        experiment_dir = BASE_OUTPUT / f"splits_{species_tag}_MSUMG_{timestamp}" / f"run_{i_part}"
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for n_prev in GRID_PREV:
            for n_new in GRID_NEW:
                if n_new > len(ft_pool_idx):
                    continue

                current_splits = {}

                # Source domains
                for src_dom in SOURCE_DOMAINS:
                    if src_dom in src_splits and n_prev > 0:
                        src_train = src_splits[src_dom]["train_idx"]
                        if len(src_train) > 0:
                            sub = subsample_dataset_stratified(
                                data=data_src[src_train],
                                labels=labels_src[src_train],
                                meta=meta_src.iloc[src_train].reset_index(drop=True),
                                n_samples=min(n_prev, len(src_train)),
                                global_indices=src_train,
                                ood=False,
                                random_state=i_part,
                            )
                            current_splits[src_dom] = {"finetuning": sub["finetuning"]["idx"]}
                        else:
                            current_splits[src_dom] = {"finetuning": np.array([], dtype=int)}
                    else:
                        current_splits[src_dom] = {"finetuning": np.array([], dtype=int)}

                # Target finetuning
                ft_idx_target = ft_pool_idx[:n_new]
                # sub_target = subsample_dataset_stratified(
                #     data=data_t[ft_pool_idx],
                #     labels=labels_t[ft_pool_idx],
                #     meta=meta_t.iloc[ft_pool_idx].reset_index(drop=True),
                #     n_samples=n_new,
                #     global_indices=ft_pool_idx,
                #     ood=False,
                #     random_state=i_part,
                # )

                # current_splits[TARGET_DOMAIN] = {
                #     "finetuning": sub_target["finetuning"]["idx"],
                #     "test":       test_idx,
                # }

                current_splits[TARGET_DOMAIN] = {
                    "finetuning": ft_idx_target,
                    "test": test_idx}

                file_name = f"prev_{n_prev}_new_{n_new}.pkl"
                with open(experiment_dir / file_name, "wb") as f:
                    pickle.dump(current_splits, f)

        pool_info = {
            "partition":      i_part,
            "random_state":   i_part,
            "species":        species,
            "target_domain":  TARGET_DOMAIN,
            "antibiotics":    ab_list,
            "dataset_path":   dataset_path,
            "source_splits":  str(splits_path),
            "ft_pool_size":   len(ft_pool_idx),
            "test_size":      len(test_idx),
            "grid_prev":      GRID_PREV,
            "grid_new":       GRID_NEW,
            "timestamp":      timestamp,
        }
        with open(experiment_dir / "pool_info.pkl", "wb") as f:
            pickle.dump(pool_info, f)

    print(f"\n  Splits saved to: {BASE_OUTPUT}/splits_{species_tag}_MSUMG_{timestamp}/")

print(f"\n===== DONE =====")
