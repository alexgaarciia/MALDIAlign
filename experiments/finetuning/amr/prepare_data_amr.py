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
from src.data.io import load_pkl


############################################################
# CONFIGURATION
############################################################
SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TARGET_DOMAIN  = "MS-UMG"
GRID_PREV      = [0]
GRID_NEW       = [50, 100, 250, 500, 1000]
N_PARTITIONS   = 10
BASE_OUTPUT    = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data")

DATASET_PATH   = "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl"
SOURCE_SPLITS  = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/data_splits.pkl")

TARGET_POOL_SIZE_PER_SPECIES = 2000 

SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae":          ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Escherichia_Coli":               ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Staphylococcus_Aureus":          ["Oxacillin", "Clindamycin", "Erythromycin"],
    "Pseudomonas_Aeruginosa":         ["Meropenem", "Amikacin"],
    "Enterococcus_Faecium":           ["Vancomycin"],
    "Enterobacter_cloacae_complex":   ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
}

ALL_ANTIBIOTICS = [
    "Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin",
    "Piperacillin-Tazobactam", "Amikacin", "Oxacillin",
    "Clindamycin", "Erythromycin", "Vancomycin",
]


############################################################
# LOAD DATASET
############################################################
print("\n===== LOADING DATASET =====")
dataset     = load_pkl(DATASET_PATH)
data_all    = dataset["data"]
amr_all     = dataset["amr"]
ab_list_raw = list(dataset["antibiotics"])
labels_all  = dataset["label"]
raw_meta    = dataset["meta"]
meta_all    = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) \
              else pd.DataFrame(raw_meta)
print(f"Total samples: {len(data_all)}")

# chrom-agar filter
if "agar" in meta_all.columns:
    chrom = ((meta_all["hospital"] == TARGET_DOMAIN) &
             (meta_all["agar"] == "chrom")).values
    keep  = ~chrom
    data_all  = data_all[keep]
    amr_all   = amr_all[keep]
    labels_all = labels_all[keep]
    meta_all  = meta_all[keep].reset_index(drop=True)
    print(f"Removed {chrom.sum()} chrom-agar samples from MS-UMG")

# antibiotic indices
ab_to_idx = {ab: ab_list_raw.index(ab) for ab in ALL_ANTIBIOTICS if ab in ab_list_raw}


############################################################
# LOAD SOURCE SPLITS
############################################################
with open(SOURCE_SPLITS, "rb") as f:
    source_splits_raw = pickle.load(f)
src_splits = source_splits_raw["splits_per_domain"]


############################################################
# BUILD MS-UMG PER-SPECIES POOLS
############################################################
print("\n===== MS-UMG STATS PER SPECIES =====")
dom_mask = (meta_all["hospital"] == TARGET_DOMAIN).values

data_msumg   = data_all[dom_mask]
amr_msumg    = amr_all[dom_mask]
labels_msumg = labels_all[dom_mask]
meta_msumg   = meta_all[dom_mask].reset_index(drop=True)

species_msumg_idx = {}
for species in SPECIES_CONFIG:
    sp_mask = (labels_msumg == species)
    idx     = np.where(sp_mask)[0]
    species_msumg_idx[species] = idx
    print(f"  {species}: {len(idx)} samples in MS-UMG")

for n_new in GRID_NEW:
    for species, idx in species_msumg_idx.items():
        if len(idx) < n_new:
            print(f"  WARNING: {species} only has {len(idx)} samples, "
                  f"n_new={n_new} will be skipped")


############################################################
# GENERATE SPLITS
############################################################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
tag       = "AllSpecies"
output_dir_base = BASE_OUTPUT / f"splits_{tag}_MSUMG_{timestamp}"
output_dir_base.mkdir(parents=True, exist_ok=True)

print(f"\n===== GENERATING SPLITS =====")
print(f"Output: {output_dir_base}")

for i_part in range(N_PARTITIONS):
    print(f"\nPartition {i_part+1}/{N_PARTITIONS}")

    run_dir = output_dir_base / f"run_{i_part}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    ft_pool_per_species = {}
    test_per_species    = {}

    rng = np.random.RandomState(i_part)

    for species, idx in species_msumg_idx.items():
        shuffled  = rng.permutation(idx)
        test_size = max(int(len(idx) * 0.2), 50)
        test_size = min(test_size, len(idx))
        pool_size = min(TARGET_POOL_SIZE_PER_SPECIES, len(idx) - test_size)
        pool_size = max(pool_size, 0)
        ft_pool_per_species[species] = shuffled[:pool_size]
        test_per_species[species]    = shuffled[pool_size:pool_size + test_size]
        print(f"  {species}: ft_pool={pool_size}, test={test_size} (total={len(idx)})")

    test_idx_global = np.concatenate(list(test_per_species.values()))

    for n_prev in GRID_PREV:
        for n_new in GRID_NEW:

            skip = False
            for species, ft_pool in ft_pool_per_species.items():
                if n_new > len(ft_pool):
                    skip = True
                    break
            if skip:
                continue

            current_splits = {}

            for src_dom in SOURCE_DOMAINS:
                if src_dom in src_splits and n_prev > 0:
                    src_train = src_splits[src_dom]["train_idx"]
                    if len(src_train) > 0:
                        sub_size = min(n_prev, len(src_train))
                        sub_idx  = rng.choice(src_train, size=sub_size, replace=False)
                        current_splits[src_dom] = {"finetuning": sub_idx}
                    else:
                        current_splits[src_dom] = {"finetuning": np.array([], dtype=int)}
                else:
                    current_splits[src_dom] = {"finetuning": np.array([], dtype=int)}

            ft_idx_per_species = {}
            for species, ft_pool in ft_pool_per_species.items():
                ft_idx_per_species[species] = ft_pool[:n_new]

            ft_idx_global = np.concatenate(list(ft_idx_per_species.values()))

            current_splits[TARGET_DOMAIN] = {
                "finetuning":            ft_idx_global,
                "finetuning_per_species": ft_idx_per_species,  
                "test":                  test_idx_global,
                "test_per_species":      test_per_species,     
            }

            file_name = f"prev_{n_prev}_new_{n_new}.pkl"
            with open(run_dir / file_name, "wb") as f:
                pickle.dump(current_splits, f)

    pool_info = {
        "partition":             i_part,
        "random_state":          i_part,
        "species_list":          list(SPECIES_CONFIG.keys()),
        "target_domain":         TARGET_DOMAIN,
        "antibiotics_per_species": SPECIES_CONFIG,
        "all_antibiotics":       ALL_ANTIBIOTICS,
        "dataset_path":          DATASET_PATH,
        "source_splits":         str(SOURCE_SPLITS),
        "ft_pool_size_per_species": {sp: len(idx) for sp, idx in ft_pool_per_species.items()},
        "test_size_per_species": {sp: len(idx) for sp, idx in test_per_species.items()},
        "test_size_global":      len(test_idx_global),
        "grid_prev":             GRID_PREV,
        "grid_new":              GRID_NEW,
        "timestamp":             timestamp,
    }
    with open(run_dir / "pool_info.pkl", "wb") as f:
        pickle.dump(pool_info, f)

print(f"\n===== DONE =====")
print(f"Splits saved to: {output_dir_base}")
