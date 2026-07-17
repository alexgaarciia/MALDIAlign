############################################################
# PATH & EXPERIMENT SETUP
############################################################
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

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

print("\n===== PATH & CONFIG =====")
print("Working directory:", os.getcwd())
print("Project root:", target)


############################################################
# IMPORTS
############################################################
from sklearn.utils import resample
from src.config.loader import load_config
from src.data.datasets import load_driams, load_msumg
from src.data.splits import subsample_dataset_stratified


############################################################
# CONFIGURATION
############################################################
domains_prev = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA", "RKI"]
domains_new  = ["DRIAMS_D", "MS-UMG"]

TARGET_SPECIES = [
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"]

N_PARTITIONS = 10
GRID_PREV = [0]
GRID_NEW  = np.arange(50, 251, 50)  

SOURCE_SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")
BASE_OUTPUT = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data")


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA & SPLITS =====")

if not SOURCE_SPLITS_PATH.exists():
    raise FileNotFoundError(f"Splits not found in: {SOURCE_SPLITS_PATH}")

with open(SOURCE_SPLITS_PATH, "rb") as f:
    full_split_data = pickle.load(f)
    source_splits = full_split_data["splits_per_domain"]

print(f"Referece splits loaded from: {SOURCE_SPLITS_PATH}")

# Create splits for unseen domains
cfg = load_config()
driams_dict_full = load_driams(cfg["data"]["DRIAMS_FULL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])

src_mask = driams_dict_full["meta"]["hospital"].isin(domains_prev)
data_src = driams_dict_full["data"][src_mask]
label_src = driams_dict_full["label"][src_mask]
meta_src = driams_dict_full["meta"][src_mask].reset_index(drop=True)

# Filter by species
sp_mask = np.isin(label_src, TARGET_SPECIES)
data_src = data_src[sp_mask]
label_src = label_src[sp_mask]
meta_src = meta_src[sp_mask].reset_index(drop=True)

# Filter DRIAMS-D
mask_hosp_D = driams_dict_full["meta"]["hospital"] == "DRIAMS_D"
mask_sp_D = np.isin(driams_dict_full["label"], TARGET_SPECIES)
mask_D = mask_hosp_D & mask_sp_D

dataD = driams_dict_full["data"][mask_D]
labelD = driams_dict_full["label"][mask_D]
metaD = driams_dict_full["meta"][mask_D].reset_index(drop=True)

# Filter MS-UMG
mask_M = np.isin(msumg_dict["label"], TARGET_SPECIES)
dataM = msumg_dict["data"][mask_M]
labelM = msumg_dict["label"][mask_M]
metaM = msumg_dict["meta"][mask_M].reset_index(drop=True)


############################################################
# OUTPUT DIRECTORY
############################################################
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = BASE_OUTPUT / f"splits_{timestamp}"
experiment_dir.mkdir(parents=True, exist_ok=True)

print("\n===== OUTPUT DIRECTORY =====")
print("Saving splits to:", experiment_dir)


############################################################
# GENERATE PARTITIONS
############################################################
domains_prev = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA", "RKI"]
domains_new  = ["DRIAMS_D", "MS-UMG"]

print("\n===== GENERATING PARTITIONS =====")
print(f"N_PARTITIONS: {N_PARTITIONS}")
print(f"GRID_PREV: {GRID_PREV}")
print(f"GRID_NEW: {GRID_NEW.tolist()}")

for i_part in range(N_PARTITIONS):
    print(f"\n{'='*60}")
    print(f"PARTITION {i_part + 1}/{N_PARTITIONS}")
    print(f"{'='*60}")
    
    partition_dir = experiment_dir / f"run_{i_part}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    
    # Split DRIAMS-D: finetuning pool vs test holdout
    local_idx_D = np.arange(len(dataD))
    d_split = subsample_dataset_stratified(
        data=dataD,
        labels=labelD,
        meta=metaD,
        n_samples=250,  
        global_indices=local_idx_D,
        ood=True,
        random_state=i_part)
    
    ft_pool_D = d_split["finetuning"]["idx"]
    test_D = d_split["test"]["idx"]
    ft_pool_D_shuffled = resample(ft_pool_D, stratify=labelD[ft_pool_D], replace=False, n_samples=len(ft_pool_D), random_state=i_part)
    
    print(f"DRIAMS-D: pool={len(ft_pool_D)}, test={len(test_D)}")
    
    # Split MS-UMG: finetuning pool vs test holdout
    local_idx_M = np.arange(len(dataM))
    m_split = subsample_dataset_stratified(
        data=dataM,
        labels=labelM,
        meta=metaM,
        n_samples=250,  
        global_indices=local_idx_M,
        ood=True,
        random_state=i_part)
    
    ft_pool_M = m_split["finetuning"]["idx"]
    test_M = m_split["test"]["idx"]
    ft_pool_M_shuffled = resample(ft_pool_M, stratify=labelM[ft_pool_M], replace=False, n_samples=len(ft_pool_M),random_state=i_part)
    
    print(f"MS-UMG: pool={len(ft_pool_M)}, test={len(test_M)}")
    
    # Generate grid of (n_prev, n_new) splits
    for n_prev in GRID_PREV:
        for n_new in GRID_NEW:
            current_splits = {}
            
            # Source domains 
            for dom in domains_prev:
                current_splits[dom] = {"finetuning": np.array([], dtype=int)}

            # Target domains DRIAMS-D, MS-UMG
            ft_idx_D = ft_pool_D_shuffled[:n_new] if n_new <= len(ft_pool_D_shuffled) else ft_pool_D_shuffled
            ft_idx_M = ft_pool_M_shuffled[:n_new] if n_new <= len(ft_pool_M_shuffled) else ft_pool_M_shuffled
            
            current_splits["DRIAMS_D"] = {
                "finetuning": ft_idx_D,
                "test": test_D}
            
            current_splits["MS-UMG"] = {
                "finetuning": ft_idx_M,
                "test": test_M}
            
            # Save split
            file_name = f"prev_{n_prev}_new_{n_new}.pkl"
            with open(partition_dir / file_name, "wb") as f:
                pickle.dump(current_splits, f)
    
    print(f"Partition {i_part}: {len(GRID_PREV) * len(GRID_NEW)} split files generated")
    
    # Save partition metadata
    partition_info = {
        "partition": i_part,
        "random_state": i_part,
        "target_species": TARGET_SPECIES,
        "grid_prev": GRID_PREV,
        "grid_new": GRID_NEW.tolist(),
        "timestamp": timestamp,
        "driams_d_pool_size": len(ft_pool_D),
        "driams_d_test_size": len(test_D),
        "msumg_pool_size": len(ft_pool_M),
        "msumg_test_size": len(test_M),
    }
    with open(partition_dir / "partition_info.pkl", "wb") as f:
        pickle.dump(partition_info, f)


############################
# SUMMARY
############################
print("\n" + "="*60)
print("===== SUMMARY =====")
print("="*60)
print(f"Total partitions: {N_PARTITIONS}")
print(f"Total split files per partition: {len(GRID_PREV) * len(GRID_NEW)}")
print(f"Total split files generated: {N_PARTITIONS * len(GRID_PREV) * len(GRID_NEW)}")
print(f"\nOutput directory: {experiment_dir}")
