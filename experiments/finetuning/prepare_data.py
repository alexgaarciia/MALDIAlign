############################
# PATH & EXPERIMENT SETUP
############################
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


############################
# IMPORTS
############################
from src.config.loader import load_config
from src.data.datasets import load_driams, load_msumg
from src.data.splits import subsample_dataset_stratified


############################
# LOAD DATA
############################
# Species used during training
TARGET_SPECIES = [
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"]

print("\n===== LOADING DATA & SPLITS =====")

# Load reference splits
SOURCE_SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")

if not SOURCE_SPLITS_PATH.exists():
    raise FileNotFoundError(f"Splits not found in: {SOURCE_SPLITS_PATH}")

with open(SOURCE_SPLITS_PATH, "rb") as f:
    full_split_data = pickle.load(f)
    source_splits = full_split_data["splits_per_domain"]

print(f"Referece splits loaded from: {SOURCE_SPLITS_PATH}")

# Create splits for unseen domains
cfg = load_config()
driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])

# Filter DRIAMS-D
# Filter by species
mask_sp_driams = np.isin(driams_dict["label"], TARGET_SPECIES)
data_driams_filtered = driams_dict["data"][mask_sp_driams]
label_driams_filtered = driams_dict["label"][mask_sp_driams]
meta_driams_filtered = driams_dict["meta"][mask_sp_driams].reset_index(drop=True)

# Filter by hospital D
mask_hosp_D = meta_driams_filtered["hospital"] == "DRIAMS_D"
dataD = data_driams_filtered[mask_hosp_D]
labelD = label_driams_filtered[mask_hosp_D]
metaD = meta_driams_filtered[mask_hosp_D].reset_index(drop=True)
local_idx_D = np.arange(len(dataD)) 

# Filter MS-UMG
mask_M = np.isin(msumg_dict["label"], TARGET_SPECIES)
global_idx_M = np.where(mask_M)[0]
dataM = msumg_dict["data"][mask_M]
labelM = msumg_dict["label"][mask_M]
metaM = msumg_dict["meta"][mask_M].reset_index(drop=True)
local_idx_M = np.arange(len(dataM))

# Generate splits
d_split = subsample_dataset_stratified(dataD, labelD, metaD, n_samples=250, global_indices=local_idx_D, ood=True)
m_split = subsample_dataset_stratified(dataM, labelM, metaM, n_samples=250, global_indices=local_idx_M, ood=True)

ood_data = {
    "DRIAMS_D": {"ft_pool": d_split["finetuning"]["idx"], "test": d_split["test"]["idx"]},
    "MS-UMG":   {"ft_pool": m_split["finetuning"]["idx"], "test": m_split["test"]["idx"]}
}


############################
# OUTPUT DIRECTORY
############################
BASE_OUTPUT = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = BASE_OUTPUT / f"splits_{timestamp}"
experiment_dir.mkdir(parents=True, exist_ok=True)

print("\n===== OUTPUT DIRECTORY =====")
print("Saving splits to:", experiment_dir)


############################
# STRATIFIED SUBSAMPLING
############################
domains_prev = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA", "RKI"]
domains_new  = ["DRIAMS_D", "MS-UMG"]

print("\n===== GENERATING SUB-SPLITS GRID =====")
grid_prev = np.arange(0, 251, 50)
grid_new = np.arange(50, 251, 50)

for n_prev in grid_prev:
    for n_new in grid_new:
        current_experiment_splits = {}

        # Process domains used in training
        for dom in domains_prev:
            vae_train = source_splits[dom]["train_idx"]
            current_experiment_splits[dom] = {
                "finetuning": vae_train[:n_prev] if n_prev > 0 else np.array([], dtype=int)
            }

        # New domains
        for dom in domains_new:
            ft_pool = ood_data[dom]["ft_pool"]
            test_idx = ood_data[dom]["test"]
            
            sub_idx_ft = ft_pool[:n_new]
            
            current_experiment_splits[dom] = {
                "finetuning": sub_idx_ft,
                "test": test_idx
            }

        file_name = f"prev_{n_prev}_new_{n_new}.pkl"
        with open(experiment_dir / file_name, "wb") as f:
            pickle.dump(current_experiment_splits, f)

        print(f"Saved to: {experiment_dir / file_name}")

print("\n===== DONE =====")
print(f"All {len(grid_prev)*len(grid_new)} split files generated")
