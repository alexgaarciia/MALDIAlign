############################
# PATH & EXPERIMENT SETUP
############################
import os
import sys
import pickle
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
# OUTPUT DIRECTORY
############################
BASE_OUTPUT = Path(
    "/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data"
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = BASE_OUTPUT / f"splits_{timestamp}"
experiment_dir.mkdir(parents=True, exist_ok=True)

print("\n===== OUTPUT DIRECTORY =====")
print("Saving splits to:", experiment_dir)


############################
# IMPORTS
############################
from utils.config import load_config
from utils.data import (
    load_driams,
    load_marisma,
    load_msumg,
    load_rki,
    subsample_dataset_stratified,
)


############################
# DATA PREPARATION
############################
cfg = load_config()

# Load datasets
driams_dict  = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
msumg_dict   = load_msumg(cfg["data"]["MSUMG_PKL"])  # already filters agar
rki_dict     = load_rki(cfg["data"]["RKI_PKL"])

# Unpack
data_driams, label_driams, meta_driams = (
    driams_dict["data"],
    driams_dict["label"],
    driams_dict["meta"],
)

data_marisma, label_marisma, meta_marisma = (
    marisma_dict["data"],
    marisma_dict["label"],
    marisma_dict["meta"],
)

data_msumg, label_msumg, meta_msumg = (
    msumg_dict["data"],
    msumg_dict["label"],
    msumg_dict["meta"],
)

data_rki, label_rki, meta_rki = (
    rki_dict["data"],
    rki_dict["label"],
    rki_dict["meta"],
)

# Split DRIAMS by hospital using masks
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA]
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB]
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC]
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD]

print("\n===== DATA LOADED =====")


############################
# STRATIFIED SUBSAMPLING
############################
print("\n===== STRATIFIED SUBSAMPLING =====")

A_split  = subsample_dataset_stratified(dataA, labelA, metaA, n_samples=200)
B_split  = subsample_dataset_stratified(dataB, labelB, metaB, n_samples=200)
C_split  = subsample_dataset_stratified(dataC, labelC, metaC, n_samples=200)
M_split  = subsample_dataset_stratified(data_marisma, label_marisma, meta_marisma, n_samples=200)
R_split  = subsample_dataset_stratified(data_rki, label_rki, meta_rki, n_samples=200)

D_split  = subsample_dataset_stratified(dataD, labelD, metaD, n_samples=1000, ood=True)
MS_split = subsample_dataset_stratified(data_msumg, label_msumg, meta_msumg, n_samples=1000, ood=True)


############################
# SAVE SPLITS
############################
splits_idx = {
    "DRIAMS_A": {"finetuning": A_split["finetuning"]["idx"]},
    "DRIAMS_B": {"finetuning": B_split["finetuning"]["idx"]},
    "DRIAMS_C": {"finetuning": C_split["finetuning"]["idx"]},
    "MARISMA":  {"finetuning": M_split["finetuning"]["idx"]},
    "RKI":      {"finetuning": R_split["finetuning"]["idx"]},
    "DRIAMS_D": {
        "finetuning": D_split["finetuning"]["idx"],
        "test": D_split["test"]["idx"],
    },
    "MS-UMG": {
        "finetuning": MS_split["finetuning"]["idx"],
        "test": MS_split["test"]["idx"],
    },
}

print("\n===== SAVING SPLITS & DATA =====")

with open(experiment_dir / "splits_idx.pkl", "wb") as f:
    pickle.dump(splits_idx, f)

print(f"Saved to: {experiment_dir / 'splits_idx.pkl'}")
print("Done.")
