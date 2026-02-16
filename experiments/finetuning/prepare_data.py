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

# Here we define what we will be using for "evaluation" and for "finetuning"
D_full_split = subsample_dataset_stratified(
    dataD,
    labelD,
    metaD,
    n_samples = 250,
    ood=True
)

# Pool for finetuning and evaluation
D_pool_idx = D_full_split["finetuning"]["idx"]
D_eval_idx = D_full_split["test"]["idx"]

# Extract data
dataD_pool  = dataD[D_pool_idx]
labelD_pool = labelD[D_pool_idx]
metaD_pool  = metaD.iloc[D_pool_idx].reset_index(drop=True)

# Same for MS-UMG
MSUMG_full_split = subsample_dataset_stratified(
    data_msumg,
    label_msumg,
    meta_msumg,
    n_samples = 250,
    ood=True
)

MSUMG_pool_idx = MSUMG_full_split["finetuning"]["idx"]
MSUMG_eval_idx = MSUMG_full_split["test"]["idx"]

data_msumg_pool  = data_msumg[MSUMG_pool_idx]
label_msumg_pool = label_msumg[MSUMG_pool_idx]
meta_msumg_pool  = meta_msumg.iloc[MSUMG_pool_idx].reset_index(drop=True)

print("\n===== DATA LOADED =====")


############################
# STRATIFIED SUBSAMPLING
############################
print("\n===== STRATIFIED SUBSAMPLING =====")

grid_prev = np.arange(0, 251, 50)
grid_new = np.arange(50, 251, 50)

print("\n===== SAVING SPLITS & DATA =====")
for n_prev in grid_prev:
    # Previous domains
    A_split  = subsample_dataset_stratified(dataA, labelA, metaA, n_samples=n_prev)
    B_split  = subsample_dataset_stratified(dataB, labelB, metaB, n_samples=n_prev)
    C_split  = subsample_dataset_stratified(dataC, labelC, metaC, n_samples=n_prev)
    M_split  = subsample_dataset_stratified(data_marisma, label_marisma, meta_marisma, n_samples=n_prev)
    R_split  = subsample_dataset_stratified(data_rki, label_rki, meta_rki, n_samples=n_prev)

    for n_new in grid_new:
        D_sub = subsample_dataset_stratified(
            dataD_pool,
            labelD_pool,
            metaD_pool,
            n_samples=n_new,
            ood=False
        )

        MSUMG_sub = subsample_dataset_stratified(
            data_msumg_pool,
            label_msumg_pool,
            meta_msumg_pool,
            n_samples=n_new,
            ood=False
        )

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
                "finetuning": D_sub["finetuning"]["idx"],
                "test": D_eval_idx,
            },
            "MS-UMG": {
                "finetuning": MSUMG_sub["finetuning"]["idx"],
                "test": MSUMG_eval_idx,
            },
        }

        file_name = f"prev_{n_prev}_new_{n_new}.pkl"
        with open(experiment_dir / file_name, "wb") as f:
            pickle.dump(splits_idx, f)

        print(f"Saved to: {experiment_dir / file_name}")

print("\n===== DONE =====")
