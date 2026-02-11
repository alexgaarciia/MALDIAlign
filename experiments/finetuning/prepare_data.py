############################
# PATH & EXPERIMENT SETUP
############################
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Find project root
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
from utils.data import load_pkl, subsample_dataset_stratified


############################
# DATA PREPARATION
############################
cfg = load_config()

driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
marisma_pkl = cfg["data"]["MARISMa_REDUCED_PKL"]
msumg_pkl = cfg["data"]["MSUMG_PKL"]
rki_pkl = cfg["data"]["RKI_PKL"]

driams = load_pkl(driams_pkl)
marisma = load_pkl(marisma_pkl)
msumg = load_pkl(msumg_pkl)
rki = load_pkl(rki_pkl)

data_driams, label_driams, meta_driams = driams["data"], driams["label"], pd.DataFrame.from_records(list(driams["meta"]))
data_marisma, label_marisma, meta_marisma = marisma["data"], marisma["label"], pd.DataFrame.from_records(list(marisma["meta"]))
data_msumg, label_msumg, meta_msumg = msumg["data"], msumg["label"], pd.DataFrame.from_records(list(msumg["meta"]))
data_rki, label_rki, meta_rki = rki["data"], rki["label"], pd.DataFrame.from_records(list(rki["meta"]))

meta_marisma.insert(0, "hospital", "MARISMA")
meta_msumg.insert(0, "hospital", "MS-UMG")
meta_rki.insert(0, "hospital", "RKI")

print("\n===== DATA LOADED =====")

# Filter the data by hospital
filtered_data = {}
for hosp in meta_driams["hospital"].unique():
    idx = np.where(meta_driams["hospital"].values == hosp)[0]
    filtered_data[hosp] = {
        "data": data_driams[idx],
        "label": label_driams[idx],
        "meta": meta_driams.iloc[idx]
    }

dataA, labelA, metaA = filtered_data["DRIAMS_A"]["data"], filtered_data["DRIAMS_A"]["label"], filtered_data["DRIAMS_A"]["meta"]
dataB, labelB, metaB = filtered_data["DRIAMS_B"]["data"], filtered_data["DRIAMS_B"]["label"], filtered_data["DRIAMS_B"]["meta"]
dataC, labelC, metaC = filtered_data["DRIAMS_C"]["data"], filtered_data["DRIAMS_C"]["label"], filtered_data["DRIAMS_C"]["meta"]
dataD, labelD, metaD = filtered_data["DRIAMS_D"]["data"], filtered_data["DRIAMS_D"]["label"], filtered_data["DRIAMS_D"]["meta"]


############################
# STRATIFIED SUBSAMPLING
############################
print("\n===== STRATIFIED SUBSAMPLING =====")
A_split = subsample_dataset_stratified(dataA, labelA, metaA, n_samples=500)
B_split = subsample_dataset_stratified(dataB, labelB, metaB, n_samples=500)
C_split = subsample_dataset_stratified(dataC, labelC, metaC, n_samples=500)
D_split = subsample_dataset_stratified(dataD, labelD, metaD, n_samples=500)
M_split = subsample_dataset_stratified(data_marisma, label_marisma, meta_marisma, n_samples=500)
R_split = subsample_dataset_stratified(data_rki, label_rki, meta_rki, n_samples=200)
MS_split = subsample_dataset_stratified(data_msumg, label_msumg, meta_msumg, n_samples=200)

print("Anchor sizes:")
print(
    f"A:{len(A_split['selected']['idx'])} "
    f"B:{len(B_split['selected']['idx'])} "
    f"C:{len(C_split['selected']['idx'])} "
    f"D:{len(D_split['selected']['idx'])} "
    f"M:{len(M_split['selected']['idx'])} "
    f"R:{len(R_split['selected']['idx'])} "
    f"MS:{len(MS_split['selected']['idx'])}"
)

print("Base sizes:")
print(
    f"A:{len(A_split['rest']['idx'])} "
    f"B:{len(B_split['rest']['idx'])} "
    f"C:{len(C_split['rest']['idx'])} "
    f"D:{len(D_split['rest']['idx'])} "
    f"M:{len(M_split['rest']['idx'])} "
    f"R:{len(R_split['rest']['idx'])} "
    f"MS:{len(MS_split['rest']['idx'])}"
)


############################
# SAVE SPLITS
############################
splits_idx = {
    "DRIAMS_A": {"base": A_split["rest"]["idx"], "anchor": A_split["selected"]["idx"]},
    "DRIAMS_B": {"base": B_split["rest"]["idx"], "anchor": B_split["selected"]["idx"]},
    "DRIAMS_C": {"base": C_split["rest"]["idx"], "anchor": C_split["selected"]["idx"]},
    "DRIAMS_D": {"base": D_split["rest"]["idx"], "anchor": D_split["selected"]["idx"]},
    "MARISMA": {"base": M_split["rest"]["idx"], "anchor": M_split["selected"]["idx"]},
    "RKI": {"base": R_split["rest"]["idx"], "anchor": R_split["selected"]["idx"]},
    "MS-UMG": {"base": MS_split["rest"]["idx"], "anchor": MS_split["selected"]["idx"]},
}

print("\n===== SAVING SPLITS & DATA =====")
with open(experiment_dir / "splits_idx.pkl", "wb") as f:
    pickle.dump(splits_idx, f)

print(f"Saved to: {experiment_dir / 'splits_idx.pkl'}")
print("Done.")
