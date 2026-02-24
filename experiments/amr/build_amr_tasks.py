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


############################################################
# IMPORTS
############################################################
import pickle
import numpy as np
import pandas as pd
from utils.data import load_pkl


############################################################
# LOAD DATASETS
############################################################
working_path = Path("/export/data_ml4ds/bacteria_id/codigoMALDIVAS/pickles/DRIAMS_study.pkl")
amr_path = Path("/export/usuarios01/agnavarr/MALDIGen-dev/pickles/DRIAMS_study_amr_ceftr_oxa.pkl")

working_driams = load_pkl(working_path)
amr_driams = load_pkl(amr_path)

working_driams_data = working_driams["data"]
working_driams_label = working_driams["label"]
working_driams_meta = pd.DataFrame.from_records(list(working_driams["meta"]))
working_driams_meta["study"] = working_driams_meta["study"].str.replace(".txt", "", regex=False)

amr_driams_data = amr_driams["data"]
amr_driams_label = amr_driams["label"]
amr_driams_meta = pd.DataFrame.from_records(list(amr_driams["meta"]))
amr_driams_meta["study"] = amr_driams_meta["study"].str.replace(".txt", "", regex=False)
amr_arr = amr_driams["amr"]                 
amr_list = amr_driams["antibiotics"]


############################################################
# FILTER WORKING DATASET BY SPECIES
############################################################
mask_choli = np.where(working_driams_label == "Escherichia_Coli")[0]
mask_pneu = np.where(working_driams_label == "Klebsiella_Pneumoniae")[0]
mask_aureus = np.where(working_driams_label == "Staphylococcus_Aureus")[0]

echoli_working = {
    "data": working_driams_data[mask_choli],
    "label": working_driams_label[mask_choli],
    "meta": working_driams_meta.iloc[mask_choli]  
}

kpneumoniae_working = {
    "data": working_driams_data[mask_pneu],
    "label": working_driams_label[mask_pneu],
    "meta": working_driams_meta.iloc[mask_pneu]
}

saureus_working = {
    "data": working_driams_data[mask_aureus],
    "label": working_driams_label[mask_aureus],
    "meta": working_driams_meta.iloc[mask_aureus]
}


############################################################
# FILTER AMR DATASET BY SPECIES
############################################################
mask_choli_amr = np.where(amr_driams_label == "Escherichia_Coli")[0]
mask_pneu_amr = np.where(amr_driams_label == "Klebsiella_Pneumoniae")[0]
mask_aureus_amr = np.where(amr_driams_label == "Staphylococcus_Aureus")[0]

echoli_amr = {
    "data": amr_driams_data[mask_choli_amr],
    "label": amr_driams_label[mask_choli_amr],
    "meta": amr_driams_meta.iloc[mask_choli_amr]
}

kpneumoniae_amr = {
    "data": amr_driams_data[mask_pneu_amr],
    "label": amr_driams_label[mask_pneu_amr],
    "meta": amr_driams_meta.iloc[mask_pneu_amr]
}

saureus_amr = {
    "data": amr_driams_data[mask_aureus_amr],
    "label": amr_driams_label[mask_aureus_amr],
    "meta": amr_driams_meta.iloc[mask_aureus_amr]
}


############################################################
# BUILD ID SETS FOR COMPARISON
############################################################
working_ids_echoli = set(echoli_working["meta"]["study"])
amr_ids_echoli = set(echoli_amr["meta"]["study"])

working_ids_kpneumoniae = set(kpneumoniae_working["meta"]["study"])
amr_ids_kpneumoniae = set(kpneumoniae_amr["meta"]["study"])

working_ids_saureus = set(saureus_working["meta"]["study"])
amr_ids_saureus = set(saureus_amr["meta"]["study"])


############################################################
# MAKE COMPARISON
############################################################
common_ids_echoli = working_ids_echoli & amr_ids_echoli
only_working_echoli = working_ids_echoli - amr_ids_echoli
only_amr_echoli = amr_ids_echoli - working_ids_echoli

common_ids_kpneumoniae = working_ids_kpneumoniae & amr_ids_kpneumoniae
only_working_kpneumoniae = working_ids_kpneumoniae - amr_ids_kpneumoniae
only_amr_kpneumoniae = amr_ids_kpneumoniae - working_ids_kpneumoniae

common_ids_saureus = working_ids_saureus & amr_ids_saureus
only_working_saureus = working_ids_saureus - amr_ids_saureus
only_amr_saureus = amr_ids_saureus - working_ids_saureus


############################################################
# PRINT RESULTS
############################################################
print("\n================ E. coli ================")
print("Same sets?", working_ids_echoli == amr_ids_echoli)
print("Total working:", len(working_ids_echoli))
print("Total AMR:", len(amr_ids_echoli))
print("Common:", len(common_ids_echoli))
print("Only in working:", len(only_working_echoli))
print("Only in AMR:", len(only_amr_echoli))

if only_working_echoli:
    print("Example only in working:", list(only_working_echoli)[:5])
if only_amr_echoli:
    print("Example only in AMR:", list(only_amr_echoli)[:5])


print("\n================ K. pneumoniae ================")
print("Same sets?", working_ids_kpneumoniae == amr_ids_kpneumoniae)
print("Total working:", len(working_ids_kpneumoniae))
print("Total AMR:", len(amr_ids_kpneumoniae))
print("Common:", len(common_ids_kpneumoniae))
print("Only in working:", len(only_working_kpneumoniae))
print("Only in AMR:", len(only_amr_kpneumoniae))

if only_working_kpneumoniae:
    print("Example only in working:", list(only_working_kpneumoniae)[:5])
if only_amr_kpneumoniae:
    print("Example only in AMR:", list(only_amr_kpneumoniae)[:5])


print("\n================ S. aureus ================")
print("Same sets?", working_ids_saureus == amr_ids_saureus)
print("Total working:", len(working_ids_saureus))
print("Total AMR:", len(amr_ids_saureus))
print("Common:", len(common_ids_saureus))
print("Only in working:", len(only_working_saureus))
print("Only in AMR:", len(only_amr_saureus))

if only_working_saureus:
    print("Example only in working:", list(only_working_saureus)[:5])
if only_amr_saureus:
    print("Example only in AMR:", list(only_amr_saureus)[:5])


############################################################
# BUILD PKLs
############################################################
out_dir = Path("experiments/amr/pickles")
out_dir.mkdir(parents=True, exist_ok=True)

# PKL E. coli
amr_ecoli = amr_arr[mask_choli_amr]  
cef_idx = amr_list.index("Ceftriaxone")

echoli_amr["amr"] = amr_ecoli[:, cef_idx]
echoli_amr["antibiotics"] = ["Ceftriaxone"]

with open(out_dir / "E-CEF.pkl", "wb") as f:
    pickle.dump(echoli_amr, f)

# PKL K. pneumoniae
amr_kpneumoniae = amr_arr[mask_pneu_amr]  

kpneumoniae_amr["amr"] = amr_kpneumoniae[:, cef_idx]
kpneumoniae_amr["antibiotics"] = ["Ceftriaxone"]

with open(out_dir / "K-CEF.pkl", "wb") as f:
    pickle.dump(kpneumoniae_amr, f)

# PKL S. aureus
amr_saureus = amr_arr[mask_aureus_amr]  
oxa_idx = amr_list.index("Oxacillin")

saureus_amr["amr"] = amr_saureus[:, oxa_idx]
saureus_amr["antibiotics"] = ["Oxacillin"]

with open(out_dir / "S-OXA.pkl", "wb") as f:
    pickle.dump(saureus_amr, f)
