# ============================================================
# PATH CONFIGURATION
# ============================================================
from pathlib import Path
import os

PROJECT_NAME = "MALDIAlign"

cwd = Path().resolve()

target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break

if target is not None and target != cwd:
    os.chdir(target)

print("Working directory:", os.getcwd())


# ============================================================
# IMPORTS
# ============================================================
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

from maldi_nn.models import MaldiTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import *
from src.evaluation.eval import make_loader

from models.baselines.mlp import MLPClassifier_Extended


# ============================================================
# ARG PARSING
# ============================================================
MODEL_SIZES = ["S", "M", "L", "XL"]

parser = argparse.ArgumentParser(description="MaldiTransformer + MLP evaluation")
parser.add_argument(
    "--size", "-s",
    type=str,
    required=True,
    choices=MODEL_SIZES,
    help="MaldiTransformer size: S, M, L or XL",
)
args = parser.parse_args()

SIZE = args.size
CHECKPOINT_PATH = Path(f"assets/MaldiTransformer{SIZE}.ckpt")
OUT_PATH = Path(f"experiments/results/classifiers/mlp/mlp_transformer_{SIZE}.csv")
N_PEAKS = 200  
EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")


# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "="*60)
print("===== LOADING DATA =====")
print("="*60)

cfg = load_config()

driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])

species_to_keep = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

mask_driams = np.isin(driams_dict["label"], species_to_keep)
data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]
meta_marisma = marisma_dict["meta"][mask_marisma].reset_index(drop=True)

mask_rki = np.isin(rki_dict["label"], species_to_keep)
data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]
meta_rki = rki_dict["meta"][mask_rki].reset_index(drop=True)

mask_msumg = np.isin(msumg_dict["label"], species_to_keep)
data_msumg = msumg_dict["data"][mask_msumg]
label_msumg = msumg_dict["label"][mask_msumg]
meta_msumg = msumg_dict["meta"][mask_msumg].reset_index(drop=True)

# Split DRIAMS
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA = data_driams[maskA], label_driams[maskA]
dataB, labelB = data_driams[maskB], label_driams[maskB]
dataC, labelC = data_driams[maskC], label_driams[maskC]
dataD, labelD = data_driams[maskD], label_driams[maskD]

# Normalize
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)

# Merge
data_list = [dataA,dataB,dataC,data_marisma,data_rki]
label_list = [labelA,labelB,labelC,label_marisma,label_rki]

data_final = np.vstack(data_list)
label_final = np.concatenate(label_list)

print("\n===== DATA LOADED =====")

# Load splits
with open(EXPERIMENT_DIR / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

print("\n===== SPLITS LOADED =====")


# ============================================================
# TRANSFORMER EMBEDDING EXTRACTION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transformer_latent(data_matrix, model, n_peaks=200, batch_size=256):
    model.eval()
    model.to(device)
    mz_axis = torch.linspace(2000, 20000, steps=6000).to(device)
    all_z = []
    
    for i in range(0, len(data_matrix), batch_size):
        batch_bin = torch.from_numpy(data_matrix[i:i+batch_size]).float().to(device)
        
        # Peak picking over the bins
        intensities, indices = torch.topk(batch_bin, k=n_peaks, dim=1)
        mzs = mz_axis[indices]
        
        batch_dict = {"mz": mzs, "intensity": intensities}
        
        with torch.no_grad():
            # Extract [CLS] token
            z_seq = model.transformer(batch_dict)
            z_cls = z_seq[:, 0, :]
            all_z.append(z_cls.cpu().numpy())
            
    return np.vstack(all_z)

print("\n===== EXTRACTING TRANSFORMER EMBEDDINGS =====")
model = MaldiTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location=device, strict=False)

Z_final = get_transformer_latent(data_final, model, n_peaks=N_PEAKS)
Z_D = get_transformer_latent(dataD, model, n_peaks=N_PEAKS)
Z_MSUMG = get_transformer_latent(data_msumg, model, n_peaks=N_PEAKS)


# ============================================================
# PREPARE DATASETS
# ============================================================
SPLIT_NAMES = {
    "A": "DRIAMS_A",
    "B": "DRIAMS_B",
    "C": "DRIAMS_C",
    "MARISMA": "MARISMA",
    "RKI": "RKI",
}
 
domains_train = {}
domains_val = {}
test_sets = {}
 
for k, sk in SPLIT_NAMES.items():
    tr_idx = splits["splits_per_domain"][sk]["train_idx"]
    va_idx = splits["splits_per_domain"][sk]["val_idx"]
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
 
    domains_train[k] = (data_final[tr_idx], label_final[tr_idx])
    domains_val[k] = (data_final[va_idx], label_final[va_idx])
    test_sets[k] = (data_final[te_idx], label_final[te_idx])
 
test_sets["D (OOD)"]= (dataD, labelD)
test_sets["MSUMG (OOD)"] = (data_msumg, label_msumg)

# Latent
domains_train_latent = {}
domains_val_latent = {}
test_sets_latent = {}
 
for k, sk in SPLIT_NAMES.items():
    tr_idx = splits["splits_per_domain"][sk]["train_idx"]
    va_idx = splits["splits_per_domain"][sk]["val_idx"]
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
 
    domains_train_latent[k] = (Z_final[tr_idx], label_final[tr_idx])
    domains_val_latent[k] = (Z_final[va_idx], label_final[va_idx])
    test_sets_latent[k] = (Z_final[te_idx], label_final[te_idx])
 
test_sets_latent["D (OOD)"] = (Z_D, labelD)
test_sets_latent["MSUMG (OOD)"] = (Z_MSUMG, label_msumg)
 

# ============================================================
# EVALUATION
# ============================================================
results = []

for train_name in domains_train.keys():
    print("\n" + "="*70)
    print(f"TRAIN DOMAIN: {train_name}")
    print("="*70)

    X_tr, y_tr = domains_train[train_name]
    X_va, y_va = domains_val[train_name]
 
    Z_tr, _ = domains_train_latent[train_name]
    Z_va, _ = domains_val_latent[train_name]

    # ============================
    # LABEL ENCODING
    # ============================
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)
 
    n_classes = len(le.classes_)

    # ============================
    # TRAIN MLP ORIGINAL
    # ============================
    print("\n--- Training MLP (original) ---")
    mlp_orig = MLPClassifier_Extended(
        input_dim=X_tr.shape[1],
        n_species=n_classes,
        epochs=50,
        lr=1e-4,
        patience=10
    )
 
    mlp_orig.trainloop(
        make_loader(X_tr, y_tr_enc, shuffle=True),
        make_loader(X_va, y_va_enc),
        device
    )
 
    # ============================
    # TRAIN MLP LATENT
    # ============================
    print("\n--- Training MLP (latent) ---")
    mlp_lat = MLPClassifier_Extended(
        input_dim=Z_tr.shape[1],
        n_species=n_classes,
        epochs=50,
        lr=1e-3,
        patience=10
    )
 
    mlp_lat.trainloop(
        make_loader(Z_tr, y_tr_enc, shuffle=True),
        make_loader(Z_va, y_va_enc),
        device
    )

    # ============================
    # TEST LOOP
    # ============================
    for test_name in test_sets.keys():
        print("\n" + "="*70)
        print(f"TEST DOMAIN: {test_name}")
        print("="*70)

        X_te, y_te = test_sets[test_name]
        Z_te, _ = test_sets_latent[test_name]
        y_te_enc = le.transform(y_te)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for space, model, X_eval in [
            ("original", mlp_orig, X_te),
            ("latent",   mlp_lat,  Z_te),
        ]:
            for fold_i, (_, idx) in enumerate(skf.split(X_eval, y_te_enc)):
                fold_loader = make_loader(X_eval[idx], y_te_enc[idx])

                metrics = metrics_report_mlp(
                    fold_loader,
                    model,
                    f"{train_name}-{test_name}-{space}-fold{fold_i}",
                    device=device,
                    class_names=le.classes_
                )

                results.append({
                    "train": train_name,
                    "test": test_name,
                    "space": space,
                    "fold": fold_i,
                    "balanced_accuracy": metrics["Balanced_Accuracy"],
                    "f1_macro": metrics["F1_Macro"],
                    "recall_macro": metrics["Recall_Macro"],
                    "specificity_macro": metrics["Specificity_Macro"],
                    "roc_auc": metrics["ROC_AUC_Macro"],
                    "cm": metrics["Confusion Matrix"],
                })


# ============================================================
# SAVE CSV
# ============================================================
print("\n===== SAVING RESULTS =====")
df = pd.DataFrame(results)
OUT_PATH.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(OUT_PATH,index=False)
print("Saved:", OUT_PATH)
