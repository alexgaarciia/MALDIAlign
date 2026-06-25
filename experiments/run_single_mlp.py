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
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import make_loader
from models.baselines.mlp import MLPClassifier_Extended

# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = Path(f"experiments/results/classifiers/mlp/{timestamp}/mlp_raw_baseline.csv")
SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Output: {OUT_PATH}")

# ============================================================
# DATA LOADING
# ============================================================
cfg = load_config()
driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])

species_to_keep = [
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

mask_driams = np.isin(driams_dict["label"], species_to_keep)
data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]

mask_rki = np.isin(rki_dict["label"], species_to_keep)
data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]

mask_msumg = np.isin(msumg_dict["label"], species_to_keep)
data_msumg = msumg_dict["data"][mask_msumg]
label_msumg = msumg_dict["label"][mask_msumg]

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
data_final = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])

with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

SPLIT_NAMES = {
    "A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C",
    "MARISMA": "MARISMA", "RKI": "RKI",
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

test_sets["D (OOD)"] = (dataD, labelD)
test_sets["MSUMG (OOD)"] = (data_msumg, label_msumg)

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

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)
    n_classes = len(le.classes_)

    print("\n--- Training MLP (raw spectra baseline) ---")
    mlp_orig = MLPClassifier_Extended(
        input_dim=X_tr.shape[1], n_species=n_classes,
        epochs=50, lr=1e-3, patience=10,
    )
    mlp_orig.trainloop(
        make_loader(X_tr, y_tr_enc, shuffle=True),
        make_loader(X_va, y_va_enc),
        device,
    )

    for test_name in test_sets.keys():
        X_te, y_te = test_sets[test_name]
        y_te_enc = le.transform(y_te)
        test_loader = make_loader(X_te, y_te_enc)

        metrics = metrics_report_mlp(
            test_loader, mlp_orig, f"{train_name}-{test_name}-raw",
            device=device, class_names=le.classes_,
        )

        results.append({
            "train": train_name, "test": test_name, "space": "original",
            "balanced_accuracy": metrics["Balanced_Accuracy"],
            "f1_macro": metrics["F1_Macro"],
            "recall_macro": metrics["Recall_Macro"],
            "specificity_macro": metrics["Specificity_Macro"],
            "roc_auc": metrics["ROC_AUC_Macro"],
            "cm": metrics["Confusion Matrix"],
        })

print("\n" + "="*70)
print("TRAIN DOMAIN: ALL (A+B+C+MARISMA+RKI pooled)")
print("="*70)

all_tr_idx = np.concatenate([
    splits["splits_per_domain"][sk]["train_idx"]
    for sk in SPLIT_NAMES.values() if sk in splits["splits_per_domain"]
])
all_va_idx = np.concatenate([
    splits["splits_per_domain"][sk]["val_idx"]
    for sk in SPLIT_NAMES.values() if sk in splits["splits_per_domain"]
])

X_tr_pool = data_final[all_tr_idx]
y_tr_pool = label_final[all_tr_idx]
X_va_pool = data_final[all_va_idx]
y_va_pool = label_final[all_va_idx]

le_pool = LabelEncoder()
y_tr_pool_enc = le_pool.fit_transform(y_tr_pool)
y_va_pool_enc = le_pool.transform(y_va_pool)
n_classes_pool = len(le_pool.classes_)

print(f"Pooled train: {len(X_tr_pool)} | Pooled val: {len(X_va_pool)}")

mlp_orig_pool = MLPClassifier_Extended(
    input_dim=X_tr_pool.shape[1], n_species=n_classes_pool,
    epochs=50, lr=1e-3, patience=10,
)
mlp_orig_pool.trainloop(
    make_loader(X_tr_pool, y_tr_pool_enc, shuffle=True),
    make_loader(X_va_pool, y_va_pool_enc),
    device,
)

for test_name in test_sets.keys():
    X_te, y_te = test_sets[test_name]
    y_te_enc = le_pool.transform(y_te)
    test_loader = make_loader(X_te, y_te_enc)

    metrics = metrics_report_mlp(
        test_loader, mlp_orig_pool, f"ALL-{test_name}-raw",
        device=device, class_names=le_pool.classes_,
    )

    results.append({
        "train": "ALL", "test": test_name, "space": "original",
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
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)
