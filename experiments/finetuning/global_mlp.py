# ============================================================
# PATH CONFIGURATION
# ============================================================
import os
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

print("Working directory:", os.getcwd())


# ============================================================
# IMPORTS
# ============================================================
from datetime import datetime

import pickle
import numpy as np
import torch

from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import make_loader, load_model, encode_latent

from models.baselines.mlp import MLPClassifier_Extended


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
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium","Enterobacter_cloacae_complex"
]

# DRIAMS
mask_driams = np.isin(driams_dict["label"], species_to_keep)
data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

# MARISMa
mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]
meta_marisma = marisma_dict["meta"][mask_marisma].reset_index(drop=True)

# MS-UMG
mask_msumg = np.isin(msumg_dict["label"], species_to_keep)
data_msumg = msumg_dict["data"][mask_msumg]
label_msumg = msumg_dict["label"][mask_msumg]
meta_msumg = msumg_dict["meta"][mask_msumg].reset_index(drop=True)

# RKI
mask_rki = np.isin(rki_dict["label"], species_to_keep)
data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]
meta_rki = rki_dict["meta"][mask_rki].reset_index(drop=True)

# Split DRIAMS per hospital
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA]
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB]
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC]
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD]

# Normalize
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_msumg = row_minmax_normalize(data_msumg)
data_rki = row_minmax_normalize(data_rki)

# Build dataset used for VAE training to ensure correct index usage
data_list  = [dataA, dataB, dataC, data_marisma, data_rki]
label_list = [labelA, labelB, labelC, label_marisma, label_rki]
meta_list  = [metaA, metaB, metaC, meta_marisma, meta_rki]

data_final = np.vstack(data_list)
label_final = np.concatenate(label_list)

print("\n===== DATA LOADED =====")


# ============================================================
# LOAD SPLITS
# ============================================================
experiment_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")

with open(experiment_dir / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

print("\n===== SPLITS LOADED =====")


# ============================================================
# VAE
# ============================================================
print("\n" + "="*60)
print("===== LOADING PRETRAINED VAE =====")
print("="*60)

from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended

data_seen  = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_seen = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])

vae = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=data_seen.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species=len(np.unique(label_seen))
    ),
    experiment_dir / "model.pth"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== LATENT ENCODING =====")
Z_final = encode_latent(vae, data_final, device)
Z_D  = encode_latent(vae, dataD, device)
Z_MSUMG = encode_latent(vae, data_msumg, device)


# ============================================================
# BUILD TRAIN / VAL FROM SPLITS (using train_idx + val_idx)
# ============================================================
print("\n" + "="*60)
print("===== BUILDING TRAIN / VAL FROM SPLITS =====")
print("="*60)

SPLIT_NAMES = {
    "A": "DRIAMS_A",
    "B": "DRIAMS_B",
    "C": "DRIAMS_C",
    "MARISMA": "MARISMA",
    "RKI": "RKI",
}
seen_keys = list(SPLIT_NAMES.keys())

# Aggregate train indices across seen domains (original space)
X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []

# Same for latent space (mismos índices, distinto espacio)
Z_train_list = []
Z_val_list   = []

for k in seen_keys:
    tr_idx = splits["splits_per_domain"][SPLIT_NAMES[k]]["train_idx"]
    va_idx = splits["splits_per_domain"][SPLIT_NAMES[k]]["val_idx"]

    X_train_list.append(data_final[tr_idx])
    y_train_list.append(label_final[tr_idx])
    X_val_list.append(data_final[va_idx])
    y_val_list.append(label_final[va_idx])

    Z_train_list.append(Z_final[tr_idx])
    Z_val_list.append(Z_final[va_idx])

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
X_val = np.vstack(X_val_list)
y_val = np.concatenate(y_val_list)
Z_train = np.vstack(Z_train_list)
Z_val = np.vstack(Z_val_list)

print(f"Train samples: {len(y_train)}")
print(f"Val samples: {len(y_val)}")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc   = le.transform(y_val)
n_classes = len(le.classes_)


# ============================================================
# TRAINING
# ============================================================
print("\n" + "="*60)
print("===== TRAINING MLPs =====")
print("="*60)

print("\nTRAINING MLP IN ORIGINAL SPACE")
mlp_orig = MLPClassifier_Extended(
    input_dim=X_train.shape[1],
    n_species=n_classes,
    epochs=50,
    lr=1e-4,
    patience=10
)

mlp_orig.trainloop(
    make_loader(X_train, y_train_enc, shuffle=True),
    make_loader(X_val,   y_val_enc),
    device
)

print("\nTRAINING MLP IN LATENT SPACE")
mlp_lat = MLPClassifier_Extended(
    input_dim=Z_train.shape[1],
    n_species=n_classes,
    epochs=50,
    lr=1e-3,
    patience=10
)

mlp_lat.trainloop(
    make_loader(Z_train, y_train_enc, shuffle=True),
    make_loader(Z_val,   y_val_enc),
    device
)


# ============================================================
# EVALUATION
# ============================================================
print("\n" + "="*60)
print("===== EVALUATING MLPs ON TEST SPLITS =====")
print("="*60)

test_sets = {
    "A":       (data_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]]),
    "B":       (data_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]]),
    "C":       (data_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]]),
    "MARISMA": (data_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
                label_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]]),
    "RKI":     (data_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
                label_final[splits["splits_per_domain"]["RKI"]["test_idx"]]),
    "D (OOD)":     (dataD, labelD),
    "MSUMG (OOD)": (data_msumg, label_msumg),
}

test_sets_latent = {
    "A":       (Z_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]]),
    "B":       (Z_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]]),
    "C":       (Z_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
                label_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]]),
    "MARISMA": (Z_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
                label_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]]),
    "RKI":     (Z_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
                label_final[splits["splits_per_domain"]["RKI"]["test_idx"]]),
    "D (OOD)":     (Z_D,     labelD),
    "MSUMG (OOD)": (Z_MSUMG, label_msumg),
}

results = []
for test_name in test_sets.keys():
    print("\n" + "="*70)
    print(f"TEST DOMAIN: {test_name}")
    print("="*70)

    X_te, y_te = test_sets[test_name]
    Z_te, _    = test_sets_latent[test_name]
    y_te_enc   = le.transform(y_te)

    test_loader_orig = make_loader(X_te, y_te_enc)
    test_loader_lat  = make_loader(Z_te, y_te_enc)

    for space, model, loader in [
        ("original", mlp_orig, test_loader_orig),
        ("latent",   mlp_lat,  test_loader_lat),
    ]:
        metrics = metrics_report_mlp(
            loader,
            model,
            f"GLOBAL-{test_name}-{space}",
            device=device,
            class_names=le.classes_
        )

        results.append({
            "train": "GLOBAL",
            "test": test_name,
            "space": space,
            "balanced_accuracy": metrics["Balanced_Accuracy"],
            "f1_macro": metrics["F1_Macro"],
            "recall_macro": metrics["Recall_Macro"],
            "specificity_macro": metrics["Specificity_Macro"],
        })

print("\n===== EVALUATION COMPLETE =====")


# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "="*60)
print("===== SAVING MODELS =====")
print("="*60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_mlp") / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

path_mlp_orig = output_dir / "mlp_original_ABC_MAR_RKI.pth"
path_mlp_lat  = output_dir / "mlp_latent_ABC_MAR_RKI.pth"

torch.save(mlp_orig.state_dict(), path_mlp_orig)
torch.save(mlp_lat.state_dict(),  path_mlp_lat)

print("Saved:", path_mlp_orig)
print("Saved:", path_mlp_lat)
