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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import make_loader

from models.baselines.mlp import MLPClassifier_Extended


# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "="*60)
print("===== LOADING DATA =====")
print("="*60)

cfg = load_config()

driams_dict = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
rki_dict = load_rki(cfg["data"]["RKI_PKL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_PKL"])

species_to_keep = ["Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus","Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"]

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

# Split DRIAMS por hospital 
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
data_list = [
    dataA,
    dataB,
    dataC,
    data_marisma,
    data_rki
]

label_list = [
    labelA,
    labelB,
    labelC,
    label_marisma,
    label_rki
]

meta_list = [
    metaA,
    metaB,
    metaC,
    meta_marisma,
    meta_rki
]

data_final = np.vstack(data_list)
label_final = np.concatenate(label_list)

print("\n===== DATA LOADED =====")


# ============================================================
# LOAD SPLITS & DEFINE DOMAINS
# ============================================================
# Load splits
experiment_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260401_162208")

with open(experiment_dir / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

print("\n===== SPLITS LOADED =====")

# Build train tests
domains_train = {
    "A": (
        data_final[splits["splits_per_domain"]["DRIAMS_A"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_A"]["train_idx"]],
    ),
    "B": (
        data_final[splits["splits_per_domain"]["DRIAMS_B"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_B"]["train_idx"]],
    ),
    "C": (
        data_final[splits["splits_per_domain"]["DRIAMS_C"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_C"]["train_idx"]],
    ),
    "MARISMA": (
        data_final[splits["splits_per_domain"]["MARISMA"]["train_idx"]],
        label_final[splits["splits_per_domain"]["MARISMA"]["train_idx"]],
    ),
    "RKI": (
        data_final[splits["splits_per_domain"]["RKI"]["train_idx"]],
        label_final[splits["splits_per_domain"]["RKI"]["train_idx"]],
    )
}

# Build test sets
test_sets = {
    "A": (
        data_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
    ),
    "B": (
        data_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
    ),
    "C": (
        data_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
    ),
    "MARISMA": (
        data_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
        label_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
    ),
    "RKI": (
        data_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
        label_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
    ),

    # OOD
    "D (OOD)": (dataD, labelD),
    "MSUMG (OOD)": (data_msumg, label_msumg),
}


# ============================================================
# VAE
# ============================================================
print("\n" + "="*60)
print("===== LOADING PRETRAINED VAE =====")
print("="*60)

from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended

def load_model(model, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def encode_latent(model, X, device, batch_size=256):
    Z = []
    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(X_tensor, batch_size=batch_size)

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            Z.append(mu.cpu().numpy())

    return np.vstack(Z)

# VAE entrenado SOLO en A+B+C+MARISMA+RKI (sin D ni MSUMG)
data_seen = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_seen = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])
vae = load_model(MultiVAE_Bernoulli_SpeciesPrior_Extended(input_dim=data_seen.shape[1], latent_dim=64, num_domains=5, n_species=len(np.unique(label_seen))), experiment_dir / "model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== LATENT ENCODING =====")

# Seen domains
Z_final = encode_latent(vae, data_final, device)

# OOD
Z_D = encode_latent(vae, dataD, device)
Z_MSUMG = encode_latent(vae, data_msumg, device)

domains_train_latent = {
    "A": (
        Z_final[splits["splits_per_domain"]["DRIAMS_A"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_A"]["train_idx"]],
    ),
    "B": (
        Z_final[splits["splits_per_domain"]["DRIAMS_B"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_B"]["train_idx"]],
    ),
    "C": (
        Z_final[splits["splits_per_domain"]["DRIAMS_C"]["train_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_C"]["train_idx"]],
    ),
    "MARISMA": (
        Z_final[splits["splits_per_domain"]["MARISMA"]["train_idx"]],
        label_final[splits["splits_per_domain"]["MARISMA"]["train_idx"]],
    ),
    "RKI": (
        Z_final[splits["splits_per_domain"]["RKI"]["train_idx"]],
        label_final[splits["splits_per_domain"]["RKI"]["train_idx"]],
    ),

    "D (OOD train)": (Z_D, labelD),
    "MSUMG (OOD train)": (Z_MSUMG, label_msumg),
}

test_sets_latent = {
    "A": (
        Z_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_A"]["test_idx"]],
    ),
    "B": (
        Z_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_B"]["test_idx"]],
    ),
    "C": (
        Z_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
        label_final[splits["splits_per_domain"]["DRIAMS_C"]["test_idx"]],
    ),
    "MARISMA": (
        Z_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
        label_final[splits["splits_per_domain"]["MARISMA"]["test_idx"]],
    ),
    "RKI": (
        Z_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
        label_final[splits["splits_per_domain"]["RKI"]["test_idx"]],
    ),

    "D (OOD)": (Z_D, labelD),
    "MSUMG (OOD)": (Z_MSUMG, label_msumg),
}

seen_keys = ["A", "B", "C", "MARISMA", "RKI"]
X_all_train = np.vstack([domains_train[k][0] for k in seen_keys])
y_all_train = np.concatenate([domains_train[k][1] for k in seen_keys])
Z_all_train = np.vstack([domains_train_latent[k][0] for k in seen_keys])

print("\n===== ENCODING COMPLETE =====")


# ============================================================
#  TRAINING
# ============================================================
print("\n" + "="*60)
print("===== TRAINING MLPs =====")
print("="*60)

le = LabelEncoder()
y_all_train_enc = le.fit_transform(y_all_train)
n_classes = len(le.classes_)

print("\nTRAINING MLP IN ORIGINAL SPACE")
X_train, X_val, y_train, y_val = train_test_split(
    X_all_train,
    y_all_train_enc,
    test_size=0.1,
    stratify=y_all_train_enc,
    random_state=42
)


mlp_orig = MLPClassifier_Extended(
    input_dim=X_all_train.shape[1],
    n_species=n_classes,
    epochs=50,
    lr=1e-4,
    patience=10
)

mlp_orig.trainloop(
    make_loader(X_train, y_train, shuffle=True),
    make_loader(X_val, y_val),
    device
)

print("\nTRAINING MLP IN LATENT SPACE")
Z_train, Z_val, y_train, y_val = train_test_split(
    Z_all_train,
    y_all_train_enc,
    test_size=0.1,
    stratify=y_all_train_enc,
    random_state=42
)

mlp_lat = MLPClassifier_Extended(
    input_dim=Z_all_train.shape[1],
    n_species=n_classes,
    epochs=50,
    lr=1e-3,
    patience=10
)

mlp_lat.trainloop(
    make_loader(Z_train, y_train, shuffle=True),
    make_loader(Z_val, y_val),
    device
)

print("\nStarting evaluation...")
results = []

for test_name in test_sets.keys():
    print("\n" + "="*70)
    print(f"TEST DOMAIN: {test_name}")
    print("="*70)

    X_te, y_te = test_sets[test_name]
    Z_te, _ = test_sets_latent[test_name]

    y_te_enc = le.transform(y_te)

    test_loader_orig = make_loader(X_te, y_te_enc)
    test_loader_lat  = make_loader(Z_te, y_te_enc)

    for space, model, loader in [
        ("original", mlp_orig, test_loader_orig),
        ("latent", mlp_lat, test_loader_lat)
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
            "specificity_macro": metrics["Specificity_Macro"]
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
torch.save(mlp_lat.state_dict(), path_mlp_lat)

print("Saved:", path_mlp_orig)
print("Saved:", path_mlp_lat)
