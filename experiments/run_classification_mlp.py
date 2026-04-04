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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import load_model, encode_latent, make_loader

from models.baselines.mlp import MLPClassifier_Extended


# ============================================================
# ARCHITECTURE TO TEST
# ============================================================
ARCHITECTURE = "coral"

if ARCHITECTURE == "vae":
    EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_bernoulli/20260403_204619")
    OUT_PATH = Path("experiments/results/classifiers/mlp/mlp_vae.csv")
elif ARCHITECTURE == "vae_multidecoder_prior":
    EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260404_062545")
    OUT_PATH = Path("experiments/results/classifiers/mlp/mlp_vae_multidecoder_prior.csv")
elif ARCHITECTURE == "dann":
    EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/dann/20260403_222152")
    OUT_PATH = Path("experiments/results/classifiers/mlp/mlp_dann.csv")
elif ARCHITECTURE == "coral":
    EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/coral/????'")
    OUT_PATH = Path("experiments/results/classifiers/mlp/mlp_coral.csv")



# ============================================================
# DATA LOADING
# ============================================================
cfg = load_config()

driams_dict = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
rki_dict = load_rki(cfg["data"]["RKI_PKL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_PKL"])

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

# Load splits
with open(EXPERIMENT_DIR / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

# Train / Test
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
from models.deep.VAEBernoulli import VAE_Bernoulli_Extended
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.deep.DANN import DANNFull_Extended
from models.deep.MultiVAECoral import MultiVAE_CORAL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== LATENT ENCODING =====")

if ARCHITECTURE == "vae_multidecoder_prior":
    vae = load_model(
        MultiVAE_Bernoulli_SpeciesPrior_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64,
            num_domains=5,
            n_species=len(np.unique(label_final))),
        EXPERIMENT_DIR / "model.pth"
    )

    Z_final = encode_latent(vae, data_final, device)
    Z_D = encode_latent(vae, dataD, device)
    Z_MSUMG = encode_latent(vae, data_msumg, device)

elif ARCHITECTURE == "vae":
    vae = load_model(
        VAE_Bernoulli_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64),
        EXPERIMENT_DIR / "model.pth"
    )

    Z_final = encode_latent(vae, data_final, device)
    Z_D = encode_latent(vae, dataD, device)
    Z_MSUMG = encode_latent(vae, data_msumg, device)

elif ARCHITECTURE == "dann":
    vae = load_model(DANNFull_Extended(
        input_dim=data_final.shape[1],
        latent_dim=64,
        n_species=len(np.unique(label_final)),
        n_domains=5), 
        EXPERIMENT_DIR / "model.pth")
    
    Z_final = encode_latent(vae, data_final, device)
    Z_D = encode_latent(vae, dataD, device)
    Z_MSUMG = encode_latent(vae, data_msumg, device)

elif ARCHITECTURE == "coral":
    vae = load_model(
        MultiVAE_CORAL(
            input_dim=data_final.shape[1],
            latent_dim=64,
            num_domains=5   
        ),
        EXPERIMENT_DIR / "model.pth"
    )

    Z_final = encode_latent(vae, data_final, device)
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
    )
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


# ============================================================
# EVALUATION
# ============================================================
results = []

for train_name in domains_train.keys():
    print("\n" + "="*70)
    print(f"TRAIN DOMAIN: {train_name}")
    print("="*70)

    X_tr, y_tr = domains_train[train_name]
    Z_tr, _ = domains_train_latent[train_name]

    # ============================
    # LABEL ENCODING
    # ============================
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    n_classes = len(le.classes_)

    # ============================
    # TRAIN MLP ORIGINAL
    # ============================
    print("\n--- Training MLP (original) ---")
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr_enc, test_size=0.1, stratify=y_tr_enc, random_state=42
    )

    mlp_orig = MLPClassifier_Extended(
        input_dim=X_tr.shape[1],
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

    # ============================
    # TRAIN MLP LATENT
    # ============================
    print("\n--- Training MLP (latent) ---")
    Z_train, Z_val, y_train, y_val = train_test_split(
        Z_tr, y_tr_enc, test_size=0.1, stratify=y_tr_enc, random_state=42
    )

    mlp_lat = MLPClassifier_Extended(
        input_dim=Z_tr.shape[1],
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

        test_loader_orig = make_loader(X_te, y_te_enc)
        test_loader_lat  = make_loader(Z_te, y_te_enc)

        for space, model, loader in [
            ("original", mlp_orig, test_loader_orig), 
            ("latent", mlp_lat, test_loader_lat)]:
            
            metrics = metrics_report_mlp(
                loader,
                model,
                f"{train_name}-{test_name}-{space}",
                device=device,
                class_names=le.classes_
            )

            results.append({
                "train": train_name,
                "test": test_name,
                "space": space,
                "balanced_accuracy": metrics["Balanced_Accuracy"],
                "f1_macro": metrics["F1_Macro"],
                "recall_macro": metrics["Recall_Macro"],
                "specificity_macro": metrics["Specificity_Macro"]
            })
    

# ============================================================
# SAVE CSV
# ============================================================
print("\n===== SAVING RESULTS =====")
df = pd.DataFrame(results)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("Saved:", OUT_PATH)
