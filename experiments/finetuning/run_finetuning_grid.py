############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys
import json
import joblib
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


############################################################
# IMPORTS
############################################################
import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config.loader import load_config
from src.data.io import load_pkl
from src.data.datasets import (
    load_driams,
    load_marisma,
    load_msumg,
    load_rki)
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.metrics import metrics_report
from experiments.finetuning.run_finetuning import run_finetuning
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


############################################################
# LOAD DATASETS
############################################################
print("\n===== LOADING DATA =====")

cfg = load_config()

# Load full datasets
driams_dict  = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
msumg_dict   = load_msumg(cfg["data"]["MSUMG_PKL"])
rki_dict     = load_rki(cfg["data"]["RKI_PKL"])

data_driams, label_driams, meta_driams = driams_dict["data"], driams_dict["label"], driams_dict["meta"]
data_marisma, label_marisma, meta_marisma = marisma_dict["data"], marisma_dict["label"], marisma_dict["meta"]
data_rki, label_rki, meta_rki = rki_dict["data"], rki_dict["label"], rki_dict["meta"]
data_msumg, label_msumg, meta_msumg = msumg_dict["data"], msumg_dict["label"], msumg_dict["meta"]

# Split DRIAMS by hospital
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA]
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB]
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC]
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD]

# Row-wise normalization
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)

print("\n===== DATA LOADED =====")


############################################################
# LOAD PRETRAINED BASELINE MODELS
############################################################
print("\n===== LOADING PRETRAINED MODELS =====")

PATH_RF_ORIGINAL = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_rf/rf_original_ABC_MAR_RKI.joblib")
PATH_RF_LATENT   = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_rf/rf_latent_ABC_MAR_RKI.joblib")
PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/finetuning_vae_multidecoder_prior/20260211_143045/model.pth")
OUTPUT_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/results")

baseline_rf_original = joblib.load(PATH_RF_ORIGINAL)
baseline_rf_latent   = joblib.load(PATH_RF_LATENT)

vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_Extended(
    input_dim=dataD.shape[1],
    latent_dim=64,
    num_domains=5,
    n_species=len(np.unique(labelD))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_pretrained.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
vae_pretrained.to(device)
vae_pretrained.eval()

print("\n===== BASELINE MODELS LOADED =====")


############################################################
# GRID SETUP
############################################################
SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data/splits_20260217_110011")

grid_prev = np.arange(0, 251, 50)
grid_new  = np.arange(50, 251, 50)

results = []


############################################################
# GRID EVALUATION LOOP (DRIAMS-D ONLY)
############################################################
for n_prev in grid_prev:
    for n_new in grid_new:

        print(f"\n--- Running n_prev={n_prev}, n_new={n_new} ---")

        # --------------------------------------------------
        # Load split indices
        # --------------------------------------------------
        split_file = SPLITS_PATH / f"prev_{n_prev}_new_{n_new}.pkl"
        splits = load_pkl(split_file)
        idx_test, idx_ft = splits["DRIAMS_D"]["test"], splits["DRIAMS_D"]["finetuning"]

        # --------------------------------------------------
        # Build TEST and FINETUNING set (fixed evaluation set)
        # --------------------------------------------------
        X_test, y_test = dataD[idx_test], labelD[idx_test]
        X_ft, y_ft = dataD[idx_ft], labelD[idx_ft]

        # ==================================================
        # A. BASELINE RF (ORIGINAL SPACE)
        # ==================================================
        metrics_orig = metrics_report(
            X_test, y_test,
            baseline_rf_original,
            "DRIAMS_D",
            np.unique(labelD)
        )

        # ==================================================
        # B. BASELINE RF (LATENT SPACE) ZERO-SHOT
        # ==================================================
        Z_test_zero = encode_latent(vae_pretrained, X_test, device)
        metrics_lat = metrics_report(
            Z_test_zero, y_test,
            baseline_rf_latent,
            "DRIAMS_D",
            np.unique(labelD)
        )

        # ==================================================
        # C. RF TRAINED ONLY ON FEW-SHOT TARGET DATA
        # ==================================================
        rf_few = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        )

        rf_few.fit(X_ft, y_ft)

        metrics_few = metrics_report(
            X_test, y_test,
            rf_few,
            "DRIAMS_D",
            np.unique(labelD)
        )

        # ==================================================
        # D. FINETUNING (FULL)
        # ==================================================
        vae_full, species_encoder_full, DOMAIN_MAP_full = run_finetuning(
            splits_path=split_file,
            target_domain="DRIAMS_D",
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            finetuning_mode="full",
            n_prev=n_prev,
            n_new=n_new,
            output_dir=OUTPUT_PATH,
            device=device,
            consider_prev_domains=(n_prev > 0)
        )

        # Encode test set with finetuned model
        Z_test_full = encode_latent(vae_full, X_test, device)

        # Evaluate with pretrained latent RF (NO retraining)
        metrics_ft_full = metrics_report(
            Z_test_full,
            y_test,
            baseline_rf_latent,
            "DRIAMS_D",
            np.unique(labelD)
        )

        # ==================================================
        # E. FINETUNING (FREEZE PRIORS)
        # ==================================================
        vae_freeze, species_encoder_freeze, DOMAIN_MAP_freeze = run_finetuning(
            splits_path=split_file,
            target_domain="DRIAMS_D",
            pretrained_model_path=PRETRAINED_MODEL_PATH,
            finetuning_mode="freeze_priors",
            n_prev=n_prev,
            n_new=n_new,
            output_dir=OUTPUT_PATH,
            device=device,
            consider_prev_domains=(n_prev > 0)
        )

        # Encode test set with freeze-priors model
        Z_test_freeze = encode_latent(vae_freeze, X_test, device)

        # Evaluate using pretrained latent RF
        metrics_ft_freeze = metrics_report(
            Z_test_freeze,
            y_test,
            baseline_rf_latent,
            "DRIAMS_D",
            np.unique(labelD)
        )

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------
        for model_name, metrics in [
            ("RF_original",        metrics_orig),
            ("RF_latent_zero",     metrics_lat),
            ("RF_few",             metrics_few),
            ("FT_full",            metrics_ft_full),
            ("FT_freeze_priors",   metrics_ft_freeze),
        ]:
            results.append({
                "n_prev": n_prev,
                "n_new": n_new,
                "model": model_name,
                "balanced_accuracy": metrics["Balanced_Accuracy"],
                "f1_macro": metrics["F1_Macro"],
                "recall_macro": metrics["Recall_Macro"],
                "specificity_macro": metrics["Specificity_Macro"],
                "confusion_matrix": json.dumps(metrics["Confusion Matrix"].tolist())
            })


############################################################
# SAVE RESULTS
############################################################
df_results = pd.DataFrame(results)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUTPUT_PATH / f"grid_results_{stamp}.csv"

df_results.to_csv(csv_path, index=False)

print("\n===== GRID RESULTS SAVED =====")
print(f"Saved to: {csv_path}")

print("\n===== GRID EVALUATION FINISHED =====")
