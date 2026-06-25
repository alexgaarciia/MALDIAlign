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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.io import load_pkl
from src.data.datasets import load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.metrics import metrics_report, metrics_report_mlp
from experiments.finetuning.run_finetuning import run_finetuning
from src.evaluation.eval import encode_latent, make_loader
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended


############################################################
# LOAD DATASETS
############################################################
print("\n===== LOADING DATA =====")

cfg = load_config()

TARGET_SPECIES = ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus", "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"]
le = LabelEncoder()
le.fit(TARGET_SPECIES)

# Load MS-UMG
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])
mask_sp = np.isin(msumg_dict["label"], TARGET_SPECIES)
data_msumg = msumg_dict["data"][mask_sp]
label_msumg = msumg_dict["label"][mask_sp]
meta_msumg = msumg_dict["meta"][mask_sp].reset_index(drop=True)
data_msumg = row_minmax_normalize(data_msumg)

print(f"MS-UMG samples (filtered by species): {len(data_msumg)}")
print("\n===== DATA LOADED =====")


############################################################
# LOAD PRETRAINED BASELINE MODELS
############################################################
print("\n===== LOADING PRETRAINED MODELS =====")

PATH_RF_ORIGINAL = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_rf/20260409_113358/rf_original_ABC_MAR_RKI.joblib")
PATH_RF_LATENT = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_rf/20260409_113358/rf_latent_ABC_MAR_RKI.joblib")
PRETRAINED_MLP_ORIGINAL = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_mlp/20260525_114715/mlp_original_ABC_MAR_RKI.pth")
PRETRAINED_MLP_LATENT = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/pretrained_mlp/20260525_114715/mlp_latent_ABC_MAR_RKI.pth")
PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/model.pth")

baseline_rf_original = joblib.load(PATH_RF_ORIGINAL)
baseline_rf_latent   = joblib.load(PATH_RF_LATENT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

baseline_mlp_original = MLPClassifier_Extended(input_dim=data_msumg.shape[1], n_species=len(TARGET_SPECIES), epochs=50, lr=1e-4, patience=10)
baseline_mlp_original.load_state_dict(torch.load(PRETRAINED_MLP_ORIGINAL))

baseline_mlp_latent = LinearProbe_Extended(latent_dim=64, n_species=len(TARGET_SPECIES), epochs=50, lr=1e-4, patience=10)
baseline_mlp_latent.load_state_dict(torch.load(PRETRAINED_MLP_LATENT))

baseline_mlp_original.to(device)
baseline_mlp_latent.to(device)

baseline_mlp_original.eval()
baseline_mlp_latent.eval()

vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_Extended(input_dim=data_msumg.shape[1], latent_dim=64, num_domains=5, n_species=len(TARGET_SPECIES))
vae_pretrained.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
vae_pretrained.to(device)
vae_pretrained.eval()

print("\n===== BASELINE MODELS LOADED =====")


############################################################
# GRID SETUP
############################################################
SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data/splits_20260525_100756")

N_PARTITIONS = 10
grid_prev = [0]
grid_new  = np.arange(50, 251, 50)

OUTPUT_PATH = Path("/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/finetuning_6species")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
RUN_LATENT_EVALUATION = False

results = []


############################################################
# GRID EVALUATION LOOP (MS-UMG ONLY)
############################################################
for i_part in range(N_PARTITIONS):
    print(f"\n{'#'*70}")
    print(f"PARTITION {i_part + 1}/{N_PARTITIONS}")
    print(f"{'#'*70}")
    
    partition_dir = SPLITS_PATH / f"run_{i_part}"

    for n_prev in grid_prev:
        for n_new in grid_new:
            print(f"\n--- Partition {i_part} | n_prev={n_prev}, n_new={n_new} ---")

            ############################################################
            # Load split indices
            ############################################################
            split_file = partition_dir / f"prev_{n_prev}_new_{n_new}.pkl"
            
            if not split_file.exists():
                print(f"Split file not found: {split_file}. Skipping.")
                continue
            
            splits = load_pkl(split_file)
            idx_test, idx_ft = splits["MS-UMG"]["test"], splits["MS-UMG"]["finetuning"]

            ############################################################
            # Build TEST and FINETUNING set 
            ############################################################
            X_test, y_test = data_msumg[idx_test], label_msumg[idx_test]
            X_ft, y_ft = data_msumg[idx_ft], label_msumg[idx_ft]

            y_test_enc = le.transform(y_test)
            y_ft_enc = le.transform(y_ft)
            counts_ft = np.bincount(y_ft_enc)

            test_loader_orig = make_loader(X_test, y_test_enc)

            ############################################################
            # TRAIN FEW-SHOT MODELS 
            ############################################################
            # RF few-shot
            rf_few = RandomForestClassifier(
                n_estimators=200, max_depth=20,
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=42
            )
            rf_few.fit(X_ft, y_ft)
    
            # MLP few-shot
            X_ft_tr, X_ft_val, y_ft_tr, y_ft_val = train_test_split(
                X_ft, y_ft_enc,
                test_size=0.2,
                stratify=y_ft_enc if np.all(counts_ft >= 2) else None,
                random_state=42
            )
            mlp_few_orig = MLPClassifier_Extended(
                input_dim=X_ft.shape[1], n_species=len(le.classes_),
                epochs=50, lr=1e-4, patience=10
            )
            mlp_few_orig.trainloop(
                make_loader(X_ft_tr, y_ft_tr, shuffle=True),
                make_loader(X_ft_val, y_ft_val),
                device
            )

            ############################################################
            # FINETUNING VAEs (once per grid cell)
            ############################################################
            vae_full, _, _ = run_finetuning(
                splits_path=split_file, target_domain="MS-UMG",
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                finetuning_mode="full", n_prev=n_prev, n_new=n_new,
                output_dir=OUTPUT_PATH, device=device,
                consider_prev_domains=(n_prev > 0),
                run_latent_evaluation=RUN_LATENT_EVALUATION,
            )
    
            vae_freeze, _, _ = run_finetuning(
                splits_path=split_file, target_domain="MS-UMG",
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                finetuning_mode="freeze_priors", n_prev=n_prev, n_new=n_new,
                output_dir=OUTPUT_PATH, device=device,
                consider_prev_domains=(n_prev > 0),
                run_latent_evaluation=RUN_LATENT_EVALUATION,
            )
    
            vae_dec, _, _ = run_finetuning(
                splits_path=split_file, target_domain="MS-UMG",
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                finetuning_mode="decoder_only", n_prev=n_prev, n_new=n_new,
                output_dir=OUTPUT_PATH, device=device,
                consider_prev_domains=False,
                run_latent_evaluation=RUN_LATENT_EVALUATION,
            )
    
            vae_partial, _, _ = run_finetuning(
                splits_path=split_file, target_domain="MS-UMG",
                pretrained_model_path=PRETRAINED_MODEL_PATH,
                finetuning_mode="partial_encoder", n_prev=n_prev, n_new=n_new,
                output_dir=OUTPUT_PATH, device=device,
                consider_prev_domains=False,
                run_latent_evaluation=RUN_LATENT_EVALUATION,
            )

            ############################################################
            # Encode test set once per model
            ############################################################
            Z_test_zero = encode_latent(vae_pretrained, X_test, device)
            Z_test_full = encode_latent(vae_full, X_test, device)
            Z_test_freeze = encode_latent(vae_freeze, X_test, device)
            Z_test_dec = encode_latent(vae_dec, X_test, device)
            Z_test_partial = encode_latent(vae_partial, X_test, device)
    
            test_loader_zero = make_loader(Z_test_zero, y_test_enc)
            test_loader_full = make_loader(Z_test_full, y_test_enc)
            test_loader_freeze = make_loader(Z_test_freeze, y_test_enc)
            test_loader_dec = make_loader(Z_test_dec, y_test_enc)
            test_loader_partial = make_loader(Z_test_partial, y_test_enc)

            ############################################################
            # EVALUATION
            ############################################################
            # Baseline RF/MLP in original space
            metrics_orig = metrics_report(X_test, y_test, baseline_rf_original, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_mlp_orig = metrics_report_mlp(test_loader_orig, baseline_mlp_original, "MS-UMG", device=device, class_names=le.classes_)
    
            # Baseline latent (zero-shot)
            metrics_lat = metrics_report(Z_test_zero, y_test, baseline_rf_latent, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_mlp_lat = metrics_report_mlp(test_loader_zero, baseline_mlp_latent, "MS-UMG", device=device, class_names=le.classes_)
    
            # Few-shot on target
            metrics_few = metrics_report(X_test, y_test, rf_few, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_mlp_few_orig = metrics_report_mlp(test_loader_orig, mlp_few_orig, "MS-UMG", device=device, class_names=le.classes_)
    
            # Finetuning full
            metrics_ft_full= metrics_report(Z_test_full, y_test, baseline_rf_latent, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_ft_full_mlp = metrics_report_mlp(test_loader_full, baseline_mlp_latent, "MS-UMG", device=device, class_names=le.classes_)
    
            # Finetuning freeze_priors
            metrics_ft_freeze = metrics_report(Z_test_freeze, y_test, baseline_rf_latent, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_ft_freeze_mlp = metrics_report_mlp(test_loader_freeze, baseline_mlp_latent, "MS-UMG", device=device, class_names=le.classes_)
    
            # Finetuning decoder_only
            metrics_ft_dec = metrics_report(Z_test_dec, y_test, baseline_rf_latent, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_ft_dec_mlp = metrics_report_mlp(test_loader_dec, baseline_mlp_latent, "MS-UMG", device=device, class_names=le.classes_)
    
            # Finetuning partial_encoder
            metrics_ft_partial = metrics_report(Z_test_partial, y_test, baseline_rf_latent, "MS-UMG", np.unique(TARGET_SPECIES))
            metrics_ft_partial_mlp = metrics_report_mlp(test_loader_partial, baseline_mlp_latent, "MS-UMG", device=device, class_names=le.classes_)
    
            ############################################################
            # Store results
            ############################################################
            for model_name, metrics in [
                ("RF_original",     metrics_orig),
                ("MLP_original",    metrics_mlp_orig),
                ("RF_latent_zero",  metrics_lat),
                ("MLP_latent_zero", metrics_mlp_lat),
                ("RF_few",          metrics_few),
                ("MLP_few",         metrics_mlp_few_orig),
                ("FT_full_RF",      metrics_ft_full),
                ("FT_full_MLP",     metrics_ft_full_mlp),
                ("FT_freeze_RF",    metrics_ft_freeze),
                ("FT_freeze_MLP",   metrics_ft_freeze_mlp),
                ("FT_decoder_RF",   metrics_ft_dec),
                ("FT_decoder_MLP",  metrics_ft_dec_mlp),
                ("FT_partial_RF",   metrics_ft_partial),
                ("FT_partial_MLP",  metrics_ft_partial_mlp),]:
                results.append({
                    "partition": i_part,
                    "n_prev": n_prev,
                    "n_new": n_new,
                    "model": model_name,
                    "balanced_accuracy": metrics["Balanced_Accuracy"],
                    "f1_macro": metrics["F1_Macro"],
                    "recall_macro": metrics["Recall_Macro"],
                    "specificity_macro": metrics["Specificity_Macro"],
                    "roc_auc_macro": metrics["ROC_AUC_Macro"],
                    "confusion_matrix": json.dumps(metrics["Confusion Matrix"].tolist()),
                })

            # Cleanup to save memory
            del vae_full, vae_freeze
            torch.cuda.empty_cache()


############################################################
# SAVE RESULTS
############################################################
df_results = pd.DataFrame(results)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = OUTPUT_PATH / f"grid_results_{stamp}.csv"
df_results.to_csv(csv_path, index=False)

print("\n===== GRID RESULTS SAVED =====")
print(f"Saved to: {csv_path}")

summary = df_results.groupby(["n_prev", "n_new", "model"]).agg({
    "balanced_accuracy": ["mean", "std"],
    "f1_macro": ["mean", "std"],
    "roc_auc_macro": ["mean", "std"]}).reset_index()

summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]
summary_path = OUTPUT_PATH / f"grid_summary_DRIAMSD_{stamp}.csv"
summary.to_csv(summary_path, index=False)

print(f"Summary saved to: {summary_path}")
print("\n===== GRID EVALUATION FINISHED =====")
