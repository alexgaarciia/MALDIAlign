############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys
import json
import pickle
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
import gc
import torch
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.io import load_pkl
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import encode_latent, make_loader
from experiments.finetuning.run_finetuning import run_finetuning

from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended

############################################################
# CONFIG
############################################################
TARGET_SPECIES = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/model.pth")
VAE_SPLITS_PATH       = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")
SPLITS_PATH           = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data/splits_20260525_100756")
OUTPUT_PATH = Path("/export/usuarios_ml4ds/agnavarr/MALDIAlign/finetuning_6species")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

N_PARTITIONS = 10
GRID_PREV    = [0]
GRID_NEW     = np.arange(50, 251, 50)
RUN_LATENT_EVALUATION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA =====")
cfg = load_config()

# Sources (for global MLP training)
driams_dict  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict     = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict   = load_msumg(cfg["data"]["MSUMG_FULL"])

def filter_species(d, species):
    mask = np.isin(d["label"], species)
    return d["data"][mask], d["label"][mask], d["meta"][mask].reset_index(drop=True)

mask_driams  = np.isin(driams_dict["label"], TARGET_SPECIES)
data_driams  = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams  = driams_dict["meta"][mask_driams].reset_index(drop=True)

data_marisma, label_marisma, _ = filter_species(marisma_dict, TARGET_SPECIES)
data_rki,     label_rki,     _ = filter_species(rki_dict,     TARGET_SPECIES)
data_msumg,   label_msumg,   _ = filter_species(msumg_dict,   TARGET_SPECIES)

maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"

dataA, labelA = data_driams[maskA], label_driams[maskA]
dataB, labelB = data_driams[maskB], label_driams[maskB]
dataC, labelC = data_driams[maskC], label_driams[maskC]

dataA        = row_minmax_normalize(dataA)
dataB        = row_minmax_normalize(dataB)
dataC        = row_minmax_normalize(dataC)
data_marisma = row_minmax_normalize(data_marisma)
data_rki     = row_minmax_normalize(data_rki)
data_msumg   = row_minmax_normalize(data_msumg)

data_final  = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])

le = LabelEncoder()
le.fit(TARGET_SPECIES)

print(f"Source samples: {len(data_final)}")
print(f"MS-UMG samples: {len(data_msumg)}")
print("===== DATA LOADED =====")

############################################################
# LOAD PRETRAINED VAE
############################################################
print("\n===== LOADING PRETRAINED VAE =====")
vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_Extended(
    input_dim=data_final.shape[1],
    latent_dim=64,
    num_domains=5,
    n_species=len(TARGET_SPECIES)
)
vae_pretrained.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
vae_pretrained.to(device)
vae_pretrained.eval()

print("\n===== ENCODING LATENT SPACE =====")
Z_final  = encode_latent(vae_pretrained, data_final,  device)
Z_MSUMG  = encode_latent(vae_pretrained, data_msumg,  device)

############################################################
# TRAIN GLOBAL BASELINE MLPs (once)
############################################################
print("\n===== TRAINING GLOBAL BASELINE MLPs =====")

with open(VAE_SPLITS_PATH, "rb") as f:
    vae_splits = pickle.load(f)

SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA", "RKI": "RKI"}

X_train_list, y_train_list = [], []
X_val_list,   y_val_list   = [], []
Z_train_list, Z_val_list   = [], []

for k, sk in SPLIT_NAMES.items():
    tr_idx = vae_splits["splits_per_domain"][sk]["train_idx"]
    va_idx = vae_splits["splits_per_domain"][sk]["val_idx"]
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
Z_val = np.concatenate(Z_val_list)

y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)

print("Training MLP (original)...")
mlp_orig = MLPClassifier_Extended(
    input_dim=X_train.shape[1], n_species=len(TARGET_SPECIES),
    epochs=50, lr=1e-3, patience=10
)
mlp_orig.trainloop(make_loader(X_train, y_train_enc, shuffle=True), make_loader(X_val, y_val_enc), device)

print("Training LinearProbe (latent)...")
mlp_lat = LinearProbe_Extended(
    latent_dim=Z_train.shape[1], n_species=len(TARGET_SPECIES),
    epochs=50, lr=1e-3, patience=10
)
mlp_lat.trainloop(make_loader(Z_train, y_train_enc, shuffle=True), make_loader(Z_val, y_val_enc), device)

mlp_orig.eval()
mlp_lat.eval()
print("===== GLOBAL MLPs TRAINED =====")

############################################################
# GRID EVALUATION LOOP (MS-UMG)
############################################################
results = []

for i_part in range(N_PARTITIONS):
    print(f"\n{'#'*70}")
    print(f"PARTITION {i_part + 1}/{N_PARTITIONS}")
    print(f"{'#'*70}")

    partition_dir = SPLITS_PATH / f"run_{i_part}"

    for n_prev in GRID_PREV:
        for n_new in GRID_NEW:
            print(f"\n--- Partition {i_part} | n_prev={n_prev}, n_new={n_new} ---")

            split_file = partition_dir / f"prev_{n_prev}_new_{n_new}.pkl"
            if not split_file.exists():
                print(f"  Split file not found: {split_file}. Skipping.")
                continue

            splits   = load_pkl(split_file)
            idx_test = splits["MS-UMG"]["test"]
            idx_ft   = splits["MS-UMG"]["finetuning"]

            X_test, y_test = data_msumg[idx_test], label_msumg[idx_test]
            X_ft,   y_ft   = data_msumg[idx_ft],   label_msumg[idx_ft]

            y_test_enc = le.transform(y_test)
            y_ft_enc   = le.transform(y_ft)
            counts_ft  = np.bincount(y_ft_enc, minlength=len(TARGET_SPECIES))

            test_loader_orig = make_loader(X_test, y_test_enc)

            # --- Few-shot RF ---
            rf_few = RandomForestClassifier(
                n_estimators=200, max_depth=20,
                class_weight="balanced_subsample",
                n_jobs=-1, random_state=42
            )
            rf_few.fit(X_ft, y_ft)

            # --- Few-shot MLP (original space) ---
            X_ft_tr, X_ft_val, y_ft_tr, y_ft_val = train_test_split(
                X_ft, y_ft_enc,
                test_size=0.2,
                stratify=y_ft_enc if np.all(counts_ft >= 2) else None,
                random_state=42
            )
            mlp_few_orig = MLPClassifier_Extended(
                input_dim=X_ft.shape[1], n_species=len(TARGET_SPECIES),
                epochs=50, lr=1e-4, patience=10
            )
            mlp_few_orig.trainloop(
                make_loader(X_ft_tr, y_ft_tr, shuffle=True),
                make_loader(X_ft_val, y_ft_val),
                device
            )

            # --- Finetuning VAEs ---
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
            # vae_dec, _, _ = run_finetuning(
            #     splits_path=split_file, target_domain="MS-UMG",
            #     pretrained_model_path=PRETRAINED_MODEL_PATH,
            #     finetuning_mode="decoder_only", n_prev=n_prev, n_new=n_new,
            #     output_dir=OUTPUT_PATH, device=device,
            #     consider_prev_domains=False,
            #     run_latent_evaluation=RUN_LATENT_EVALUATION,
            # )
            # vae_partial, _, _ = run_finetuning(
            #     splits_path=split_file, target_domain="MS-UMG",
            #     pretrained_model_path=PRETRAINED_MODEL_PATH,
            #     finetuning_mode="partial_encoder", n_prev=n_prev, n_new=n_new,
            #     output_dir=OUTPUT_PATH, device=device,
            #     consider_prev_domains=False,
            #     run_latent_evaluation=RUN_LATENT_EVALUATION,
            # )

            # --- Encode test set ---
            Z_test_zero    = encode_latent(vae_pretrained, X_test, device)
            Z_test_full    = encode_latent(vae_full,       X_test, device)
            Z_test_freeze  = encode_latent(vae_freeze,     X_test, device)
            # Z_test_dec     = encode_latent(vae_dec,        X_test, device)
            # Z_test_partial = encode_latent(vae_partial,    X_test, device)

            test_loader_zero    = make_loader(Z_test_zero,    y_test_enc)
            test_loader_full    = make_loader(Z_test_full,    y_test_enc)
            test_loader_freeze  = make_loader(Z_test_freeze,  y_test_enc)
            # test_loader_dec     = make_loader(Z_test_dec,     y_test_enc)
            # test_loader_partial = make_loader(Z_test_partial, y_test_enc)

            # --- Evaluate all models ---
            evals = [
                ("MLP_original",    metrics_report_mlp(test_loader_orig,    mlp_orig,     "MS-UMG", device=device, class_names=le.classes_)),
                ("MLP_latent_zero", metrics_report_mlp(test_loader_zero,    mlp_lat,      "MS-UMG", device=device, class_names=le.classes_)),
                ("MLP_few",         metrics_report_mlp(test_loader_orig,    mlp_few_orig, "MS-UMG", device=device, class_names=le.classes_)),
                ("FT_full_MLP",     metrics_report_mlp(test_loader_full,    mlp_lat,      "MS-UMG", device=device, class_names=le.classes_)),
                ("FT_freeze_MLP",   metrics_report_mlp(test_loader_freeze,  mlp_lat,      "MS-UMG", device=device, class_names=le.classes_)),
                # ("FT_decoder_MLP",  metrics_report_mlp(test_loader_dec,     mlp_lat,      "MS-UMG", device=device, class_names=le.classes_)),
                # ("FT_partial_MLP",  metrics_report_mlp(test_loader_partial, mlp_lat,      "MS-UMG", device=device, class_names=le.classes_)),
            ]

            for model_name, metrics in evals:
                results.append({
                    "partition":         i_part,
                    "n_prev":            n_prev,
                    "n_new":             n_new,
                    "model":             model_name,
                    "balanced_accuracy": metrics["Balanced_Accuracy"],
                    "f1_macro":          metrics["F1_Macro"],
                    "recall_macro":      metrics["Recall_Macro"],
                    "specificity_macro": metrics["Specificity_Macro"],
                    "roc_auc_macro":     metrics["ROC_AUC_Macro"],
                    "confusion_matrix":  json.dumps(metrics["Confusion Matrix"].tolist()),
                })

            mlp_few_orig.cpu()
            del mlp_few_orig, rf_few
            del Z_test_zero, Z_test_full, Z_test_freeze
            del test_loader_zero, test_loader_full, test_loader_freeze, test_loader_orig
            vae_full.cpu()
            vae_freeze.cpu()
            del vae_full, vae_freeze
            torch.cuda.empty_cache()
            gc.collect()
            

############################################################
# SAVE RESULTS
############################################################
df_results = pd.DataFrame(results)
stamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

csv_path = OUTPUT_PATH / f"grid_results_MSUMG_{stamp}.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nSaved results: {csv_path}")

summary = df_results.groupby(["n_prev", "n_new", "model"]).agg(
    balanced_accuracy_mean=("balanced_accuracy", "mean"),
    balanced_accuracy_std =("balanced_accuracy", "std"),
    f1_macro_mean         =("f1_macro",          "mean"),
    f1_macro_std          =("f1_macro",          "std"),
    roc_auc_mean          =("roc_auc_macro",      "mean"),
    roc_auc_std           =("roc_auc_macro",      "std"),
).reset_index()

summary_path = OUTPUT_PATH / f"grid_summary_MSUMG_{stamp}.csv"
summary.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")
print("\n===== DONE =====")
