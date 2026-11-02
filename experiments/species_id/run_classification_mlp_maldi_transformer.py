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
from datetime import datetime

from maldi_nn.models import MaldiTransformer

from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import *
from src.evaluation.eval import make_loader
from src.finetuning.malditransformer import get_transformer_latent

from models.baselines.mlp_latent import LinearProbe_Extended


# ============================================================
# ARG PARSING
# ============================================================
MODEL_SIZES = ["S", "M", "L", "XL"]

parser = argparse.ArgumentParser(description="MaldiTransformer + linear probe evaluation (frozen or fine-tuned encoder)")
parser.add_argument("--size", "-s", type=str, required=True, choices=MODEL_SIZES, help="MaldiTransformer size: S, M, L or XL")
parser.add_argument("--finetuned", action="store_true", help="use the fine-tuned checkpoint instead of the frozen pretrained one")
parser.add_argument("--ckpt", type=str, default=None, help="fine-tuned checkpoint path (only with --finetuned); default assets/MaldiTransformer{SIZE}_ft.ckpt")
args = parser.parse_args()

SIZE = args.size
ENCODER = "finetuned" if args.finetuned else "frozen"
if args.finetuned:
    CHECKPOINT_PATH = Path(args.ckpt) if args.ckpt else Path(f"assets/MaldiTransformer{SIZE}_ft.ckpt")
else:
    CHECKPOINT_PATH = Path(f"assets/MaldiTransformer{SIZE}.ckpt")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = Path(f"experiments/species_id/results/{timestamp}/mlp_transformer_{SIZE}{'_ft' if args.finetuned else ''}.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
RUNS_PATH = OUT_PATH.with_name(OUT_PATH.stem + "_runs.csv")

N_PEAKS = 200
SEEDS = [42, 123, 456, 789, 1011]
EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")
BASE = "/export/data_ml4ds/bacteria_id/MALDIAlign_Alex"

print(f"Encoder: {ENCODER} | checkpoint: {CHECKPOINT_PATH}")
print(f"Output: {OUT_PATH} | seeds: {SEEDS}")


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
meta_driams["hospital"] = meta_driams["hospital"].str.replace("-", "_", regex=False)

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
data_list = [dataA, dataB, dataC, data_marisma, data_rki]
label_list = [labelA, labelB, labelC, label_marisma, label_rki]

data_final = np.vstack(data_list)
label_final = np.concatenate(label_list)

# Load splits
with open(EXPERIMENT_DIR / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)


# ============================================================
# BUILD TEST SETS FROM SPLITS
# ============================================================
SPLIT_NAMES = {
    "A": "DRIAMS_A",
    "B": "DRIAMS_B",
    "C": "DRIAMS_C",
    "MARISMA": "MARISMA",
    "RKI": "RKI",
}

test_sets = {}
for k, sk in SPLIT_NAMES.items():
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
    test_sets[k] = (data_final[te_idx], label_final[te_idx])

# OOD test sets
test_sets["D (OOD)"] = (dataD, labelD)
test_sets["MSUMG (OOD)"] = (data_msumg, label_msumg)


# ============================================================
# TRANSFORMER EMBEDDING EXTRACTION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n===== EXTRACTING TRANSFORMER EMBEDDINGS =====")
transformer_model = MaldiTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location=device, strict=False)

ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
Z_final = get_transformer_latent(data_final, transformer_model, n_peaks=N_PEAKS, device=device)
Z_D = get_transformer_latent(dataD, transformer_model, n_peaks=N_PEAKS, device=device)
Z_MSUMG = get_transformer_latent(data_msumg, transformer_model, n_peaks=N_PEAKS, device=device)


# ============================================================
# TRAIN SETS AND TEST SETS (latent)
# ============================================================
train_sets = {k: (splits["splits_per_domain"][sk]["train_idx"], splits["splits_per_domain"][sk]["val_idx"]) for k, sk in SPLIT_NAMES.items()}
train_sets["ALL"] = (
    np.concatenate([v[0] for v in train_sets.values()]),
    np.concatenate([v[1] for v in train_sets.values()]),
)

test_latent, test_labels = {}, {}
for k, sk in SPLIT_NAMES.items():
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
    test_latent[k], test_labels[k] = Z_final[te_idx], label_final[te_idx]

test_latent["D (OOD)"], test_labels["D (OOD)"] = Z_D, labelD
test_latent["MSUMG (OOD)"], test_labels["MSUMG (OOD)"] = Z_MSUMG, label_msumg


# ============================================================
# EVALUATION: each train set, one linear probe per seed
# ============================================================
run_rows = []

for train_name, (tr_idx, va_idx) in train_sets.items():
    print("\n" + "="*70)
    print(f"TRAIN DOMAIN: {train_name} | train: {len(tr_idx)} | val: {len(va_idx)}")
    print("="*70)

    le = LabelEncoder().fit(species_to_keep)
    y_tr_enc = le.transform(label_final[tr_idx])
    y_va_enc = le.transform(label_final[va_idx])
    n_classes = len(le.classes_)

    for seed in SEEDS:
        print(f"\n--- train={train_name} | seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        mlp_lat = LinearProbe_Extended(latent_dim=Z_final.shape[1], n_species=n_classes, epochs=50, lr=1e-3, patience=10)
        mlp_lat.trainloop(make_loader(Z_final[tr_idx], y_tr_enc, shuffle=True), make_loader(Z_final[va_idx], y_va_enc), device)

        for test_name in test_latent.keys():
            y_te_enc = le.transform(test_labels[test_name])

            metrics = metrics_report_mlp(make_loader(test_latent[test_name], y_te_enc), mlp_lat, f"{train_name}-{test_name}-s{seed}", device=device, class_names=le.classes_,)

            run_rows.append({
                "train": train_name,
                "test": test_name,
                "space": "latent",
                "encoder": ENCODER,
                "seed": seed,
                "balanced_accuracy": metrics["Balanced_Accuracy"],
                "f1_macro": metrics["F1_Macro"],
                "recall_macro": metrics["Recall_Macro"],
                "specificity_macro": metrics["Specificity_Macro"],
                "roc_auc": metrics["ROC_AUC_Macro"],
                "cm": metrics["Confusion Matrix"],
            })

        pd.DataFrame(run_rows).to_csv(RUNS_PATH, index=False)


# ============================================================
# SUMMARY: mean and SD over seeds
# ============================================================
metric_cols = ["balanced_accuracy", "f1_macro", "recall_macro", "specificity_macro", "roc_auc"]
df_runs = pd.DataFrame(run_rows)

df_summary = df_runs.groupby(["train", "test", "space", "encoder"], sort=False)[metric_cols].agg(["mean", "std"])
df_summary.columns = [f"{m}_{s}" for m, s in df_summary.columns]
df_summary = df_summary.reset_index()
df_summary.insert(4, "n_seeds", len(SEEDS))

print("\n===== BALANCED ACCURACY: MEAN ± SD OVER SEEDS =====")
for _, r in df_summary.iterrows():
    print(f"{r['train']} -> {r['test']} [{r['encoder']}]: {r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}")


# ============================================================
# SAVE CSV
# ============================================================
print("\n===== SAVING RESULTS =====")
df_runs.to_csv(RUNS_PATH, index=False)
df_summary.to_csv(OUT_PATH, index=False)
print("Saved summary:", OUT_PATH)
print("Saved per-seed runs:", RUNS_PATH)
