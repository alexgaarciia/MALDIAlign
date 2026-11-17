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


# ============================================================
# IMPORTS
# ============================================================
import pickle
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import load_model, encode_latent, make_loader

from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

DALMA_EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")
OUT_PATH = Path(f"experiments/spectral_similarity/results/{timestamp}/domain_clf.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
N_FOLDS = 5
SPACES = ("original", "latent")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA LOADING
# ============================================================
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

maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA = data_driams[maskA], label_driams[maskA]
dataB, labelB = data_driams[maskB], label_driams[maskB]
dataC, labelC = data_driams[maskC], label_driams[maskC]
dataD, labelD = data_driams[maskD], label_driams[maskD]

dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)


# ============================================================
# POOL ALL 7 DOMAINS
# ============================================================
domain_blocks = [
    ("A", dataA, labelA),
    ("B", dataB, labelB),
    ("C", dataC, labelC),
    ("MARISMA", data_marisma, label_marisma),
    ("RKI", data_rki, label_rki),
    ("D", dataD, labelD),
    ("MSUMG", data_msumg, label_msumg),
]

data_pool = np.vstack([blk[1] for blk in domain_blocks])
species_pool = np.concatenate([blk[2] for blk in domain_blocks])
domain_pool = np.concatenate([np.full(len(blk[1]), blk[0]) for blk in domain_blocks])

strat_label = np.array([f"{d}_{s}" for d, s in zip(domain_pool, species_pool)])

le_domain = LabelEncoder().fit(domain_pool)
y_domain = le_domain.transform(domain_pool)
n_domains = len(le_domain.classes_)


# ============================================================
# DALMA LATENT ENCODING
# ============================================================
print("\n===== LATENT ENCODING (DALMA) =====")

n_species_ckpt = len(np.unique(species_pool))

dalma = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=data_pool.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species=n_species_ckpt),
    DALMA_EXPERIMENT_DIR / "model.pth"
)

Z_pool = encode_latent(dalma, data_pool, device)


# ============================================================
# DOMAIN CLASSIFIER: 5-FOLD STRATIFIED CV
# ============================================================
run_rows = []

for space, X_pool in [("original", data_pool), ("latent", Z_pool)]:
    for seed in SEEDS:
        print(f"\n--- space={space} | seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_skf.split(X_pool, strat_label)):
            tr_strat = strat_label[tr_idx]
            inner_skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
            inner_tr_rel, inner_va_rel = next(inner_skf.split(tr_idx, tr_strat))
            tr_idx_final, va_idx = tr_idx[inner_tr_rel], tr_idx[inner_va_rel]

            if space == "original":
                model = MLPClassifier_Extended(
                    input_dim=X_pool.shape[1], n_species=n_domains,
                    epochs=200, lr=1e-3, patience=10,
                )
            else:
                model = LinearProbe_Extended(
                    latent_dim=X_pool.shape[1], n_species=n_domains,
                    epochs=200, lr=1e-3, patience=10,
                )

            model.trainloop(
                make_loader(X_pool[tr_idx_final], y_domain[tr_idx_final], shuffle=True),
                make_loader(X_pool[va_idx], y_domain[va_idx]),
                device,
            )

            metrics = metrics_report_mlp(
                make_loader(X_pool[te_idx], y_domain[te_idx]),
                model,
                f"{space}-s{seed}-fold{fold_idx}",
                device=device,
                class_names=le_domain.classes_,
            )

            run_rows.append({
                "space": space,
                "seed": seed,
                "fold": fold_idx,
                "balanced_accuracy": metrics["Balanced_Accuracy"],
                "f1_macro": metrics["F1_Macro"],
                "recall_macro": metrics["Recall_Macro"],
                "cm": metrics["Confusion Matrix"],
            })

        pd.DataFrame(run_rows).to_csv(OUT_PATH.with_name(OUT_PATH.stem + "_runs.csv"), index=False)


# ============================================================
# SUMMARY
# ============================================================
df_runs = pd.DataFrame(run_rows)
df_summary = df_runs.groupby(["space"], sort=False)[["balanced_accuracy", "f1_macro", "recall_macro"]].agg(["mean", "std"])
df_summary.columns = [f"{m}_{s}" for m, s in df_summary.columns]
df_summary = df_summary.reset_index()

print("\n===== DOMAIN CLASSIFIER: BALANCED ACCURACY =====")
for _, r in df_summary.iterrows():
    print(f"space={r['space']}: {r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}")

RUNS_PATH = OUT_PATH.with_name(OUT_PATH.stem + "_runs.csv")
df_runs.to_csv(RUNS_PATH, index=False)
df_summary.to_csv(OUT_PATH, index=False)
print("Saved summary:", OUT_PATH)
print("Saved per-fold runs:", RUNS_PATH)
