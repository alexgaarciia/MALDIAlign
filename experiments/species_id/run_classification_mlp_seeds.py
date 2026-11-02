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

from sklearn.preprocessing import LabelEncoder

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.metrics import metrics_report_mlp
from src.evaluation.eval import load_model, encode_latent, make_loader

from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended


# ============================================================
# ARCHITECTURE TO TEST
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

ARCH_CONFIGS = {
    "vae": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_bernoulli/20260614_235638",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_vae.csv",
    },
    "vae_prior": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_bernoulli/20260614_235109",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_vae_prior.csv",
    },
    "vae_multidecoder": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_bernoulli/20260614_235237",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_vae_multidecoder.csv",
    },
    "vae_multidecoder_prior": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_vae_multidecoder_prior.csv",
    },
    "dann": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/dann/20260412_160450",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_dann.csv",
    },
    "coral": {
        "experiment_dir": "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_coral/20260412_160952",
        "out_path": f"experiments/species_id/results/{timestamp}/mlp_coral.csv",
    },
}

parser = argparse.ArgumentParser(description="MLP evaluation across architectures")
parser.add_argument(
    "--architecture", "-a",
    type=str,
    required=True,
    choices=list(ARCH_CONFIGS.keys()),
    help="Architecture to evaluate",
)
args = parser.parse_args()

ARCHITECTURE = args.architecture
EXPERIMENT_DIR = Path(ARCH_CONFIGS[ARCHITECTURE]["experiment_dir"])
OUT_PATH = Path(ARCH_CONFIGS[ARCHITECTURE]["out_path"])
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1011]
SPACES = ("original", "latent")

print(f"Architecture: {ARCHITECTURE}")
print(f"Experiment: {EXPERIMENT_DIR}")
print(f"Output: {OUT_PATH}")
print(f"Seeds: {SEEDS}")


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

print(f"A={len(dataA)} B={len(dataB)} C={len(dataC)} D={len(dataD)} MARISMA={len(data_marisma)} RKI={len(data_rki)} MS-UMG={len(data_msumg)}")
assert min(len(dataA), len(dataB), len(dataC), len(dataD)) > 0, "A DRIAMS hospital is empty: check the hospital names in meta"

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

max_idx = max(int(np.max(v[key])) for v in splits["splits_per_domain"].values() for key in ("train_idx", "val_idx", "test_idx") if len(v[key]) > 0)
assert max_idx < len(data_final), f"split index {max_idx} out of range for data_final ({len(data_final)})"
print(f"data_final: {data_final.shape} | max split index: {max_idx}")

test_sets = {}
for k, sk in SPLIT_NAMES.items():
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
    test_sets[k] = (data_final[te_idx], label_final[te_idx])

# OOD test sets
test_sets["D (OOD)"] = (dataD, labelD)
test_sets["MSUMG (OOD)"] = (data_msumg, label_msumg)


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

elif ARCHITECTURE == "vae":
    vae = load_model(
        VAE_Bernoulli_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64),
        EXPERIMENT_DIR / "model.pth"
    )

elif ARCHITECTURE == "vae_prior":
    vae = load_model(
        VAE_Bernoulli_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64,
            use_species_prior=True,
            n_species=len(np.unique(label_final)),
            num_domains=1,
        ),
        EXPERIMENT_DIR / "model.pth"
    )

elif ARCHITECTURE == "vae_multidecoder":
    vae = load_model(
        VAE_Bernoulli_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64,
            use_species_prior=False,
            num_domains=5,
        ),
        EXPERIMENT_DIR / "model.pth"
    )

elif ARCHITECTURE == "dann":
    vae = load_model(
        DANNFull_Extended(
            input_dim=data_final.shape[1],
            latent_dim=64,
            n_species=len(np.unique(label_final)),
            n_domains=5),
        EXPERIMENT_DIR / "model.pth"
    )

elif ARCHITECTURE == "coral":
    vae = load_model(
        MultiVAE_CORAL(
            input_dim=data_final.shape[1],
            latent_dim=64,
            num_domains=5),
        EXPERIMENT_DIR / "model.pth"
    )

Z_final = encode_latent(vae, data_final, device)
Z_D = encode_latent(vae, dataD, device)
Z_MSUMG = encode_latent(vae, data_msumg, device)


# ============================================================
# BUILD LATENT TEST SETS
# ============================================================
test_sets_latent = {}
for k, sk in SPLIT_NAMES.items():
    te_idx = splits["splits_per_domain"][sk]["test_idx"]
    test_sets_latent[k] = (Z_final[te_idx], label_final[te_idx])

test_sets_latent["D (OOD)"] = (Z_D, labelD)
test_sets_latent["MSUMG (OOD)"] = (Z_MSUMG, label_msumg)


# ============================================================
# EVALUATION: train sets (each source domain + all pooled), one run per seed
# ============================================================
train_sets = {k: (splits["splits_per_domain"][sk]["train_idx"], splits["splits_per_domain"][sk]["val_idx"]) for k, sk in SPLIT_NAMES.items()}
train_sets["ALL"] = (
    np.concatenate([v[0] for v in train_sets.values()]),
    np.concatenate([v[1] for v in train_sets.values()]),
)

run_rows = []

for train_name, (tr_idx, va_idx) in train_sets.items():
    print("\n" + "=" * 70)
    print(f"TRAIN DOMAIN: {train_name} | train: {len(tr_idx)} | val: {len(va_idx)}")
    print("=" * 70)

    le = LabelEncoder().fit(label_final[tr_idx])
    y_tr_enc = le.transform(label_final[tr_idx])
    y_va_enc = le.transform(label_final[va_idx])
    n_classes = len(le.classes_)

    for seed in SEEDS:
        print(f"\n--- train={train_name} | seed={seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        mlp_orig = MLPClassifier_Extended(
            input_dim=data_final.shape[1], n_species=n_classes,
            epochs=50, lr=1e-3, patience=10,
        )
        mlp_orig.trainloop(
            make_loader(data_final[tr_idx], y_tr_enc, shuffle=True),
            make_loader(data_final[va_idx], y_va_enc),
            device,
        )

        mlp_lat = LinearProbe_Extended(
            latent_dim=Z_final.shape[1], n_species=n_classes,
            epochs=50, lr=1e-3, patience=10,
        )
        mlp_lat.trainloop(
            make_loader(Z_final[tr_idx], y_tr_enc, shuffle=True),
            make_loader(Z_final[va_idx], y_va_enc),
            device,
        )

        for test_name in test_sets.keys():
            y_te_enc = le.transform(test_sets[test_name][1])

            for space, model, X_eval in [
                ("original", mlp_orig, test_sets[test_name][0]),
                ("latent",   mlp_lat,  test_sets_latent[test_name][0]),
            ]:
                metrics = metrics_report_mlp(
                    make_loader(X_eval, y_te_enc),
                    model,
                    f"{train_name}-{test_name}-{space}-s{seed}",
                    device=device,
                    class_names=le.classes_,
                )

                run_rows.append({
                    "train": train_name,
                    "test": test_name,
                    "space": space,
                    "seed": seed,
                    "balanced_accuracy": metrics["Balanced_Accuracy"],
                    "f1_macro": metrics["F1_Macro"],
                    "recall_macro": metrics["Recall_Macro"],
                    "specificity_macro": metrics["Specificity_Macro"],
                    "roc_auc": metrics["ROC_AUC_Macro"],
                    "cm": metrics["Confusion Matrix"],
                })

        pd.DataFrame(run_rows).to_csv(OUT_PATH.with_name(OUT_PATH.stem + "_runs.csv"), index=False)


# ============================================================
# SUMMARY: mean and SD over seeds
# ============================================================
metric_cols = ["balanced_accuracy", "f1_macro", "recall_macro", "specificity_macro", "roc_auc"]
df_runs = pd.DataFrame(run_rows)

df_summary = df_runs.groupby(["train", "test", "space"], sort=False)[metric_cols].agg(["mean", "std"])
df_summary.columns = [f"{m}_{s}" for m, s in df_summary.columns]
df_summary = df_summary.reset_index()
df_summary.insert(3, "n_seeds", len(SEEDS))

print("\n===== BALANCED ACCURACY: MEAN ± SD OVER SEEDS =====")
for _, r in df_summary.iterrows():
    print(f"{r['train']} -> {r['test']} [{r['space']}]: {r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f}")


# ============================================================
# SAVE CSV
# ============================================================
print("\n===== SAVING RESULTS =====")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
RUNS_PATH = OUT_PATH.with_name(OUT_PATH.stem + "_runs.csv")
df_runs.to_csv(RUNS_PATH, index=False)
df_summary.to_csv(OUT_PATH, index=False)

print("Saved summary:", OUT_PATH)
print("Saved per-seed runs:", RUNS_PATH)
