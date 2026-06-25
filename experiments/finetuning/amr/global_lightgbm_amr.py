
############################################################
#  PATH CONFIGURATION
############################################################
import os
import sys
import pickle
import joblib
import json
from pathlib import Path
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

print("Working directory:", os.getcwd())


############################################################
# IMPORTS
############################################################
import torch
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.io import load_pkl
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head


############################################################
# CONFIGURATION
############################################################
# Pretrained AMR VAE (trained on A+B+C+MARISMA)
PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260410_105028/model.pth")
PRETRAINED_SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260410_105028/data_splits.pkl")
DATASET_PATH = "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl"

SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]

ANTIBIOTICS = [
    "Amikacin", "Ceftazidime", "Ceftriaxone", "Clindamycin",
    "Erythromycin", "Meropenem", "Piperacillin-Tazobactam",
    "Tetracycline", "Vancomycin"
]

LATENT_DIM = 128
NUM_DOMAINS = 4
N_SPECIES = 6
N_ANTIBIOTICS = 9
LAMBDA_AMR = 25

MIN_TRAIN = 10
MIN_PER_CLASS = 3


############################################################
# LOAD DATASET
############################################################
print("\n" + "=" * 60)
print("===== LOADING DATA =====")
print("=" * 60)

dataset = load_pkl(DATASET_PATH)
data = dataset["data"]
amr = dataset["amr"]
ab_list_raw = list(dataset["antibiotics"])
labels = dataset["label"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

print(f"Raw shape: {data.shape}")

# Filter antibiotics (Columnas)
keep_idx = [ab_list_raw.index(n) for n in ANTIBIOTICS if n in ab_list_raw]
ab_list = [n for n in ANTIBIOTICS if n in ab_list_raw]
amr = amr[:, keep_idx]

# Filter chrom agar (Filas) - Debe coincidir con el recorte original del split
if "agar" in meta.columns:
    chrom_mask = (meta["hospital"] == "MS-UMG") & (meta["agar"] == "chrom")
    if chrom_mask.sum() > 0:
        keep = ~chrom_mask.values
        data, amr, labels = data[keep], amr[keep], labels[keep]
        meta = meta.loc[keep].reset_index(drop=True)
        print(f"Removed {chrom_mask.sum()} chrom-agar")

# Normalize
data_norm = row_minmax_normalize(data)

# Filter empty antibiotics (Columnas)
valid_ab = ~np.all(np.isnan(amr), axis=0)
ab_list = [a for a, v in zip(ab_list, valid_ab) if v]
amr = amr[:, valid_ab]

print(f"Antibiotics ({len(ab_list)}): {ab_list}")
print(f"Final data shape: {data_norm.shape}")


############################################################
# LOAD SPLITS
############################################################
print("\n===== LOADING SPLITS =====")

with open(PRETRAINED_SPLITS_PATH, "rb") as f:
    saved_splits = pickle.load(f)

domain_splits = saved_splits.get("splits_per_domain", {})

source_train_idx = []
source_test_idx = []
for src in SOURCE_DOMAINS:
    if src in domain_splits:
        source_train_idx.append(domain_splits[src]["train_idx"])
        source_test_idx.append(domain_splits[src]["test_idx"])
        print(f"{src}: train={len(domain_splits[src]['train_idx'])}, test={len(domain_splits[src]['test_idx'])}")

source_train_idx = np.concatenate(source_train_idx)
source_test_idx = np.concatenate(source_test_idx)
print(f"Total — train: {len(source_train_idx)}, test: {len(source_test_idx)}")

assert source_train_idx.max() < len(data_norm), "Split indices exceed dataset size!"


############################################################
# LOAD VAE & ENCODE LATENT
############################################################
print("\n===== LOADING VAE & ENCODING =====")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head(
    input_dim=data_norm.shape[1],
    latent_dim=LATENT_DIM,
    num_domains=NUM_DOMAINS,
    n_species=N_SPECIES,
    n_antibiotics=len(ab_list),
    lambda_amr=LAMBDA_AMR,
)
vae.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
vae.to(device)
vae.eval()

Z_all = np.asarray(encode_latent(vae, data_norm, device))
print(f"Latent shape: {Z_all.shape}")


############################################################
# TRAIN LGBM PER (SPECIES, ANTIBIOTIC) PAIR
############################################################
print("\n" + "=" * 60)
print("===== TRAINING BASELINE LGBM CLASSIFIERS =====")
print("=" * 60)

species_list = sorted(np.unique(labels))
classifiers_original = {}
classifiers_latent = {}
training_stats = []

for species_name in species_list:
    species_idx = np.where(labels == species_name)[0]
    train_sp_idx = np.intersect1d(source_train_idx, species_idx)

    for atb_idx, atb_name in enumerate(ab_list):
        y_col = amr[:, atb_idx]
        y_tr = y_col[train_sp_idx]
        m_tr = ~np.isnan(y_tr)

        X_tr_orig = data_norm[train_sp_idx[m_tr]]
        Z_tr_lat = Z_all[train_sp_idx[m_tr]]
        y_tr_clean = y_tr[m_tr].astype(int)

        n_pos = int((y_tr_clean == 1).sum())
        n_neg = int((y_tr_clean == 0).sum())

        if len(y_tr_clean) < MIN_TRAIN or n_pos < MIN_PER_CLASS or n_neg < MIN_PER_CLASS:
            print(f"SKIP {species_name} × {atb_name}: n={len(y_tr_clean)} (R={n_pos}, S={n_neg})")
            continue

        scale_w = n_neg / n_pos
        pair_key = (species_name, atb_name)

        # Original
        lgbm_orig = LGBMClassifier(
            n_estimators=300, scale_pos_weight=scale_w,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        lgbm_orig.fit(X_tr_orig, y_tr_clean)
        classifiers_original[pair_key] = lgbm_orig

        # Latent
        lgbm_lat = LGBMClassifier(
            n_estimators=300, scale_pos_weight=scale_w,
            random_state=42, verbosity=-1, n_jobs=-1,
        )
        lgbm_lat.fit(Z_tr_lat, y_tr_clean)
        classifiers_latent[pair_key] = lgbm_lat

        print(f"{species_name} × {atb_name}: n={len(y_tr_clean)} (R={n_pos}, S={n_neg}) ✓")

        training_stats.append({
            "species": species_name,
            "antibiotic": atb_name,
            "n_train": len(y_tr_clean),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "scale_pos_weight": round(scale_w, 2),
        })

print(f"\nClassifiers trained: {len(classifiers_original)} pairs (original + latent)")


############################################################
# VALIDATION ON SOURCE TEST SPLITS
############################################################
print("\n" + "=" * 60)
print("===== VALIDATION ON SOURCE TEST =====")
print("=" * 60)

validation_results = []

for pair_key in classifiers_original:
    species_name, atb_name = pair_key
    lgbm_orig = classifiers_original[pair_key]
    lgbm_lat = classifiers_latent[pair_key]

    species_idx = np.where(labels == species_name)[0]
    test_sp_idx = np.intersect1d(source_test_idx, species_idx)

    atb_idx = ab_list.index(atb_name)
    y_col = amr[:, atb_idx]
    y_te = y_col[test_sp_idx]
    m_te = ~np.isnan(y_te)

    X_te = data_norm[test_sp_idx[m_te]]
    Z_te = Z_all[test_sp_idx[m_te]]
    y_te_clean = y_te[m_te].astype(int)

    if len(y_te_clean) < 10 or len(np.unique(y_te_clean)) < 2:
        continue

    y_prob_orig = lgbm_orig.predict_proba(X_te)[:, 1]
    auc_orig = roc_auc_score(y_te_clean, y_prob_orig)
    pr_orig = average_precision_score(y_te_clean, y_prob_orig)

    y_prob_lat = lgbm_lat.predict_proba(Z_te)[:, 1]
    auc_lat = roc_auc_score(y_te_clean, y_prob_lat)
    pr_lat = average_precision_score(y_te_clean, y_prob_lat)

    print(f"  {species_name} × {atb_name}: "
          f"ORIG AUC={auc_orig:.3f} PR={pr_orig:.3f} | "
          f"LAT  AUC={auc_lat:.3f} PR={pr_lat:.3f}")

    validation_results.append({
        "species": species_name,
        "antibiotic": atb_name,
        "auc_original": auc_orig,
        "pr_original": pr_orig,
        "auc_latent": auc_lat,
        "pr_latent": pr_lat,
        "n_test": len(y_te_clean),
    })

df_val = pd.DataFrame(validation_results)
if len(df_val) > 0:
    print(f"\n--- Source test summary ---")
    print(f"Mean AUC — Original: {df_val['auc_original'].mean():.3f} | Latent: {df_val['auc_latent'].mean():.3f}")
    print(f"Mean PR  — Original: {df_val['pr_original'].mean():.3f} | Latent: {df_val['pr_latent'].mean():.3f}")


############################################################
# SAVE
############################################################
print("\n" + "=" * 60)
print("===== SAVING =====")
print("=" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/pretrained_lgbm") / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

path_orig = output_dir / "lgbm_original_ABCM.joblib"
path_lat = output_dir / "lgbm_latent_ABCM.joblib"

joblib.dump(classifiers_original, path_orig)
joblib.dump(classifiers_latent, path_lat)

print(f"Saved: {path_orig}")
print(f"Saved: {path_lat}")

# Stats
df_stats = pd.DataFrame(training_stats)
df_stats.to_csv(output_dir /"training_stats.csv", index=False)
print(f"Saved: {output_dir /'training_stats.csv'}")

if len(df_val) > 0:
    df_val.to_csv(output_dir /"validation_source_test.csv", index=False)
    print(f"Saved: {output_dir /'validation_source_test.csv'}")

# Metadata
meta_info = {
    "pretrained_model": str(PRETRAINED_MODEL_PATH),
    "pretrained_splits": str(PRETRAINED_SPLITS_PATH),
    "dataset": DATASET_PATH,
    "source_domains": SOURCE_DOMAINS,
    "antibiotics": ab_list,
    "latent_dim": LATENT_DIM,
    "n_classifiers": len(classifiers_original),
    "total_train_samples": int(len(source_train_idx)),
    "pairs_trained": [f"{s}×{a}" for s, a in classifiers_original.keys()],
    "timestamp": timestamp,
}
with open(output_dir / "metadata.json", "w") as f:
    json.dump(meta_info, f, indent=2)
print(f"Saved: {output_dir / 'metadata.json'}")

print(f"\n===== DONE. All saved in: {output_dir} =====")
