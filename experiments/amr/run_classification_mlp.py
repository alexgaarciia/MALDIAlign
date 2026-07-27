############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys

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
import pickle
from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import evaluate_amr_head
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ
from models.deep.MultiVAEPriorAMRHeadZEmb import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb
from models.baselines.mlps_amr import AMRProbeRaw, AMRProbeRawNoSpecies


############################################################
# CONFIG
############################################################
cfg = load_config()

VAE_AMR_HEAD_EMB_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150807")
VAE_AMR_HEAD_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449")
SPLITS_PATH = VAE_AMR_HEAD_DIR / "data_splits.pkl"

timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"/export/usuarios_ml4ds/agnavarr/MALDIAlign/experiments/results/amr_zero_shot/{timestamp}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae":          ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Escherichia_Coli":               ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Enterobacter_cloacae_complex":   ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Pseudomonas_Aeruginosa":         ["Meropenem", "Amikacin"],
    "Staphylococcus_Aureus":          ["Oxacillin", "Clindamycin", "Erythromycin"],
    "Enterococcus_Faecium":           ["Vancomycin"],
}

ALL_ANTIBIOTICS = [
    "Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin",
    "Piperacillin-Tazobactam", "Amikacin", "Oxacillin",
    "Clindamycin", "Erythromycin", "Vancomycin",
]

SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA"}


############################################################
# LOAD VAE 
############################################################
print("\n===== LOADING VAE AMR HEAD =====")
state_amr = torch.load(VAE_AMR_HEAD_DIR / "model.pth", map_location="cpu")

vae_amr_head = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
    input_dim=6000, latent_dim=128, num_domains=4,
    n_species=6, n_antibiotics=10, lambda_amr=1,
    antibiotic_names=ALL_ANTIBIOTICS,
)
vae_amr_head.load_state_dict(state_amr)
vae_amr_head.to(device).eval()
for param in vae_amr_head.parameters():
    param.requires_grad = False

print("\n===== LOADING VAE AMR HEAD (species embedding) =====")
state_amr_emb = torch.load(VAE_AMR_HEAD_EMB_DIR / "model.pth", map_location="cpu")
vae_amr_head_emb = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb(
    input_dim=6000, latent_dim=128, num_domains=4,
    n_species=6, n_antibiotics=10, lambda_amr=1,
    antibiotic_names=ALL_ANTIBIOTICS,
    species_emb_dim=30,
)
vae_amr_head_emb.load_state_dict(state_amr_emb)
vae_amr_head_emb.to(device).eval()
for param in vae_amr_head_emb.parameters():
    param.requires_grad = False


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA =====")

with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)
domain_splits = splits.get("splits_per_domain", {})

with open("/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl", "rb") as f:
    amr_pkl = pickle.load(f)

data_raw   = amr_pkl["data"]
labels_all = amr_pkl["label"]
amr_matrix = amr_pkl["amr"]
amr_cols   = amr_pkl["antibiotics"]
meta_all   = pd.DataFrame(amr_pkl["meta"])

data_raw = row_minmax_normalize(data_raw)
ab_to_idx_full = {ab: i for i, ab in enumerate(amr_cols)}
all_ab_indices = [ab_to_idx_full[ab] for ab in ALL_ANTIBIOTICS if ab in ab_to_idx_full]

le_species = LabelEncoder()
le_species.fit(sorted(SPECIES_CONFIG.keys()))
print(f"Species order: {list(le_species.classes_)}")
species_encoded_all = le_species.transform(labels_all)

all_tr_idx = np.concatenate([
    domain_splits[sk]["train_idx"]
    for sk in SPLIT_NAMES.values() if sk in domain_splits
])
all_va_idx = np.concatenate([
    domain_splits[sk]["val_idx"]
    for sk in SPLIT_NAMES.values() if sk in domain_splits
])

mask_msumg = meta_all["hospital"] == "MS-UMG"
print(f"Train pool: {len(all_tr_idx)} | Val pool: {len(all_va_idx)}")
print(f"MS-UMG: {mask_msumg.sum()}")


############################################################
# TRAIN RAW MLP GLOBAL (all species + onehot)
############################################################
print("\n===== TRAINING RAW MLP GLOBAL (all species + species onehot) =====")

X_tr_raw_all   = data_raw[all_tr_idx]
X_va_raw_all   = data_raw[all_va_idx]
y_tr_all       = amr_matrix[all_tr_idx][:, all_ab_indices]
y_va_all       = amr_matrix[all_va_idx][:, all_ab_indices]
species_tr_all = species_encoded_all[all_tr_idx]
species_va_all = species_encoded_all[all_va_idx]

raw_mlp_global = AMRProbeRaw(
    input_dim=6000,
    n_species=len(le_species.classes_),
    species_emb_dim=30,
    n_antibiotics=len(ALL_ANTIBIOTICS),
    epochs=1200, lr=1e-3, patience=15,
)
raw_mlp_global.trainloop(
    X_tr_raw_all, y_tr_all, X_va_raw_all, y_va_all,
    species_tr_all, species_va_all,
    device,
)
print("Raw MLP global trained.")


############################################################
# EVALUATION HELPERS
############################################################
def evaluate_auroc(probe, X_te, y_te, antibiotics, device):
    probs = probe.predict_proba(X_te, device)
    results = {}
    for j, ab in enumerate(antibiotics):
        y_j  = y_te[:, j]
        mask = ~np.isnan(y_j)
        if mask.sum() < 10 or len(np.unique(y_j[mask])) < 2:
            results[ab] = float("nan")
        else:
            results[ab] = roc_auc_score(y_j[mask], probs[mask, j])
    return results

def evaluate_auroc_global(probe, X_te, y_te_full, species_te, antibiotics, device):
    probs_full = probe.predict_proba(X_te, species_te, device)
    results = {}
    for ab in antibiotics:
        if ab not in ALL_ANTIBIOTICS:
            results[ab] = float("nan")
            continue
        j_global = ALL_ANTIBIOTICS.index(ab)
        j_full   = ab_to_idx_full[ab]
        y_j  = y_te_full[:, j_full]
        mask = ~np.isnan(y_j)
        if mask.sum() < 10 or len(np.unique(y_j[mask])) < 2:
            results[ab] = float("nan")
        else:
            results[ab] = roc_auc_score(y_j[mask], probs_full[mask, j_global])
    return results

def evaluate_amr_head_auroc(model, X_te, y_te, antibiotics, ab_to_idx, device, species_te=None, n_species=None):
    ab_col_indices = [ab_to_idx[ab] for ab in antibiotics if ab in ab_to_idx]
    y_te_sub = y_te[:, ab_col_indices]
    results_raw = evaluate_amr_head(
        model, X_te, y_te_sub, antibiotics, device,
        species=species_te, n_species=n_species,
    )
    return {ab: results_raw[ab]["auc"] if ab in results_raw else float("nan")
            for ab in antibiotics}


############################################################
# MAIN EVALUATION LOOP
############################################################
print("\n===== ZERO-SHOT AMR EVALUATION (MS-UMG only) =====")

all_results = []

for species, antibiotics in SPECIES_CONFIG.items():
    print(f"\n{'='*70}")
    print(f"  SPECIES: {species}")
    print(f"  Antibiotics: {antibiotics}")
    print(f"{'='*70}")

    ab_col_indices = [ab_to_idx_full[ab] for ab in antibiotics if ab in ab_to_idx_full]
    n_ab = len(ab_col_indices)

    mask_sp = (labels_all == species)

    mask_tr = np.zeros(len(labels_all), dtype=bool)
    mask_tr[all_tr_idx] = True
    mask_tr &= mask_sp

    mask_va = np.zeros(len(labels_all), dtype=bool)
    mask_va[all_va_idx] = True
    mask_va &= mask_sp

    X_tr_sp = data_raw[mask_tr]
    X_va_sp = data_raw[mask_va]
    y_tr    = amr_matrix[mask_tr][:, ab_col_indices]
    y_va    = amr_matrix[mask_va][:, ab_col_indices]

    print(f"  Train (species): {mask_tr.sum()} | Val (species): {mask_va.sum()}")

    mask_msumg_sp = mask_msumg.values & mask_sp

    X_msumg_raw       = data_raw[mask_msumg_sp]
    y_msumg_full      = amr_matrix[mask_msumg_sp]
    y_msumg           = y_msumg_full[:, ab_col_indices]
    species_msumg_enc = species_encoded_all[mask_msumg_sp]

    print(f"\n  --- Raw spectra (MLP global + species onehot) ---")
    results_msumg_raw = evaluate_auroc_global(
        raw_mlp_global, X_msumg_raw, y_msumg_full, species_msumg_enc, antibiotics, device
    )
    for ab in antibiotics:
        all_results.append({
            "species": species, "model": "Raw (global + species onehot)", "antibiotic": ab,
            "MSUMG_AUROC": results_msumg_raw.get(ab, float("nan")),})
        print(f"    {ab:35s}  MSUMG={results_msumg_raw.get(ab, float('nan')):.3f}")

    print(f"\n  --- Raw spectra (MLP per species, no onehot) ---")
    probe_sp = AMRProbeRawNoSpecies(
        n_antibiotics=n_ab, epochs=1200, lr=1e-3, patience=15
    )
    probe_sp.trainloop(X_tr_sp, y_tr, X_va_sp, y_va, device)

    results_msumg_sp = evaluate_auroc(probe_sp, X_msumg_raw, y_msumg, antibiotics, device)
    for ab in antibiotics:
        all_results.append({
            "species": species, "model": "Raw (per species, no onehot)", "antibiotic": ab,
            "MSUMG_AUROC": results_msumg_sp.get(ab, float("nan")),
        })
        print(f"    {ab:35s}  MSUMG={results_msumg_sp.get(ab, float('nan')):.3f}")

    print(f"\n  --- VAE + AMR head ---")
    results_msumg_head = evaluate_amr_head_auroc(
        vae_amr_head, X_msumg_raw, y_msumg_full, antibiotics, ab_to_idx_full, device,
        species_te=species_msumg_enc, n_species=len(le_species.classes_),
    )
    for ab in antibiotics:
        all_results.append({
            "species": species, "model": "VAE AMR head", "antibiotic": ab,
            "MSUMG_AUROC": results_msumg_head.get(ab, float("nan")),
        })
        print(f"    {ab:35s}  MSUMG={results_msumg_head.get(ab, float('nan')):.3f}")

    print(f"\n  --- VAE + AMR head (species embedding) ---")
    results_msumg_head_emb = evaluate_amr_head_auroc(
        vae_amr_head_emb, X_msumg_raw, y_msumg_full, antibiotics, ab_to_idx_full, device,
        species_te=species_msumg_enc, n_species=None, 
    )
    for ab in antibiotics:
        all_results.append({
            "species": species, "model": "VAE AMR head (species emb)", "antibiotic": ab,
            "MSUMG_AUROC": results_msumg_head_emb.get(ab, float("nan")),
        })
        print(f"    {ab:35s}  MSUMG={results_msumg_head_emb.get(ab, float('nan')):.3f}")


############################################################
# SAVE RESULTS
############################################################
df_results = pd.DataFrame(all_results)
out_csv = OUTPUT_DIR / "amr_zero_shot_results.csv"
df_results.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

print(f"\n===== MSUMG_AUROC =====")
pivot = df_results.pivot_table(
    index=["species", "antibiotic"],
    columns="model",
    values="MSUMG_AUROC",
)
print(pivot.to_string())

print(f"\n===== DONE — results in {OUTPUT_DIR} =====")
