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
if target is not None:
    os.chdir(target)
print("Working directory:", os.getcwd())


# ============================================================
# IMPORTS
# ============================================================
import pickle
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.config.loader import *
from src.data.io import *
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.visualization.viz import *
from src.evaluation.eval import load_model, encode_latent
from src.evaluation.spectrum_similarity import *
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_DIR = Path(f"experiments/spectra_similarity/results/{timestamp}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
VAE_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")


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
mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
mask_rki = np.isin(rki_dict["label"], species_to_keep)
mask_msumg = np.isin(msumg_dict["label"], species_to_keep)

data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]
meta_marisma = marisma_dict["meta"][mask_marisma].reset_index(drop=True)

data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]
meta_rki = rki_dict["meta"][mask_rki].reset_index(drop=True)

data_msumg = msumg_dict["data"][mask_msumg]
label_msumg = msumg_dict["label"][mask_msumg]
meta_msumg = msumg_dict["meta"][mask_msumg].reset_index(drop=True)

# Split DRIAMS
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA].reset_index(drop=True)
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB].reset_index(drop=True)
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC].reset_index(drop=True)
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD].reset_index(drop=True)

# Normalize
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)

data_final = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])
meta_final = pd.concat([metaA, metaB, metaC, meta_marisma, meta_rki], ignore_index=True)

# Load splits
with open(VAE_DIR / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)
print("Available domains in splits:", list(splits["splits_per_domain"].keys()))

SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA", "RKI": "RKI"}
all_te_idx = np.concatenate([
    splits["splits_per_domain"][sk]["test_idx"]
    for sk in SPLIT_NAMES.values()
    if sk in splits["splits_per_domain"]
])

data_test = data_final[all_te_idx]
label_test = label_final[all_te_idx]
meta_test = meta_final.iloc[all_te_idx].reset_index(drop=True)


# ============================================================
# LOAD DALMA + ENCODE LATENT
# ============================================================
vae = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=data_final.shape[1], latent_dim=64, num_domains=5, n_species=6
    ),
    VAE_DIR / "model.pth"
)

Z_test = encode_latent(model=vae, X=data_test, device=device)
Z_D = encode_latent(model=vae, X=dataD, device=device)
Z_MSUMG = encode_latent(model=vae, X=data_msumg, device=device)

data_combined = np.vstack([data_test, dataD, data_msumg])
label_combined = np.concatenate([label_test, labelD, label_msumg])
meta_combined = pd.concat([meta_test, metaD, meta_msumg], ignore_index=True)
Z_combined = np.vstack([Z_test, Z_D, Z_MSUMG])


# ============================================================
# COMPUTE SIMILARITY
# ============================================================
cosine_results_raw = compute_all_domain_species_cosine(data_combined, meta_combined, label_combined)
cosine_results_latent = compute_all_domain_species_cosine(Z_combined, meta_combined, label_combined)

cosine_results_raw_norm = {sp: normalize_diagonal_to_one(mat) for sp, mat in cosine_results_raw.items()}
cosine_results_latent_norm = {sp: normalize_diagonal_to_one(mat) for sp, mat in cosine_results_latent.items()}

all_hospitals = sorted(meta_combined["hospital"].unique())
agg_cosine_raw_mean = aggregate_across_species(cosine_results_raw_norm, all_hospitals)
agg_cosine_latent_mean = aggregate_across_species(cosine_results_latent_norm, all_hospitals)


# ============================================================
# PLOTS
# ============================================================
plot_similarity_heatmap_combined(
    agg_cosine_raw_mean, agg_cosine_latent_mean,
    save_path=EXPERIMENT_DIR / "similarity_cosine_combined.png",
    vmin=0.5, vmax=1
)


# ============================================================
# STATS
# ============================================================
stats_raw = off_diagonal_stats(agg_cosine_raw_mean)
stats_latent = off_diagonal_stats(agg_cosine_latent_mean)

print("\nOriginal space (cosine, normalized):")
print(f"  Mean off-diagonal: {stats_raw['mean_off_diag']:.3f}")
print(f"  Std off-diagonal:  {stats_raw['std_off_diag']:.3f}")
print(f"  Min off-diagonal:  {stats_raw['min_off_diag']:.3f}")

print("\nDALMA latent space (cosine, normalized):")
print(f"  Mean off-diagonal: {stats_latent['mean_off_diag']:.3f}")
print(f"  Std off-diagonal:  {stats_latent['std_off_diag']:.3f}")
print(f"  Min off-diagonal:  {stats_latent['min_off_diag']:.3f}")
