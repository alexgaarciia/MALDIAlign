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
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.config.loader import *
from src.data.io import *
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.data.datasets import load_msumg
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


# ============================================================
# DATA LOADING
# ============================================================
cfg = load_config()

driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])

species_to_keep = ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus", "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"]

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
dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA].reset_index(drop=True)
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB].reset_index(drop=True)
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC].reset_index(drop=True)
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD].reset_index(drop=True)

# Normalize
data_driams = row_minmax_normalize(data_driams)
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)

data_all = np.vstack([data_driams, data_marisma, data_msumg, data_rki])
label_all = np.concatenate([label_driams, label_marisma, label_msumg, label_rki])
meta_all = pd.concat([meta_driams, meta_marisma, meta_msumg, meta_rki], ignore_index=True)


# ============================================================
# LOAD DALMA + ENCODE LATENT
# ============================================================
vae_path = "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009/model.pth"
backbone = MultiVAE_Bernoulli_SpeciesPrior_Extended(input_dim=data_all.shape[1], latent_dim=64, num_domains=5, n_species=6)
vae = load_model(backbone, vae_path)

Z_all = encode_latent(model=vae, X=data_all, device=device)


# ============================================================
# PLOTS
# ============================================================
cosine_results_raw = compute_all_domain_species_cosine(data_all, meta_all, label_all)
cosine_results_latent = compute_all_domain_species_cosine(Z_all, meta_all, label_all)

cosine_results_raw_norm = {sp: normalize_diagonal_to_one(mat) for sp, mat in cosine_results_raw.items()}
cosine_results_latent_norm = {sp: normalize_diagonal_to_one(mat) for sp, mat in cosine_results_latent.items()}

all_hospitals = sorted(meta_all["hospital"].unique())
agg_cosine_raw_mean = aggregate_across_species(cosine_results_raw_norm, all_hospitals)
agg_cosine_latent_mean = aggregate_across_species(cosine_results_latent_norm, all_hospitals)

plot_similarity_heatmap_combined(agg_cosine_raw_mean, agg_cosine_latent_mean, save_path=EXPERIMENT_DIR/"similarity_cosine_combined.png", vmin=0.5, vmax=1)

# ============================================================
# COMPUTE STATS
# ============================================================
stats_raw = off_diagonal_stats(agg_cosine_raw_mean)
stats_latent = off_diagonal_stats(agg_cosine_latent_mean)

print("Original space (cosine):")
print(f"  Mean off-diagonal: {stats_raw['mean_off_diag']:.3f}")
print(f"  Std off-diagonal:  {stats_raw['std_off_diag']:.3f}")
print(f"  Min off-diagonal:  {stats_raw['min_off_diag']:.3f}")

print("\nDALMA latent space (cosine):")
print(f"  Mean off-diagonal: {stats_latent['mean_off_diag']:.3f}")
print(f"  Std off-diagonal:  {stats_latent['std_off_diag']:.3f}")
print(f"  Min off-diagonal:  {stats_latent['min_off_diag']:.3f}")
