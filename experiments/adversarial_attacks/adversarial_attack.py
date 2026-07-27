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
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.eval import load_model, encode_latent, make_loader
from src.evaluation.adversarial_attacks import *
from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PRETRAINED_DALMA = Path("experiments/results/vae_multidecoder_prior/20260409_100009/model.pth")
SPLITS_PATH = Path("experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")
EXPERIMENT_DIR = Path(f"experiments/adversarial_attacks/results/{timestamp}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXPERIMENT_DIR / "adversarial_robustness.csv"

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

species_to_keep = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

mask_driams = np.isin(driams_dict["label"], species_to_keep)
mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
mask_rki = np.isin(rki_dict["label"], species_to_keep)
mask_msumg = np.isin(msumg_dict["label"], species_to_keep)

data_driams,  label_driams,  meta_driams = driams_dict["data"][mask_driams], driams_dict["label"][mask_driams], driams_dict["meta"][mask_driams].reset_index(drop=True)
data_marisma, label_marisma, meta_marisma = marisma_dict["data"][mask_marisma], marisma_dict["label"][mask_marisma], marisma_dict["meta"][mask_marisma].reset_index(drop=True)
data_rki, label_rki, meta_rki = rki_dict["data"][mask_rki], rki_dict["label"][mask_rki], rki_dict["meta"][mask_rki].reset_index(drop=True)
data_msumg, label_msumg, meta_msumg = msumg_dict["data"][mask_msumg], msumg_dict["label"][mask_msumg], msumg_dict["meta"][mask_msumg].reset_index(drop=True)

# Split DRIAMS
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"

dataA, labelA = data_driams[maskA], label_driams[maskA]
dataB, labelB = data_driams[maskB], label_driams[maskB]
dataC, labelC = data_driams[maskC], label_driams[maskC]

# Normalize
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)

# Merge source domains
data_final  = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])

print(f"Source data: {data_final.shape} | MS-UMG OOD: {data_msumg.shape}")


# ============================================================
# LOAD SPLITS
# ============================================================
with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

SPLIT_NAMES = {
    "A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C",
    "MARISMA": "MARISMA", "RKI": "RKI",
}

all_tr_idx = np.concatenate([
    splits["splits_per_domain"][sk]["train_idx"]
    for sk in SPLIT_NAMES.values()
])
all_va_idx = np.concatenate([
    splits["splits_per_domain"][sk]["val_idx"]
    for sk in SPLIT_NAMES.values()
])

X_tr_pool = data_final[all_tr_idx]
y_tr_pool = label_final[all_tr_idx]
X_va_pool = data_final[all_va_idx]
y_va_pool = label_final[all_va_idx]

le_pool = LabelEncoder()
y_tr_pool_enc = le_pool.fit_transform(y_tr_pool)
y_va_pool_enc = le_pool.transform(y_va_pool)
n_classes_pool = len(le_pool.classes_)
print(f"Train: {len(X_tr_pool)} | Val: {len(X_va_pool)} | Classes: {n_classes_pool}")


# ============================================================
# LOAD DALMA + ENCODE LATENT
# ============================================================
print("\n===== LOADING DALMA =====")
vae = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=data_final.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species=n_classes_pool,
    ), PRETRAINED_DALMA)
vae.to(device)
vae.eval()

Z_tr_pool = encode_latent(vae, X_tr_pool, device)
Z_va_pool = encode_latent(vae, X_va_pool, device)
Z_msumg   = encode_latent(vae, data_msumg, device)
print(f"Z_train: {Z_tr_pool.shape} | Z_val: {Z_va_pool.shape} | Z_msumg: {Z_msumg.shape}")


# ============================================================
# TRAIN MLPs
# ============================================================
print("\n===== TRAINING MLP RAW (ALL pooled) =====")
mlp_raw = MLPClassifier_Extended(
    input_dim=X_tr_pool.shape[1],
    n_species=n_classes_pool,
    epochs=50, lr=1e-3, patience=10,
)
mlp_raw.trainloop(
    make_loader(X_tr_pool, y_tr_pool_enc, shuffle=True),
    make_loader(X_va_pool, y_va_pool_enc),
    device,
)

print("\n===== TRAINING LINEAR PROBE (ALL pooled) =====")
probe = LinearProbe_Extended(
    latent_dim=Z_tr_pool.shape[1],
    n_species=n_classes_pool,
    epochs=50, lr=1e-3, patience=10,
)
probe.trainloop(
    make_loader(Z_tr_pool, y_tr_pool_enc, shuffle=True),
    make_loader(Z_va_pool, y_va_pool_enc),
    device,
)


# ============================================================
# VAEProbe WRAPPER — encoder + probe end-to-end
# ============================================================
class VAEProbe(nn.Module):
    """
    Wrapper encoder (VAE) + probe in a single nn.Module.
    Allows end-to-end backprop from logits to x_raw.
    """
    def __init__(self, vae, probe):
        super().__init__()
        self.vae   = vae
        self.probe = probe

    def forward(self, x):
        mu, _ = self.vae.encoder(x)
        return self.probe(mu)

pipeline = VAEProbe(vae, probe).to(device)
pipeline.eval()


# ============================================================
# RUN SWEEP
# ============================================================
EPSILON_LIST = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]
PGD_STEPS = (10, 40)

y_msumg_enc = le_pool.transform(label_msumg)

models_to_compare = {
    "MLP raw":   mlp_raw,   
    "VAE+Probe": pipeline,}

print("\n===== ADVERSARIAL ROBUSTNESS SWEEP =====")
df_adv = sweep_attacks(
    models_to_compare,
    data_msumg, y_msumg_enc,
    eps_list=EPSILON_LIST,
    pgd_steps_list=PGD_STEPS,
    device=device,
)

df_adv.to_csv(OUT_PATH, index=False)
print(f"\nResults saved: {OUT_PATH}")
print("\n===== Experiment completed successfully =====")
