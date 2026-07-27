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
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from src.config.loader import load_config
from src.data.datasets import *
from src.data.preprocessing import *
from src.evaluation.eval import load_model, make_loader
from src.evaluation.adversarial_attacks import *
from models.baselines.mlp import MLPClassifier_Extended
from models.baselines.mlp_latent import LinearProbe_Extended
from models.deep.MultiVAEPriorAdv import MultiVAE_Bernoulli_SpeciesPrior_Adv_Extended


# ============================================================
# CONFIG
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

PRETRAINED_DALMA_ADV = Path("experiments/results/vae_multidecoder_prior_adv/20260718_094728/model.pth")
SPLITS_PATH          = Path("experiments/results/vae_multidecoder_prior/20260409_100009/data_splits.pkl")

EXPERIMENT_DIR = Path(f"experiments/adversarial_attacks/results/{timestamp}")
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = EXPERIMENT_DIR / "adversarial_robustness_adv.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

ADV_EPS    = 0.02
ADV_LAMBDA = 0.5


# ============================================================
# DATA LOADING
# ============================================================
cfg = load_config()
driams_dict  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict     = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict   = load_msumg(cfg["data"]["MSUMG_FULL"])

species_to_keep = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

mask_driams  = np.isin(driams_dict["label"],  species_to_keep)
mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
mask_rki     = np.isin(rki_dict["label"],     species_to_keep)
mask_msumg   = np.isin(msumg_dict["label"],   species_to_keep)

data_driams,  label_driams,  meta_driams  = driams_dict["data"][mask_driams],   driams_dict["label"][mask_driams],   driams_dict["meta"][mask_driams].reset_index(drop=True)
data_marisma, label_marisma, meta_marisma = marisma_dict["data"][mask_marisma], marisma_dict["label"][mask_marisma], marisma_dict["meta"][mask_marisma].reset_index(drop=True)
data_rki,     label_rki,     meta_rki     = rki_dict["data"][mask_rki],         rki_dict["label"][mask_rki],         rki_dict["meta"][mask_rki].reset_index(drop=True)
data_msumg,   label_msumg,   meta_msumg   = msumg_dict["data"][mask_msumg],     msumg_dict["label"][mask_msumg],     msumg_dict["meta"][mask_msumg].reset_index(drop=True)

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


# ============================================================
# LOAD SPLITS
# ============================================================
with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

SPLIT_NAMES = {
    "A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C",
    "MARISMA": "MARISMA", "RKI": "RKI",
}
all_tr_idx = np.concatenate([splits["splits_per_domain"][sk]["train_idx"] for sk in SPLIT_NAMES.values()])
all_va_idx = np.concatenate([splits["splits_per_domain"][sk]["val_idx"]   for sk in SPLIT_NAMES.values()])

X_tr_pool = data_final[all_tr_idx]
y_tr_pool = label_final[all_tr_idx]
X_va_pool = data_final[all_va_idx]
y_va_pool = label_final[all_va_idx]

le_pool = LabelEncoder()
y_tr_pool_enc = le_pool.fit_transform(y_tr_pool)
y_va_pool_enc = le_pool.transform(y_va_pool)
n_classes_pool = len(le_pool.classes_)


# ============================================================
# HELPERS
# ============================================================
def train_adversarial(model, trainloader, validloader, device,
                      eps=0.02, adv_lambda=0.5, epochs=50, lr=1e-3, patience=10):
    optimizer        = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None
    model.to(device)

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_clean = F.cross_entropy(model(x), y)
            x_adv = x.clone().detach().requires_grad_(True)
            F.cross_entropy(model(x_adv), y).backward()
            with torch.no_grad():
                x_adv = x + eps * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = x_adv.detach()
            optimizer.zero_grad()
            loss_adv = F.cross_entropy(model(x_adv), y)
            loss = (1.0 - adv_lambda) * loss_clean + adv_lambda * loss_adv
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(trainloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(device), y.to(device)
                val_loss += F.cross_entropy(model(x), y).item()
        val_loss /= len(validloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | tr={tr_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_probe_adversarial(vae, probe, trainloader, validloader, device, eps=0.02, adv_lambda=0.5, epochs=50, lr=1e-3, patience=10):
    optimizer        = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-5)
    best_val_loss    = float("inf")
    patience_counter = 0
    best_state       = None

    vae.to(device).eval()
    probe.to(device)

    for epoch in range(epochs):
        probe.train()
        tr_loss = 0.0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                mu, _ = vae.encoder(x)
            optimizer.zero_grad()
            loss_clean = F.cross_entropy(probe(mu), y)

            x_adv = x.clone().detach().requires_grad_(True)
            mu_adv, _ = vae.encoder(x_adv)
            F.cross_entropy(probe(mu_adv), y).backward()
            with torch.no_grad():
                x_adv = x + eps * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            x_adv = x_adv.detach()

            optimizer.zero_grad()
            with torch.no_grad():
                mu_adv, _ = vae.encoder(x_adv)
            loss_adv = F.cross_entropy(probe(mu_adv), y)

            loss = (1.0 - adv_lambda) * loss_clean + adv_lambda * loss_adv
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(trainloader)

        probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in validloader:
                x, y = x.to(device), y.to(device)
                mu, _ = vae.encoder(x)
                val_loss += F.cross_entropy(probe(mu), y).item()
        val_loss /= len(validloader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | tr={tr_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = copy.deepcopy(probe.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe

class VAEProbe(nn.Module):
    def __init__(self, vae, probe):
        super().__init__()
        self.vae   = vae
        self.probe = probe

    def forward(self, x):
        mu, _ = self.vae.encoder(x)
        return self.probe(mu)


# ============================================================
# LOAD DALMA ADVERSARIAL
# ============================================================
print("\n===== LOADING DALMA ADVERSARIAL =====")
vae_adv = load_model(
    MultiVAE_Bernoulli_SpeciesPrior_Adv_Extended(
        input_dim=data_final.shape[1],
        latent_dim=64, num_domains=5, n_species=n_classes_pool,
        adv_eps=ADV_EPS, adv_lambda=ADV_LAMBDA,
    ), PRETRAINED_DALMA_ADV)
vae_adv.to(device).eval()


# ============================================================
# TRAIN MLP RAW ADVERSARIAL
# ============================================================
print("\n===== TRAINING MLP RAW ADVERSARIAL =====")
mlp_raw_adv = MLPClassifier_Extended(
    input_dim=X_tr_pool.shape[1], n_species=n_classes_pool,
    epochs=50, lr=1e-3, patience=10,
)
train_adversarial(
    mlp_raw_adv,
    make_loader(X_tr_pool, y_tr_pool_enc, shuffle=True),
    make_loader(X_va_pool, y_va_pool_enc),
    device, eps=ADV_EPS, adv_lambda=ADV_LAMBDA,
)


# ============================================================
# TRAIN LINEAR PROBE ADVERSARIAL (over DALMA adversarial)
# ============================================================
print("\n===== TRAINING LINEAR PROBE ADVERSARIAL (DALMA adversarial) =====")
probe_adv = LinearProbe_Extended(
    latent_dim=64, n_species=n_classes_pool,
    epochs=50, lr=1e-3, patience=10,
)
train_probe_adversarial(
    vae_adv, probe_adv,
    make_loader(X_tr_pool, y_tr_pool_enc, shuffle=True),
    make_loader(X_va_pool, y_va_pool_enc),
    device, eps=ADV_EPS, adv_lambda=ADV_LAMBDA,
)

pipeline_adv_adv = VAEProbe(vae_adv, probe_adv).to(device).eval()


# ============================================================
# RUN SWEEP
# ============================================================
EPSILON_LIST = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1]
PGD_STEPS = (10, 40)

y_msumg_enc = le_pool.transform(label_msumg)

models_to_compare = {
    "MLP raw (adversarial)":       mlp_raw_adv,
    "VAE+Probe adv / probe adv":   pipeline_adv_adv,
}

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
