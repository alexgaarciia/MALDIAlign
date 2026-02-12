####################
# PATH & SETUP
####################
from pathlib import Path
import os, sys
from datetime import datetime
import pickle

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

print("\n===== INITIALIZING EXPERIMENT (MS-UMG FINETUNE) =====")

EXP_NAME = "finetuning_vae_multidecoder_prior_ABC_MAR_RKI"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = Path("experiments/finetuning/results") / EXP_NAME / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)

print("Experiment directory created.")


####################
# IMPORTS
####################
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd

from utils.config import load_config
from utils.data import load_driams, load_marisma, load_msumg, load_rki, row_minmax_normalize
from utils.eval import eval_model, run_tsne_evaluation
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from sklearn.model_selection import train_test_split


####################
# LOAD DATASETS
####################
print("\n===== LOADING DATA =====")

cfg = load_config()

driams_dict  = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
msumg_dict   = load_msumg(cfg["data"]["MSUMG_PKL"])  # ya filtra agar
rki_dict     = load_rki(cfg["data"]["RKI_PKL"])

data_driams, label_driams, meta_driams = driams_dict["data"], driams_dict["label"], driams_dict["meta"]
data_marisma, label_marisma, meta_marisma = marisma_dict["data"], marisma_dict["label"], marisma_dict["meta"]
data_rki, label_rki, meta_rki = rki_dict["data"], rki_dict["label"], rki_dict["meta"]
data_msumg, label_msumg, meta_msumg = msumg_dict["data"], msumg_dict["label"], msumg_dict["meta"]


####################
# SPLIT DRIAMS BY HOSPITAL
####################
maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA]
dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB]
dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC]
dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD]


####################
# NORMALIZATION (row min-max)
####################
dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
dataD = row_minmax_normalize(dataD)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)
data_msumg = row_minmax_normalize(data_msumg)


####################
# LOAD FINETUNING SPLITS
####################
print("Loading finetuning indices...")

with open("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/output_data/splits_20260212_222230/splits_idx.pkl", "rb") as f:
    splits_idx = pickle.load(f)

def select(data, label, meta, idx):
    return data[idx], label[idx], meta.iloc[idx].reset_index(drop=True)


####################
# BUILD FINETUNING DATASET (MS-UMG ONLY as new domain)
####################
print("Building finetuning dataset...")

dataA_ft, labelA_ft, metaA_ft = select(dataA, labelA, metaA, splits_idx["DRIAMS_A"]["finetuning"])
dataB_ft, labelB_ft, metaB_ft = select(dataB, labelB, metaB, splits_idx["DRIAMS_B"]["finetuning"])
dataC_ft, labelC_ft, metaC_ft = select(dataC, labelC, metaC, splits_idx["DRIAMS_C"]["finetuning"])
dataM_ft, labelM_ft, metaM_ft = select(data_marisma, label_marisma, meta_marisma, splits_idx["MARISMA"]["finetuning"])
dataR_ft, labelR_ft, metaR_ft = select(data_rki, label_rki, meta_rki, splits_idx["RKI"]["finetuning"])
dataMS_ft, labelMS_ft, metaMS_ft = select(data_msumg, label_msumg, meta_msumg, splits_idx["MS-UMG"]["finetuning"])

X_ft = np.vstack([dataA_ft, dataB_ft, dataC_ft, dataM_ft, dataR_ft, dataMS_ft])
y_ft = np.concatenate([labelA_ft, labelB_ft, labelC_ft, labelM_ft, labelR_ft, labelMS_ft])
meta_ft = pd.concat([metaA_ft, metaB_ft, metaC_ft, metaM_ft, metaR_ft, metaMS_ft], ignore_index=True)

DOMAIN_MAP = {
    "DRIAMS_A": 0,
    "DRIAMS_B": 1,
    "DRIAMS_C": 2,
    "MARISMA":  3,
    "RKI":      4,
    "MS-UMG":   5,  
}

domain_ft = meta_ft["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
_, species_ft = np.unique(y_ft, return_inverse=True)
species_ft = species_ft.astype(np.int64)


####################
# 7. LOAD BASE MODEL
####################
print("Loading base pretrained model...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = MultiVAE_Bernoulli_SpeciesPrior_Extended(
    input_dim=X_ft.shape[1],
    latent_dim=64,
    num_domains=5,   # base: A,B,C,M,R
    n_species=6,
)

vae.load_state_dict(
    torch.load(
        "/export/usuarios01/agnavarr/MALDIAlign/experiments/results/finetuning_vae_multidecoder_prior/20260211_143045/model.pth",
        map_location=device,
    )
)
vae.to(device)


####################
# 8. ADD NEW DECODER (MS-UMG)
####################
print("Adding new decoder for MS-UMG...")

old_decoders = vae.decoder.net
latent_dim = old_decoders[0][0].in_features
output_dim = old_decoders[0][-2].out_features

new_decoder = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Linear(1024, output_dim),
    nn.Sigmoid(),
)

vae.decoder.net = nn.ModuleList(list(old_decoders) + [new_decoder])
vae.decoder.num_domains = 6  # 5 base + 1 nuevo
vae.to(device)


####################
# TRAIN / VALID SPLIT
####################
print("Preparing training split...")

X_tr, X_val, d_tr, d_val, s_tr, s_val = train_test_split(
    X_ft,
    domain_ft,
    species_ft,
    test_size=0.2,
    stratify=species_ft,
    random_state=42
)

train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(d_tr, dtype=torch.long),
        torch.tensor(s_tr, dtype=torch.long),
    ),
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(d_val, dtype=torch.long),
        torch.tensor(s_val, dtype=torch.long),
    ),
    batch_size=32,
    shuffle=False,
)


####################
# FINETUNING (FULL UNFREEZE)
####################
print("\n===== STARTING FINETUNING =====")

vae.prior.mu_embed.weight.requires_grad = False
vae.prior.logvar_embed.weight.requires_grad = False

vae.epochs = 30
vae.lr = 1e-5
vae.annealing_epochs = 10
vae.patience = 10

vae.optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, vae.parameters()),
    lr=vae.lr,
    weight_decay=1e-5
)

vae.trainloop(
    trainloader=train_loader,
    validloader=val_loader,
    device=device
)

torch.save(
    vae.state_dict(),
    experiment_dir / "model_finetuned_MS_only.pth",
)

print("Finetuning complete. Model saved.")


####################
# LATENT EVALUATION + t-SNE
####################
print("\n===== RUNNING LATENT EVALUATION =====")

X_eval = np.vstack([dataA, dataB, dataC, data_marisma, data_rki, dataMS_ft])
y_eval = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki, labelMS_ft])
meta_eval = pd.concat([metaA, metaB, metaC, meta_marisma, meta_rki, metaMS_ft], ignore_index=True)
meta_eval["year"] = meta_eval["year"].fillna("unknown").astype(str)

domain_eval = meta_eval["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
_, species_eval = np.unique(y_eval, return_inverse=True)
species_eval = species_eval.astype(np.int64)

eval_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_eval, dtype=torch.float32),
        torch.tensor(domain_eval, dtype=torch.long),
        torch.tensor(species_eval, dtype=torch.long),
    ),
    batch_size=256,
    shuffle=False
)

vae.eval()
mus_all = eval_model(vae, eval_loader, device)

run_tsne_evaluation(
    mus_all=mus_all,
    label_final=y_eval,
    meta_final=meta_eval,
    output_dir=experiment_dir,
    prefix="FINETUNED_MS_ONLY"
)

print("Latent evaluation finished.")
print("\n===== EXPERIMENT COMPLETE =====")
