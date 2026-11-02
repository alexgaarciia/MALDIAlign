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
import torch
import pickle
import argparse
import numpy as np

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from datetime import datetime

from maldi_nn.models import MaldiTransformer

from src.config.loader import *
from src.data.datasets import *
from src.data.preprocessing import *
from src.finetuning.malditransformer import FTDataset, FTDataModule


# ============================================================
# ARG PARSING
# ============================================================
MODEL_SIZES = ["S", "M", "L", "XL"]

parser = argparse.ArgumentParser(description="MaldiTransformer fine-tuning (FT)")
parser.add_argument(
    "--size", "-s",
    type=str,
    required=True,
    choices=MODEL_SIZES,
    help="MaldiTransformer size: S, M, L or XL",
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

SIZE = args.size
CHECKPOINT_PATH = Path(f"assets/MaldiTransformer{SIZE}.ckpt")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"experiments/species_id/results/{timestamp}")
N_PEAKS = 200
EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")


# ============================================================
# DATA LOADING
# ============================================================
print("\n" + "="*60)
print("===== LOADING DATA =====")
print("="*60)

cfg = load_config()

driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict = load_rki(cfg["data"]["RKI_FULL"])

species_to_keep = ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus", "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"]

mask_driams = np.isin(driams_dict["label"], species_to_keep)
data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

mask_marisma = np.isin(marisma_dict["label"], species_to_keep)
data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]

mask_rki = np.isin(rki_dict["label"], species_to_keep)
data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]

maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"

dataA, labelA = data_driams[maskA], label_driams[maskA]
dataB, labelB = data_driams[maskB], label_driams[maskB]
dataC, labelC = data_driams[maskC], label_driams[maskC]

dataA = row_minmax_normalize(dataA)
dataB = row_minmax_normalize(dataB)
dataC = row_minmax_normalize(dataC)
data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)

data_list = [dataA, dataB, dataC, data_marisma, data_rki]
label_list = [labelA, labelB, labelC, label_marisma, label_rki]

data_final = np.vstack(data_list)
label_final = np.concatenate(label_list)

print("\n===== DATA LOADED =====")

with open(EXPERIMENT_DIR / "data_splits.pkl", "rb") as f:
    splits = pickle.load(f)

print("\n===== SPLITS LOADED =====")


# ============================================================
# POOLED TRAIN/VAL
# ============================================================
SPLIT_NAMES = {
    "A": "DRIAMS_A",
    "B": "DRIAMS_B",
    "C": "DRIAMS_C",
    "MARISMA": "MARISMA",
    "RKI": "RKI",
}

all_tr_idx = np.concatenate([
    splits["splits_per_domain"][sk]["train_idx"]
    for sk in SPLIT_NAMES.values()
    if sk in splits["splits_per_domain"]
])
all_va_idx = np.concatenate([
    splits["splits_per_domain"][sk]["val_idx"]
    for sk in SPLIT_NAMES.values()
    if sk in splits["splits_per_domain"]
])

species_to_idx = {s: i for i, s in enumerate(species_to_keep)}
y_tr_ft = np.array([species_to_idx[s] for s in label_final[all_tr_idx]])
y_va_ft = np.array([species_to_idx[s] for s in label_final[all_va_idx]])

print(f"Pooled train: {len(all_tr_idx)} | Pooled val: {len(all_va_idx)}")

train_ds = FTDataset(data_final[all_tr_idx], y_tr_ft, n_peaks=N_PEAKS)
val_ds = FTDataset(data_final[all_va_idx], y_va_ft, n_peaks=N_PEAKS)
dm = FTDataModule(train_ds, val_ds, batch_size=args.batch_size)


# ============================================================
# MODEL: pretrained backbone + new head of 6 classes
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size_to_layer_dims = {"S": [160, 4], "M": [184, 6], "L": [232, 8], "XL": [304, 10]}
dim, depth = size_to_layer_dims[SIZE]

model = MaldiTransformer(
    depth, dim, n_classes=len(species_to_keep), n_heads=8, dropout=0.2,
    p=0.5, clf=True, clf_train_p=1/100, lmbda=1.0,
    lr=1e-5, weight_decay=0, lr_decay_factor=1, warmup_steps=250,
)

pretrained = MaldiTransformer.load_from_checkpoint(CHECKPOINT_PATH, map_location=device, strict=False)
backbone_state = {k: v for k, v in pretrained.transformer.state_dict().items() if not k.startswith("output_head")}
model.transformer.load_state_dict({**model.transformer.state_dict(), **backbone_state})
model.output_head.load_state_dict(pretrained.output_head.state_dict())


# ============================================================
# FINETUNING
# ============================================================
print("\n===== FINE-TUNING MALDI TRANSFORMER =====")

ASSETS_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
FT_CKPT_PATH = ASSETS_DIR / f"MaldiTransformer{SIZE}_ft.ckpt"

checkpoint_callback = ModelCheckpoint(
    dirpath=ASSETS_DIR,
    filename=f"MaldiTransformer{SIZE}_ft",
    monitor="val_clfacc",
    mode="max",
    save_top_k=1,
)

trainer = L.Trainer(
    max_epochs=args.epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    gradient_clip_val=1,
    callbacks=[checkpoint_callback],
)
trainer.fit(model, dm)

print("\n===== SAVED =====")
print("Best FT checkpoint:", checkpoint_callback.best_model_path)
