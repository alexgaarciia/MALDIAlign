############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys
import copy
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


############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from src.data.io import load_pkl
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import evaluate_amr_head
from models.deep.MultiVAEPriorAMRHeadZ import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ
from models.deep.MultiVAEPriorAMRHeadZEmb import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb
from models.baselines.mlps_amr import AMRProbeRaw, AMRProbeRawNoSpecies
from torch.utils.data import DataLoader, TensorDataset


############################################################
# CONFIGURATION
############################################################
SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TARGET_DOMAIN  = "MS-UMG"

DATASET_PATH  = "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl"
SPLITS_DIR    = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_AllSpecies_MSUMG_20260717_080505")
SOURCE_SPLITS = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/data_splits.pkl")

VAE_NO_SPECIES_PATH  = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150449/model.pth")
VAE_EMB_SPECIES_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr_all_species_all_abs/20260716_150807/model.pth")

OUTPUT_PATH = Path("/export/usuarios_ml4ds/agnavarr/MALDIAlign/finetuning_amr_all_species_msumg_global")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

GRID_PREV       = [0]
GRID_NEW        = [50, 100, 250, 500, 1000]
N_PARTITIONS    = 10
FT_LR           = 1e-5
FT_EPOCHS       = 50
FT_PATIENCE     = 15
LATENT_DIM      = 128
LAMBDA_AMR      = 100
NUM_DOMAINS     = 4
SPECIES_EMB_DIM = 30

SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae":          ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Escherichia_Coli":               ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
    "Staphylococcus_Aureus":          ["Oxacillin", "Clindamycin", "Erythromycin"],
    "Pseudomonas_Aeruginosa":         ["Meropenem", "Amikacin"],
    "Enterococcus_Faecium":           ["Vancomycin"],
    "Enterobacter_cloacae_complex":   ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
}

ALL_ANTIBIOTICS = [
    "Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin",
    "Piperacillin-Tazobactam", "Amikacin", "Oxacillin",
    "Clindamycin", "Erythromycin", "Vancomycin",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


############################################################
# LOAD DATASET
############################################################
print("\n===== LOADING DATASET =====")
dataset    = load_pkl(DATASET_PATH)
data_raw   = dataset["data"]
amr_raw    = dataset["amr"]
ab_list_raw = list(dataset["antibiotics"])
labels_all = dataset["label"]
raw_meta   = dataset["meta"]
meta_all   = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) \
             else pd.DataFrame(raw_meta)

if "agar" in meta_all.columns:
    chrom = ((meta_all["hospital"] == TARGET_DOMAIN) &
             (meta_all["agar"] == "chrom")).values
    keep       = ~chrom
    data_raw   = data_raw[keep]
    amr_raw    = amr_raw[keep]
    labels_all = labels_all[keep]
    meta_all   = meta_all[keep].reset_index(drop=True)
    print(f"Removed {chrom.sum()} chrom-agar samples")

data_norm = row_minmax_normalize(data_raw)
ab_to_idx = {ab: ab_list_raw.index(ab) for ab in ALL_ANTIBIOTICS if ab in ab_list_raw}
all_ab_indices = [ab_to_idx[ab] for ab in ALL_ANTIBIOTICS if ab in ab_to_idx]
print(f"Total samples: {len(data_norm)}")


############################################################
# LABEL ENCODER
############################################################
le_species = LabelEncoder()
le_species.fit(sorted(SPECIES_CONFIG.keys()))
print(f"Species order: {list(le_species.classes_)}")
species_encoded_all = le_species.transform(labels_all)


############################################################
# SEPARAR MS-UMG Y SOURCE
############################################################
mask_msumg  = (meta_all["hospital"] == TARGET_DOMAIN).values
mask_source = meta_all["hospital"].isin(SOURCE_DOMAINS).values

data_msumg    = data_norm[mask_msumg]
amr_msumg     = amr_raw[mask_msumg]
labels_msumg  = labels_all[mask_msumg]
species_msumg = species_encoded_all[mask_msumg]

data_source    = data_norm[mask_source]
amr_source     = amr_raw[mask_source]
labels_source  = labels_all[mask_source]
species_source = species_encoded_all[mask_source]

print(f"MS-UMG: {len(data_msumg)} | Source: {len(data_source)}")

##########
# ##################################################
# LOAD SOURCE SPLITS
############################################################
with open(SOURCE_SPLITS, "rb") as f:
    source_splits_raw = pickle.load(f)
src_splits = source_splits_raw["splits_per_domain"]

DOMAIN_MAP = {d: i for i, d in enumerate(SOURCE_DOMAINS)}
DOMAIN_MAP[TARGET_DOMAIN] = len(SOURCE_DOMAINS)


############################################################
# HELPERS
############################################################
def load_vae_no_species(model_path):
    state = torch.load(model_path, map_location="cpu")
    model = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZ(
        input_dim=6000, latent_dim=LATENT_DIM, num_domains=NUM_DOMAINS,
        n_species=len(le_species.classes_), n_antibiotics=len(ALL_ANTIBIOTICS),
        lambda_amr=LAMBDA_AMR, antibiotic_names=ALL_ANTIBIOTICS,
    )
    model.load_state_dict(state)
    return model

def load_vae_emb(model_path):
    state = torch.load(model_path, map_location="cpu")
    model = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZEmb(
        input_dim=6000, latent_dim=LATENT_DIM, num_domains=NUM_DOMAINS,
        n_species=len(le_species.classes_), n_antibiotics=len(ALL_ANTIBIOTICS),
        lambda_amr=LAMBDA_AMR, antibiotic_names=ALL_ANTIBIOTICS,
        species_emb_dim=SPECIES_EMB_DIM,
    )
    model.load_state_dict(state)
    return model

def add_new_decoder(vae):
    n_existing = len(vae.decoder.net)
    if DOMAIN_MAP[TARGET_DOMAIN] >= n_existing:
        old     = vae.decoder.net[0]
        lat_dim = old[0].in_features
        out_dim = old[-2].out_features
        new_dec = nn.Sequential(
            nn.Linear(lat_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024),   nn.ReLU(),
            nn.Linear(1024, 2048),  nn.ReLU(),
            nn.Linear(2048, out_dim), nn.Sigmoid(),
        )
        vae.decoder.net = nn.ModuleList(list(vae.decoder.net) + [new_dec])
        vae.decoder.num_domains = len(vae.decoder.net)
    return vae

def freeze_for_enc_dec_amr(vae, has_emb=False):
    for p in vae.parameters():
        p.requires_grad = False
    for p in vae.encoder.parameters():
        p.requires_grad = True
    for p in vae.decoder.net[DOMAIN_MAP[TARGET_DOMAIN]].parameters():
        p.requires_grad = True
    for head in vae.amr_heads:
        for p in head.parameters():
            p.requires_grad = True
    if has_emb and hasattr(vae, "species_emb"):
        for p in vae.species_emb.parameters():
            p.requires_grad = True

def freeze_for_amr_head_only(vae, has_emb=False):
    for p in vae.parameters():
        p.requires_grad = False
    for head in vae.amr_heads:
        for p in head.parameters():
            p.requires_grad = True
    if has_emb and hasattr(vae, "species_emb"):
        for p in vae.species_emb.parameters():
            p.requires_grad = True

def freeze_for_enc_amr(vae, has_emb=False):
    for p in vae.parameters():
        p.requires_grad = False
    for p in vae.encoder.parameters():
        p.requires_grad = True
    for head in vae.amr_heads:
        for p in head.parameters():
            p.requires_grad = True
    if has_emb and hasattr(vae, "species_emb"):
        for p in vae.species_emb.parameters():
            p.requires_grad = True

def build_ft_loader(X, domain_ids, species_ids, amr, batch_size=64, shuffle=True):
    return DataLoader(
        TensorDataset(
            torch.tensor(X,           dtype=torch.float32),
            torch.tensor(domain_ids,  dtype=torch.long),
            torch.tensor(species_ids, dtype=torch.long),
            torch.tensor(amr,         dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=shuffle,
    )

def auroc_raw_global(probe, X_te, amr_te_full, labels_te, species_te):
    results = {}
    for sp, abs_sp in SPECIES_CONFIG.items():
        sp_mask = (labels_te == sp)
        X_sp    = X_te[sp_mask]
        sp_sp   = species_te[sp_mask]
        if len(X_sp) == 0:
            continue
        probs = probe.predict_proba(X_sp, sp_sp, device)
        for ab in abs_sp:
            if ab not in ab_to_idx:
                continue
            j_global = ALL_ANTIBIOTICS.index(ab)
            y_j      = amr_te_full[sp_mask][:, ab_to_idx[ab]]
            nan_mask = ~np.isnan(y_j)
            if nan_mask.sum() < 10 or len(np.unique(y_j[nan_mask])) < 2:
                continue
            y_true = y_j[nan_mask].astype(int)
            y_prob = probs[nan_mask, j_global]
            y_pred = (y_prob >= 0.5).astype(int)
            results[(sp, ab)] = {
                "auc":     roc_auc_score(y_true, y_prob),
                "bal_acc": balanced_accuracy_score(y_true, y_pred),
            }
    return results

def auroc_raw_per_species(probe, sp, abs_sp, X_te, amr_te_full, labels_te):
    results = {}
    sp_mask = (labels_te == sp)
    X_sp    = X_te[sp_mask]
    if len(X_sp) == 0:
        return results
    probs = probe.predict_proba(X_sp, device)
    for j, ab in enumerate(abs_sp):
        if ab not in ab_to_idx:
            continue
        y_j      = amr_te_full[sp_mask][:, ab_to_idx[ab]]
        nan_mask = ~np.isnan(y_j)
        if nan_mask.sum() < 10 or len(np.unique(y_j[nan_mask])) < 2:
            continue
        y_true = y_j[nan_mask].astype(int)
        y_prob = probs[nan_mask, j]
        y_pred = (y_prob >= 0.5).astype(int)
        results[(sp, ab)] = {
            "auc":     roc_auc_score(y_true, y_prob),
            "bal_acc": balanced_accuracy_score(y_true, y_pred),
        }
    return results

def auroc_per_species(model, X_te, amr_te_full, labels_te, species_te, has_emb=False):
    results = {}
    for sp, abs_sp in SPECIES_CONFIG.items():
        sp_mask  = (labels_te == sp)
        if sp_mask.sum() == 0:
            continue
        X_sp     = X_te[sp_mask]
        sp_te_sp = species_te[sp_mask]
        ab_idx   = [ab_to_idx[ab] for ab in abs_sp if ab in ab_to_idx]
        amr_sp   = amr_te_full[sp_mask][:, ab_idx]

        res = evaluate_amr_head(
            model, X_sp, amr_sp, abs_sp, device,
            species=sp_te_sp if has_emb else None,
            n_species=None,
        )
        for ab, m in res.items():
            results[(sp, ab)] = {
                "auc":     m["auc"],
                "bal_acc": m["bal_acc"],
            }
    return results

def append_results(all_results, res_dict, partition, n_prev, n_new, model_name):
    for (sp, ab), metrics in res_dict.items():
        all_results.append({
            "partition":  partition,
            "n_prev":     n_prev,
            "n_new":      n_new,
            "model":      model_name,
            "species":    sp,
            "antibiotic": ab,
            "auc":        metrics["auc"],
            "bal_acc":    metrics["bal_acc"],
        })


############################################################
# TRAIN RAW MLP BASELINES
############################################################
print("\n===== TRAINING RAW MLP BASELINES =====")

all_tr_idx = np.concatenate([
    src_splits[sk]["train_idx"]
    for sk in SOURCE_DOMAINS if sk in src_splits
])
all_va_idx = np.concatenate([
    src_splits[sk]["val_idx"]
    for sk in SOURCE_DOMAINS if sk in src_splits
])

mask_6sp = np.isin(labels_source, list(SPECIES_CONFIG.keys()))

X_tr_raw  = data_source[all_tr_idx][mask_6sp[all_tr_idx]]
X_va_raw  = data_source[all_va_idx][mask_6sp[all_va_idx]]
y_tr_raw  = amr_raw[mask_source][all_tr_idx][mask_6sp[all_tr_idx]][:, all_ab_indices]
y_va_raw  = amr_raw[mask_source][all_va_idx][mask_6sp[all_va_idx]][:, all_ab_indices]
sp_tr_raw = species_encoded_all[mask_source][all_tr_idx][mask_6sp[all_tr_idx]]
sp_va_raw = species_encoded_all[mask_source][all_va_idx][mask_6sp[all_va_idx]]

print(f"Raw MLP global — train: {len(X_tr_raw)} | val: {len(X_va_raw)}")

raw_mlp_global = AMRProbeRaw(
    input_dim=6000, n_species=len(le_species.classes_),
    n_antibiotics=len(ALL_ANTIBIOTICS), species_emb_dim=SPECIES_EMB_DIM,
    epochs=50, lr=1e-3, patience=15,
)
raw_mlp_global.trainloop(X_tr_raw, y_tr_raw, X_va_raw, y_va_raw,
                         sp_tr_raw, sp_va_raw, device)
print("Raw MLP global trained.")

raw_mlp_per_species = {}
for sp in SPECIES_CONFIG:
    sp_mask   = (labels_source == sp)
    ab_idx_sp = [ab_to_idx[ab] for ab in SPECIES_CONFIG[sp] if ab in ab_to_idx]
    X_tr_sp   = data_source[all_tr_idx][sp_mask[all_tr_idx]]
    X_va_sp   = data_source[all_va_idx][sp_mask[all_va_idx]]
    y_tr_sp   = amr_raw[mask_source][all_tr_idx][sp_mask[all_tr_idx]][:, ab_idx_sp]
    y_va_sp   = amr_raw[mask_source][all_va_idx][sp_mask[all_va_idx]][:, ab_idx_sp]

    probe = AMRProbeRawNoSpecies(
        n_antibiotics=len(ab_idx_sp), epochs=50, lr=1e-3, patience=15
    )
    probe.trainloop(X_tr_sp, y_tr_sp, X_va_sp, y_va_sp, device)
    raw_mlp_per_species[sp] = probe
    print(f"  Raw per species ({sp}): {len(X_tr_sp)} train samples")


############################################################
# MAIN LOOP
############################################################
print("\n===== FINETUNING LOOP =====")
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
all_results = []

for i_part in range(N_PARTITIONS):
    print(f"\n{'*'*60}")
    print(f"  PARTITION {i_part+1}/{N_PARTITIONS}")
    print(f"{'*'*60}")

    for n_prev in GRID_PREV:
        for n_new in GRID_NEW:

            split_file = SPLITS_DIR / f"run_{i_part}" / f"prev_{n_prev}_new_{n_new}.pkl"
            if not split_file.exists():
                print(f"Split not found: {split_file}. Skipping.")
                continue

            with open(split_file, "rb") as f:
                splits = pickle.load(f)

            idx_ft_global = splits[TARGET_DOMAIN]["finetuning"]
            idx_ft_per_sp = splits[TARGET_DOMAIN]["finetuning_per_species"]
            idx_test      = splits[TARGET_DOMAIN]["test"]

            X_ft          = data_msumg[idx_ft_global]
            amr_ft        = amr_raw[mask_msumg][idx_ft_global][:, all_ab_indices]
            sp_ft         = species_msumg[idx_ft_global]
            dom_ft        = np.full(len(X_ft), DOMAIN_MAP[TARGET_DOMAIN], dtype=np.int64)

            X_test        = data_msumg[idx_test]
            amr_test_full = amr_raw[mask_msumg][idx_test]
            labels_test   = labels_msumg[idx_test]
            sp_test       = species_msumg[idx_test]

            print(f"\nPart={i_part} | n_prev={n_prev} | n_new={n_new} "
                  f"| FT={len(X_ft)} | Test={len(X_test)}")

            # train/val split compartido para VAE y Raw global
            X_tr, X_va, d_tr, d_va, s_tr, s_va, a_tr, a_va = train_test_split(
                X_ft, dom_ft, sp_ft, amr_ft,
                test_size=0.2, random_state=i_part,
            )
            tr_loader = build_ft_loader(X_tr, d_tr, s_tr, a_tr)
            va_loader = build_ft_loader(X_va, d_va, s_va, a_va, shuffle=False)

            # ──────────────────────────────────────────────
            # ZERO-SHOT
            # ──────────────────────────────────────────────
            if n_new == GRID_NEW[0] and n_prev == 0:

                vae_no_sp = load_vae_no_species(VAE_NO_SPECIES_PATH)
                vae_no_sp.to(device).eval()
                append_results(all_results,
                    auroc_per_species(vae_no_sp, X_test, amr_test_full,
                                      labels_test, sp_test, has_emb=False),
                    i_part, n_prev, 0)
                del vae_no_sp
                torch.cuda.empty_cache()

                vae_emb = load_vae_emb(VAE_EMB_SPECIES_PATH)
                vae_emb.to(device).eval()
                append_results(all_results,
                    auroc_per_species(vae_emb, X_test, amr_test_full,
                                    labels_test, sp_test, has_emb=True,
                                    model_name="VAE_species_emb_zero_shot"),
                    i_part, n_prev, 0)
                del vae_emb
                torch.cuda.empty_cache()

                append_results(all_results,
                    auroc_raw_global(raw_mlp_global, X_test, amr_test_full,
                                     labels_test, sp_test),
                    i_part, n_prev, 0, "Raw_global_zero_shot")

                for sp, probe in raw_mlp_per_species.items():
                    append_results(all_results,
                        auroc_raw_per_species(probe, sp, SPECIES_CONFIG[sp],
                                              X_test, amr_test_full, labels_test),
                        i_part, n_prev, 0, "Raw_per_species_zero_shot")

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/o SPECIES
            # ──────────────────────────────────────────────
            vae_no_sp = load_vae_no_species(VAE_NO_SPECIES_PATH)
            vae_no_sp = add_new_decoder(vae_no_sp)
            freeze_for_enc_dec_amr(vae_no_sp, has_emb=False)
            vae_no_sp.to(device)
            vae_no_sp.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_no_sp.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_no_sp.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_no_sp.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_no_sp.epochs   = FT_EPOCHS
            vae_no_sp.patience = FT_PATIENCE
            vae_no_sp.trainloop(tr_loader, va_loader, device)
            vae_no_sp.eval()
            append_results(all_results,
                auroc_per_species(vae_no_sp, X_test, amr_test_full,
                                  labels_test, sp_test, has_emb=False),
                i_part, n_prev, n_new)
            del vae_no_sp
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/ SPECIES EMB
            # ──────────────────────────────────────────────
            vae_emb = load_vae_emb(VAE_EMB_SPECIES_PATH)
            vae_emb = add_new_decoder(vae_emb)
            freeze_for_enc_dec_amr(vae_emb, has_emb=True)
            vae_emb.to(device)
            vae_emb.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_emb.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_emb.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_emb.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_emb.epochs   = FT_EPOCHS
            vae_emb.patience = FT_PATIENCE
            vae_emb.trainloop(tr_loader, va_loader, device)
            vae_emb.eval()
            append_results(all_results,
                auroc_per_species(vae_emb, X_test, amr_test_full,
                                labels_test, sp_test, has_emb=True,
                                model_name="VAE_species_emb_ft"),
                i_part, n_prev, n_new)
            del vae_emb
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/o SPECIES — AMR HEAD ONLY
            # ──────────────────────────────────────────────
            vae_no_sp = load_vae_no_species(VAE_NO_SPECIES_PATH)
            freeze_for_amr_head_only(vae_no_sp, has_emb=False)
            vae_no_sp.to(device)
            vae_no_sp.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_no_sp.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_no_sp.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_no_sp.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_no_sp.epochs   = FT_EPOCHS
            vae_no_sp.patience = FT_PATIENCE
            vae_no_sp.trainloop(tr_loader, va_loader, device)
            vae_no_sp.eval()
            append_results(all_results,
                auroc_per_species(vae_no_sp, X_test, amr_test_full,
                                  labels_test, sp_test, has_emb=False),
                i_part, n_prev, n_new)
            del vae_no_sp
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/ SPECIES EMB — AMR HEAD ONLY
            # ──────────────────────────────────────────────
            vae_emb = load_vae_emb(VAE_EMB_SPECIES_PATH)
            freeze_for_amr_head_only(vae_emb, has_emb=True)
            vae_emb.to(device)
            vae_emb.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_emb.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_emb.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_emb.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_emb.epochs   = FT_EPOCHS
            vae_emb.patience = FT_PATIENCE
            vae_emb.trainloop(tr_loader, va_loader, device)
            vae_emb.eval()
            append_results(all_results,
                auroc_per_species(vae_emb, X_test, amr_test_full,
                                  labels_test, sp_test, has_emb=True),
                i_part, n_prev, n_new)
            del vae_emb
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/o SPECIES — enc + amr head
            # ──────────────────────────────────────────────
            vae_no_sp = load_vae_no_species(VAE_NO_SPECIES_PATH)
            vae_no_sp = add_new_decoder(vae_no_sp)  
            freeze_for_enc_amr(vae_no_sp, has_emb=False)
            vae_no_sp.to(device)
            vae_no_sp.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_no_sp.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_no_sp.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_no_sp.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_no_sp.epochs   = FT_EPOCHS
            vae_no_sp.patience = FT_PATIENCE
            vae_no_sp.trainloop(tr_loader, va_loader, device)
            vae_no_sp.eval()
            append_results(all_results,
                auroc_per_species(vae_no_sp, X_test, amr_test_full,
                                  labels_test, sp_test, has_emb=False),
                i_part, n_prev, n_new)
            del vae_no_sp
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING VAE w/ SPECIES EMB — enc + amr head
            # ──────────────────────────────────────────────
            vae_emb = load_vae_emb(VAE_EMB_SPECIES_PATH)
            vae_emb = add_new_decoder(vae_emb)
            freeze_for_enc_amr(vae_emb, has_emb=True)
            vae_emb.to(device)
            vae_emb.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, vae_emb.parameters()),
                lr=FT_LR, weight_decay=1e-5,
            )
            vae_emb.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                vae_emb.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6,
            )
            vae_emb.epochs   = FT_EPOCHS
            vae_emb.patience = FT_PATIENCE
            vae_emb.trainloop(tr_loader, va_loader, device)
            vae_emb.eval()
            append_results(all_results,
                auroc_per_species(vae_emb, X_test, amr_test_full,
                                  labels_test, sp_test, has_emb=True),
                i_part, n_prev, n_new)
            del vae_emb
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING RAW MLP GLOBAL
            # ──────────────────────────────────────────────
            mlp_global_ft = copy.deepcopy(raw_mlp_global)
            mlp_global_ft.epochs   = FT_EPOCHS
            mlp_global_ft.lr       = FT_LR
            mlp_global_ft.patience = FT_PATIENCE
            mlp_global_ft.trainloop(
                X_tr, a_tr, X_va, a_va, s_tr, s_va, device,
            )
            mlp_global_ft.eval()
            append_results(all_results,
                auroc_raw_global(mlp_global_ft, X_test, amr_test_full,
                                 labels_test, sp_test),
                i_part, n_prev, n_new, "Raw_global_ft")
            del mlp_global_ft
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # FINE-TUNING RAW MLP PER SPECIES
            # ──────────────────────────────────────────────
            for sp, probe_orig in raw_mlp_per_species.items():
                abs_sp    = SPECIES_CONFIG[sp]
                ab_idx_sp = [ab_to_idx[ab] for ab in abs_sp if ab in ab_to_idx]
                idx_sp    = idx_ft_per_sp[sp]

                X_ft_sp = data_msumg[idx_sp]
                y_ft_sp = amr_raw[mask_msumg][idx_sp][:, ab_idx_sp]

                X_tr_sp, X_va_sp, a_tr_sp, a_va_sp = train_test_split(
                    X_ft_sp, y_ft_sp, test_size=0.2, random_state=i_part,
                )

                probe_ft = copy.deepcopy(probe_orig)
                probe_ft.epochs   = FT_EPOCHS
                probe_ft.lr       = FT_LR
                probe_ft.patience = FT_PATIENCE
                probe_ft.trainloop(X_tr_sp, a_tr_sp, X_va_sp, a_va_sp, device)
                probe_ft.eval()

                append_results(all_results,
                    auroc_raw_per_species(probe_ft, sp, abs_sp,
                                         X_test, amr_test_full, labels_test),
                    i_part, n_prev, n_new, "Raw_per_species_ft")
                del probe_ft
                torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # MLP FROM SCRATCH GLOBAL
            # ──────────────────────────────────────────────
            mlp_scratch_global = AMRProbeRaw(
                input_dim=6000, n_species=len(le_species.classes_),
                n_antibiotics=len(ALL_ANTIBIOTICS), species_emb_dim=SPECIES_EMB_DIM,
                epochs=50, lr=1e-3, patience=15,
            )
            X_tr_sg, X_va_sg, sp_tr_sg, sp_va_sg, a_tr_sg, a_va_sg = train_test_split(
                X_ft, sp_ft, amr_ft, test_size=0.2, random_state=i_part,
            )
            mlp_scratch_global.trainloop(
                X_tr_sg, a_tr_sg, X_va_sg, a_va_sg,
                sp_tr_sg, sp_va_sg, device,
            )
            mlp_scratch_global.eval()
            append_results(all_results,
                auroc_raw_global(mlp_scratch_global, X_test, amr_test_full,
                                 labels_test, sp_test),
                i_part, n_prev, n_new, "Raw_global_scratch")
            del mlp_scratch_global
            torch.cuda.empty_cache()

            # ──────────────────────────────────────────────
            # MLP FROM SCRATCH PER SPECIES
            # ──────────────────────────────────────────────
            for sp in SPECIES_CONFIG:
                abs_sp    = SPECIES_CONFIG[sp]
                ab_idx_sp = [ab_to_idx[ab] for ab in abs_sp if ab in ab_to_idx]
                idx_sp    = idx_ft_per_sp[sp]

                X_ft_sp = data_msumg[idx_sp]
                y_ft_sp = amr_raw[mask_msumg][idx_sp][:, ab_idx_sp]

                if len(X_ft_sp) < 20:
                    continue

                X_tr_sp, X_va_sp, a_tr_sp, a_va_sp = train_test_split(
                    X_ft_sp, y_ft_sp, test_size=0.2, random_state=i_part,
                )

                probe_scratch = AMRProbeRawNoSpecies(
                    n_antibiotics=len(ab_idx_sp), epochs=50, lr=1e-3, patience=15
                )
                probe_scratch.trainloop(X_tr_sp, a_tr_sp, X_va_sp, a_va_sp, device)
                probe_scratch.eval()

                append_results(all_results,
                    auroc_raw_per_species(probe_scratch, sp, abs_sp,
                                         X_test, amr_test_full, labels_test),
                    i_part, n_prev, n_new, "Raw_per_species_scratch")
                del probe_scratch
                torch.cuda.empty_cache()

    df_partial = pd.DataFrame(all_results)
    df_partial.to_csv(
        OUTPUT_PATH / f"partial_results_part{i_part}_{timestamp}.csv", index=False
    )
    print(f"Partial results saved (partition {i_part})")


############################################################
# SAVE FINAL RESULTS
############################################################
df_results = pd.DataFrame(all_results)
csv_path   = OUTPUT_PATH / f"amr_finetuning_global_{timestamp}.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nFull results saved: {csv_path}")


############################################################
# SUMMARY
############################################################
print("\n===== SUMMARY =====")

MODEL_ORDER = [
    "VAE_no_species_zero_shot",       "VAE_species_emb_zero_shot",
    "Raw_global_zero_shot",           "Raw_per_species_zero_shot",
    "VAE_no_species_ft",              "VAE_species_emb_ft",
    "VAE_no_species_enc_amr_ft",      "VAE_species_emb_enc_amr_ft",
    "VAE_no_species_amr_head_ft",     "VAE_species_emb_amr_head_ft",
    "Raw_global_ft",                  "Raw_per_species_ft",
    "Raw_global_scratch",             "Raw_per_species_scratch",
]

for sp in SPECIES_CONFIG:
    sub_sp = df_results[df_results["species"] == sp]
    if len(sub_sp) == 0:
        continue
    print(f"\n{'='*60}\n  {sp}\n{'='*60}")

    for ab in SPECIES_CONFIG[sp]:
        sub_ab = sub_sp[sub_sp["antibiotic"] == ab]
        if len(sub_ab) == 0:
            continue

        for metric, metric_label in [("auc", "AUROC"), ("bal_acc", "Balanced Accuracy")]:
            print(f"\n  {ab} — {metric_label}:")
            summary_m = (
                sub_ab.groupby(["n_new", "model"])[metric]
                .agg(["mean", "std"])
                .round(3)
                .reset_index()
            )
            summary_m["mean±std"] = (
                summary_m["mean"].map(lambda x: f"{x:.3f}") + " ± " +
                summary_m["std"].map(lambda x: f"{x:.3f}")
            )
            pivot_m = summary_m.pivot_table(
                index="n_new", columns="model", values="mean±std", aggfunc="first"
            )
            existing = [m for m in MODEL_ORDER if m in pivot_m.columns]
            print(pivot_m[existing].to_string())

print("\n===== DONE =====")
