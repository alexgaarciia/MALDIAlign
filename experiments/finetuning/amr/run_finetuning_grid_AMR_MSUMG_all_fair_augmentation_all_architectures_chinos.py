############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys
import pickle
import copy

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
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data.io import load_pkl
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import evaluate_amr_head

from experiments.finetuning.amr.run_finetuning_amr_augmentation_chen import run_finetuning_amr
from models.baselines.mlp_chen import ChenMLP_Extended
from models.baselines.mlp_chen_multihead import ChenMLP_MultiHead_Extended
from models.deep.MultiVAEPriorAMRHeadZChen import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZChinos
from models.deep.IWAEAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_IWAE_Extended

############################################################
# CONFIGURATION
############################################################
SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TARGET_DOMAIN  = "MS-UMG"
OUTPUT_PATH    = Path("/export/usuarios01/agnavarr/MALDIAlign/finetuning_amr_all_species_msumg")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FT_MODES        = ["enc_dec_amr"]
GRID_PREV       = [0]
GRID_NEW        = [50, 100, 250, 500, 1000, 1500, 2000]
GRID_AMR_SAMPLES = [1]

N_PARTITIONS       = 10
RUN_EVAL           = False
LATENT_DIM         = 128
LAMBDA_AMR         = 100
NUM_DOMAINS        = 4
MLP_POOLED_FT_EPOCHS = 50

# ── Baseline inventory ────────────────────────────────────────────────────────
#
#  VAE/IWAE (DALMA):
#    {model}_zero_shot          — pretrained on source, evaluated directly on target
#    {model}_FT_enc_dec_amr_aug1 — finetuned on n_new target samples
#
#  MLP multi-head (arquitectura Chen, un head por antibiótico):
#    MLP_pooled_source          — trained on SOURCE domains (A+B+C+MARISMA), zero-shot on target
#    MLP_pooled_finetuned       — same model, finetuned on n_new target samples (lr=1e-4)
#    MLP_from_scratch           — trained from scratch on n_new target samples only (lr=1e-3)
#
#  Chen per-pair (arquitectura Chen, un modelo por especie×antibiótico):
#    Chen_pooled_source         — trained on SOURCE domains, zero-shot on target
#    Chen_pooled_finetuned      — same model, finetuned on n_new target samples (lr=1e-4)
#    Chen_from_scratch          — trained from scratch on n_new target samples only (lr=1e-3)
#
# ─────────────────────────────────────────────────────────────────────────────

SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae": {
        "antibiotics": ["Imipenem", "Meropenem", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "vae_z":          Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103048/model.pth"),
        "vae_z_annealing": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103633/model.pth"),
        "splits_path": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_223514/data_splits.pkl"),
        "splits_dir":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_KlebsiellaPneumoniae_MSUMG_20260531_102427"),
    },
    "Escherichia_Coli": {
        "antibiotics": ["Ampicillin", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "vae_z":          Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103412/model.pth"),
        "vae_z_annealing": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103836/model.pth"),
        "splits_path": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_223855/data_splits.pkl"),
        "splits_dir":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_EscherichiaColi_MSUMG_20260531_201956"),
    },
    "Staphylococcus_Aureus": {
        "antibiotics": ["Oxacillin", "Clindamycin", "Erythromycin"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "vae_z":          Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103452/model.pth"),
        "vae_z_annealing": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103918/model.pth"),
        "splits_path": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_224204/data_splits.pkl"),
        "splits_dir":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_StaphylococcusAureus_MSUMG_20260531_102427"),
    },
    "Enterococcus_Faecium": {
        "antibiotics": ["Vancomycin"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "vae_z":          Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103603/model.pth"),
        "vae_z_annealing": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_104023/model.pth"),
        "splits_path": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_224514/data_splits.pkl"),
        "splits_dir":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_EnterococcusFaecium_MSUMG_20260531_102427"),
    },
    "Pseudomonas_Aeruginosa": {
        "antibiotics": ["Meropenem", "Amikacin"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final_v3.pkl",
        "vae_z":          Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103534/model.pth"),
        "vae_z_annealing": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260606_103950/model.pth"),
        "splits_path": Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260527_224359/data_splits.pkl"),
        "splits_dir":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_PseudomonasAeruginosa_MSUMG_20260531_102427"),
    },
}

############################################################
# HELPERS
############################################################
MODEL_KEYS = ["vae_z", "vae_z_annealing"]

def is_iwae_model(model_name):    return model_name.startswith("iwae")
def uses_fixed_prior(model_name): return "gaussian" in model_name

def make_loader_2d(X, y, shuffle=False):
    """Multi-head loader: y is (N, n_antibiotics)."""
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=64, shuffle=shuffle,
    )

def make_loader_1d(X, y, shuffle=False):
    """Per-pair loader: y is (N,)."""
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=64, shuffle=shuffle,
    )

def instantiate_pretrained_model(model_name, model_path, input_dim, n_antibiotics, antibiotic_names):
    state = torch.load(model_path, map_location="cpu")
    model_class = (
        MultiVAE_Bernoulli_SpeciesPrior_AMR_IWAE_Extended
        if is_iwae_model(model_name)
        else MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZChinos
    )
    common_kwargs = dict(
        input_dim=input_dim, latent_dim=LATENT_DIM, num_domains=NUM_DOMAINS,
        n_species=1, n_antibiotics=n_antibiotics, lambda_amr=LAMBDA_AMR,
        antibiotic_names=antibiotic_names, use_fixed_prior=uses_fixed_prior(model_name),
    )
    model = model_class(**common_kwargs, n_iwae_samples=5) if is_iwae_model(model_name) \
            else model_class(**common_kwargs)
    model.load_state_dict(state)
    return model

def append_result(all_results, species, i_part, n_prev, n_new, model_name, atb, m, n_test,
                  base_model="baseline", n_amr_samples=None):
    row = {
        "species": species, "partition": i_part,
        "n_prev": n_prev, "n_new": n_new,
        "base_model": base_model, "model": model_name,
        "antibiotic": atb, "auc": m["auc"],
        "pr_auc": m["pr_auc"], "n_test": m.get("n", n_test),
    }
    if n_amr_samples is not None:
        row["n_amr_samples"] = n_amr_samples
    all_results.append(row)

############################################################
# MAIN
############################################################
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

all_results = []
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

for species, cfg in SPECIES_CONFIG.items():
    antibiotics  = cfg["antibiotics"]
    splits_dir   = cfg["splits_dir"]
    dataset_path = cfg["dataset_path"]

    print(f"\n{'#'*70}")
    print(f"  SPECIES: {species}")
    print(f"  Antibiotics: {antibiotics}")
    print(f"{'#'*70}")

    # ── Load dataset ───────────────────────────────────────────────────────────
    dataset    = load_pkl(dataset_path)
    data_raw   = dataset["data"]
    amr_raw    = dataset["amr"]
    ab_list_raw = list(dataset["antibiotics"])
    labels_raw = dataset["label"]
    meta_raw   = pd.DataFrame(dataset["meta"]) if not isinstance(dataset["meta"], pd.DataFrame) \
                 else dataset["meta"]
    data_norm  = row_minmax_normalize(data_raw)

    # Remove chrom-agar from MS-UMG (different culture medium → spectral shift)
    if "agar" in meta_raw.columns:
        chrom = ((meta_raw["hospital"] == TARGET_DOMAIN) & (meta_raw["agar"] == "chrom")).values
        keep  = ~chrom
        data_norm, amr_raw, labels_raw = data_norm[keep], amr_raw[keep], labels_raw[keep]
        meta_raw = meta_raw[keep].reset_index(drop=True)
        print(f"Removed {chrom.sum()} chrom-agar samples from MS-UMG")

    keep_idx = [ab_list_raw.index(n) for n in antibiotics if n in ab_list_raw]
    ab_list  = [n for n in antibiotics if n in ab_list_raw]

    # ── Target sub-dataset (MS-UMG, species filtered) ─────────────────────────
    local_mask = (meta_raw["hospital"] == TARGET_DOMAIN).values & (labels_raw == species)
    data_t     = data_norm[local_mask]
    amr_t      = amr_raw[:, keep_idx][local_mask]

    print(f"\nMS-UMG ({species}): {len(data_t)} samples")
    for j, atb in enumerate(ab_list):
        y = amr_t[:, j]; valid = ~np.isnan(y)
        n_r, n_s = int((y[valid] == 1).sum()), int((y[valid] == 0).sum())
        if n_r + n_s > 0:
            print(f"  {atb}: S={n_s}, R={n_r}, prev={100*n_r/(n_r+n_s):.1f}%")

    # ── Build source pooled data (A+B+C+MARISMA) ──────────────────────────────
    with open(cfg["splits_path"], "rb") as f:
        vae_splits = pickle.load(f)
    domain_splits = vae_splits.get("splits_per_domain", {})

    sp_mask_src = (labels_raw == species)
    data_sp     = data_norm[sp_mask_src]
    amr_sp      = amr_raw[:, keep_idx][sp_mask_src]

    all_X_tr, all_y_tr, all_X_vl, all_y_vl = [], [], [], []
    for dom in SOURCE_DOMAINS:
        if dom not in domain_splits:
            continue
        sp     = domain_splits[dom]
        tr_idx = sp["train_idx"]
        vl_idx = sp.get("val_idx", np.array([], dtype=int))
        if len(tr_idx) > 0:
            all_X_tr.append(data_sp[tr_idx])
            all_y_tr.append(amr_sp[tr_idx].astype(float))
        if len(vl_idx) > 0:
            all_X_vl.append(data_sp[vl_idx])
            all_y_vl.append(amr_sp[vl_idx].astype(float))

    X_pool_tr = np.vstack(all_X_tr); y_pool_tr = np.vstack(all_y_tr)
    X_pool_vl = np.vstack(all_X_vl); y_pool_vl = np.vstack(all_y_vl)

    print(f"\nSource pool (A+B+C+MARISMA): n={len(X_pool_tr)}")
    for j, atb in enumerate(ab_list):
        col = y_pool_tr[:, j]; valid = ~np.isnan(col)
        print(f"  {atb}: n={valid.sum()}, R={(col[valid]==1).sum():.0f}, S={(col[valid]==0).sum():.0f}")

    # ── Train MLP_pooled_source ONCE per species ───────────────────────────────
    # Multi-head (one head per antibiotic), Chen backbone (512→256→128)
    # Trained on source domains (A+B+C+MARISMA), evaluated zero-shot on MS-UMG
    print(f"\nTraining MLP_pooled_source (multi-head, source domains)...")
    mlp_pooled = ChenMLP_MultiHead_Extended(
        input_dim=X_pool_tr.shape[1], n_antibiotics=len(ab_list),
        antibiotic_names=ab_list, epochs=100, lr=1e-3, patience=20,
    )
    mlp_pooled.trainloop(
        make_loader_2d(X_pool_tr, y_pool_tr, shuffle=True),
        make_loader_2d(X_pool_vl, y_pool_vl),
        device,
    )
    mlp_pooled.eval()
    print("  MLP_pooled_source trained.")

    # ── Train Chen_pooled_source ONCE per species (one model per antibiotic) ───
    # Per-pair (one model per antibiotic), Chen backbone (512→256→128→1)
    # Trained on source domains (A+B+C+MARISMA), evaluated zero-shot on MS-UMG
    print(f"\nTraining Chen_pooled_source (per-pair, source domains)...")
    chen_pooled_per_pair = {}
    for j, atb in enumerate(ab_list):
        y_tr_j = y_pool_tr[:, j]; valid_tr = ~np.isnan(y_tr_j)
        y_vl_j = y_pool_vl[:, j]; valid_vl = ~np.isnan(y_vl_j)
        if valid_tr.sum() < 20 or len(np.unique(y_tr_j[valid_tr].astype(int))) < 2:
            chen_pooled_per_pair[atb] = None
            continue
        chen_src = ChenMLP_Extended(input_dim=X_pool_tr.shape[1], epochs=200, lr=1e-3, patience=20)
        chen_src.trainloop(
            make_loader_1d(X_pool_tr[valid_tr], y_tr_j[valid_tr].astype(float), shuffle=True),
            make_loader_1d(X_pool_vl[valid_vl], y_vl_j[valid_vl].astype(float)),
            device,
        )
        chen_src.eval()
        chen_pooled_per_pair[atb] = chen_src
        print(f"  Chen_pooled_source trained: {atb}")

    # ── Grid loop ──────────────────────────────────────────────────────────────
    for i_part in range(N_PARTITIONS):
        print(f"\n{'*'*60}")
        print(f" {species} | PARTITION {i_part + 1}/{N_PARTITIONS}")
        print(f"{'*'*60}")

        for n_prev in GRID_PREV:
            for n_new in GRID_NEW:
                split_file = splits_dir / f"run_{i_part}" / f"prev_{n_prev}_new_{n_new}.pkl"
                if not split_file.exists():
                    print(f"Split not found: {split_file}. Skipping.")
                    continue

                splits   = load_pkl(split_file)
                idx_ft   = splits[TARGET_DOMAIN]["finetuning"]
                idx_test = splits[TARGET_DOMAIN]["test"]

                X_ft   = data_t[idx_ft];  amr_ft   = amr_t[idx_ft]
                X_test = data_t[idx_test]; amr_test = amr_t[idx_test]

                print(f"\nPartition={i_part} | N_PREV={n_prev} | N_NEW={n_new} "
                      f"| FT={len(idx_ft)} | Test={len(idx_test)}")

                # ==============================================================
                # DALMA VAE / IWAE
                # ==============================================================
                for model_name in MODEL_KEYS:
                    model_path      = cfg[model_name]
                    use_fixed_prior = uses_fixed_prior(model_name)

                    print(f"\n{'='*80}")
                    print(f"  MODEL: {model_name} | fixed_prior={use_fixed_prior}")
                    print(f"{'='*80}")

                    vae_pretrained = instantiate_pretrained_model(
                        model_name, model_path, data_norm.shape[1], len(ab_list), ab_list
                    )
                    vae_pretrained.to(device).eval()
                    vae_pretrained.n_amr_samples = 1

                    # Zero-shot: pretrained on source, no adaptation
                    if n_prev == 0:
                        for atb, m in evaluate_amr_head(
                            vae_pretrained, X_test, amr_test, ab_list, device
                        ).items():
                            append_result(all_results, species, i_part, n_prev, n_new,
                                          f"{model_name}_zero_shot", atb, m, len(idx_test),
                                          base_model=model_name, n_amr_samples=1)
                            print(f"  {model_name}_zero_shot {atb}: AUC={m['auc']:.3f}")

                    # Finetuned: adapted on n_new target samples
                    for mode in FT_MODES:
                        for n_amr_samples in GRID_AMR_SAMPLES:
                            try:
                                vae_ft, _, _ = run_finetuning_amr(
                                    splits_path=split_file, target_domain=TARGET_DOMAIN,
                                    pretrained_model_path=model_path, finetuning_mode=mode,
                                    n_prev=n_prev, n_new=n_new, output_dir=OUTPUT_PATH,
                                    device=device, target_antibiotics=antibiotics,
                                    species=species, consider_prev_domains=(n_prev > 0),
                                    lambda_amr=LAMBDA_AMR, run_latent_evaluation=RUN_EVAL,
                                    ft_lr=1e-5, dataset_path=dataset_path,
                                    n_amr_samples=n_amr_samples, random_state=i_part,
                                    model_type=model_name, use_fixed_prior=use_fixed_prior,
                                    annealing_epochs=None,
                                )
                                vae_ft.eval(); vae_ft.n_amr_samples = 1
                                result_name = f"{model_name}_FT_{mode}_aug{n_amr_samples}"
                                for atb, m in evaluate_amr_head(
                                    vae_ft, X_test, amr_test, ab_list, device
                                ).items():
                                    append_result(all_results, species, i_part, n_prev, n_new,
                                                  result_name, atb, m, len(idx_test),
                                                  base_model=model_name, n_amr_samples=n_amr_samples)
                                    print(f"  {result_name} {atb}: AUC={m['auc']:.3f}")
                                del vae_ft; torch.cuda.empty_cache()
                            except Exception as e:
                                print(f"ERROR {model_name} {mode} aug{n_amr_samples}: {e}")
                                import traceback; traceback.print_exc()

                    del vae_pretrained; torch.cuda.empty_cache()

                # ==============================================================
                # BASELINES
                # ==============================================================

                # ── MLP_pooled_source (multi-head, source domains, zero-shot) ──
                # Entrenado en A+B+C+MARISMA, evaluado directamente en MS-UMG
                for atb, m in mlp_pooled.evaluate(X_test, amr_test, device).items():
                    append_result(all_results, species, i_part, n_prev, n_new,
                                  "MLP_pooled_source", atb, m, len(idx_test))
                    print(f"  MLP_pooled_source {atb}: AUC={m['auc']:.3f}")

                if len(X_ft) >= 20:

                    # ── MLP_pooled_finetuned (multi-head, source→target) ────────
                    # Mismo modelo que MLP_pooled_source, finetuneado en n_new muestras
                    # del target con lr=1e-4 (protocolo Chen et al.)
                    X_tr_p, X_val_p, a_tr_p, a_val_p = train_test_split(
                        X_ft, amr_ft, test_size=0.2, random_state=i_part
                    )
                    mlp_pooled_ft = copy.deepcopy(mlp_pooled)
                    mlp_pooled_ft.epochs = MLP_POOLED_FT_EPOCHS
                    mlp_pooled_ft.trainloop(
                        make_loader_2d(X_tr_p, a_tr_p, shuffle=True),
                        make_loader_2d(X_val_p, a_val_p),
                        device, finetune=True,  # lr=1e-4
                    )
                    for atb, m in mlp_pooled_ft.evaluate(X_test, amr_test, device).items():
                        append_result(all_results, species, i_part, n_prev, n_new,
                                      "MLP_pooled_finetuned", atb, m, len(idx_test))
                        print(f"  MLP_pooled_finetuned {atb}: AUC={m['auc']:.3f}")
                    del mlp_pooled_ft; torch.cuda.empty_cache()

                    # ── MLP_from_scratch (multi-head, target only) ──────────────
                    # Entrenado desde cero únicamente con n_new muestras de MS-UMG
                    # Sin ningún pretraining en source. lr=1e-3
                    X_tr_s, X_val_s, a_tr_s, a_val_s = train_test_split(
                        X_ft, amr_ft, test_size=0.2, random_state=i_part
                    )
                    mlp_scratch = ChenMLP_MultiHead_Extended(
                        input_dim=X_tr_s.shape[1], n_antibiotics=len(ab_list),
                        antibiotic_names=ab_list, epochs=100, lr=1e-3, patience=20,
                    )
                    mlp_scratch.trainloop(
                        make_loader_2d(X_tr_s, a_tr_s, shuffle=True),
                        make_loader_2d(X_val_s, a_val_s),
                        device,
                    )
                    for atb, m in mlp_scratch.evaluate(X_test, amr_test, device).items():
                        append_result(all_results, species, i_part, n_prev, n_new,
                                      "MLP_from_scratch", atb, m, len(idx_test))
                        print(f"  MLP_from_scratch {atb}: AUC={m['auc']:.3f}")
                    del mlp_scratch; torch.cuda.empty_cache()

                # ── Chen per-pair baselines ─────────────────────────────────────
                for j, atb in enumerate(ab_list):
                    chen_src = chen_pooled_per_pair.get(atb)
                    if chen_src is None:
                        continue

                    y_te_j   = amr_test[:, j]; valid_te = ~np.isnan(y_te_j)
                    has_test = valid_te.sum() >= 10 and len(np.unique(y_te_j[valid_te].astype(int))) == 2

                    y_ft_j   = amr_ft[:, j]; valid_ft = ~np.isnan(y_ft_j)
                    has_ft   = (
                        len(X_ft) >= 20
                        and valid_ft.sum() >= 10
                        and len(np.unique(y_ft_j[valid_ft].astype(int))) == 2
                    )

                    # Chen_pooled_source (per-pair, source domains, zero-shot)
                    # Entrenado en A+B+C+MARISMA para este antibiótico, evaluado en MS-UMG
                    if has_test:
                        m = chen_src.evaluate(X_test[valid_te], y_te_j[valid_te].astype(int), device)
                        if m:
                            append_result(all_results, species, i_part, n_prev, n_new,
                                          "Chen_pooled_source", atb, m, len(idx_test))
                            print(f"  Chen_pooled_source {atb}: AUC={m['auc']:.3f}")

                    if has_ft:
                        X_tr_c, X_val_c, y_tr_c, y_val_c = train_test_split(
                            X_ft[valid_ft], y_ft_j[valid_ft].astype(float),
                            test_size=0.2, random_state=i_part
                        )

                        # Chen_pooled_finetuned (per-pair, source→target)
                        # Mismo modelo que Chen_pooled_source, finetuneado con lr=1e-4
                        chen_ft = copy.deepcopy(chen_src)
                        chen_ft.epochs = MLP_POOLED_FT_EPOCHS
                        chen_ft.trainloop(
                            make_loader_1d(X_tr_c, y_tr_c, shuffle=True),
                            make_loader_1d(X_val_c, y_val_c),
                            device, finetune=True,  # lr=1e-4
                        )
                        if has_test:
                            m = chen_ft.evaluate(X_test[valid_te], y_te_j[valid_te].astype(int), device)
                            if m:
                                append_result(all_results, species, i_part, n_prev, n_new,
                                              "Chen_pooled_finetuned", atb, m, len(idx_test))
                                print(f"  Chen_pooled_finetuned {atb}: AUC={m['auc']:.3f}")
                        del chen_ft

                        # Chen_from_scratch (per-pair, target only)
                        # Entrenado desde cero únicamente con n_new muestras de MS-UMG. lr=1e-3
                        chen_scratch = ChenMLP_Extended(
                            input_dim=X_tr_c.shape[1], epochs=200, lr=1e-3, patience=20,
                        )
                        chen_scratch.trainloop(
                            make_loader_1d(X_tr_c, y_tr_c, shuffle=True),
                            make_loader_1d(X_val_c, y_val_c),
                            device, finetune=False,
                        )
                        if has_test:
                            m = chen_scratch.evaluate(X_test[valid_te], y_te_j[valid_te].astype(int), device)
                            if m:
                                append_result(all_results, species, i_part, n_prev, n_new,
                                              "Chen_from_scratch", atb, m, len(idx_test))
                                print(f"  Chen_from_scratch {atb}: AUC={m['auc']:.3f}")
                        del chen_scratch
                        torch.cuda.empty_cache()

        # Partial save after each partition
        pd.DataFrame(all_results).to_csv(
            OUTPUT_PATH / f"partial_{species.replace('_','')}_{timestamp}.csv", index=False
        )

    pd.DataFrame(all_results).to_csv(
        OUTPUT_PATH / f"partial_results_{species.replace('_', '')}_{timestamp}.csv", index=False
    )
    print(f"\nIntermediate results saved for {species}")

    del mlp_pooled
    for m in chen_pooled_per_pair.values():
        if m is not None: del m
    chen_pooled_per_pair = {}
    torch.cuda.empty_cache()

############################################################
# SAVE FINAL RESULTS
############################################################
df_results = pd.DataFrame(all_results)
csv_path   = OUTPUT_PATH / f"amr_finetuning_all_species_MSUMG_{timestamp}.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nFull results saved: {csv_path}")

############################################################
# SUMMARY
############################################################
col_order = []
for model_name in MODEL_KEYS:
    col_order.append(f"{model_name}_zero_shot")
    for mode in FT_MODES:
        for k in GRID_AMR_SAMPLES:
            col_order.append(f"{model_name}_FT_{mode}_aug{k}")

col_order += [
    "MLP_pooled_source",    # multi-head, source domains, zero-shot
    "MLP_pooled_finetuned", # multi-head, source→target finetuned
    "MLP_from_scratch",     # multi-head, target only
    "Chen_pooled_source",   # per-pair,   source domains, zero-shot
    "Chen_pooled_finetuned",# per-pair,   source→target finetuned
    "Chen_from_scratch",    # per-pair,   target only
]

for species in SPECIES_CONFIG.keys():
    sub_sp  = df_results[df_results["species"] == species]
    ab_list = SPECIES_CONFIG[species]["antibiotics"]
    if len(sub_sp) == 0:
        continue

    print(f"\n{'='*70}\n  {species}\n{'='*70}")

    for atb in ab_list:
        sub_atb = sub_sp[sub_sp["antibiotic"] == atb]
        if len(sub_atb) == 0:
            continue
        print(f"\n--- {atb} ---")
        per_partition = sub_atb.groupby(
            ["partition", "n_prev", "n_new", "model"]
        )["auc"].mean().reset_index()
        atb_summary = per_partition.groupby(
            ["n_prev", "n_new", "model"]
        )["auc"].agg(["mean", "std"]).reset_index()
        atb_summary["mean_std"] = (
            atb_summary["mean"].round(3).astype(str)
            + " ± "
            + atb_summary["std"].round(3).astype(str)
        )
        atb_summary.to_csv(
            OUTPUT_PATH / f"summary_{species.replace('_','')}_{atb}_{timestamp}.csv",
            index=False
        )
        pivot    = atb_summary.pivot_table(
            index=["n_prev", "n_new"], columns="model",
            values="mean_std", aggfunc="first"
        )
        existing = [c for c in col_order if c in pivot.columns]
        print(pivot[existing].to_string())

print(f"\n===== DONE =====")
