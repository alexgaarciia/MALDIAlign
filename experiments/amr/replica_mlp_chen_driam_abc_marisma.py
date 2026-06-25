############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

output_dir     = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space          = "original/chen_replication"
timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)
print(f"\nExperiment directory: {experiment_dir}")

############################################################
# IMPORTS
############################################################
import numpy as np
import pandas as pd
import torch
import copy
import yaml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.training.data_pipeline import load_pkl
from src.data.preprocessing import row_minmax_normalize
from models.baselines.mlp_chen import ChenMLP_Extended

from src.config.loader import load_config

############################################################
# CONFIG
############################################################
PAIRS = [
    ("Enterococcus_Faecium",    "Vancomycin"),
    ("Escherichia_Coli",        "Ceftazidime"),
    ("Escherichia_Coli",        "Ciprofloxacin"),
    ("Escherichia_Coli",        "Piperacillin-Tazobactam"),
    ("Klebsiella_Pneumoniae",   "Ceftazidime"),
    ("Klebsiella_Pneumoniae",   "Ciprofloxacin"),
    ("Klebsiella_Pneumoniae",   "Piperacillin-Tazobactam"),
    ("Pseudomonas_Aeruginosa",  "Meropenem"),
    ("Staphylococcus_Aureus",   "Erythromycin"),
    ("Staphylococcus_Aureus",   "Oxacillin"),
    ("Staphylococcus_Aureus",   "Tetracycline"),
]

# Chen et al. seeds
SEEDS = [42, 123, 456, 789, 1024]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

############################################################
# LOAD DATA
############################################################
cfg = load_config()

species_to_keep = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]

# ── DRIAMS A+B+C ───────────────────────────────────────────
driams_dict  = load_pkl(cfg["data"]["DRIAMS_FULL"])
mask_driams  = np.isin(driams_dict["label"], species_to_keep)
data_driams  = row_minmax_normalize(driams_dict["data"][mask_driams])
label_driams = driams_dict["label"][mask_driams]
amr_driams   = driams_dict["amr"][mask_driams]
ab_driams    = list(driams_dict["antibiotics"])
meta_driams  = pd.DataFrame.from_records(list(driams_dict["meta"]))[mask_driams].reset_index(drop=True)

# Excluir DRIAMS-D
mask_abc = ~(meta_driams["hospital"] == "DRIAMS_D").values
data_driams  = data_driams[mask_abc]
label_driams = label_driams[mask_abc]
amr_driams   = amr_driams[mask_abc]
meta_driams  = meta_driams[mask_abc].reset_index(drop=True)
print(f"DRIAMS A+B+C: {len(data_driams)} samples")
print(f"  por hospital: {meta_driams['hospital'].value_counts().to_dict()}")

# ── MARISMA ────────────────────────────────────────────────
marisma_dict  = load_pkl(cfg["data"]["MARISMa_FULL"])
mask_marisma  = np.isin(marisma_dict["label"], species_to_keep)
data_marisma  = row_minmax_normalize(marisma_dict["data"][mask_marisma])
label_marisma = marisma_dict["label"][mask_marisma]
amr_marisma   = marisma_dict["amr"][mask_marisma]
ab_marisma    = list(marisma_dict["antibiotics"])
meta_marisma  = pd.DataFrame.from_records(list(marisma_dict["meta"]))[mask_marisma].reset_index(drop=True)
meta_marisma.insert(0, "hospital", "MARISMA")
print(f"MARISMA: {len(data_marisma)} samples")

# ── Concatenar fuentes de entrenamiento ────────────────────
# Los antibióticos pueden diferir entre datasets — necesitamos alinear columnas
def align_amr(amr_src, ab_src, ab_ref):
    """Reordena/rellena con NaN las columnas de amr_src para que coincidan con ab_ref."""
    out = np.full((len(amr_src), len(ab_ref)), np.nan)
    for j, ab in enumerate(ab_ref):
        if ab in ab_src:
            out[:, j] = amr_src[:, ab_src.index(ab)]
    return out

# ab_driams es la referencia — MARISMA se alinea a ella
amr_marisma_aligned = align_amr(amr_marisma, ab_marisma, ab_driams)

data_train  = np.concatenate([data_driams,  data_marisma],          axis=0)
label_train = np.concatenate([label_driams, label_marisma],         axis=0)
amr_train   = np.concatenate([amr_driams,   amr_marisma_aligned],   axis=0)
ab_train    = ab_driams  # referencia unificada

print(f"\nTrain pool total: {len(data_train)} samples")
print(f"  DRIAMS A+B+C: {len(data_driams)}")
print(f"  MARISMA:      {len(data_marisma)}")

# ── MS-UMG ─────────────────────────────────────────────────
msumg_dict  = load_pkl(cfg["data"]["MSUMG_FULL"])
mask_msumg  = np.isin(msumg_dict["label"], species_to_keep)
data_msumg  = row_minmax_normalize(msumg_dict["data"][mask_msumg])
label_msumg = msumg_dict["label"][mask_msumg]
amr_msumg   = msumg_dict["amr"][mask_msumg]
ab_msumg    = list(msumg_dict["antibiotics"])
meta_msumg  = pd.DataFrame.from_records(list(msumg_dict["meta"]))[mask_msumg].reset_index(drop=True)
meta_msumg.insert(0, "hospital", "MS-UMG")
print(f"MS-UMG total: {len(data_msumg)} samples")

if "agar" in meta_msumg.columns:
    chrom_mask_msumg = (meta_msumg["agar"] == "chrom").values
    agar_mask_msumg  = ~chrom_mask_msumg
    print(f"  chrom: {chrom_mask_msumg.sum()}  |  agar: {agar_mask_msumg.sum()}")
else:
    chrom_mask_msumg = np.zeros(len(data_msumg), dtype=bool)
    agar_mask_msumg  = np.ones(len(data_msumg),  dtype=bool)

############################################################
# HELPERS
############################################################

def get_ab_idx(ab_name, ab_list):
    for i, a in enumerate(ab_list):
        if a == ab_name:
            return i
    return None


def make_loader(X, y, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=256, shuffle=shuffle
    )


def train_source_only(X_tr, y_tr):
    """Pretrain on DRIAMS-A with internal 90/10 val split."""
    n_val    = max(int(len(y_tr) * 0.1), 4)
    X_vl, y_vl = X_tr[-n_val:], y_tr[-n_val:]
    X_tr, y_tr = X_tr[:-n_val], y_tr[:-n_val]

    model = ChenMLP_Extended(input_dim=X_tr.shape[1])
    model.trainloop(make_loader(X_tr, y_tr, shuffle=True),
                    make_loader(X_vl, y_vl), device)
    return model

def target_only(X_ft, y_ft, seed):
    """
    Train from scratch on target data — no pretraining.
    Same 70/10/20 split as finetune(), lr=1e-3.
    Returns (model, X_test, y_test).
    """
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss_outer.split(X_ft, y_ft.astype(int)))

    X_trainval, y_trainval = X_ft[trainval_idx], y_ft[trainval_idx]
    X_test,     y_test     = X_ft[test_idx],     y_ft[test_idx]

    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=seed)
    tr_idx, vl_idx = next(sss_inner.split(X_trainval, y_trainval.astype(int)))

    X_tr, y_tr = X_trainval[tr_idx], y_trainval[tr_idx]
    X_vl, y_vl = X_trainval[vl_idx], y_trainval[vl_idx]

    # Modelo nuevo desde cero, lr=1e-3 (no finetune)
    model = ChenMLP_Extended(input_dim=X_tr.shape[1])
    model.trainloop(
        make_loader(X_tr, y_tr, shuffle=True),
        make_loader(X_vl, y_vl),
        device,
        finetune=False,  # lr=1e-3
    )
    return model, X_test, y_test


def finetune(pretrained_model, X_ft, y_ft, seed):
    """
    Fine-tune a deep copy of pretrained_model on (X_ft, y_ft).
    Split 70/10/20 → train/val/test following Chen et al.
    Returns (finetuned_model, X_test, y_test).
    """
    # Stratified 80/20, then 70/10 from the 80
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(sss_outer.split(X_ft, y_ft.astype(int)))

    X_trainval, y_trainval = X_ft[trainval_idx], y_ft[trainval_idx]
    X_test, y_test = X_ft[test_idx],     y_ft[test_idx]

    # 70/10
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=1/8, random_state=seed)
    tr_idx, vl_idx = next(sss_inner.split(X_trainval, y_trainval.astype(int)))

    X_tr, y_tr = X_trainval[tr_idx], y_trainval[tr_idx]
    X_vl, y_vl = X_trainval[vl_idx], y_trainval[vl_idx]

    ft_model = copy.deepcopy(pretrained_model)
    ft_model.trainloop(
        make_loader(X_tr, y_tr, shuffle=True),
        make_loader(X_vl, y_vl),
        device,
        finetune=True,   # lr=1e-4
    )
    return ft_model, X_test, y_test


def safe_evaluate(model, X, y):
    valid = ~np.isnan(y)
    if valid.sum() < 10 or len(np.unique(y[valid].astype(int))) < 2:
        return None
    return model.evaluate(X[valid], y[valid].astype(int), device)


############################################################
# MAIN LOOP
############################################################
results = []

for species, antibiotic in PAIRS:
    print(f"\n{'='*60}")
    print(f"  {species} | {antibiotic}")
    print(f"{'='*60}")

    # ── DRIAMS-A train data ────────────────────────────────
    sp_mask_tr = (label_train == species)
    ab_idx_tr  = get_ab_idx(antibiotic, ab_train)
    if ab_idx_tr is None:
        print(f"  '{antibiotic}' not in DRIAMS. Skipping.")
        continue

    y_tr_raw = amr_train[sp_mask_tr, ab_idx_tr]
    valid_tr = ~np.isnan(y_tr_raw)
    X_tr_src = data_train[sp_mask_tr][valid_tr]
    y_tr_src = y_tr_raw[valid_tr]

    if len(y_tr_src) < 20 or len(np.unique(y_tr_src.astype(int))) < 2:
        print(f"  Not enough DRIAMS-A data ({len(y_tr_src)}). Skipping.")
        continue

    print(f"  Train DRIAMS-A: n={len(y_tr_src)}  "
          f"R={int(y_tr_src.sum())} / S={int((y_tr_src==0).sum())}")

    # ── Pretrain source-only model ─────────────────────────
    source_model = train_source_only(X_tr_src, y_tr_src)

    # ── MS-UMG data for this pair ──────────────────────────
    sp_mask_te = (label_msumg == species)
    ab_idx_te  = get_ab_idx(antibiotic, ab_msumg)
    if ab_idx_te is None:
        print(f"  '{antibiotic}' not in MS-UMG. Skipping eval.")
        continue

    y_te_all = amr_msumg[sp_mask_te, ab_idx_te]
    X_te_all = data_msumg[sp_mask_te]

    # ── Three test subsets ─────────────────────────────────
    subsets = {
        "all":   np.ones(sp_mask_te.sum(), dtype=bool),
        "agar":  agar_mask_msumg[sp_mask_te],
        "chrom": chrom_mask_msumg[sp_mask_te],
    }

    for subset_name, subset_mask in subsets.items():
        if subset_mask.sum() == 0:
            continue

        X_sub = X_te_all[subset_mask]
        y_sub = y_te_all[subset_mask]

        # Filter NaN once
        valid_sub = ~np.isnan(y_sub)
        X_sub_clean = X_sub[valid_sub]
        y_sub_clean = y_sub[valid_sub]

        if len(y_sub_clean) < 10 or len(np.unique(y_sub_clean.astype(int))) < 2:
            print(f"  [{subset_name}] Not enough data. Skipping.")
            continue

        # ── SOURCE-ONLY: evaluate on full subset ──────────
        so_metrics = safe_evaluate(source_model, X_sub_clean, y_sub_clean)
        if so_metrics:
            print(f"  [{subset_name:5s}] source_only  "
                  f"n={so_metrics['n']:4d}  AUC={so_metrics['auc']:.3f}  "
                  f"PR-AUC={so_metrics['pr_auc']:.3f}")
            results.append({
                "species":    species,
                "antibiotic": antibiotic,
                "subset":     subset_name,
                "method":     "source_only",
                "seed":       None,
                "n_test":     so_metrics["n"],
                "roc_auc":    round(so_metrics["auc"],    4),
                "pr_auc":     round(so_metrics["pr_auc"], 4),
            })

        # ── FINE-TUNING: 5 seeds, test set per seed ───────
        ft_aucs, ft_praucs = [], []
        for seed in SEEDS:
            # Need enough samples for stratified 70/10/20
            if len(y_sub_clean) < 20:
                break

            ft_model, X_test_seed, y_test_seed = finetune(
                source_model, X_sub_clean, y_sub_clean, seed
            )
            ft_metrics = safe_evaluate(ft_model, X_test_seed, y_test_seed)
            if ft_metrics is None:
                continue

            ft_aucs.append(ft_metrics["auc"])
            ft_praucs.append(ft_metrics["pr_auc"])

            results.append({
                "species":    species,
                "antibiotic": antibiotic,
                "subset":     subset_name,
                "method":     "finetuned",
                "train_source": "DRIAMS_ABC_MARISMA",
                "seed":       seed,
                "n_test":     ft_metrics["n"],
                "roc_auc":    round(ft_metrics["auc"],    4),
                "pr_auc":     round(ft_metrics["pr_auc"], 4),
            })

        if ft_aucs:
            print(f"  [{subset_name:5s}] finetuned    "
                  f"AUC={np.mean(ft_aucs):.3f}±{np.std(ft_aucs):.3f}  "
                  f"PR-AUC={np.mean(ft_praucs):.3f}±{np.std(ft_praucs):.3f}")
    
        # ── TARGET-ONLY: 5 seeds, mismo split que finetuned ───
        to_aucs, to_praucs = [], []
        for seed in SEEDS:
            if len(y_sub_clean) < 20:
                break

            to_model, X_test_seed, y_test_seed = target_only(
                X_sub_clean, y_sub_clean, seed
            )
            to_metrics = safe_evaluate(to_model, X_test_seed, y_test_seed)
            if to_metrics is None:
                continue

            to_aucs.append(to_metrics["auc"])
            to_praucs.append(to_metrics["pr_auc"])

            results.append({
                "species":    species,
                "antibiotic": antibiotic,
                "subset":     subset_name,
                "method":     "target_only",
                "seed":       seed,
                "n_test":     to_metrics["n"],
                "roc_auc":    round(to_metrics["auc"],    4),
                "pr_auc":     round(to_metrics["pr_auc"], 4),
            })

        if to_aucs:
            print(f"  [{subset_name:5s}] target_only  "
                  f"AUC={np.mean(to_aucs):.3f}±{np.std(to_aucs):.3f}  "
                  f"PR-AUC={np.mean(to_praucs):.3f}±{np.std(to_praucs):.3f}")

    # Partial save
    pd.DataFrame(results).to_csv(
        experiment_dir / "chen_replication_partial.csv", index=False
    )

############################################################
# SAVE & SUMMARY
############################################################
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "chen_replication.csv", index=False)
print(f"\nResults saved: {experiment_dir / 'chen_replication.csv'}")

# Summary table: mean±std over seeds for finetuned, single value for source_only
summary = (
    df.groupby(["species", "antibiotic", "subset", "method"])
    .agg(
        auc_mean=("roc_auc", "mean"),
        auc_std=("roc_auc", "std"),
        prauc_mean=("pr_auc", "mean"),
        prauc_std=("pr_auc", "std"),
    )
    .round(3)
    .reset_index()
)
summary["auc_mean_std"]   = summary.apply(
    lambda r: f"{r.auc_mean:.3f}" if pd.isna(r.auc_std)
              else f"{r.auc_mean:.3f} ± {r.auc_std:.3f}", axis=1
)
summary["prauc_mean_std"] = summary.apply(
    lambda r: f"{r.prauc_mean:.3f}" if pd.isna(r.prauc_std)
              else f"{r.prauc_mean:.3f} ± {r.prauc_std:.3f}", axis=1
)

print("\n" + summary[["species","antibiotic","subset","method",
                       "auc_mean_std","prauc_mean_std"]].to_string(index=False))
summary.to_csv(experiment_dir / "chen_replication_summary.csv", index=False)