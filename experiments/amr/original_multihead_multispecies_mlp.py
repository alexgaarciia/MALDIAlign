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
space          = "original/mlp_crosscenter"
timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)
print(f"\nExperiment directory: {experiment_dir}")

############################################################
# IMPORTS
############################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.training.data_pipeline import load_pkl
from src.data.preprocessing import row_minmax_normalize
from models.baselines.amr_mlp import SimpleAMRMLP_Extended

############################################################
# SPECIES CONFIG
############################################################
SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae": {
        "antibiotics":  ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260514_101045/data_splits.pkl"),
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl",
    },
    "Escherichia_Coli": {
        "antibiotics":  ["Ampicillin", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260513_152147/data_splits.pkl"),
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl",
    },
    "Staphylococcus_Aureus": {
        "antibiotics":  ["Oxacillin", "Clindamycin", "Erythromycin"],
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260519_103809/data_splits.pkl"),
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_v2.pkl",
    },
}


SKIP_CENTERS = ["DRIAMS_D"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

############################################################
# HELPERS
############################################################

def load_species_data(cfg, species):
    """Load dataset filtered to one species."""
    dataset  = load_pkl(cfg["dataset_path"])
    data     = row_minmax_normalize(dataset["data"])
    amr      = dataset["amr"]
    ab_list  = list(dataset["antibiotics"])
    labels   = dataset["label"]
    raw_meta = dataset["meta"]
    meta     = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) \
               else pd.DataFrame(raw_meta)

    if "agar" in meta.columns:
        chrom = ((meta["hospital"] == "MS-UMG") & (meta["agar"] == "chrom")).values
        keep  = ~chrom
        data, amr, labels = data[keep], amr[keep], labels[keep]
        meta = meta[keep].reset_index(drop=True)

    sp_mask = (labels == species)
    data    = data[sp_mask]
    amr     = amr[sp_mask]
    labels  = labels[sp_mask]
    meta    = meta[sp_mask].reset_index(drop=True)

    valid   = ~np.all(np.isnan(amr), axis=0)
    ab_list = [a for a, k in zip(ab_list, valid) if k]
    amr     = amr[:, valid]

    print(f"  Loaded {len(data)} samples for {species}")
    return data, amr, ab_list, labels, meta


def get_center_arrays(center, domain_splits, meta, data, amr, ab_indices):
    """
    Return train/val/test arrays for one center.
    amr is already filtered to ab_indices columns.
    """
    amr_sel = amr[:, ab_indices]

    if center in domain_splits:
        sp        = domain_splits[center]
        train_idx = sp["train_idx"]
        val_idx   = sp.get("val_idx", np.array([], dtype=int))
        test_idx  = sp["test_idx"]
        is_ood    = False
    else:
        train_idx = np.array([], dtype=int)
        val_idx   = np.array([], dtype=int)
        test_idx  = np.where(meta["hospital"] == center)[0]
        is_ood    = True

    def get(idx):
        if len(idx) == 0:
            return (np.array([]).reshape(0, data.shape[1]),
                    np.array([]).reshape(0, len(ab_indices)))
        return data[idx], amr_sel[idx].astype(float)

    X_tr, y_tr = get(train_idx)
    X_vl, y_vl = get(val_idx)
    X_te, y_te = get(test_idx)

    return X_tr, y_tr, X_vl, y_vl, X_te, y_te, is_ood


def train_mlp(X_tr, y_tr, X_vl, y_vl, antibiotics):
    """Train one SimpleAMRMLP_Extended for all antibiotics of a species."""
    if len(y_vl) < 4:
        n_val = max(int(len(y_tr) * 0.1), 1)
        X_vl, y_vl = X_tr[-n_val:], y_tr[-n_val:]
        X_tr, y_tr = X_tr[:-n_val], y_tr[:-n_val]

    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr.astype(float), dtype=torch.float32),
        ),
        batch_size=256, shuffle=True
    )
    vl_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_vl, dtype=torch.float32),
            torch.tensor(y_vl.astype(float), dtype=torch.float32),
        ),
        batch_size=256, shuffle=False
    )

    model = SimpleAMRMLP_Extended(
        input_dim        = X_tr.shape[1],
        latent_dim       = 128,
        n_antibiotics    = len(antibiotics),
        antibiotic_names = antibiotics,
    )
    model.trainloop(tr_loader, vl_loader, device)
    return model


############################################################
# MAIN LOOP — one MLP per (species, train_center)
############################################################
results = []

for species, cfg in SPECIES_CONFIG.items():
    print(f"\n{'#'*70}")
    print(f"  SPECIES: {species}")
    print(f"{'#'*70}")

    data, amr, ab_list, labels, meta = load_species_data(cfg, species)

    with open(cfg["splits_path"], "rb") as f:
        saved_splits = pickle.load(f)
    domain_splits = saved_splits.get("splits_per_domain", {})

    antibiotics = [a for a in cfg["antibiotics"] if a in ab_list]
    if not antibiotics:
        print("  No antibiotics found, skipping.")
        continue
    ab_indices = [ab_list.index(a) for a in antibiotics]

    print(f"  Antibiotics: {antibiotics}")
    print(f"  Domains in splits: {list(domain_splits.keys())}")

    all_centers = [c for c in sorted(meta["hospital"].unique())
                   if c not in SKIP_CENTERS]

    # Build per-center arrays once
    center_data = {}
    for center in all_centers:
        X_tr, y_tr, X_vl, y_vl, X_te, y_te, is_ood = get_center_arrays(
            center, domain_splits, meta, data, amr, ab_indices
        )
        # Need test data with at least one antibiotic having 2 classes
        has_valid_test = any(
            len(np.unique(y_te[~np.isnan(y_te[:, j]), j])) >= 2
            for j in range(len(antibiotics))
        ) if len(y_te) > 0 else False

        if not has_valid_test:
            print(f"  Skipping {center}: no valid test data")
            continue

        center_data[center] = dict(
            X_tr=X_tr, y_tr=y_tr,
            X_vl=X_vl, y_vl=y_vl,
            X_te=X_te, y_te=y_te,
            is_ood=is_ood,
        )
        print(f"  {center}: train={len(y_tr)}, val={len(y_vl)}, test={len(y_te)}, ood={is_ood}")

    # Train one MLP per source center, test on all centers
    for train_center, cd in center_data.items():
        if cd["is_ood"]:
            continue
        if len(cd["y_tr"]) < 10:
            print(f"  Skipping {train_center}: not enough train data")
            continue

        print(f"\n  Training on {train_center}: n={len(cd['y_tr'])}")

        mlp = train_mlp(cd["X_tr"], cd["y_tr"], cd["X_vl"], cd["y_vl"], antibiotics)

        # Evaluate on all centers
        for test_center, td in center_data.items():
            eval_res = mlp.evaluate(td["X_te"], td["y_te"], device)

            for ab_name, metrics in eval_res.items():
                results.append({
                    "species":      species,
                    "antibiotic":   ab_name,
                    "train_center": train_center,
                    "test_center":  test_center,
                    "same_center":  train_center == test_center,
                    "n_test":       metrics["n"],
                    "roc_auc":      round(metrics["auc"], 4),
                    "pr_auc":       round(metrics["pr_auc"], 4),
                })
                print(f"    → {test_center} | {ab_name}: "
                      f"AUC={metrics['auc']:.3f}  PRAUC={metrics['pr_auc']:.3f}")

        # Save partial results
        pd.DataFrame(results).to_csv(
            experiment_dir / "cross_center_amr_original_partial.csv", index=False
        )

############################################################
# SAVE FINAL CSV
############################################################
df = pd.DataFrame(results)
csv_path = experiment_dir / "cross_center_amr_original.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved: {csv_path}")

############################################################
# HEATMAPS — one per (species, antibiotic)
############################################################
print("\n===== GENERATING HEATMAPS =====")

for (species, atb), grp in df.groupby(["species", "antibiotic"]):
    all_c = sorted(set(grp["train_center"].tolist() + grp["test_center"].tolist()))
    mat   = grp.pivot(index="train_center", columns="test_center", values="roc_auc")
    mat   = mat.reindex(index=all_c, columns=all_c)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis",
                vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5)
    ax.set_facecolor("#333333")
    ax.set_title(f"AMR — Original space\n{atb} | {species}")
    ax.set_xlabel("Test center")
    ax.set_ylabel("Train center")
    plt.tight_layout()

    clean = lambda s: s.replace(" ", "_").replace("/", "_")
    fig.savefig(
        experiment_dir / f"heatmap_{clean(species)}_{clean(atb)}.png",
        dpi=300
    )
    plt.close()

print(f"Heatmaps saved in: {experiment_dir}")
print("\nDONE.")
