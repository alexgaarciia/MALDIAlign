############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "original/mlp_pooled"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)
print(f"\nExperiment directory: {experiment_dir}")

############################################################
# IMPORTS
############################################################
import numpy as np
import pandas as pd
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

SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TEST_DOMAIN    = "MS-UMG"

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


def get_pooled_train(domain_splits, meta, data, amr, ab_indices):
    """
    Pool train and val from all source domains.
    Returns X and amr matrix with only the selected antibiotic columns.
    """
    all_X_tr, all_y_tr = [], []
    all_X_vl, all_y_vl = [], []

    for domain in SOURCE_DOMAINS:
        if domain not in domain_splits:
            continue
        sp = domain_splits[domain]

        tr_idx = sp["train_idx"]
        vl_idx = sp.get("val_idx", np.array([], dtype=int))

        if len(tr_idx) > 0:
            all_X_tr.append(data[tr_idx])
            all_y_tr.append(amr[tr_idx][:, ab_indices])

        if len(vl_idx) > 0:
            all_X_vl.append(data[vl_idx])
            all_y_vl.append(amr[vl_idx][:, ab_indices])

    X_tr = np.vstack(all_X_tr) if all_X_tr else np.array([]).reshape(0, data.shape[1])
    y_tr = np.vstack(all_y_tr) if all_y_tr else np.array([]).reshape(0, len(ab_indices))
    X_vl = np.vstack(all_X_vl) if all_X_vl else np.array([]).reshape(0, data.shape[1])
    y_vl = np.vstack(all_y_vl) if all_y_vl else np.array([]).reshape(0, len(ab_indices))

    return X_tr, y_tr, X_vl, y_vl


def get_test_data(meta, data, amr, ab_indices):
    """Get test data from MS-UMG."""
    test_mask = (meta["hospital"] == TEST_DOMAIN).values
    test_idx  = np.where(test_mask)[0]
    if len(test_idx) == 0:
        return np.array([]).reshape(0, data.shape[1]), \
               np.array([]).reshape(0, len(ab_indices))
    return data[test_idx], amr[test_idx][:, ab_indices]


############################################################
# MAIN LOOP — one MLP per species, predicting all antibiotics
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

    # Only keep antibiotics that exist in this dataset
    antibiotics = [a for a in cfg["antibiotics"] if a in ab_list]
    if not antibiotics:
        print("  No antibiotics found, skipping.")
        continue
    ab_indices = [ab_list.index(a) for a in antibiotics]

    print(f"  Antibiotics: {antibiotics}")

    # Pool all source domains
    X_tr, y_tr, X_vl, y_vl = get_pooled_train(
        domain_splits, meta, data, amr, ab_indices
    )

    print(f"  Pooled train: n={len(y_tr)}")
    for j, ab_name in enumerate(antibiotics):
        col = y_tr[:, j]
        valid = ~np.isnan(col)
        print(f"    {ab_name}: n={valid.sum()}, "
              f"R={(col[valid]==1).sum():.0f}, S={(col[valid]==0).sum():.0f}")

    # Val fallback
    if len(y_vl) < 4:
        n_val = max(int(len(y_tr) * 0.1), 1)
        X_vl, y_vl = X_tr[-n_val:], y_tr[-n_val:]
        X_tr, y_tr = X_tr[:-n_val], y_tr[:-n_val]

    # Build DataLoaders
    tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        ),
        batch_size=256, shuffle=True
    )
    vl_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_vl, dtype=torch.float32),
            torch.tensor(y_vl, dtype=torch.float32),
        ),
        batch_size=256, shuffle=False
    )

    # Train one MLP for all antibiotics of this species
    model = SimpleAMRMLP_Extended(
        input_dim        = X_tr.shape[1],
        latent_dim       = 128,
        n_antibiotics    = len(antibiotics),
        antibiotic_names = antibiotics,
    )
    model.trainloop(tr_loader, vl_loader, device)

    # Evaluate on MS-UMG
    X_te, y_te = get_test_data(meta, data, amr, ab_indices)

    if len(X_te) == 0:
        print(f"  No test data in {TEST_DOMAIN}, skipping.")
        continue

    eval_results = model.evaluate(X_te, y_te, device)

    print(f"\n  Results on {TEST_DOMAIN}:")
    for ab_name, metrics in eval_results.items():
        print(f"    {ab_name}: AUC={metrics['auc']:.3f}  PRAUC={metrics['pr_auc']:.3f}  n={metrics['n']}")
        results.append({
            "species":       species,
            "antibiotic":    ab_name,
            "train_domains": "+".join(SOURCE_DOMAINS),
            "test_domain":   TEST_DOMAIN,
            "n_train":       len(y_tr),
            "n_test":        metrics["n"],
            "roc_auc":       round(metrics["auc"], 4),
            "pr_auc":        round(metrics["pr_auc"], 4),
            "method":        "MLP_pooled_original",
        })

############################################################
# SAVE
############################################################
df = pd.DataFrame(results)
csv_path = experiment_dir / "pooled_amr_original.csv"
df.to_csv(csv_path, index=False)
print(f"\nResults saved: {csv_path}")

############################################################
# SUMMARY
############################################################
print(f"\n{'='*60}")
print("SUMMARY: Pooled source → MS-UMG (Original space)")
print(f"{'='*60}")
print(f"{'Species':<25} {'Antibiotic':<30} {'AUC':>7} {'PRAUC':>7}")
print("-"*60)
for _, row in df.iterrows():
    print(f"  {row['species']:<23} {row['antibiotic']:<30} "
          f"{row['roc_auc']:>7.3f} {row['pr_auc']:>7.3f}")

print(f"\nMean AUC:   {df['roc_auc'].mean():.3f}")
print(f"Mean PRAUC: {df['pr_auc'].mean():.3f}")
print("\nDONE.")
