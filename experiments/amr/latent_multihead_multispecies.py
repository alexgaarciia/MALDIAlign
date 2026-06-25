############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

train_exp_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260410_105028")
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "latent/multihead"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory created at:\n{experiment_dir}")
print("-" * 60)


############################################################
# IMPORTS
############################################################
import torch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.training.data_pipeline import load_pkl
from src.evaluation.eval import encode_latent
from src.data.preprocessing import row_minmax_normalize
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended


############################################################
# MINIMUM SAMPLE THRESHOLDS
############################################################
MIN_TRAIN = 10
MIN_TEST = 10
MIN_PER_CLASS_TRAIN = 3
MIN_PER_CLASS_TEST = 3

MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_PATIENCE = 10
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA AND SPLITS =====")

dataset = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl")
data = dataset["data"]
amr = dataset["amr"]
ab_list = dataset["antibiotics"]
labels = dataset["label"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

print(f"Raw data shape: {data.shape}")


############################################################
# FILTER ANTIBIOTICS BY NAME
############################################################
antibiotics_filter = ["Amikacin", "Ceftazidime", "Ceftriaxone", "Clindamycin", "Erythromycin", "Meropenem", "Piperacillin-Tazobactam", "Tetracycline", "Vancomycin"]

if antibiotics_filter is not None:
    keep_idx = []
    keep_names = []
    for name in antibiotics_filter:
        if name in ab_list:
            keep_idx.append(ab_list.index(name))
            keep_names.append(name)
        else:
            print(f"WARNING: antibiotic '{name}' not found in dataset. Available: {ab_list}")

    if len(keep_idx) == 0:
        raise ValueError(f"None of the requested antibiotics found! Requested: {antibiotics_filter}, Available: {ab_list}")

    # Filtrar las columnas de la matriz AMR
    if amr.ndim > 1:
        amr = amr[:, keep_idx]
    
    # Actualizar la lista de nombres
    ab_list = keep_names

    print(f"\nFiltered antibiotics: {len(keep_names)} selected from total")
    print(f"  Selected: {keep_names}")


############################################################
# FILTER CHROM AGAR FROM THE START
############################################################
if "agar" in meta.columns:
    chrom_mask = (meta["hospital"] == "MS-UMG") & (meta["agar"] == "chrom")
    n_chrom = chrom_mask.sum()
    keep_mask = ~chrom_mask.values

    data = data[keep_mask]
    amr = amr[keep_mask]
    labels = labels[keep_mask]
    meta = meta.loc[keep_mask].reset_index(drop=True)

    # Build old→new index mapping for remapping splits
    old_to_new = np.full(len(keep_mask), -1, dtype=int)
    old_to_new[np.where(keep_mask)[0]] = np.arange(keep_mask.sum())

    print(f"Removed {n_chrom} MS-UMG chrom-agar samples. Remaining: {len(data)}")
else:
    old_to_new = None
    print("No 'agar' column — no chrom filtering applied")


############################################################
# NORMALIZE
############################################################
data = row_minmax_normalize(data)


############################################################
# LOAD AND REMAP SPLITS
############################################################
with open(train_exp_dir / "data_splits.pkl", "rb") as f:
    saved_splits = pickle.load(f)

domain_splits = saved_splits.get("splits_per_domain", {})

if old_to_new is not None:
    for center, indices in domain_splits.items():
        for key in ["train_idx", "val_idx", "test_idx"]:
            if key in indices:
                old_idx = indices[key]
                new_idx = old_to_new[old_idx]
                indices[key] = new_idx[new_idx >= 0]

all_centers = sorted(meta["hospital"].unique())
print(f"Data shape: {data.shape}")
print(f"Centers: {all_centers}")
print(f"Centers in splits: {sorted(domain_splits.keys())}")
print(f"Centers NOT in splits (OOD): {[c for c in all_centers if c not in domain_splits]}")


############################################################
# LOAD VAE & ENCODE LATENT
############################################################
print("\n===== LOADING PRETRAINED MODEL =====")

LATENT_DIM = 128
vae = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
    input_dim=data.shape[1],
    latent_dim=LATENT_DIM,
    num_domains=4,
    n_species=6,
    n_antibiotics=9,
    lambda_amr=100,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.load_state_dict(torch.load(train_exp_dir / "model.pth", map_location=device))
vae.to(device)
vae.eval()

total_params = sum(p.numel() for p in vae.parameters())
print(f"Model loaded on {device} | latent_dim={LATENT_DIM} | params={total_params:,}")

print("\n===== ENCODING LATENT =====")
Z_all = np.asarray(encode_latent(vae, data, device))


############################################################
# MLP ARCHITECTURE
############################################################
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
    def forward(self, x): return self.net(x)

def masked_bce_loss(logits, targets, pos_weights):
    mask = ~torch.isnan(targets)
    weight_mask = torch.ones_like(targets)
    for i in range(targets.shape[1]): weight_mask[:, i] = pos_weights[i]
    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    raw_loss = loss_func(logits, targets.nan_to_num(0))
    raw_loss = torch.where(targets == 1, raw_loss * weight_mask, raw_loss)
    return (raw_loss * mask.float()).sum() / (mask.sum() + 1e-8)

def train_multi_mlp(X_tr, Y_tr, X_val, Y_val, input_dim):
    output_dim = Y_tr.shape[1]
    model = MultiOutputMLP(input_dim, output_dim).to(device)
    pos_weights = torch.tensor([(np.nansum(Y_tr[:,i]==0)/max(np.nansum(Y_tr[:,i]==1),1)) for i in range(output_dim)], dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    
    X_tr_t, Y_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device), torch.tensor(Y_tr, dtype=torch.float32).to(device)
    X_val_t, Y_val_t = torch.tensor(X_val, dtype=torch.float32).to(device), torch.tensor(Y_val, dtype=torch.float32).to(device)
    
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, Y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    best_loss, patience, best_state = float("inf"), 0, None

    for epoch in range(MLP_EPOCHS):
        model.train()
        train_l = 0
        for xb, yb in loader:
            optimizer.zero_grad(); loss = masked_bce_loss(model(xb), yb, pos_weights); loss.backward(); optimizer.step()
            train_l += loss.item()
        
        model.eval()
        with torch.no_grad(): val_l = masked_bce_loss(model(X_val_t), Y_val_t, pos_weights).item()
        
        if epoch % 10 == 0: print(f"      Epoch {epoch:02d} | Train: {train_l/len(loader):.4f} | Val: {val_l:.4f}")
        if val_l < best_loss:
            best_loss, patience, best_state = val_l, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else: patience += 1
        if patience >= MLP_PATIENCE: break
            
    model.load_state_dict(best_state); return model


############################################################
# FILTER EMPTY ANTIBIOTICS
############################################################
if amr.ndim > 1:
    valid_mask = ~np.all(np.isnan(amr), axis=0)
    if amr.shape[1] > 1:
        ab_list = [a for a, keep in zip(ab_list, valid_mask) if keep]
    amr = amr[:, valid_mask]

print(f"\nValid antibiotics ({len(ab_list)}): {ab_list}")


############################################################
# CROSS-CENTER EVALUATION
############################################################
print("\n===== STARTING EVALUATION =====")
all_centers = sorted(meta["hospital"].unique())
species_list = sorted(np.unique(labels))
results = []
skipped_test = 0
skipped_train = 0

for species_name in species_list:
    print(f"\n{'#'*60}\n### SPECIES: {species_name}\n{'#'*60}")
    species_mask = (labels == species_name)
    species_idx_global = np.where(species_mask)[0]

    # --- Pre-train Multi-MLP per species once per center ---
    mlp_models_per_center = {}
    source_centers_present = [c for c in all_centers if c in domain_splits]
    for tr_c in source_centers_present:
        idx_c = domain_splits[tr_c]
        tr_idx = np.intersect1d(idx_c['train_idx'], species_idx_global)
        va_idx = np.intersect1d(idx_c['val_idx'], species_idx_global)
        if len(tr_idx) >= MIN_TRAIN:
            print(f"  Training MLP for {species_name} on {tr_c}...")
            mlp_models_per_center[tr_c] = train_multi_mlp(Z_all[tr_idx], amr[tr_idx], Z_all[va_idx], amr[va_idx], LATENT_DIM)

    # --- Antibiotic Loop ---
    for atb_idx, atb_name in enumerate(ab_list):
        y_atb_total = amr[species_idx_global, atb_idx]
        if np.all(np.isnan(y_atb_total)): continue

        print(f"\n  {atb_name}")
        center_data = {}
        for center in all_centers:
            if center in domain_splits:
                indices = domain_splits[center]
                train_idx = np.intersect1d(indices['train_idx'], species_idx_global)
                test_idx = np.intersect1d(indices['test_idx'], species_idx_global)
                is_ood = False
            else:
                train_idx = np.array([], dtype=int)
                test_idx = np.where((meta["hospital"] == center).values & species_mask)[0]
                is_ood = True

            y_col = amr[:, atb_idx]
            m_tr, m_te = ~np.isnan(y_col[train_idx]), ~np.isnan(y_col[test_idx])
            
            X_tr_clean, y_tr_clean = Z_all[train_idx][m_tr], y_col[train_idx][m_tr].astype(int)
            X_te_clean, y_te_clean = Z_all[test_idx][m_te], y_col[test_idx][m_te].astype(int)

            if len(y_te_clean) >= MIN_TEST and (y_te_clean==1).sum() >= MIN_PER_CLASS_TEST and (y_te_clean==0).sum() >= MIN_PER_CLASS_TEST:
                center_data[center] = {"train_data": X_tr_clean, "train_labels": y_tr_clean, "test_data": X_te_clean, "test_labels": y_te_clean, "is_ood": is_ood}
            elif len(y_te_clean) > 0: skipped_test += 1

        valid_train_centers = [c for c, d in center_data.items() if not d["is_ood"] and len(d["train_labels"]) >= MIN_TRAIN and (d["train_labels"]==1).sum() >= MIN_PER_CLASS_TRAIN]
        
        # Conteo de skips de entrenamiento
        n_skipped_here = sum(1 for c, d in center_data.items() if not d["is_ood"] and len(d["train_labels"]) > 0 and c not in valid_train_centers)
        skipped_train += n_skipped_here

        for train_center in valid_train_centers:
            d_tr_obj = center_data[train_center]
            lgbm = LGBMClassifier(n_estimators=300, scale_pos_weight=(d_tr_obj["train_labels"]==0).sum()/max((d_tr_obj["train_labels"]==1).sum(),1), random_state=42, verbosity=-1)
            lgbm.fit(d_tr_obj["train_data"], d_tr_obj["train_labels"])

            for test_center, d_te in center_data.items():
                # LGBM
                p_l = lgbm.predict_proba(d_te["test_data"])[:, 1]
                auc_l = roc_auc_score(d_te["test_labels"], p_l)
                pr_l = average_precision_score(d_te["test_labels"], p_l)
                results.append({"model": "LightGBM", "species": species_name, "antibiotic": atb_name, "train_center": train_center, "test_center": test_center, "is_ood": d_te["is_ood"], "roc_auc": auc_l, "pr_auc": pr_l})

                # MLP
                auc_m, pr_m = np.nan, np.nan
                if train_center in mlp_models_per_center:
                    m_mlp = mlp_models_per_center[train_center]
                    m_mlp.eval()
                    with torch.no_grad():
                        p_m_all = torch.sigmoid(m_mlp(torch.tensor(d_te["test_data"], dtype=torch.float32).to(device))).cpu().numpy()
                        p_m = p_m_all[:, atb_idx]
                        auc_m = roc_auc_score(d_te["test_labels"], p_m)
                        pr_m = average_precision_score(d_te["test_labels"], p_m)
                        results.append({"model": "MultiMLP", "species": species_name, "antibiotic": atb_name, "train_center": train_center, "test_center": test_center, "is_ood": d_te["is_ood"], "roc_auc": auc_m, "pr_auc": pr_m})

                prefix = "[OOD]" if d_te["is_ood"] else "     "
                print(f"    {prefix} {train_center} -> {test_center}: LGBM_AUC={auc_l:.3f} | MLP_AUC={auc_m:.3f}")

# SAVE
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "results_latent_comparison.csv", index=False)
print(f"\n{'='*60}\nResults saved: {experiment_dir / 'results_latent_comparison.csv'}")
print(f"Total evaluations: {len(results)}\nSkipped test: {skipped_test}\nSkipped train: {skipped_train}")


############################################################
# HEATMAPS COMPARATIVOS
############################################################
print("\n===== GENERATING COMPARATIVE HEATMAPS =====")
source_centers = [c for c in all_centers if c in domain_splits]
all_centers_ordered = source_centers + [c for c in all_centers if c not in domain_splits]

for metric in ["roc_auc", "pr_auc"]:
    metric_dir = experiment_dir / metric; metric_dir.mkdir(parents=True, exist_ok=True)
    for (sp, atb), df_g in df.groupby(['species', 'antibiotic']):
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        for i, mod in enumerate(["LightGBM", "MultiMLP"]):
            sub = df_g[df_g['model'] == mod]
            if not sub.empty:
                matrix = sub.pivot(index="train_center", columns="test_center", values=metric).reindex(index=source_centers, columns=all_centers_ordered)
                sns.heatmap(matrix, annot=True, fmt=".3f", cmap=("viridis" if metric=="roc_auc" else "magma"), ax=axes[i], vmin=(0.4 if metric=="roc_auc" else 0))
            axes[i].set_title(f"Model: {mod}"); axes[i].set_facecolor("#333333")
        plt.suptitle(f"LATENT SPACE {metric.upper()}: {atb} in {sp}")
        plt.savefig(metric_dir / f"comp_heatmap_{sp}_{atb}.png", dpi=300); plt.close()

print("\nDONE.")
