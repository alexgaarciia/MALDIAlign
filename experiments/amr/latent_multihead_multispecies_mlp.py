############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

train_exp_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260429_184236")
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "latent/mlp"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory created at:\n{experiment_dir}")
print("-" * 60)


############################################################
# IMPORTS
############################################################
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from src.training.data_pipeline import load_pkl
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head


############################################################
# MLP CLASSIFIER FOR BINARY AMR
############################################################
class AMR_MLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_amr_mlp(X_train, y_train, X_val, y_val, device, epochs=200, lr=1e-3, patience=20, pos_weight=None):
    """
    Train a binary MLP for a single antibiotic.
    Uses val split from data_splits.pkl for early stopping.
    """
    model = AMR_MLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    pw = torch.tensor([min(pos_weight, 20.0)], device=device) if pos_weight else None

    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_vl = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_vl = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_auc = -1.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_logits = model(X_tr)
        loss = nn.functional.binary_cross_entropy_with_logits(
            train_logits, y_tr, pos_weight=pw
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl)
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                val_logits, y_vl, pos_weight=pw
            ).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            train_probs = torch.sigmoid(train_logits.detach()).cpu().numpy()

        if len(np.unique(y_val)) >= 2:
            val_auc = roc_auc_score(y_val, val_probs)
        else:
            val_auc = 0.5

        if len(np.unique(y_train)) >= 2:
            train_auc = roc_auc_score(y_train, train_probs)
        else:
            train_auc = 0.5

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 20 == 0:
            print(f"        Ep {epoch+1:3d}/{epochs} | "
                  f"Train loss={loss.item():.4f} AUC={train_auc:.4f} | "
                  f"Val loss={val_loss:.4f} AUC={val_auc:.4f} | "
                  f"Best={best_auc:.4f} wait={wait}")

        if wait >= patience:
            print(f"        Early stop at epoch {epoch+1} | Best val AUC={best_auc:.4f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_auc


############################################################
# LOAD DATA & SPLITS
############################################################
from src.data.preprocessing import row_minmax_normalize

print("\n===== LOADING DATA AND SPLITS =====")

dataset = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl")
data = dataset["data"]
data = row_minmax_normalize(data)
amr = dataset["amr"]
ab_list = dataset["antibiotics"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

print(f"Data shape: {data.shape}")
print(f"AMR shape: {amr.shape}")
print(f"Labels: {len(dataset['label'])} samples, {len(np.unique(dataset['label']))} species")

# Mark MS-UMG chrom-agar samples for exclusion (indices stay valid)
exclude_mask = np.zeros(len(data), dtype=bool)
if "agar" in meta.columns:
    chrom_mask = (meta["hospital"] == "MS-UMG") & (meta["agar"] == "chrom")
    exclude_mask = chrom_mask.values
    print(f"Marked {exclude_mask.sum()} MS-UMG chrom-agar samples for exclusion")
else:
    print("No 'agar' column found — no chrom filtering applied")

with open(train_exp_dir / "data_splits.pkl", "rb") as f:
    saved_splits = pickle.load(f)

all_centers = sorted(meta["hospital"].unique())
print(f"Centers identified: {all_centers}")


############################################################
# LOAD VAE & ENCODE LATENT
############################################################
print("\n===== LOADING PRETRAINED MODEL =====")

LATENT_DIM = 128

model_path = train_exp_dir / "model.pth"
vae = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head(
    input_dim=data.shape[1],
    latent_dim=LATENT_DIM,
    num_domains=4,
    n_species=1,
    n_antibiotics=1,
    lambda_amr=100,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.to(device)
vae.eval()

total_params = sum(p.numel() for p in vae.parameters())
print(f"Model loaded on {device} | latent_dim={LATENT_DIM} | params={total_params:,}")

print("\n===== ENCODING LATENT =====")
Z_all = np.asarray(encode_latent(vae, data, device))
print(f"Latent shape: {Z_all.shape} (expected: ({data.shape[0]}, {LATENT_DIM}))")


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
# CROSS-CENTER EVALUATION WITH MLP
############################################################
print("\n===== STARTING CROSS-CENTER EVALUATION =====")
print("=" * 60)

centers = sorted(meta["hospital"].unique())
species_list = sorted(np.unique(dataset["label"]))
results = []
domain_splits = saved_splits.get("splits_per_domain", {})

for species_name in species_list:
    print(f"\n{'#'*60}")
    print(f"### SPECIES: {species_name}")
    print('#' * 60)

    species_mask_global = (dataset["label"] == species_name)
    species_indices_global = np.where(species_mask_global)[0]
    print(f"    Total samples for species: {len(species_indices_global)}")

    for atb_idx, atb_name in enumerate(ab_list):
        y_atb_total = amr[species_indices_global, atb_idx]
        if np.all(np.isnan(y_atb_total)):
            continue

        n_valid = (~np.isnan(y_atb_total)).sum()
        n_pos = int((y_atb_total == 1).sum())
        n_neg = int((y_atb_total == 0).sum())
        print(f"\n  Antibiotic: {atb_name} | valid={n_valid} (R={n_pos}, S={n_neg})")

        center_data = {}
        for center in centers:
            if center in domain_splits:
                indices = domain_splits[center]
                train_idx = np.intersect1d(indices['train_idx'], species_indices_global)
                val_idx = np.intersect1d(indices['val_idx'], species_indices_global) if 'val_idx' in indices else np.array([], dtype=int)
                test_idx = np.intersect1d(indices['test_idx'], species_indices_global)
                is_ood = False
            else:
                # OOD center
                train_idx = np.array([], dtype=int)
                val_idx = np.array([], dtype=int)
                hospital_mask = (meta["hospital"] == center)
                test_idx = np.where(hospital_mask & species_mask_global)[0]
                # Exclude chrom agar from MS-UMG
                test_idx = test_idx[~exclude_mask[test_idx]]
                is_ood = True

            # Extract
            y_col = amr[:, atb_idx]
            y_tr = y_col[train_idx] if len(train_idx) > 0 else np.array([])
            y_vl = y_col[val_idx] if len(val_idx) > 0 else np.array([])
            y_te = y_col[test_idx]

            Z_tr = Z_all[train_idx] if len(train_idx) > 0 else np.zeros((0, Z_all.shape[1]))
            Z_vl = Z_all[val_idx] if len(val_idx) > 0 else np.zeros((0, Z_all.shape[1]))
            Z_te = Z_all[test_idx]

            # Clean NaNs
            m_tr = ~np.isnan(y_tr) if len(y_tr) > 0 else np.array([], dtype=bool)
            m_vl = ~np.isnan(y_vl) if len(y_vl) > 0 else np.array([], dtype=bool)
            m_te = ~np.isnan(y_te)

            Z_train_clean, y_train_clean = Z_tr[m_tr], y_tr[m_tr].astype(int)
            Z_val_clean = Z_vl[m_vl] if np.any(m_vl) else np.zeros((0, Z_all.shape[1]))
            y_val_clean = y_vl[m_vl].astype(int) if np.any(m_vl) else np.array([])
            Z_test_clean, y_test_clean = Z_te[m_te], y_te[m_te].astype(int)

            if len(np.unique(y_test_clean)) >= 2:
                center_data[center] = {
                    "train_z": Z_train_clean,
                    "train_y": y_train_clean,
                    "val_z": Z_val_clean,
                    "val_y": y_val_clean,
                    "test_z": Z_test_clean,
                    "test_y": y_test_clean,
                    "is_ood": is_ood,
                }

        # Print center summary
        for c, cd in center_data.items():
            ood_tag = " [OOD]" if cd["is_ood"] else ""
            tr_n = len(cd["train_y"])
            vl_n = len(cd["val_y"])
            te_n = len(cd["test_y"])
            te_pos = int((cd["test_y"] == 1).sum())
            print(f"    {c}{ood_tag}: train={tr_n} val={vl_n} test={te_n} (R={te_pos}, S={te_n - te_pos})")

        # Train and test cross-center
        valid_train_centers = [
            c for c, d in center_data.items()
            if not d["is_ood"] and len(d["train_y"]) > 0 and len(np.unique(d["train_y"])) >= 2
        ]

        if len(valid_train_centers) == 0:
            print(f"    >> No valid train centers for {atb_name}. Skipping.")
            continue

        for train_center in valid_train_centers:
            cd = center_data[train_center]
            Z_tr_mlp = cd["train_z"]
            y_tr_mlp = cd["train_y"]

            # Use val from splits; fallback to 10% of train
            if len(cd["val_y"]) > 0 and len(np.unique(cd["val_y"])) >= 2:
                Z_vl_mlp = cd["val_z"]
                y_vl_mlp = cd["val_y"]
                val_source = "splits"
            else:
                n_val = max(int(len(y_tr_mlp) * 0.1), 1)
                Z_vl_mlp = Z_tr_mlp[-n_val:]
                y_vl_mlp = y_tr_mlp[-n_val:]
                Z_tr_mlp = Z_tr_mlp[:-n_val]
                y_tr_mlp = y_tr_mlp[:-n_val]
                val_source = f"fallback (last {n_val})"

            n_pos = int((y_tr_mlp == 1).sum())
            n_neg = int((y_tr_mlp == 0).sum())
            pw = n_neg / n_pos if n_pos > 0 else 1.0

            print(f"\n    Training MLP: {train_center} (train={len(y_tr_mlp)}, val={len(y_vl_mlp)} [{val_source}], R={n_pos}, S={n_neg}, pw={pw:.2f})")

            mlp, best_val_auc = train_amr_mlp(
                Z_tr_mlp, y_tr_mlp,
                Z_vl_mlp, y_vl_mlp,
                device=device,
                pos_weight=pw,
            )
            print(f"    Best val AUC: {best_val_auc:.4f}")

            # Test on all centers
            for test_center in center_data.keys():
                Z_te_mlp = center_data[test_center]["test_z"]
                y_te_mlp = center_data[test_center]["test_y"]

                mlp.eval()
                with torch.no_grad():
                    Z_te_t = torch.tensor(Z_te_mlp, dtype=torch.float32, device=device)
                    y_prob = torch.sigmoid(mlp(Z_te_t)).cpu().numpy()

                auc = roc_auc_score(y_te_mlp, y_prob)
                ood_tag = " [OOD]" if center_data[test_center]["is_ood"] else ""
                print(f"      {train_center} -> {test_center}{ood_tag}: AUC={auc:.4f}")

                results.append({
                    "species": species_name,
                    "antibiotic": atb_name,
                    "train_center": train_center,
                    "test_center": test_center,
                    "roc_auc": auc,
                })

# SAVE
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "cross_center_results_mlp.csv", index=False)
print(f"\n{'='*60}")
print(f"Results saved: {experiment_dir / 'cross_center_results_mlp.csv'}")
print(f"Total evaluations: {len(results)}")


############################################################
# HEATMAPS
############################################################
print("\n===== GENERATING HEATMAPS =====")

n_heatmaps = 0
for (species, atb), df_group in df.groupby(['species', 'antibiotic']):
    roc_matrix = df_group.pivot(index="train_center", columns="test_center", values="roc_auc")
    roc_matrix = roc_matrix.reindex(index=all_centers, columns=all_centers)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(roc_matrix, annot=True, fmt=".3f", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_facecolor("#333333")
    plt.title(f"MLP on Latent (z={LATENT_DIM}) - AUC: {atb} in {species}")
    plt.xlabel("Test Center")
    plt.ylabel("Train Center")

    clean_sp = species.replace(" ", "_")
    clean_atb = atb.replace("/", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(experiment_dir / f"heatmap_MLP_{clean_sp}_{clean_atb}.png", dpi=300)
    plt.close()
    n_heatmaps += 1

print(f"Generated {n_heatmaps} heatmaps")
print(f"\nDONE. All results saved in: {experiment_dir}")
