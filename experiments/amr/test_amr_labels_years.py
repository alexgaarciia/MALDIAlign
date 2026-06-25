############################################################
# PATH CONFIGURATION
############################################################
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "original/mlp_temporal"
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.training.data_pipeline import load_pkl
from src.data.preprocessing import row_minmax_normalize


############################################################
# GLOBAL VARIABLES
############################################################
MIN_TRAIN, MIN_TEST               = 10, 10
MIN_PER_CLASS_TRAIN               = 3
MIN_PER_CLASS_TEST                = 3
MLP_EPOCHS, MLP_LR, MLP_PATIENCE = 200, 1e-3, 20
TEST_SIZE, VAL_SIZE, SEED         = 0.2, 0.1, 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA =====")

dataset    = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl")
X_original = dataset["data"]
amr        = dataset["amr"]
ab_list    = dataset["antibiotics"]
labels     = dataset["label"]
raw_meta   = dataset["meta"]
meta       = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)
print(meta[meta["hospital"] == "MARISMA"]["year"].value_counts(dropna=False))

assert "year" in meta.columns, "meta debe tener columna 'year'"
meta["year"] = meta["year"].astype(int)

print(f"Raw data shape: {X_original.shape}")


############################################################
# FILTER CHROM AGAR
############################################################
if "agar" in meta.columns:
    chrom_mask = (meta["hospital"] == "MS-UMG") & (meta["agar"] == "chrom")
    n_chrom    = chrom_mask.sum()
    keep_mask  = ~chrom_mask.values
    X_original = X_original[keep_mask]
    amr        = amr[keep_mask]
    labels     = labels[keep_mask]
    meta       = meta.loc[keep_mask].reset_index(drop=True)
    print(f"Removed {n_chrom} MS-UMG chrom-agar samples. Remaining: {len(X_original)}")


############################################################
# NORMALIZE
############################################################
X_original = row_minmax_normalize(X_original)


############################################################
# FILTER EMPTY ANTIBIOTICS
############################################################
if amr.ndim > 1:
    valid_mask = ~np.all(np.isnan(amr), axis=0)
    ab_list    = [a for a, keep in zip(ab_list, valid_mask) if keep]
    amr        = amr[:, valid_mask]

print(f"Valid antibiotics ({len(ab_list)}): {ab_list}")

all_centers = sorted(meta["hospital"].unique())
all_years   = sorted(meta["year"].unique())
print(f"Centers : {all_centers}")
print(f"Years   : {all_years}")


############################################################
# CREAR SPLITS POR (center, year)
# Cada celda tiene su propio train/val/test estratificado
# por especie, completamente independiente del resto
############################################################
print("\n===== CREATING SPLITS BY (center, year) =====")

# Centros que quieres usar solo como test (OOD)
# Déjalo vacío si todos los centros participan en train
OOD_CENTERS = []

domain_splits = {}

for center in all_centers:
    for year in all_years:
        cell_mask = ((meta["hospital"] == center) & (meta["year"] == year)).values
        cell_idx  = np.where(cell_mask)[0]

        if len(cell_idx) == 0:
            continue

        cell_labels = labels[cell_idx]
        unique_cls  = np.unique(cell_labels)
        print(f"[{center} | {year}] n={len(cell_idx)}, especies={len(unique_cls)}: {unique_cls}")

        # Pocas muestras o centro OOD -> solo test, sin train
        if center in OOD_CENTERS or len(cell_idx) < 20:
            domain_splits[(center, year)] = {
                "train_idx": np.array([], dtype=int),
                "val_idx":   np.array([], dtype=int),
                "test_idx":  cell_idx,
                "is_ood":    True,
            }
            print(f"  [{center} | {year}] OOD -> test only: {len(cell_idx)} samples")
            continue

        try:
            idx_temp, idx_test = train_test_split(
                cell_idx, test_size=TEST_SIZE, random_state=SEED, stratify=cell_labels
            )
            relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
            idx_train, idx_val = train_test_split(
                idx_temp, test_size=relative_val, random_state=SEED, stratify=labels[idx_temp]
            )
        except ValueError:
            idx_temp, idx_test = train_test_split(cell_idx, test_size=TEST_SIZE, random_state=SEED)
            relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
            idx_train, idx_val = train_test_split(idx_temp, test_size=relative_val, random_state=SEED)

        domain_splits[(center, year)] = {
            "train_idx": idx_train,
            "val_idx":   idx_val,
            "test_idx":  idx_test,
            "is_ood":    False,
        }
        print(f"  [{center} | {year}] Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

print(f"\nTotal cells: {len(domain_splits)}")


############################################################
# MLP
############################################################
class AMR_MLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024),      nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512),       nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128),        nn.BatchNorm1d(128),  nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


############################################################
# CROSS-CENTER x CROSS-YEAR EVALUATION
############################################################
species_list = sorted(np.unique(labels))
results      = []
skipped_test = 0

for species_name in species_list:
    print(f"\n{'#'*60}\n### SPECIES: {species_name}\n{'#'*60}")
    species_mask = (labels == species_name)

    for atb_idx, atb_name in enumerate(ab_list):
        y_atb_total = amr[species_mask, atb_idx]
        if np.all(np.isnan(y_atb_total)):
            continue

        print(f"\n  {atb_name}")
        y_col = amr[:, atb_idx]

        # ── Construir cell_data[(center, year)] ──────────────────────────
        cell_data = {}

        for (center, year), sp in domain_splits.items():
            train_idx = sp["train_idx"]
            val_idx   = sp["val_idx"]
            test_idx  = sp["test_idx"]
            is_ood    = sp["is_ood"]

            # Filtrar por especie dentro de cada split
            train_idx = train_idx[species_mask[train_idx]] if len(train_idx) > 0 else train_idx
            val_idx   = val_idx[species_mask[val_idx]]     if len(val_idx)   > 0 else val_idx
            test_idx  = test_idx[species_mask[test_idx]]   if len(test_idx)  > 0 else test_idx

            # Limpiar NaNs AMR
            m_tr = ~np.isnan(y_col[train_idx]) if len(train_idx) > 0 else np.array([], dtype=bool)
            m_vl = ~np.isnan(y_col[val_idx])   if len(val_idx)   > 0 else np.array([], dtype=bool)
            m_te = ~np.isnan(y_col[test_idx])   if len(test_idx)  > 0 else np.array([], dtype=bool)

            X_tr, y_tr = X_original[train_idx][m_tr], y_col[train_idx][m_tr].astype(int)
            X_vl, y_vl = X_original[val_idx][m_vl],   y_col[val_idx][m_vl].astype(int)
            X_te, y_te = X_original[test_idx][m_te],   y_col[test_idx][m_te].astype(int)

            if (len(y_te) >= MIN_TEST
                    and (y_te == 1).sum() >= MIN_PER_CLASS_TEST
                    and (y_te == 0).sum() >= MIN_PER_CLASS_TEST):
                cell_data[(center, year)] = {
                    "X_tr": X_tr, "y_tr": y_tr,
                    "X_vl": X_vl, "y_vl": y_vl,
                    "X_te": X_te, "y_te": y_te,
                    "is_ood": is_ood,
                }
            elif len(y_te) > 0:
                skipped_test += 1

        # Print detalles de la celda de train
        for (train_center, train_year), cd in cell_data.items():
            if cd["is_ood"]:
                continue
            print(f"\n  [{train_center} | {train_year}] {species_name} / {atb_name}")
            print(f"    TRAIN: {len(cd['y_tr'])} | S={( cd['y_tr']==0).sum()} | R={(cd['y_tr']==1).sum()}")
            print(f"    VAL:   {len(cd['y_vl'])} | S={(cd['y_vl']==0).sum()} | R={(cd['y_vl']==1).sum()}")
            print(f"    TEST:  {len(cd['y_te'])} | S={(cd['y_te']==0).sum()} | R={(cd['y_te']==1).sum()} → 5 folds (~{len(cd['y_te'])//5} muestras/fold)")

        # Entrenar sobre celdas válidas y evaluar sobre todas
        for (train_center, train_year), cd in cell_data.items():
            if cd["is_ood"]:
                continue

            X_tr_mlp, y_tr_mlp = cd["X_tr"], cd["y_tr"]

            if (len(X_tr_mlp) < MIN_TRAIN
                    or (y_tr_mlp == 1).sum() < MIN_PER_CLASS_TRAIN
                    or (y_tr_mlp == 0).sum() < MIN_PER_CLASS_TRAIN):
                continue

            # Fallback val
            if len(cd["y_vl"]) >= 2 and (cd["y_vl"] == 1).sum() >= 1 and (cd["y_vl"] == 0).sum() >= 1:
                X_vl_mlp, y_vl_mlp = cd["X_vl"], cd["y_vl"]
            else:
                n_val    = max(int(len(y_tr_mlp) * 0.1), 1)
                X_vl_mlp = X_tr_mlp[-n_val:];  y_vl_mlp = y_tr_mlp[-n_val:]
                X_tr_mlp = X_tr_mlp[:-n_val];  y_tr_mlp = y_tr_mlp[:-n_val]

            pw        = min((y_tr_mlp == 0).sum() / max((y_tr_mlp == 1).sum(), 1), 20.0)
            model     = AMR_MLP(X_tr_mlp.shape[1]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
            pw_tensor = torch.tensor([pw], device=device)

            X_tr_t = torch.tensor(X_tr_mlp, dtype=torch.float32, device=device)
            y_tr_t = torch.tensor(y_tr_mlp, dtype=torch.float32, device=device)
            X_vl_t = torch.tensor(X_vl_mlp, dtype=torch.float32, device=device)

            best_auc, best_state, wait = -1.0, None, 0

            for epoch in range(MLP_EPOCHS):
                model.train()
                optimizer.zero_grad()
                loss = nn.functional.binary_cross_entropy_with_logits(model(X_tr_t), y_tr_t, pos_weight=pw_tensor)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(X_vl_t)).cpu().numpy()

                val_auc = roc_auc_score(y_vl_mlp, val_probs) if len(np.unique(y_vl_mlp)) >= 2 else 0.5
                if val_auc > best_auc:
                    best_auc   = val_auc
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                if wait >= MLP_PATIENCE:
                    break

            model.load_state_dict(best_state)
            model.eval()

            K = 5
            skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)

            for (test_center, test_year), td in cell_data.items():
                X_te_all = td["X_te"]
                y_te_all = td["y_te"]

                fold_aucs = []
                fold_prs  = []

                for fold_idx, (_, te_idx) in enumerate(skf.split(X_te_all, y_te_all)):
                    X_fold = X_te_all[te_idx]
                    y_fold = y_te_all[te_idx]

                    if len(np.unique(y_fold)) < 2:
                        fold_aucs.append(np.nan)
                        fold_prs.append(np.nan)
                        continue

                    with torch.no_grad():
                        X_te_t = torch.tensor(X_fold, dtype=torch.float32, device=device)
                        y_prob  = torch.sigmoid(model(X_te_t)).cpu().numpy()

                    fold_aucs.append(roc_auc_score(y_fold, y_prob))
                    fold_prs.append(average_precision_score(y_fold, y_prob))

                results.append({
                    "species":      species_name,
                    "antibiotic":   atb_name,
                    "train_center": train_center,
                    "train_year":   train_year,
                    "test_center":  test_center,
                    "test_year":    test_year,
                    "same_center":  train_center == test_center,
                    "same_year":    train_year == test_year,
                    "year_delta":   test_year - train_year,
                    "n_test":       len(y_te_all),
                    "n_test_R":     (y_te_all == 1).sum(),
                    "n_test_S":     (y_te_all == 0).sum(),
                    "auc_fold_1":   fold_aucs[0],
                    "auc_fold_2":   fold_aucs[1],
                    "auc_fold_3":   fold_aucs[2],
                    "auc_fold_4":   fold_aucs[3],
                    "auc_fold_5":   fold_aucs[4],
                    "pr_fold_1":    fold_prs[0],
                    "pr_fold_2":    fold_prs[1],
                    "pr_fold_3":    fold_prs[2],
                    "pr_fold_4":    fold_prs[3],
                    "pr_fold_5":    fold_prs[4],
                })

                valid_aucs = [a for a in fold_aucs if not np.isnan(a)]
                mean_auc   = np.mean(valid_aucs) if valid_aucs else np.nan
                print(f"    ({train_center}, {train_year}) -> ({test_center}, {test_year}): "
                    f"AUC={mean_auc:.3f} ({sum(np.isnan(fold_aucs))}/5 NaN folds)")


############################################################
# SAVE
############################################################
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "cross_center_temporal_results.csv", index=False)
print(f"\nResults saved: {experiment_dir / 'cross_center_temporal_results.csv'}")
print(f"Total evaluations : {len(results)}")
print(f"Skipped test cells: {skipped_test}")
