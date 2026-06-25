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
space = "original/multihead"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory created at:\n{experiment_dir}")
print("-" * 60)


############################################################
# IMPORTS
############################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.training.data_pipeline import load_pkl
from src.data.preprocessing import row_minmax_normalize


############################################################
# GLOBAL VARIABLES
############################################################
MIN_TRAIN, MIN_TEST = 10, 10
MIN_PER_CLASS_TRAIN, MIN_PER_CLASS_TEST = 3, 3

MLP_EPOCHS, MLP_LR, MLP_PATIENCE, BATCH_SIZE = 50, 1e-3, 10, 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 

############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA AND SPLITS =====")

dataset = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl")
X_original = dataset["data"]
amr = dataset["amr"]
ab_list = dataset["antibiotics"]
labels = dataset["label"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

print(f"Raw data shape: {X_original.shape}")


############################################################
# FILTER ANTIBIOTICS BY NAME (NUEVO BLOQUE)
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

    X_original = X_original[keep_mask]
    amr = amr[keep_mask]
    labels = labels[keep_mask]
    meta = meta.loc[keep_mask].reset_index(drop=True)

    # Build old→new index mapping for remapping splits
    old_to_new = np.full(len(keep_mask), -1, dtype=int)
    old_to_new[np.where(keep_mask)[0]] = np.arange(keep_mask.sum())

    print(f"Removed {n_chrom} MS-UMG chrom-agar samples. Remaining: {len(X_original)}")
else:
    old_to_new = None
    print("No 'agar' column — no chrom filtering applied")


############################################################
# NORMALIZE
############################################################
X_original = row_minmax_normalize(X_original)


############################################################
# LOAD AND REMAP SPLITS
############################################################
with open(train_exp_dir / "data_splits.pkl", "rb") as f:
    saved_splits = pickle.load(f)

domain_splits = saved_splits.get("splits_per_domain", {})

# Remap split indices to account for removed rows
if old_to_new is not None:
    for center, indices in domain_splits.items():
        for key in ["train_idx", "val_idx", "test_idx"]:
            if key in indices:
                old_idx = indices[key]
                new_idx = old_to_new[old_idx]
                indices[key] = new_idx[new_idx >= 0]

print(f"Data shape: {X_original.shape}")
all_centers = sorted(meta["hospital"].unique())
print(f"Centers: {all_centers}")
print(f"Centers in splits: {sorted(domain_splits.keys())}")
print(f"Centers NOT in splits (OOD): {[c for c in all_centers if c not in domain_splits]}")


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
# MLP ARCHITECTURE & TRAINING
############################################################
class MultiOutputMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),      
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )
    def forward(self, x): 
        return self.net(x)

def masked_bce_loss(logits, targets, pos_weights):
    mask = ~torch.isnan(targets)
    weight_mask = torch.ones_like(targets)

    for i in range(targets.shape[1]): 
        weight_mask[:, i] = pos_weights[i]

    loss_func = nn.BCEWithLogitsLoss(reduction='none')
    raw_loss = loss_func(logits, targets.nan_to_num(0))
    raw_loss = torch.where(targets == 1, raw_loss * weight_mask, raw_loss)
    return (raw_loss * mask.float()).sum() / (mask.sum() + 1e-8)

def train_multi_mlp(X_tr, Y_tr, X_val, Y_val, input_dim):
    output_dim = Y_tr.shape[1]
    model = MultiOutputMLP(input_dim, output_dim).to(device)
    
    # Calcular pesos para clases desbalanceadas
    pos_weights = torch.tensor([(np.nansum(Y_tr[:,i]==0)/max(np.nansum(Y_tr[:,i]==1),1)) 
                               for i in range(output_dim)], dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr_t, Y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    
    best_loss, patience, best_state = float("inf"), 0, None
    
    for epoch in range(MLP_EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = masked_bce_loss(model(xb), yb, pos_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(loader)
        
        model.eval()
        with torch.no_grad():
            v_loss = masked_bce_loss(model(X_val_t), Y_val_t, pos_weights).item()
        
        # PRINTS DE PROGRESO
        if epoch % 10 == 0 or epoch == MLP_EPOCHS - 1:
            print(f"      Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss < best_loss:
            best_loss, patience, best_state = v_loss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        
        if patience >= MLP_PATIENCE:
            print(f"      Early stopping at epoch {epoch}")
            break
            
    model.load_state_dict(best_state)
    return model


############################################################
# CROSS-CENTER EVALUATION
############################################################
centers = sorted(meta["hospital"].unique())
species_list = sorted(np.unique(labels))
results = []
skipped_test = 0
skipped_train = 0

for species_name in species_list:
    print(f"\n{'#'*60}\n### SPECIES: {species_name}\n{'#'*60}")
    species_mask = (labels == species_name)
    species_idx_global = np.where(species_mask)[0]

    # --- Pre-entrenar MLP para esta especie (Multi-output) ---
    # Para mantener el MLP Multi-output, necesitamos entrenarlo una vez por centro de origen
    mlp_models_per_center = {}
    
    source_centers = [c for c in all_centers if c in domain_splits]
    for tr_c in source_centers:
        idx_c = domain_splits[tr_c]
        tr_idx = np.intersect1d(idx_c['train_idx'], species_idx_global)
        va_idx = np.intersect1d(idx_c['val_idx'], species_idx_global)
        
        if len(tr_idx) >= MIN_TRAIN:
            print(f"  Training MLP for species {species_name} on {tr_c}...")
            mlp_models_per_center[tr_c] = train_multi_mlp(
                X_original[tr_idx], amr[tr_idx], 
                X_original[va_idx], amr[va_idx], X_original.shape[1]
            )

    # --- Bucle de antibióticos (Tu loop tal cual) ---
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
            X_tr_clean, y_tr_clean = X_original[train_idx][~np.isnan(y_col[train_idx])], y_col[train_idx][~np.isnan(y_col[train_idx])].astype(int)
            X_te_clean, y_te_clean = X_original[test_idx][~np.isnan(y_col[test_idx])], y_col[test_idx][~np.isnan(y_col[test_idx])].astype(int)

            if len(y_te_clean) >= MIN_TEST and (y_te_clean==1).sum() >= MIN_PER_CLASS_TEST and (y_te_clean==0).sum() >= MIN_PER_CLASS_TEST:
                center_data[center] = {"train_data": X_tr_clean, "train_labels": y_tr_clean, "test_data": X_te_clean, "test_labels": y_te_clean, "is_ood": is_ood, "full_test_idx": test_idx}
            elif len(y_te_clean) > 0: skipped_test += 1

        valid_train_centers = [c for c, d in center_data.items() if not d["is_ood"] and len(d["train_labels"]) >= MIN_TRAIN and (d["train_labels"]==1).sum() >= MIN_PER_CLASS_TRAIN]

        for train_center in valid_train_centers:
            d_train_obj = center_data[train_center] 

            n_pos = (d_train_obj["train_labels"] == 1).sum()
            n_neg = (d_train_obj["train_labels"] == 0).sum()
            scale_weight = n_neg / max(n_pos, 1)

            lgbm = LGBMClassifier(
                n_estimators=300, 
                scale_pos_weight=scale_weight, 
                random_state=42, 
                verbosity=-1,
                n_jobs=-1
            )
            lgbm.fit(d_train_obj["train_data"], d_train_obj["train_labels"])

            for test_center, d_test in center_data.items():
                y_prob_lgbm = lgbm.predict_proba(d_test["test_data"])[:, 1]
                auc_lgbm = roc_auc_score(d_test["test_labels"], y_prob_lgbm)
                pr_lgbm = average_precision_score(d_test["test_labels"], y_prob_lgbm)

                results.append({"model": "LightGBM", "species": species_name, "antibiotic": atb_name, "train_center": train_center, "test_center": test_center, "is_ood": d_test["is_ood"], "roc_auc": auc_lgbm, "pr_auc": pr_lgbm})

                # --- EVALUACIÓN MLP (Si existe para este centro) ---
                if train_center in mlp_models_per_center:
                    model_mlp = mlp_models_per_center[train_center]
                    model_mlp.eval()
                    with torch.no_grad():
                        # Usamos los mismos datos de test que el LGBM
                        X_test_t = torch.tensor(d_test["test_data"], dtype=torch.float32).to(device)
                        y_prob_mlp = torch.sigmoid(model_mlp(X_test_t)).cpu().numpy()
                        # Si el MLP es multi-output, extraemos solo la columna de este atb
                        y_prob_mlp_atb = y_prob_mlp[:, atb_idx]
                        
                        auc_mlp = roc_auc_score(d_test["test_labels"], y_prob_mlp_atb)
                        pr_mlp = average_precision_score(d_test["test_labels"], y_prob_mlp_atb)
                        results.append({"model": "MultiMLP", "species": species_name, "antibiotic": atb_name, "train_center": train_center, "test_center": test_center, "is_ood": d_test["is_ood"], "roc_auc": auc_mlp, "pr_auc": pr_mlp})

                print(f"      {train_center} -> {test_center}: LGBM_AUC={auc_lgbm:.3f} | MLP_AUC={auc_mlp:.3f}")

# SAVE
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "results_original_comparison.csv", index=False)
print(f"\n{'='*60}")
print(f"Results saved: {experiment_dir / 'cross_center_results_original.csv'}")
print(f"Total evaluations: {len(results)}")
print(f"Skipped test centers (too few samples): {skipped_test}")
print(f"Skipped train centers (too few samples): {skipped_train}")


############################################################
# HEATMAPS COMPARATIVOS (ROC-AUC y PR-AUC)
############################################################
print("\n===== GENERATING COMPARATIVE HEATMAPS =====")

# Definimos los centros para los ejes
source_centers = [c for c in all_centers if c in domain_splits]
all_centers_ordered = source_centers + [c for c in all_centers if c not in domain_splits]

metrics = ["roc_auc", "pr_auc"]
n_heatmaps = 0

for metric in metrics:
    # Crear subcarpeta para la métrica
    metric_dir = experiment_dir / metric
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    # Agrupamos por especie y antibiótico
    for (species, atb), df_group in df.groupby(['species', 'antibiotic']):
        
        # Crear figura con 2 subplots (LGBM vs MLP)
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        models_plot = ["LightGBM", "MultiMLP"]
        
        # Configuración visual según métrica
        v_min = 0.4 if metric == "roc_auc" else 0.0
        cmap = "viridis" if metric == "roc_auc" else "magma"
        
        for i, model_name in enumerate(models_plot):
            model_data = df_group[df_group['model'] == model_name]
            
            if model_data.empty:
                axes[i].text(0.5, 0.5, f"No data for {model_name}", ha='center')
                axes[i].set_title(f"{model_name} (No data)")
                continue
                
            # Pivotar matriz: Filas=Train, Columnas=Test
            matrix = model_data.pivot(index="train_center", columns="test_center", values=metric)
            # Reindexar para asegurar que los OOD salgan a la derecha y solo fuentes en las filas
            matrix = matrix.reindex(index=source_centers, columns=all_centers_ordered)

            sns.heatmap(
                matrix, 
                annot=True, 
                fmt=".3f", 
                cmap=cmap, 
                vmin=v_min, 
                vmax=1.0, 
                ax=axes[i],
                cbar=(i == 1)
            )
            
            axes[i].set_facecolor("#333333") # Gris oscuro para celdas sin datos
            axes[i].set_title(f"Model: {model_name}")
            axes[i].set_xlabel("Test Center (Target)")
            axes[i].set_ylabel("Train Center (Source)" if i == 0 else "")

        plt.suptitle(f"{metric.upper()} ORIGINAL SPACE: {atb} in {species}", fontsize=16)
        
        # Limpieza de nombres para archivo
        clean_sp = species.replace(" ", "_")
        clean_atb = atb.replace("/", "_").replace(" ", "_")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(metric_dir / f"comp_{metric}_{clean_sp}_{clean_atb}.png", dpi=300)
        plt.close()
        n_heatmaps += 1

print(f"\nDONE. Results and plots saved in: {experiment_dir}")
