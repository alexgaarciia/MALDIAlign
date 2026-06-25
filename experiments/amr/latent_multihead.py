############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

train_exp_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260406_164700")
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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.training.data_pipeline import load_pkl
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head


############################################################
# LOAD DATA & SPLITS
############################################################
print("\n===== LOADING DATA AND SPLITS =====")

dataset = load_pkl("/export/data_ml4ds/bacteria_id/MALDIAlign_Alex/driams_marisma_common.pkl")
data = dataset["data"]
amr = dataset["amr"]
ab_list = dataset["antibiotics"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

with open(train_exp_dir / "data_splits.pkl", "rb") as f:
    saved_splits = pickle.load(f)

print("Data shape:", data.shape)
all_centers = sorted(meta["hospital"].unique())
print("Centers identified:", all_centers)


############################################################
# LOAD MODEL
############################################################
print("\n===== LOADING PRETRAINED MODEL =====")

model_path = train_exp_dir / "model.pth"

vae = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head(
    input_dim=data.shape[1],
    latent_dim=128,
    num_domains=4,
    n_species=1,
    n_antibiotics=3,
    lambda_amr=25
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.load_state_dict(torch.load(model_path, map_location=device))
vae.to(device)
vae.eval()

print("Model loaded.")


############################################################
# ENCODE LATENT
############################################################
print("\n===== ENCODING LATENT =====")

Z_all = encode_latent(vae, data, device)
Z_all = np.asarray(Z_all)

print("Latent shape:", Z_all.shape)


############################################################
# FILTER EMPTY ANTIBIOTICS
############################################################
if amr.ndim > 1:
    valid_mask = ~np.all(np.isnan(amr), axis=0)

    if amr.shape[1] > 1:
        ab_list = [a for a, keep in zip(ab_list, valid_mask) if keep]

    amr = amr[:, valid_mask]

print("\nValid antibiotics:", ab_list)


############################################################
# CROSS-CENTER EVALUATION
############################################################
############################################################
# CROSS-CENTER EVALUATION (Including OOD Centers like DRIAMS_D)
############################################################
centers = sorted(meta["hospital"].unique())
results = []

domain_splits = saved_splits.get("splits_per_domain", {})

for atb_idx, atb_name in enumerate(ab_list):

    print("\n" + "="*60)
    print(f"ANTIBIOTIC: {atb_name}")
    print("="*60)

    center_data = {}
    for center in centers:
        # 1. Identificar si el centro es OOD (Out-of-Distribution)
        if center in domain_splits:
            indices = domain_splits[center]
            train_idx = indices['train_idx']
            test_idx = indices['test_idx']
            is_ood = False
        else:
            # Centro NO visto en VAE: Todo es Test
            train_idx = np.array([], dtype=int)
            test_idx = meta.index[meta["hospital"] == center].tolist()
            is_ood = True

        # 2. Extraer datos
        if amr.ndim == 1:
            y_tr = amr[train_idx] if len(train_idx) > 0 else np.array([])
            y_te = amr[test_idx]
        else:
            y_tr = amr[train_idx, atb_idx] if len(train_idx) > 0 else np.array([])
            y_te = amr[test_idx, atb_idx]

        X_tr = Z_all[train_idx] if len(train_idx) > 0 else np.array([]).reshape(0, Z_all.shape[1])
        X_te = Z_all[test_idx]

        # 3. Limpiar NaNs
        m_tr = ~np.isnan(y_tr) if len(y_tr) > 0 else np.array([], dtype=bool)
        m_te = ~np.isnan(y_te)

        X_train_clean, y_train_clean = X_tr[m_tr], y_tr[m_tr].astype(int)
        X_test_clean, y_test_clean = X_te[m_te], y_te[m_te].astype(int)

        # LOG de clases
        tr_0 = (y_train_clean == 0).sum() if len(y_train_clean) > 0 else 0
        tr_1 = (y_train_clean == 1).sum() if len(y_train_clean) > 0 else 0
        te_0 = (y_test_clean == 0).sum()
        te_1 = (y_test_clean == 1).sum()

        ood_label = "[OOD]" if is_ood else "[Source]"
        print(f"{center.ljust(10)} {ood_label} | Train: {len(y_train_clean):>5} (0:{tr_0:>5}, 1:{tr_1:>4}) | "
              f"Test: {len(y_test_clean):>5} (0:{te_0:>5}, 1:{te_1:>3})")

        if len(np.unique(y_test_clean)) >= 2:
            center_data[center] = {
                "train_data": X_train_clean,
                "train_labels": y_train_clean,
                "test_data": X_test_clean,
                "test_labels": y_test_clean,
                "is_ood": is_ood
            }
        else:
            print(f"  {center} | Saltado: Sin suficientes clases en TEST para AUC.")

    valid_train_centers = [
        c for c, d in center_data.items() 
        if not d["is_ood"] and len(np.unique(d["train_labels"])) >= 2
    ]

    for train_center in valid_train_centers:
        X_train_raw = center_data[train_center]["train_data"]
        y_train_lgbm = center_data[train_center]["train_labels"]

        # Convertimos a DataFrame para evitar el UserWarning de LightGBM
        X_train_lgbm = pd.DataFrame(X_train_raw)

        model = LGBMClassifier(
            objective="binary", 
            n_estimators=300, 
            random_state=42, 
            verbosity=-1
        )
        model.fit(X_train_lgbm, y_train_lgbm)

        # Evaluamos en TODOS los centros registrados (Sources y OOD)
        for test_center in center_data.keys():
            X_test_raw = center_data[test_center]["test_data"]
            y_test_lgbm = center_data[test_center]["test_labels"]

            # Convertimos a DataFrame para consistencia
            X_test_lgbm = pd.DataFrame(X_test_raw)

            y_prob = model.predict_proba(X_test_lgbm)[:, 1]
            auc = roc_auc_score(y_test_lgbm, y_prob)

            print(f"Train: {train_center}, Test: {test_center} | AUC: {auc:.3f}")

            results.append({
                "antibiotic": atb_name,
                "train_center": train_center,
                "test_center": test_center,
                "roc_auc": auc
            })

# SAVE RESULTS
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "cross_center_results.csv", index=False)


############################################################
# HEATMAPS CON HUECOS EN GRIS
############################################################
print("\n===== GENERATING HEATMAPS =====")

for atb in ab_list:
    df_s = df[df["antibiotic"] == atb]
    
    # Si no hay ningún resultado para este antibiótico, saltar
    if df_s.empty: continue

    # Crear la matriz pivotada
    roc_matrix = df_s.pivot(index="train_center", columns="test_center", values="roc_auc")

    # TRUCO: Reindexar con todos los hospitales para forzar NaNs en los huecos
    roc_matrix = roc_matrix.reindex(index=all_centers, columns=all_centers)

    plt.figure(figsize=(10, 8))
    
    # El facecolor "gray" se verá en las casillas donde roc_matrix tiene NaN
    ax = sns.heatmap(
        roc_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="viridis", 
        vmin=0.0, # ROC-AUC < 0.5 es peor que el azar
        vmax=1.0, 
        cbar_kws={'label': 'ROC-AUC'}
    )
    
    # Pintamos el fondo de gris para las celdas vacías
    ax.set_facecolor("#333333") 
    
    plt.title(f"Cross-Center Robustness: {atb}\n(Gray cells = Insufficient data / Not evaluated)")
    plt.xlabel("Test Center (Unseen VAE Split)")
    plt.ylabel("Train Center (VAE Latent)")
    
    safe_name = atb.replace("/", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(experiment_dir / f"heatmap_auc_{safe_name}.png", dpi=300)
    plt.close()

print(f"\nDONE. Results and heatmaps saved in: {experiment_dir}")
