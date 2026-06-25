############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Usamos la carpeta del experimento del VAE para sacar los mismos splits
train_exp_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260329_210528")
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
import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.training.data_pipeline import load_pkl


############################################################
# LOAD DATA & SPLITS
############################################################
print("\n===== LOADING DATA AND SPLITS =====")

dataset = load_pkl("/export/usuarios01/agnavarr/MALDIGen-dev/pickles/DRIAMS_study_amr_klebsiella2.pkl")
X_original = dataset["data"]  
amr = dataset["amr"]
ab_list = dataset["antibiotics"]
raw_meta = dataset["meta"]
meta = pd.DataFrame(list(raw_meta)) if isinstance(raw_meta, (list, np.ndarray)) else pd.DataFrame(raw_meta)

with open(train_exp_dir / "data_splits.pkl", "rb") as f:
    saved_splits = pickle.load(f)

print(f"Original Data shape: {X_original.shape}")
all_centers = sorted(meta["hospital"].unique())
print("Centers identified:", all_centers)


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
# CROSS-CENTER EVALUATION (ORIGINAL SPACE)
############################################################
results = []
domain_splits = saved_splits.get("splits_per_domain", {})

for atb_idx, atb_name in enumerate(ab_list):

    print("\n" + "="*60)
    print(f"ANTIBIOTIC: {atb_name} (ORIGINAL SPACE)")
    print("="*60)

    center_data = {}
    for center in all_centers:
        if center not in domain_splits:
            continue

        indices = domain_splits[center]
        train_idx = indices['train_idx']
        test_idx = indices['test_idx']

        X_tr = X_original[train_idx]
        X_te = X_original[test_idx]

        if amr.ndim == 1:
            y_tr = amr[train_idx]
            y_te = amr[test_idx]
        else:
            y_tr = amr[train_idx, atb_idx]
            y_te = amr[test_idx, atb_idx]

        m_tr, m_te = ~np.isnan(y_tr), ~np.isnan(y_te)
        X_train_clean, y_train_clean = X_tr[m_tr], y_tr[m_tr].astype(int)
        X_test_clean, y_test_clean = X_te[m_te], y_te[m_te].astype(int)

        # Cálculo de conteos por clase
        tr_0, tr_1 = (y_train_clean == 0).sum(), (y_train_clean == 1).sum()
        te_0, te_1 = (y_test_clean == 0).sum(), (y_test_clean == 1).sum()

        print(f"{center.ljust(10)} | Train: {len(y_train_clean):>5} (0:{tr_0:>5}, 1:{tr_1:>4}) | "
              f"Test: {len(y_test_clean):>5} (0:{te_0:>5}, 1:{te_1:>3})")

        # Verificamos diversidad en Train para poder entrenar el LGBM
        if len(np.unique(y_train_clean)) < 2:
            print(f"{center} saltado: Falta diversidad en Train.")
            continue
        
        center_data[center] = {
            "train_data": X_train_clean, 
            "train_labels": y_train_clean,
            "test_data": X_test_clean, 
            "test_labels": y_test_clean
        }

    valid_train_centers = sorted(center_data.keys())

    for train_center in valid_train_centers:
        print(f"\nEntrenando LGBM (Original) en {train_center}...")

        X_train_lgbm = center_data[train_center]["train_data"]
        y_train_lgbm = center_data[train_center]["train_labels"]

        model = LGBMClassifier(
            objective="binary", 
            n_estimators=300, 
            random_state=42, 
            verbosity=-1,
            n_jobs=-1 
        )
        model.fit(X_train_lgbm, y_train_lgbm)

        for test_center in all_centers:
            # Solo evaluamos si tenemos el test_split guardado del centro
            if test_center not in center_data:
                continue
            
            X_test_lgbm = center_data[test_center]["test_data"]
            y_test_lgbm = center_data[test_center]["test_labels"]

            # Solo evaluamos si el test tiene ambas clases para el AUC
            if len(np.unique(y_test_lgbm)) < 2:
                continue

            y_prob = model.predict_proba(X_test_lgbm)[:, 1]
            auc = roc_auc_score(y_test_lgbm, y_prob)

            print(f"Test in {test_center} | AUC: {auc:.3f}")

            results.append({
                "antibiotic": atb_name,
                "train_center": train_center,
                "test_center": test_center,
                "roc_auc": auc
            })


# GUARDAR RESULTADOS CSV
df = pd.DataFrame(results)
df.to_csv(experiment_dir / "cross_center_results_original.csv", index=False)


############################################################
# HEATMAPS CON HUECOS EN GRIS
############################################################
print("\n===== GENERATING HEATMAPS =====")

for atb in ab_list:
    df_s = df[df["antibiotic"] == atb]
    
    if df_s.empty: 
        continue

    # Matriz pivotada
    roc_matrix = df_s.pivot(index="train_center", columns="test_center", values="roc_auc")

    # Reindexar con TODOS los hospitales para que los huecos aparezcan como NaNs
    roc_matrix = roc_matrix.reindex(index=all_centers, columns=all_centers)

    plt.figure(figsize=(10, 8))
    
    # Configuramos el fondo gris oscuro (#333333) para los huecos
    ax = sns.heatmap(
        roc_matrix, 
        annot=True, 
        fmt=".3f", 
        cmap="viridis", 
        vmin=0.0, 
        vmax=1.0, 
        cbar_kws={'label': 'ROC-AUC'}
    )
    
    ax.set_facecolor("#333333") 
    
    plt.title(f"ORIGINAL SPACE Robustness: {atb}\n(Gray cells = Insufficient data / Not evaluated)")
    plt.xlabel("Test Center (Unseen Split)")
    plt.ylabel("Train Center")
    
    safe_name = atb.replace("/", "_").replace(" ", "_")
    plt.tight_layout()
    plt.savefig(experiment_dir / f"heatmap_original_auc_{safe_name}.png", dpi=300)
    plt.close()

print(f"\nDONE. Original space results saved in: {experiment_dir}")
