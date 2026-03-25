############################################################
# PATH CONFIGURATION
############################################################
import sys
import pickle
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Create experiment directory 
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "latent/clf_head"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

experiment_dir = output_dir / space / timestamp
experiment_dir.mkdir(parents=True, exist_ok=True)

print(f"\nExperiment directory created at:\n{experiment_dir}")
print("-"*60)


############################################################
# IMPORTS
############################################################
import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, confusion_matrix

from src.training.data_pipeline import load_pkl
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended



############################################################
# LOAD DATASETS
############################################################
print("\n" + "="*60)
print("===== LOADING DATA =====")
print("="*60)

ecef = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/pickles/E-CEF.pkl")
kcef = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/pickles/K-CEF.pkl")
soxa = load_pkl("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/pickles/S-OXA.pkl")

ecef_data, ecef_label, ecef_meta, ecef_amr = ecef["data"], ecef["label"], ecef["meta"], ecef["amr"]
kcef_data, kcef_label, kcef_meta, kcef_amr = kcef["data"], kcef["label"], kcef["meta"], kcef["amr"]
soxa_data, soxa_label, soxa_meta, soxa_amr = soxa["data"], soxa["label"], soxa["meta"], soxa["amr"]

print("Datasets loaded successfully.")


############################################################
# LOAD PRETRAINED VAEs AND SPLITS
############################################################
print("\n===== LOADING PRETRAINED MODELS AND SPLITS =====")

PRETRAINED_MODEL_ECEF = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_221257/model.pth")
PRETRAINED_MODEL_KCEF = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_224119/model.pth")
PRETRAINED_MODEL_SOXA = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_224502/model.pth")

SPLITS_ECEF = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_221257/data_splits.pkl")
SPLITS_KCEF = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_224119/data_splits.pkl")
SPLITS_SOXA = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260325_224502/data_splits.pkl")

# Instantiate models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# E-CEF
vae_pretrained_ecef = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
    input_dim=ecef_data.shape[1], latent_dim=64, num_domains=4, n_species=1, n_antibiotics=1
)
vae_pretrained_ecef.load_state_dict(torch.load(PRETRAINED_MODEL_ECEF, map_location=device))
vae_pretrained_ecef.to(device).eval()

# K-CEF
vae_pretrained_kcef = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
    input_dim=kcef_data.shape[1], latent_dim=64, num_domains=4, n_species=1, n_antibiotics=1
)
vae_pretrained_kcef.load_state_dict(torch.load(PRETRAINED_MODEL_KCEF, map_location=device))
vae_pretrained_kcef.to(device).eval()
    
# S-OXA
vae_pretrained_soxa = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
    input_dim=soxa_data.shape[1], latent_dim=64, num_domains=3, n_species=1, n_antibiotics=1
)
vae_pretrained_soxa.load_state_dict(torch.load(PRETRAINED_MODEL_SOXA, map_location=device))
vae_pretrained_soxa.to(device).eval()

print("\n===== MODELS LOADED SUCCESSFULLY =====")


############################################################
# DEFINE SCENARIOS
############################################################
scenarios = {
    "E-CEF": {
        "data": encode_latent(vae_pretrained_ecef, ecef_data, device), "label": ecef_label, "meta": ecef_meta, "amr": ecef_amr, 
        "model": vae_pretrained_ecef, "splits_path": SPLITS_ECEF
    },
    "K-CEF": {
        "data": encode_latent(vae_pretrained_kcef, kcef_data, device), "label": kcef_label, "meta": kcef_meta, "amr": kcef_amr,
        "model": vae_pretrained_kcef, "splits_path": SPLITS_KCEF
    },
    "S-OXA": {
        "data": encode_latent(vae_pretrained_soxa, soxa_data, device), "label": soxa_label, "meta": soxa_meta, "amr": soxa_amr,
        "model": vae_pretrained_soxa, "splits_path": SPLITS_SOXA
    },
}

results = []

param_grid = {
    "n_estimators": np.arange(300, 901, 200),   
    "num_leaves": np.arange(31, 98, 22),        
    "max_depth": np.arange(6, 15, 3),           
}


############################################################
# RUN EXPERIMENTS
############################################################
for scenario_name, s_dict in scenarios.items():
    print("\n" + "="*60)
    print(f"===== SCENARIO: {scenario_name} =====")
    print("="*60)

    Z_all = s_dict["data"]
    amr = s_dict["amr"]
    meta = s_dict["meta"]

    centers = sorted(meta["hospital"].unique())
    print(f"Centers available: {centers}")

    with open(s_dict["splits_path"], "rb") as f:
            saved_splits = pickle.load(f)
        
    domain_splits = saved_splits.get("splits_per_domain", {})

    # Split data per hospital
    center_data = {}

    for center in centers:
        indices = domain_splits[center]
        train_idx = indices['train_idx']
        test_idx = indices['test_idx']

        y_train_raw = amr[train_idx]
        y_test_raw = amr[test_idx]

        Z_train_raw = Z_all[train_idx]
        Z_test_raw = Z_all[test_idx]

        m_tr = ~np.isnan(y_train_raw)
        m_te = ~np.isnan(y_test_raw)

        Z_train_clean, y_train_clean = Z_train_raw[m_tr], y_train_raw[m_tr]
        Z_test_clean, y_test_clean = Z_test_raw[m_te], y_test_raw[m_te]


        if len(np.unique(y_train_clean)) < 2:
            print(f"Skipping {center}")
            continue

        center_data[center] = {
            "train_data": Z_train_clean,
            "train_labels": y_train_clean,
            "test_data": Z_test_clean,
            "test_labels": y_test_clean,
        }

        tr_0, tr_1 = (y_train_clean == 0).sum(), (y_train_clean == 1).sum()
        print(f"{center.ljust(10)} | Train N: {len(y_train_clean)} (0:{tr_0}, 1:{tr_1})")

    # Cross-site evaluation
    valid_train_centers = sorted(center_data.keys())

    for train_center in centers:
        print("\n" + "-"*50)
        print(f"Training on {train_center}")
        print("-"*50)

        Z_train = center_data[train_center]["train_data"]
        y_train = center_data[train_center]["train_labels"]

        base_clf = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, n_jobs=1)
        grid_clf = GridSearchCV(base_clf, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=8)

        grid_clf.fit(Z_train, y_train)
        best_model = grid_clf.best_estimator_

        # Evaluate on centers
        for test_center in valid_train_centers:
            Z_test = center_data[test_center]["test_data"]
            y_test = center_data[test_center]["test_labels"]

            if len(Z_test) == 0 or len(np.unique(y_test)) < 2:
                continue

            y_pred = best_model.predict(Z_test)
            y_prob = best_model.predict_proba(Z_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            ba  = balanced_accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred)  
            
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            eval_type = "INTRA-SITE" if test_center == train_center else "CROSS-SITE"

            print(
                f"  [{eval_type}] "
                f"Test {test_center} | "
                f"AUC: {auc:.3f} | BA: {ba:.3f} | "
                f"Sens: {sens:.3f} | Spec: {spec:.3f}"
            )

            results.append({
                "scenario": scenario_name,
                "train_center": train_center,
                "test_center": test_center,
                "evaluation": "intra_site" if test_center == train_center else "cross_site",
                "roc_auc": auc,
                "balanced_acc": ba,
                "sensitivity": sens,
                "specificity": spec
            })

df_results = pd.DataFrame(results)
df_results.to_csv(experiment_dir / "results.csv", index=False)


############################################################
# GENERATE AND SAVE CROSS-SITE HEATMAPS
############################################################
print("\n" + "="*60)
print("===== GENERATING CROSS-SITE HEATMAPS =====")
print("="*60)

for scenario_name in df_results["scenario"].unique():
    print(f"Creating plot for {scenario_name}")

    df_s = df_results[df_results["scenario"] == scenario_name]

    # Obtain original centers to keep grid constant (shows empty boxes if missing)
    scenario_centers = sorted(scenarios[scenario_name]["meta"]["hospital"].unique())

    # Pivot table
    roc_matrix = df_s.pivot(
        index="train_center",
        columns="test_center",
        values="roc_auc"
    )

    # Reindex to force all hospitals to appear
    roc_matrix = roc_matrix.reindex(
        index=scenario_centers,
        columns=scenario_centers
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # ROC heatmap
    sns.heatmap(
        roc_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "ROC-AUC"},
        ax=ax
    )
    
    # Set dark background for missing values (NaN)
    ax.set_facecolor("#333333") 
    ax.set_title(f"{scenario_name} - ROC-AUC\n(Gray cells = Insufficient data)")
    ax.set_xlabel("Test Center")
    ax.set_ylabel("Train Center")

    plt.tight_layout()

    # Save image
    png_path = experiment_dir / f"{scenario_name}_heatmap_roc.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")


############################################################
# EXPERIMENT FINISHED
############################################################
print("\n" + "="*60)
print("===== EXPERIMENT FINISHED SUCCESSFULLY =====")
print("="*60)
print(f"Results saved in: {experiment_dir}")
print("="*60)
