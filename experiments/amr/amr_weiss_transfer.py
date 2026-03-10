############################################################
# PATH CONFIGURATION
############################################################
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Create experiment directory 
output_dir = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/amr/results")
space = "latent"
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

from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, recall_score, confusion_matrix

from src.training.data_pipeline import load_pkl
from src.evaluation.eval import encode_latent
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.deep.MultiVAEPriorAMR import MultiVAE_Bernoulli_SpeciesPrior_AMR_Extended



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
# LOAD PRETRAINED BASELINE MODEL
############################################################
print("\n===== LOADING PRETRAINED MODEL =====")

PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_amr/20260308_202522/model.pth")

vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_AMR_Extended(
    input_dim=ecef_data.shape[1],
    latent_dim=64,
    num_domains=4, # important to change when S-OXA
    n_species=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_pretrained.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
vae_pretrained.to(device)
vae_pretrained.eval()

print("\n===== MULTIDECODER LOADED SUCCESSFULLY =====")


############################################################
# DEFINE SCENARIOS
############################################################
scenarios = {
    "E-CEF": {"data": ecef_data, "label": ecef_label, "meta": ecef_meta, "amr": ecef_amr},
    #"K-CEF": {"data": kcef_data, "label": kcef_label, "meta": kcef_meta, "amr": kcef_amr},
    #"S-OXA": {"data": soxa_data, "label": soxa_label, "meta": soxa_meta, "amr": soxa_amr},
}

# Container for global results
results = []


############################################################
# RUN EXPERIMENTS
############################################################
for scenario_name, scenario_dict in scenarios.items():
    print("\n" + "="*60)
    print(f"===== SCENARIO: {scenario_name} =====")
    print("="*60)

    data = scenario_dict["data"]
    label = scenario_dict["label"]
    meta = scenario_dict["meta"]
    amr  = scenario_dict["amr"]

    centers = sorted(meta["hospital"].unique())
    print(f"Centers available: {centers}")

    center_data = {}

    # ------------------------------------------------------
    # Split data per hospital
    # ------------------------------------------------------
    print("\nSplitting data per hospital")

    for center in centers:
        mask = meta["hospital"] == center
        
        X_raw = data[mask]
        y = amr[mask]

        # There is *one* sample in S-OXA with an emtpy value
        valid_mask = ~np.isnan(y)
        X_raw = X_raw[valid_mask]
        y = y[valid_mask]

        # encode una sola vez
        Z = encode_latent(vae_pretrained, X_raw, device)

        Z_train, Z_test, y_train, y_test = train_test_split(
            Z, y, stratify=y, test_size=0.2, random_state=42
        )

        center_data[center] = {
            "train_data": Z_train,
            "train_labels": y_train,
            "test_data": Z_test,
            "test_labels": y_test,
        }

    # ------------------------------------------------------
    # Hyperparameter grid
    # ------------------------------------------------------
    param_grid = {
        "n_estimators": np.arange(300, 901, 200),   
        "num_leaves": np.arange(31, 98, 22),        
        "max_depth": np.arange(6, 15, 3),           
    }

    # ------------------------------------------------------
    # Cross-site evaluation
    # ------------------------------------------------------
    for train_center in centers:
        print("\n" + "-"*50)
        print(f"Training on {train_center}")
        print("-"*50)

        X_train = center_data[train_center]["train_data"]
        y_train = center_data[train_center]["train_labels"]

        base_clf = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, n_jobs=1)

        grid_clf = GridSearchCV(base_clf, param_grid=param_grid, cv=5, scoring="roc_auc", n_jobs=8)
        grid_clf.fit(X_train, y_train)
    
        print(f"Best CV ROC-AUC: {grid_clf.best_score_:.3f}")
        print(f"Best params: {grid_clf.best_params_}")

        best_model = grid_clf.best_estimator_

        # ---------------------------------------------
        # Evaluate on centers
        # ---------------------------------------------
        for test_center in centers:
            X_test = center_data[test_center]["test_data"]
            y_test = center_data[test_center]["test_labels"]

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
            pr = average_precision_score(y_test, y_prob)
    
            ba  = balanced_accuracy_score(y_test, y_pred)
            sens = recall_score(y_test, y_pred)  

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp)

            eval_type = "INTRA-SITE" if test_center == train_center else "CROSS-SITE"

            print(
                f"[{eval_type}] "
                f"Train {train_center} → Test {test_center} | "
                f"ROC-AUC: {auc:.3f} | "
                f"PR-AUC: {pr:.3f} | "
                f"BA: {ba:.3f} | "
                f"Sens: {sens:.3f} | "
                f"Spec: {spec:.3f}"
            )

            results.append({
                "scenario": scenario_name,
                "train_center": train_center,
                "test_center": test_center,
                "evaluation": "intra_site" if test_center == train_center else "cross_site",
                "roc_auc": auc,
                "pr_auc": pr,
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

    print(f"Creating plots for {scenario_name}")

    df_s = df_results[df_results["scenario"] == scenario_name]

    # Pivot tables
    roc_matrix = df_s.pivot(
        index="train_center",
        columns="test_center",
        values="roc_auc"
    )

    pr_matrix = df_s.pivot(
        index="train_center",
        columns="test_center",
        values="pr_auc"
    )

    # Order centers
    centers_sorted = sorted(df_s["train_center"].unique())
    roc_matrix = roc_matrix.reindex(
        index=centers_sorted,
        columns=centers_sorted
    )

    pr_matrix = pr_matrix.reindex(
        index=centers_sorted,
        columns=centers_sorted
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC heatmap
    sns.heatmap(
        roc_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.4,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "ROC-AUC"},
        ax=axes[0]
    )
    axes[0].set_title(f"{scenario_name} - ROC-AUC")
    axes[0].set_xlabel("Test Center")
    axes[0].set_ylabel("Train Center")

    # PR heatmap
    sns.heatmap(
        pr_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0.4,
        vmax=1.0,
        linewidths=0.5,
        cbar_kws={"label": "PR-AUC"},
        ax=axes[1]
    )
    axes[1].set_title(f"{scenario_name} - PR-AUC")
    axes[1].set_xlabel("Test Center")
    axes[1].set_ylabel("")

    plt.tight_layout()

    # Save image
    png_path = experiment_dir / f"{scenario_name}_heatmap.png"
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
