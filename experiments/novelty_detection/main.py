############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys

PROJECT_NAME = "MALDIAlign"
cwd = Path().resolve()
target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break
if target is not None and target != cwd:
    os.chdir(target)
    sys.path.append(str(target))


############################################################
# IMPORTS
############################################################
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import encode_latent, make_loader, predict, predict_proba, compute_all_metrics
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.baselines.mlp_latent import LinearProbe_Extended


############################################################
# CONFIG
############################################################
cfg = load_config()

TARGET_SPECIES = ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus", "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex",]

EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")
MODEL_PATH  = EXPERIMENT_DIR / "model.pth"
SPLITS_PATH = EXPERIMENT_DIR / "data_splits.pkl"

SWEEP_PERCENTILES = [0.01, 0.05, 0.1, 0.5]   # percent scale, of the source-train log-likelihood

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/novelty_detection/combined") / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA", "RKI": "RKI"}

############################################################
# LOAD MODEL
############################################################
print("\n===== LOADING MODEL =====")
state = torch.load(MODEL_PATH, map_location="cpu")
decoder_indices = {int(k.split(".")[2]) for k in state if k.startswith("decoder.net.")}
num_domains = max(decoder_indices) + 1

model = MultiVAE_Bernoulli_SpeciesPrior_Extended(input_dim=6000, latent_dim=64, num_domains=num_domains, n_species=len(TARGET_SPECIES))
model.load_state_dict(state)
model.to(device).eval()
print(f"Model loaded — num_domains={num_domains}")


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA =====")

driams_dict  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict     = load_rki(cfg["data"]["RKI_FULL"])

mask_driams  = np.isin(driams_dict["label"],  TARGET_SPECIES)
mask_marisma = np.isin(marisma_dict["label"], TARGET_SPECIES)
mask_rki     = np.isin(rki_dict["label"],     TARGET_SPECIES)

data_driams   = driams_dict["data"][mask_driams]
label_driams  = driams_dict["label"][mask_driams]
meta_driams   = driams_dict["meta"][mask_driams].reset_index(drop=True)
meta_driams["hospital"] = meta_driams["hospital"].str.replace("-", "_", regex=False)

data_marisma  = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]

data_rki      = rki_dict["data"][mask_rki]
label_rki     = rki_dict["label"][mask_rki]

maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA  = row_minmax_normalize(data_driams[maskA])
labelA = label_driams[maskA]
dataB  = row_minmax_normalize(data_driams[maskB])
labelB = label_driams[maskB]
dataC  = row_minmax_normalize(data_driams[maskC])
labelC = label_driams[maskC]
dataD  = row_minmax_normalize(data_driams[maskD])
labelD = label_driams[maskD]

data_marisma = row_minmax_normalize(data_marisma)
data_rki     = row_minmax_normalize(data_rki)

data_final  = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])
print(f"data_final: {data_final.shape}")

msumg_dict   = load_msumg(cfg["data"]["MSUMG_FULL"])
mask_msumg   = np.isin(msumg_dict["label"], TARGET_SPECIES)
data_msumg   = row_minmax_normalize(msumg_dict["data"][mask_msumg])
labels_msumg = msumg_dict["label"][mask_msumg]
labels_D     = labelD

print(f"DRIAMS-D: {len(dataD)} | MS-UMG: {len(data_msumg)}")

with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

domain_splits = splits.get("splits_per_domain", {})

all_X_tr, all_y_tr, all_X_vl, all_y_vl, all_X_te, all_y_te = [], [], [], [], [], []
for k, sk in SPLIT_NAMES.items():
    if sk not in domain_splits:
        continue
    tr_idx = domain_splits[sk]["train_idx"]
    vl_idx = domain_splits[sk].get("val_idx",  np.array([], dtype=int))
    te_idx = domain_splits[sk].get("test_idx", np.array([], dtype=int))
    if len(tr_idx) > 0:
        all_X_tr.append(data_final[tr_idx]); all_y_tr.append(label_final[tr_idx])
    if len(vl_idx) > 0:
        all_X_vl.append(data_final[vl_idx]); all_y_vl.append(label_final[vl_idx])
    if len(te_idx) > 0:
        all_X_te.append(data_final[te_idx]); all_y_te.append(label_final[te_idx])

X_train = np.vstack(all_X_tr); y_train = np.concatenate(all_y_tr)
X_val   = np.vstack(all_X_vl); y_val   = np.concatenate(all_y_vl)
X_test  = np.vstack(all_X_te); y_test  = np.concatenate(all_y_te)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test (source): {len(X_test)}")


############################################################
# ENCODE LATENT
############################################################
Z_train = encode_latent(model, X_train, device)
Z_val = encode_latent(model, X_val, device)
Z_test = encode_latent(model, X_test, device)
Z_D = encode_latent(model, dataD, device)
Z_msumg = encode_latent(model, data_msumg, device)
print(f"Z_train={Z_train.shape} | Z_test={Z_test.shape} | Z_D={Z_D.shape} | Z_msumg={Z_msumg.shape}")


############################################################
# FIT GMM
############################################################
print("\n===== FITTING GMM =====")
gmm = GaussianMixture(
    n_components=len(TARGET_SPECIES), covariance_type="full",
    max_iter=200, random_state=42, verbose=1)
gmm.fit(Z_train)
print(f"GMM converged: {gmm.converged_}")

ll_train = gmm.score_samples(Z_train)
ll_test = gmm.score_samples(Z_test)
ll_D = gmm.score_samples(Z_D)
ll_msumg = gmm.score_samples(Z_msumg)

# Thresholds are percentiles of the SOURCE-TRAIN log-likelihood only; applied unchanged to the targets
thr_gmm = [np.percentile(ll_train, p) for p in SWEEP_PERCENTILES]

print("\nThresholds (GMM, percentiles of source-train log-likelihood):")
for i, p in enumerate(SWEEP_PERCENTILES):
    print(f"  P{p} (GMM)  →  GMM={thr_gmm[i]:.2f}")


############################################################
# TRAIN MLP in latent space
############################################################
print("\n===== TRAINING MLP =====")
le = LabelEncoder()
le.fit(TARGET_SPECIES)
y_tr_enc = le.transform(y_train)
y_vl_enc = le.transform(y_val)
n_classes = len(le.classes_)

mlp_lat = LinearProbe_Extended(
    latent_dim=Z_train.shape[1], n_species=n_classes,
    epochs=50, lr=1e-3, patience=10,
)
mlp_lat.trainloop(
    make_loader(Z_train, y_tr_enc, shuffle=True),
    make_loader(Z_val,   y_vl_enc),
    device,
)


############################################################
# BUILD OOD FLAGS FOR THE SELECTED PERCENTILE THRESHOLDS
############################################################
preds_test = predict(mlp_lat, Z_test, device)
preds_D = predict(mlp_lat, Z_D, device)
preds_msumg = predict(mlp_lat, Z_msumg, device)
y_test_enc = le.transform(y_test)
y_D_enc = le.transform(labels_D)
y_msumg_enc = le.transform(labels_msumg)
probs_test = predict_proba(mlp_lat, Z_test, device)
probs_D = predict_proba(mlp_lat, Z_D, device)
probs_msumg = predict_proba(mlp_lat, Z_msumg, device)

metric_names = ["BA", "F1", "Precision", "Recall", "Specificity", "ROC AUC"]
metric_keys  = ["ba", "f1", "precision", "recall", "specificity", "roc_auc"]

criteria_results = {}

for ds_name, ll, y_enc, preds, probs in [
    ("Test-source", ll_test,  y_test_enc,  preds_test,  probs_test),
    ("DRIAMS-D",    ll_D,     y_D_enc,     preds_D,     probs_D),
    ("MS-UMG",      ll_msumg, y_msumg_enc, preds_msumg, probs_msumg),
]:
    criteria_results[ds_name] = {}

    rows_gmm = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        accepted = ll >= thr_gmm[i]
        n_total = len(accepted)
        n_kept = int(accepted.sum())
        coverage = 100 * accepted.mean()
        y_true = y_enc[accepted]; y_pred = preds[accepted]; y_prob = probs[accepted]
        if accepted.sum() > 10 and len(np.unique(y_true)) >= 2:
            m = compute_all_metrics(y_true, y_pred, y_prob, le.classes_)
        else:
            m = {k: np.nan for k in metric_keys}
        row = {"n_total": n_total, "n_kept": n_kept, "coverage": coverage}
        for name, key in zip(metric_names, metric_keys):
            row[name] = m[key]
        rows_gmm.append(row)
    criteria_results[ds_name]["GMM"] = rows_gmm


############################################################
# SUMMARY TABLE — Percentile sweep, GMM, all metrics
############################################################
print("\n===== COMBINED COVERAGE TABLE — GMM ONLY =====")

for ds_name in ["Test-source", "DRIAMS-D", "MS-UMG"]:
    print(f"\n{ds_name}:")
    rows = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        row = {"percentile_gmm": p}
        r = criteria_results[ds_name]["GMM"][i]
        n_total = r["n_total"]
        n_kept = r["n_kept"]
        n_disc = n_total - n_kept
        row["total"] = n_total
        row["kept"] = n_kept
        row["discarded"] = n_disc
        row["coverage"] = round(r["coverage"], 1)
        for metric in metric_names:
            val = r[metric]
            row[metric] = round(val, 3) if not np.isnan(val) else val
        rows.append(row)
    df_gmm_only = pd.DataFrame(rows)
    print(df_gmm_only.to_string(index=False))
    df_gmm_only.to_csv(OUTPUT_DIR / f"coverage_table_GMM_only_{ds_name.replace('-','_')}.csv", index=False)


############################################################
# RISK-COVERAGE CURVES
############################################################
print("\n===== RISK-COVERAGE CURVES =====")
coverages = np.round(np.arange(0.05, 1.0001, 0.01), 2)
rc_curves = {}
op_points = {}

for ds_name, ll, y_enc, preds in [
    ("DRIAMS-D", ll_D,     y_D_enc,     preds_D),
    ("MS-UMG",   ll_msumg, y_msumg_enc, preds_msumg)]:
    order = np.argsort(-ll)  
    rows = []
    for cov in coverages:
        n_kept = int(round(cov * len(ll)))
        idx = order[:n_kept]
        if len(np.unique(y_enc[idx])) >= 2:
            ba = balanced_accuracy_score(y_enc[idx], preds[idx])
        else:
            ba = np.nan
        rows.append({"coverage": cov, "n_kept": n_kept, "BA": ba, "ll_cut": ll[order[n_kept - 1]]})
    df_rc = pd.DataFrame(rows)
    rc_curves[ds_name] = df_rc
    df_rc.to_csv(OUTPUT_DIR / f"risk_coverage_{ds_name.replace('-','_')}.csv", index=False)

    # operating points defined by the source-train percentiles 
    pts = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        acc = ll >= thr_gmm[i]
        if acc.sum() > 10 and len(np.unique(y_enc[acc])) >= 2:
            pts.append((acc.mean(), balanced_accuracy_score(y_enc[acc], preds[acc]), p))
    op_points[ds_name] = pts

colors  = {"DRIAMS-D": "#3B82F6", "MS-UMG": "#EF4444"}
markers = {"DRIAMS-D": "o", "MS-UMG": "s"}
below   = {("MS-UMG", 0.1), ("MS-UMG", 0.5)}  

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for ds_name, df_rc in rc_curves.items():
    pts = op_points[ds_name]
    cov_min = min(c for c, _, _ in pts) 
    df_plot = df_rc[df_rc["coverage"] >= cov_min - 0.01].dropna(subset=["BA"])
    ax.plot(100 * df_plot["coverage"], df_plot["BA"], color=colors[ds_name], lw=1.8, label=ds_name)
    for cov, ba, p in pts:
        ax.scatter(100 * cov, ba, color=colors[ds_name], marker=markers[ds_name], s=28,
                   edgecolor="black", linewidth=0.6, zorder=3)
        if (ds_name, p) in below:
            ax.annotate(f"{p}", (100 * cov, ba), textcoords="offset points", xytext=(0, -9),
                        ha="center", va="top", fontsize=7, color=colors[ds_name])
        else:
            ax.annotate(f"{p}", (100 * cov, ba), textcoords="offset points", xytext=(5, 0),
                        ha="left", va="center", fontsize=7, color=colors[ds_name])

ax.set_xlim(101, min(min(c for c, _, _ in op_points[d]) for d in op_points) * 100 - 2) 
ax.set_xlabel("Coverage (%)")
ax.set_ylabel("Balanced accuracy")
ax.grid(alpha=0.3)
leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=1.0,
                facecolor="white", edgecolor="white", fontsize=8)
leg.set_zorder(5)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "risk_coverage.png", dpi=300)
plt.close(fig)


############################################################
# PER-SPECIES OOD RESULTS — TARGET_SPECIES on Test-source, DRIAMS-D, and MS-UMG
############################################################
print("\n===== PER-SPECIES OOD =====")
per_species_rows = {}
for ds_name, ll, labels in [
    ("Test-source", ll_test,  y_test),
    ("DRIAMS-D",    ll_D,     labels_D),
    ("MS-UMG",      ll_msumg, labels_msumg),
]:
    rows = []
    for sp in TARGET_SPECIES:
        mask = (labels == sp)
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        for i, p in enumerate(SWEEP_PERCENTILES):
            ood_gmm = ll[mask] < thr_gmm[i]

            row = {"species": sp.split("_")[0], "n_total": n_total, "percentile": p}
            for crit_name, ood_flags in [
                ("GMM", ood_gmm),
            ]:
                n_ood = int(ood_flags.sum())
                n_indist = n_total - n_ood
                row[f"{crit_name}_n_ood"] = n_ood
                row[f"{crit_name}_n_indist"] = n_indist
                row[f"{crit_name}_pct_ood"] = round(100 * ood_flags.mean(), 1)
            rows.append(row)
    df_sp = pd.DataFrame(rows)
    per_species_rows[ds_name] = df_sp
    df_sp.to_csv(OUTPUT_DIR / f"per_species_ood_{ds_name.replace('-','_')}.csv", index=False)
    print(f"\n{ds_name}:")
    print(df_sp.to_string(index=False))

print(f"\n===== DONE — results in {OUTPUT_DIR} =====")
