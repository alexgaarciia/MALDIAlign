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
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import encode_latent, make_loader
from src.evaluation.reconstruction_error import reconstruction_error, recon_ood_score
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

SWEEP_PERCENTILES = [0.01, 0.05, 0.1, 0.5, 1]
SWEEP_PERCENTILES_RECON = [100 - p for p in SWEEP_PERCENTILES]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/novelty_detection/combined") / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DOMAIN_IDS = {"DRIAMS_A": 0, "DRIAMS_B": 1, "DRIAMS_C": 2, "MARISMA": 3, "RKI": 4}
SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA", "RKI": "RKI"}
RELIABLE_DECODERS = {"A": "DRIAMS_A", "MARISMA": "MARISMA"}

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
Z_train  = encode_latent(model, X_train,    device)
Z_val    = encode_latent(model, X_val,      device)
Z_test   = encode_latent(model, X_test,     device)
Z_D      = encode_latent(model, dataD,      device)
Z_msumg  = encode_latent(model, data_msumg, device)
print(f"Z_train={Z_train.shape} | Z_test={Z_test.shape} | Z_D={Z_D.shape} | Z_msumg={Z_msumg.shape}")


############################################################
# FIT GMM
############################################################
print("\n===== FITTING GMM =====")
gmm = GaussianMixture(
    n_components=len(TARGET_SPECIES), covariance_type="full",
    max_iter=200, random_state=42, verbose=1,
)
gmm.fit(Z_train)
print(f"GMM converged: {gmm.converged_}")

ll_train = gmm.score_samples(Z_train)
ll_test  = gmm.score_samples(Z_test)
ll_D     = gmm.score_samples(Z_D)
ll_msumg = gmm.score_samples(Z_msumg)


############################################################
# FIT LOG-NORMAL BY DECODER
############################################################
print("\n===== FITTING LOG-NORMAL PER DECODER =====")

domain_lognormals = {}
errors_per_domain = {}

for k, sk in SPLIT_NAMES.items():
    tr_idx = domain_splits[sk]["train_idx"]
    X_tr_d = data_final[tr_idx]
    domain_id = DOMAIN_IDS[sk]
    errs = reconstruction_error(model, X_tr_d, domain_id, device)
    log_errs = np.log(errs + 1e-10)
    mu_log, sig_log = norm.fit(log_errs)
    domain_lognormals[domain_id] = (mu_log, sig_log)
    errors_per_domain[k] = errs
    print(f"  {sk}: n={len(errs)}  mu_log={mu_log:.4f}  sigma_log={sig_log:.4f}")

scores_test_by_decoder = {}
scores_D_by_decoder = {}
scores_msumg_by_decoder = {}
scores_train_by_decoder = {}

for k, sk in RELIABLE_DECODERS.items():
    domain_id = DOMAIN_IDS[sk]
    mu_log, sig_log = domain_lognormals[domain_id]

    errs_test  = reconstruction_error(model, X_test,    domain_id, device)
    errs_D     = reconstruction_error(model, dataD,      domain_id, device)
    errs_msumg = reconstruction_error(model, data_msumg, domain_id, device)
    scores_test_by_decoder[k]  = recon_ood_score(errs_test,  mu_log, sig_log)
    scores_D_by_decoder[k]     = recon_ood_score(errs_D,     mu_log, sig_log)
    scores_msumg_by_decoder[k] = recon_ood_score(errs_msumg, mu_log, sig_log)
    scores_train_by_decoder[k] = recon_ood_score(errors_per_domain[k], mu_log, sig_log)

thr_recon_per_decoder = {}
for k in RELIABLE_DECODERS:
    thr_recon_per_decoder[k] = [np.percentile(scores_train_by_decoder[k], p) for p in SWEEP_PERCENTILES_RECON]

thr_gmm = [np.percentile(ll_train, p) for p in SWEEP_PERCENTILES]

print("\nThresholds per criterion:")
for i, p in enumerate(SWEEP_PERCENTILES):
    print(f"  P{p} (GMM) / P{SWEEP_PERCENTILES_RECON[i]} (recon)  →  "
          f"GMM={thr_gmm[i]:.2f}  A={thr_recon_per_decoder['A'][i]:.3f}  MARISMA={thr_recon_per_decoder['MARISMA'][i]:.3f}")


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
# HELPERS
############################################################
def get_predictions(mlp, X, device, batch_size=512):
    mlp.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = mlp(X_t[i:i+batch_size].to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)

def get_probs(mlp, X, device, batch_size=512):
    mlp.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            logits = mlp(X_t[i:i+batch_size].to(device))
            probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs_list, axis=0)

def compute_all_metrics(y_true, y_pred, y_probs, classes):
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ["ba", "f1", "precision", "recall", "specificity", "roc_auc"]}
    ba = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    specs = []
    for c in range(len(classes)):
        tn = ((y_true != c) & (y_pred != c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
    specificity = np.nanmean(specs)
    try:
        classes_present = np.unique(y_true)
        if len(classes_present) < 2:
            roc_auc = np.nan
        else:
            y_bin = label_binarize(y_true, classes=list(range(len(classes))))
            roc_auc = roc_auc_score(y_bin[:, classes_present], y_probs[:, classes_present],
                                     average="macro", multi_class="ovr")
    except Exception:
        roc_auc = np.nan
    return {"ba": ba, "f1": f1, "precision": precision, "recall": recall,
            "specificity": specificity, "roc_auc": roc_auc}


############################################################
# BUILD OOD FLAGS FOR THE SELECTED PERCENTILE THRESHOLDS
############################################################
preds_test  = get_predictions(mlp_lat, Z_test, device)
preds_D     = get_predictions(mlp_lat, Z_D, device)
preds_msumg = get_predictions(mlp_lat, Z_msumg, device)
y_test_enc  = le.transform(y_test)
y_D_enc     = le.transform(labels_D)
y_msumg_enc = le.transform(labels_msumg)
probs_test  = get_probs(mlp_lat, Z_test, device)
probs_D     = get_probs(mlp_lat, Z_D, device)
probs_msumg = get_probs(mlp_lat, Z_msumg, device)

metric_names = ["BA", "F1", "Precision", "Recall", "Specificity", "ROC AUC"]
metric_keys  = ["ba", "f1", "precision", "recall", "specificity", "roc_auc"]

criteria_results = {}

for ds_name, ll, scores_dec, y_enc, preds, probs in [
    ("Test-source", ll_test,  scores_test_by_decoder,  y_test_enc,  preds_test,  probs_test),
    ("DRIAMS-D",    ll_D,     scores_D_by_decoder,     y_D_enc,     preds_D,     probs_D),
    ("MS-UMG",      ll_msumg, scores_msumg_by_decoder, y_msumg_enc, preds_msumg, probs_msumg),
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

    rows_A = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        accepted = scores_dec["A"] <= thr_recon_per_decoder["A"][i]
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
        rows_A.append(row)
    criteria_results[ds_name]["Decoder A"] = rows_A

    rows_M = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        accepted = scores_dec["MARISMA"] <= thr_recon_per_decoder["MARISMA"][i]
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
        rows_M.append(row)
    criteria_results[ds_name]["MARISMA"] = rows_M

    # Intersection
    rows_int = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        ood_gmm = ll < thr_gmm[i]
        ood_A   = scores_dec["A"] > thr_recon_per_decoder["A"][i]
        ood_M   = scores_dec["MARISMA"] > thr_recon_per_decoder["MARISMA"][i]
        ood_combined = ood_gmm & ood_A & ood_M
        accepted = ~ood_combined

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
        rows_int.append(row)
    criteria_results[ds_name]["Intersection (3 agree)"] = rows_int

    # Union
    rows_union = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        ood_gmm = ll < thr_gmm[i]
        ood_A   = scores_dec["A"] > thr_recon_per_decoder["A"][i]
        ood_M   = scores_dec["MARISMA"] > thr_recon_per_decoder["MARISMA"][i]
        ood_combined = ood_gmm | ood_A | ood_M
        accepted = ~ood_combined

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
        rows_union.append(row)
    criteria_results[ds_name]["Union (at least 1 agrees)"] = rows_union


############################################################
# SUMMARY TABLE — Percentile sweep, five criteria, all metrics
############################################################
criteria_colors = {
    "GMM": "#3B82F6", "Decoder A": "#10B981",
    "MARISMA": "#F59E0B", "Intersection (3 agree)": "#EF4444",
    "Union (at least 1 agrees)": "#8B5CF6",
}

print("\n===== COMBINED COVERAGE TABLE =====")

for ds_name in ["Test-source", "DRIAMS-D", "MS-UMG"]:
    print(f"\n{ds_name}:")
    rows = []
    for i, p in enumerate(SWEEP_PERCENTILES):
        row = {"percentile_gmm": p, "percentile_recon": SWEEP_PERCENTILES_RECON[i]}
        for crit_name in criteria_colors.keys():
            r = criteria_results[ds_name][crit_name][i]
            n_total = r["n_total"]
            n_kept = r["n_kept"]
            n_disc = n_total - n_kept
            row[f"{crit_name}_total"] = n_total
            row[f"{crit_name}_kept"] = n_kept
            row[f"{crit_name}_discarded"] = n_disc
            row[f"{crit_name}_cov"] = round(r["coverage"], 1)
            for metric in metric_names:
                val = r[metric]
                row[f"{crit_name}_{metric}"] = round(val, 3) if not np.isnan(val) else val
        rows.append(row)
    df_combined = pd.DataFrame(rows)
    print(df_combined.to_string(index=False))
    df_combined.to_csv(OUTPUT_DIR / f"combined_coverage_table_{ds_name.replace('-','_')}.csv", index=False)

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
# PER-SPECIES OOD RESULTS BY CRITERION — TARGET_SPECIES on Test-source, DRIAMS-D, and MS-UMG
############################################################
print("\n===== PER-SPECIES OOD PER CRITERION =====")
per_species_rows = {}
for ds_name, ll, scores_dec, labels in [
    ("Test-source", ll_test,  scores_test_by_decoder,  y_test),
    ("DRIAMS-D",    ll_D,     scores_D_by_decoder,     labels_D),
    ("MS-UMG",      ll_msumg, scores_msumg_by_decoder, labels_msumg),
]:
    rows = []
    for sp in TARGET_SPECIES:
        mask = (labels == sp)
        n_total = int(mask.sum())
        if n_total == 0:
            continue
        for i, p in enumerate(SWEEP_PERCENTILES):
            ood_gmm   = ll[mask] < thr_gmm[i]
            ood_A     = scores_dec["A"][mask] > thr_recon_per_decoder["A"][i]
            ood_M     = scores_dec["MARISMA"][mask] > thr_recon_per_decoder["MARISMA"][i]
            ood_int   = ood_gmm & ood_A & ood_M
            ood_union = ood_gmm | ood_A | ood_M

            row = {"species": sp.split("_")[0], "n_total": n_total, "percentile": p}
            for crit_name, ood_flags in [
                ("GMM", ood_gmm), ("Decoder A", ood_A),
                ("MARISMA", ood_M), ("Intersection (3 agree)", ood_int),
                ("Union (at least 1 agrees)", ood_union),
            ]:
                n_ood = int(ood_flags.sum())
                n_indist = n_total - n_ood
                row[f"{crit_name}_n_ood"] = n_ood
                row[f"{crit_name}_n_indist"] = n_indist
                row[f"{crit_name}_pct_ood"] = round(100 * ood_flags.mean(), 1)
            rows.append(row)
    df_sp = pd.DataFrame(rows)
    per_species_rows[ds_name] = df_sp
    df_sp.to_csv(OUTPUT_DIR / f"per_species_ood_by_criterion_{ds_name.replace('-','_')}.csv", index=False)
    print(f"\n{ds_name}:")
    print(df_sp.to_string(index=False))


############################################################
# PLOT — Grouped bar chart: species × criterion, one panel per percentile, by dataset
############################################################
criteria_list = ["GMM", "Decoder A", "MARISMA", "Intersection (3 agree)", "Union (at least 1 agrees)"]
criteria_colors_list = [criteria_colors[c] for c in criteria_list]

for ds_name in ["Test-source", "DRIAMS-D", "MS-UMG"]:
    df_sp = per_species_rows[ds_name]
    species_list = sorted(df_sp["species"].unique())

    fig, axes = plt.subplots(1, len(SWEEP_PERCENTILES), figsize=(7 * len(SWEEP_PERCENTILES), 5.5), sharey=True)

    for ax, p in zip(axes, SWEEP_PERCENTILES):
        sub = df_sp[df_sp["percentile"] == p].set_index("species").loc[species_list]
        x = np.arange(len(species_list))
        width = 0.8 / len(criteria_list)
        for i, (crit, color) in enumerate(zip(criteria_list, criteria_colors_list)):
            vals = sub[f"{crit}_pct_ood"].values
            ax.bar(x + i * width, vals, width, color=color, label=crit)

        ax.set_xticks(x + width * (len(criteria_list) - 1) / 2)
        ax.set_xticklabels(species_list, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        ax.set_title(f"P{p}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        ax.legend(fontsize=6, loc="upper right")

    axes[0].set_ylabel("% OOD")
    fig.suptitle(f"Per-species OOD detection by criterion — {ds_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / f"plot_per_species_ood_by_criterion_{ds_name.replace('-','_')}.png", dpi=150, bbox_inches="tight")
    plt.show()


############################################################
# UNSEEN SPECIES — OOD RESULTS BY CRITERION (GMM, A, MARISMA, Intersection, Union)
############################################################
print("\n===== UNSEEN SPECIES — OOD PER CRITERION =====")

MIN_N_UNSEEN = 10

driams_dict_all  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict_all = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict_all     = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict_all   = load_msumg(cfg["data"]["MSUMG_FULL"])

unseen_data_list  = []
unseen_label_list = []

for name, d in [
    ("DRIAMS", driams_dict_all),
    ("MARISMA", marisma_dict_all),
    ("RKI", rki_dict_all),
    ("MS-UMG", msumg_dict_all),
]:
    mask_unseen = ~np.isin(d["label"], TARGET_SPECIES)
    if mask_unseen.sum() == 0:
        continue
    unseen_data_list.append(row_minmax_normalize(d["data"][mask_unseen]))
    unseen_label_list.append(d["label"][mask_unseen])

data_unseen   = np.vstack(unseen_data_list)
labels_unseen = np.concatenate(unseen_label_list)
unique_unseen_species = np.unique(labels_unseen)

print(f"Found {len(unique_unseen_species)} unseen species, n={len(data_unseen)} samples")

Z_unseen  = encode_latent(model, data_unseen, device)
ll_unseen = gmm.score_samples(Z_unseen)

scores_unseen_by_decoder = {}
for k, sk in RELIABLE_DECODERS.items():
    domain_id = DOMAIN_IDS[sk]
    mu_log, sig_log = domain_lognormals[domain_id]
    errs_unseen = reconstruction_error(model, data_unseen, domain_id, device)
    scores_unseen_by_decoder[k] = recon_ood_score(errs_unseen, mu_log, sig_log)

rows_unseen = []
for sp in unique_unseen_species:
    mask = (labels_unseen == sp)
    n_total = int(mask.sum())
    if n_total < MIN_N_UNSEEN:
        continue
    for i, p in enumerate(SWEEP_PERCENTILES):
        ood_gmm   = ll_unseen[mask] < thr_gmm[i]
        ood_A     = scores_unseen_by_decoder["A"][mask] > thr_recon_per_decoder["A"][i]
        ood_M     = scores_unseen_by_decoder["MARISMA"][mask] > thr_recon_per_decoder["MARISMA"][i]
        ood_int   = ood_gmm & ood_A & ood_M
        ood_union = ood_gmm | ood_A | ood_M

        row = {"species": sp, "n_total": n_total, "percentile": p}
        for crit_name, ood_flags in [
            ("GMM", ood_gmm), ("Decoder A", ood_A),
            ("MARISMA", ood_M), ("Intersection (3 agree)", ood_int),
            ("Union (at least 1 agrees)", ood_union),
        ]:
            n_ood = int(ood_flags.sum())
            n_indist = n_total - n_ood
            row[f"{crit_name}_n_ood"] = n_ood
            row[f"{crit_name}_n_indist"] = n_indist
            row[f"{crit_name}_pct_ood"] = round(100 * ood_flags.mean(), 1)
        rows_unseen.append(row)

df_unseen_crit = pd.DataFrame(rows_unseen)
df_unseen_crit.to_csv(OUTPUT_DIR / "unseen_species_ood_by_criterion.csv", index=False)
print(f"\nEspecies con n>={MIN_N_UNSEEN}: {df_unseen_crit['species'].nunique()}")


############################################################
# PLOT — Grouped bar chart: species × criterion, multiple panels, one percentile per figure
############################################################
species_list_unseen = sorted(df_unseen_crit["species"].unique())
n_species = len(species_list_unseen)
n_per_panel = 20
n_panels = int(np.ceil(n_species / n_per_panel))
n_cols = 2
n_rows = int(np.ceil(n_panels / n_cols))

for p in SWEEP_PERCENTILES:
    df_p = df_unseen_crit[df_unseen_crit["percentile"] == p].set_index("species").loc[species_list_unseen]

    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    axes_grid = np.array(axes_grid).reshape(-1)

    for panel_i in range(n_panels):
        start = panel_i * n_per_panel
        end = min(start + n_per_panel, n_species)
        sub_species = species_list_unseen[start:end]
        sub = df_p.loc[sub_species]

        ax = axes_grid[panel_i]
        x = np.arange(len(sub_species))
        width = 0.8 / len(criteria_list)

        for i, (crit, color) in enumerate(zip(criteria_list, criteria_colors_list)):
            vals = sub[f"{crit}_pct_ood"].values
            ax.bar(x + i * width, vals, width, color=color, label=crit)

        ax.set_xticks(x + width * (len(criteria_list) - 1) / 2)
        ax.set_xticklabels([s.replace("_", " ") for s in sub_species], fontsize=8, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.set_ylabel("% OOD")
        ax.set_title(f"Panel {panel_i+1}/{n_panels}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

        fig_ind, ax_ind = plt.subplots(figsize=(16, 6.5))
        for i, (crit, color) in enumerate(zip(criteria_list, criteria_colors_list)):
            vals = sub[f"{crit}_pct_ood"].values
            ax_ind.bar(x + i * width, vals, width, color=color, label=crit)
        ax_ind.set_xticks(x + width * (len(criteria_list) - 1) / 2)
        ax_ind.set_xticklabels([s.replace("_", " ") for s in sub_species], fontsize=9, rotation=45, ha="right")
        ax_ind.set_ylim(0, 105)
        ax_ind.set_ylabel("% OOD")
        ax_ind.set_title(f"Unseen species — P{p} — panel {panel_i+1}/{n_panels} (n≥{MIN_N_UNSEEN})", fontsize=11, fontweight="bold", pad=8)
        ax_ind.legend(fontsize=7, loc="upper right")
        ax_ind.grid(True, axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.close(fig_ind)

    for j in range(n_panels, len(axes_grid)):
        axes_grid[j].axis("off")

    fig.suptitle(f"Unseen species — OOD by criterion (P{p})", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(OUTPUT_DIR / f"plot_unseen_species_ood_criterion_P{str(p).replace('.','')}_grid.png", dpi=150, bbox_inches="tight")
    plt.show()

print(f"\n===== DONE — results in {OUTPUT_DIR} =====")
