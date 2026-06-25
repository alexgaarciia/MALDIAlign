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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import encode_latent, make_loader
from src.evaluation.metrics import metrics_report_mlp
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended
from models.baselines.mlp_latent import LinearProbe_Extended


############################################################
# CONFIG
############################################################
cfg = load_config()

TARGET_SPECIES = ["Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus", "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex",]

SPECIES_COLORS = {
    "Enterobacter_cloacae_complex": "#F28B82",
    "Escherichia_Coli":             "#B39DDB",
    "Enterococcus_Faecium":         "#A5D6A7",
    "Klebsiella_Pneumoniae":        "#90CAF9",
    "Staphylococcus_Aureus":        "#FFCC80",
    "Pseudomonas_Aeruginosa":       "#80CBC4",
}

EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")
MODEL_PATH  = EXPERIMENT_DIR / "model.pth"
SPLITS_PATH = EXPERIMENT_DIR / "data_splits.pkl"
RUN_PLOTS = False
RUN_RAW = False

SWEEP_PERCENTILES = [0.01, 0.05, 0.1, 0.5, 1]
REFERENCE_PERCENTILE = 0.1  # percentil usado para confusion matrices y t-SNE

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/novelty_detection/results") / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Device: {device}")

############################################################
# LOAD MODEL
############################################################
print("\n===== LOADING MODEL =====")
state = torch.load(MODEL_PATH, map_location="cpu")

model = MultiVAE_Bernoulli_SpeciesPrior_Extended(
    input_dim=6000,
    latent_dim=64,
    num_domains=5,
    n_species=len(TARGET_SPECIES),
)
model.load_state_dict(state)
model.to(device).eval()
print("Model loaded — num_domains=5")

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

dataA  = data_driams[maskA]
labelA = label_driams[maskA]
dataB  = data_driams[maskB]
labelB = label_driams[maskB]
dataC  = data_driams[maskC]
labelC = label_driams[maskC]
dataD  = data_driams[maskD]
labelD = label_driams[maskD]

dataA        = row_minmax_normalize(dataA)
dataB        = row_minmax_normalize(dataB)
dataC        = row_minmax_normalize(dataC)
dataD        = row_minmax_normalize(dataD)
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

print(f"DRIAMS-D: {len(dataD)}")
print(f"MS-UMG:   {len(data_msumg)}")

with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

domain_splits = splits.get("splits_per_domain", {})
SPLIT_NAMES   = {
    "A":       "DRIAMS_A",
    "B":       "DRIAMS_B",
    "C":       "DRIAMS_C",
    "MARISMA": "MARISMA",
    "RKI":     "RKI",
}

all_X_tr = []
all_y_tr = []
all_X_vl = []
all_y_vl = []
all_X_te = []
all_y_te = []

for k, sk in SPLIT_NAMES.items():
    if sk not in domain_splits:
        continue
    tr_idx = domain_splits[sk]["train_idx"]
    vl_idx = domain_splits[sk].get("val_idx",  np.array([], dtype=int))
    te_idx = domain_splits[sk].get("test_idx", np.array([], dtype=int))
    if len(tr_idx) > 0:
        all_X_tr.append(data_final[tr_idx])
        all_y_tr.append(label_final[tr_idx])
    if len(vl_idx) > 0:
        all_X_vl.append(data_final[vl_idx])
        all_y_vl.append(label_final[vl_idx])
    if len(te_idx) > 0:
        all_X_te.append(data_final[te_idx])
        all_y_te.append(label_final[te_idx])

X_train = np.vstack(all_X_tr)
y_train = np.concatenate(all_y_tr)
X_val   = np.vstack(all_X_vl)
y_val   = np.concatenate(all_y_vl)
X_test  = np.vstack(all_X_te)
y_test  = np.concatenate(all_y_te)

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
    n_components=len(TARGET_SPECIES),
    covariance_type="full",
    max_iter=200,
    random_state=42,
    verbose=1,
)
gmm.fit(Z_train)
print(f"GMM converged: {gmm.converged_}")

ll_train = gmm.score_samples(Z_train)
ll_test  = gmm.score_samples(Z_test)
ll_D     = gmm.score_samples(Z_D)
ll_msumg = gmm.score_samples(Z_msumg)

############################################################
# OOD DETECTION + PER-SPECIES — barrido de percentiles
############################################################
print("\n===== OOD DETECTION (SWEEP) =====")

for p in SWEEP_PERCENTILES:
    thr_p = np.percentile(ll_train, p)

    ood_test_p  = (ll_test  < thr_p)
    ood_D_p     = (ll_D     < thr_p)
    ood_msumg_p = (ll_msumg < thr_p)

    print(f"\n{'='*70}")
    print(f"  PERCENTILE P{p}  (threshold={thr_p:.2f})")
    print(f"{'='*70}")
    print(f"Test (source) OOD: {ood_test_p.sum()}/{len(ood_test_p)} ({100*ood_test_p.mean():.1f}%)")
    print(f"DRIAMS-D OOD: {ood_D_p.sum()}/{len(ood_D_p)} ({100*ood_D_p.mean():.1f}%)")
    print(f"MS-UMG OOD: {ood_msumg_p.sum()}/{len(ood_msumg_p)} ({100*ood_msumg_p.mean():.1f}%)")

    results_ood = []
    for sp in TARGET_SPECIES:
        row = {"species": sp}
        for name, ll, yy in [
            ("test",     ll_test,  y_test),
            ("driams_d", ll_D,     labels_D),
            ("msumg",    ll_msumg, labels_msumg),
        ]:
            mask = (yy == sp)
            if mask.sum() == 0:
                row[f"{name}_n"]       = 0
                row[f"{name}_pct_ood"] = np.nan
            else:
                ll_sp = ll[mask]
                row[f"{name}_n"]       = mask.sum()
                row[f"{name}_pct_ood"] = 100 * (ll_sp < thr_p).mean()
        results_ood.append(row)

    df_ood_p = pd.DataFrame(results_ood)
    print(f"\n--- Per-species % OOD at P{p} ---")
    print(df_ood_p.to_string(index=False))
    df_ood_p.to_csv(OUTPUT_DIR / f"novelty_detection_per_species_p{str(p).replace('.','')}.csv", index=False)

    if p == REFERENCE_PERCENTILE:
        thr_ref = thr_p
        ood_test = ood_test_p
        ood_D    = ood_D_p
        ood_msumg = ood_msumg_p

np.savetxt(OUTPUT_DIR / "ood_indices_driams_d_reference.txt", np.where(ood_D)[0], fmt="%d")
np.savetxt(OUTPUT_DIR / "ood_indices_msumg_reference.txt", np.where(ood_msumg)[0], fmt="%d")

############################################################
# PLOT 1 — distribuciones log-likelihood por especie (percentil de referencia)
############################################################
fig, axes = plt.subplots(1, len(TARGET_SPECIES),
                         figsize=(4 * len(TARGET_SPECIES), 4), sharey=False)

for ax, sp in zip(axes, TARGET_SPECIES):
    for ll, yy, color, label in [
        (ll_test,  y_test,     "#66A182", "Test (source)"),
        (ll_D,     labels_D,   "#3B82F6", "DRIAMS-D"),
        (ll_msumg, labels_msumg, "#EF4444", "MS-UMG"),
    ]:
        mask = (yy == sp)
        if mask.sum() > 0:
            ax.hist(ll[mask], bins=30, alpha=0.5, color=color,
                    label=label, density=True)

    ax.axvline(thr_ref, color="black", ls="--", lw=1.5, label=f"P{REFERENCE_PERCENTILE} threshold")
    ax.set_title(sp.replace("_", " "), fontsize=9, fontweight="bold")
    ax.set_xlabel("Log-likelihood")
    ax.grid(True, linestyle=":", alpha=0.4)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="upper center",
           bbox_to_anchor=(0.5, 1.03), ncol=5, fontsize=9)
fig.suptitle(f"GMM Log-Likelihood Distributions per Species (P{REFERENCE_PERCENTILE} threshold)",
             fontsize=13, fontweight="bold", y=1.07)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "novelty_ll_distributions_reference.png", dpi=150, bbox_inches="tight")
plt.show()

############################################################
# TRAIN MLP in latent space
############################################################
print("\n===== TRAINING MLP =====")
le = LabelEncoder()
le.fit(TARGET_SPECIES)

y_tr_enc  = le.transform(y_train)
y_vl_enc  = le.transform(y_val)
n_classes = len(le.classes_)

print("Training MLP latent space...")
mlp_lat = LinearProbe_Extended(
    latent_dim=Z_train.shape[1],
    n_species=n_classes,
    epochs=50,
    lr=1e-3,
    patience=10,
)
mlp_lat.trainloop(
    make_loader(Z_train, y_tr_enc, shuffle=True),
    make_loader(Z_val,   y_vl_enc),
    device,
)


############################################################
# HELPER
############################################################
def get_predictions(mlp, X, device, batch_size=512):
    mlp.eval()
    X_t   = torch.tensor(X, dtype=torch.float32)
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
            y_bin_present = y_bin[:, classes_present]
            y_probs_present = y_probs[:, classes_present]
            roc_auc = roc_auc_score(y_bin_present, y_probs_present, average="macro", multi_class="ovr")
    except Exception:
        roc_auc = np.nan
    return {"ba": ba, "f1": f1, "precision": precision, "recall": recall,
            "specificity": specificity, "roc_auc": roc_auc}


############################################################
# OOD vs CLASSIFICATION ERRORS — barrido de percentiles
############################################################
print("\n===== OOD vs CLASSIFICATION ERRORS (SWEEP) =====")

summary_rows = []

for p in SWEEP_PERCENTILES:
    thr_p = np.percentile(ll_train, p)

    ood_test_p  = (ll_test  < thr_p)
    ood_D_p     = (ll_D     < thr_p)
    ood_msumg_p = (ll_msumg < thr_p)

    print(f"\n{'='*70}")
    print(f"  PERCENTILE P{p}  (threshold={thr_p:.2f})")
    print(f"{'='*70}")

    for ds_name, X_te, Z_te, y_te, ood_flags, ll_te in [
        ("Test-source", X_test,     Z_test,  y_test,       ood_test_p,  ll_test),
        ("DRIAMS-D",    dataD,      Z_D,     labels_D,     ood_D_p,     ll_D),
        ("MS-UMG",      data_msumg, Z_msumg, labels_msumg, ood_msumg_p, ll_msumg),
    ]:
        y_te_enc = le.transform(y_te)
        preds = get_predictions(mlp_lat, Z_te, device)
        correct = (preds == y_te_enc)
        probs = get_probs(mlp_lat, Z_te, device)

        m_all = compute_all_metrics(y_te_enc, preds, probs, le.classes_)
        m_ind = compute_all_metrics(y_te_enc[~ood_flags], preds[~ood_flags], probs[~ood_flags], le.classes_) if (~ood_flags).sum() > 0 else {k: np.nan for k in ["ba","f1","precision","recall","specificity","roc_auc"]}
        m_ood = compute_all_metrics(y_te_enc[ood_flags],  preds[ood_flags],  probs[ood_flags],  le.classes_) if  ood_flags.sum()  > 0 else {k: np.nan for k in ["ba","f1","precision","recall","specificity","roc_auc"]}

        print(f"\n{ds_name}")
        print(f"  {'':20s}  {'Bal.Acc':>8}  {'F1':>8}  {'Prec':>8}  {'Recall':>8}  {'Spec':>8}  {'ROC AUC':>8}  {'n':>6}")
        for label, m, n in [
            ("ALL",            m_all, len(correct)),
            ("In-distribution",m_ind, (~ood_flags).sum()),
            ("OOD",            m_ood, ood_flags.sum()),
        ]:
            print(f"  {label:20s}  {m['ba']:8.3f}  {m['f1']:8.3f}  {m['precision']:8.3f}  {m['recall']:8.3f}  {m['specificity']:8.3f}  {m['roc_auc']:8.3f}  {n:>6}")

        for sp in TARGET_SPECIES:
            m = (y_te == sp)
            if m.sum() == 0:
                continue
            summary_rows.append({
                "percentile": p,
                "dataset": ds_name,
                "species": sp,
                "n_total": m.sum(),
                "n_ind": (m & ~ood_flags).sum(),
                "n_ood": (m & ood_flags).sum(),
                "acc_all": correct[m].mean(),
                "acc_ind": correct[m & ~ood_flags].mean() if (m & ~ood_flags).sum() > 0 else np.nan,
                "acc_ood": correct[m & ood_flags].mean()  if (m & ood_flags).sum()  > 0 else np.nan,
            })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUTPUT_DIR / "ood_vs_accuracy_sweep.csv", index=False)
print("\n", df_summary.to_string(index=False))


############################################################
# UNSEEN SPECIES — especies completamente fuera de TARGET_SPECIES
############################################################
print("\n===== UNSEEN SPECIES — NOVELTY DETECTION =====")

# cargar TODAS las especies del pkl (sin filtrar por TARGET_SPECIES)
# usamos los mismos hospitales source para detectar qué especies hay fuera de la lista
driams_dict_all  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict_all = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict_all     = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict_all   = load_msumg(cfg["data"]["MSUMG_FULL"])

unseen_data_list  = []
unseen_label_list = []
unseen_source_list = []

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
    unseen_source_list.append(np.full(mask_unseen.sum(), name))

data_unseen   = np.vstack(unseen_data_list)
labels_unseen = np.concatenate(unseen_label_list)
source_unseen = np.concatenate(unseen_source_list)

unique_unseen_species = np.unique(labels_unseen)
print(f"Found {len(unique_unseen_species)} unseen species, n={len(data_unseen)} samples")
for sp in unique_unseen_species:
    n_sp = (labels_unseen == sp).sum()
    print(f"  {sp}: n={n_sp}")

# encode + log-likelihood bajo el GMM ya entrenado
Z_unseen  = encode_latent(model, data_unseen, device)
ll_unseen = gmm.score_samples(Z_unseen)

print(f"\nZ_unseen={Z_unseen.shape}")
print(f"Log-likelihood unseen — mean={ll_unseen.mean():.2f}  std={ll_unseen.std():.2f}")

# sweep de percentiles — % detectado como OOD
print("\n--- Unseen species: % OOD across thresholds ---")
rows_unseen_summary = []
for p in SWEEP_PERCENTILES:
    thr_p = np.percentile(ll_train, p)
    ood_unseen_p = (ll_unseen < thr_p)
    pct = 100 * ood_unseen_p.mean()
    rows_unseen_summary.append({"percentile": p, "threshold": thr_p, "pct_ood": round(pct, 2), "n_ood": int(ood_unseen_p.sum()), "n_total": len(ll_unseen)})
    print(f"  P{p}: OOD={pct:.1f}%  ({ood_unseen_p.sum()}/{len(ll_unseen)})")

df_unseen_summary = pd.DataFrame(rows_unseen_summary)
df_unseen_summary.to_csv(OUTPUT_DIR / "unseen_species_ood_sweep_summary.csv", index=False)

# tabla per-species — % OOD a cada percentil, desglosado por fuente
print("\n--- Unseen species: % OOD per species per threshold, by source ---")
rows_unseen_sp = []
for sp in unique_unseen_species:
    for src in np.unique(source_unseen):
        mask_sp = (labels_unseen == sp) & (source_unseen == src)
        n_sp = mask_sp.sum()
        if n_sp == 0:
            continue
        row = {"species": sp, "source": src, "n": n_sp}
        for p in SWEEP_PERCENTILES:
            thr_p = np.percentile(ll_train, p)
            pct = 100 * (ll_unseen[mask_sp] < thr_p).mean()
            row[f"P{p}"] = round(pct, 1)
        rows_unseen_sp.append(row)

df_unseen_per_species = pd.DataFrame(rows_unseen_sp).set_index(["species", "source"])
print(df_unseen_per_species.to_string())
df_unseen_per_species.to_csv(OUTPUT_DIR / "unseen_species_ood_per_species_by_source.csv")


############################################################
# RAW SPECTRA — todas las especies, solo binning sin preprocesado
############################################################
if RUN_RAW:
    print("\n===== RAW SPECTRA — NOVELTY DETECTION (ALL SPECIES) =====")

    RAW_PKL = {
        "DRIAMS": cfg["data"]["DRIAMS_FULL_RAW"],
        "MARISMA": cfg["data"]["MARISMa_FULL_RAW"],
        "RKI": cfg["data"]["RKI_FULL_RAW"],
        "MS-UMG": cfg["data"]["MSUMG_FULL_RAW"],
    }

    raw_data_list   = []
    raw_label_list  = []
    raw_source_list = []

    for name, pkl_path in RAW_PKL.items():
        if name == "DRIAMS":
            d = load_driams(pkl_path)
        elif name == "MARISMA":
            d = load_marisma(pkl_path)
        elif name == "RKI":
            d = load_rki(pkl_path)
        elif name == "MS-UMG":
            d = load_msumg(pkl_path)

        raw_data_list.append(row_minmax_normalize(d["data"]))
        raw_label_list.append(d["label"])
        raw_source_list.append(np.full(len(d["label"]), name))

    data_raw_all   = np.vstack(raw_data_list)
    labels_raw_all = np.concatenate(raw_label_list)
    source_raw_all = np.concatenate(raw_source_list)

    unique_raw_species = np.unique(labels_raw_all)
    print(f"Found {len(unique_raw_species)} species in raw pkl, n={len(data_raw_all)} samples")
    for sp in unique_raw_species:
        n_sp = (labels_raw_all == sp).sum()
        print(f"  {sp}: n={n_sp}")

    # encode + log-likelihood bajo el GMM ya entrenado (mismo modelo)
    Z_raw_all  = encode_latent(model, data_raw_all, device)
    ll_raw_all = gmm.score_samples(Z_raw_all)

    print(f"\nZ_raw_all={Z_raw_all.shape}")
    print(f"Log-likelihood raw — mean={ll_raw_all.mean():.2f}  std={ll_raw_all.std():.2f}")

    # sweep de percentiles
    print("\n--- Raw spectra: % OOD across thresholds (all species) ---")
    rows_raw_summary = []
    for p in SWEEP_PERCENTILES:
        thr_p = np.percentile(ll_train, p)
        ood_raw_p = (ll_raw_all < thr_p)
        pct = 100 * ood_raw_p.mean()
        rows_raw_summary.append({"percentile": p, "threshold": thr_p, "pct_ood": round(pct, 2), "n_ood": int(ood_raw_p.sum()), "n_total": len(ll_raw_all)})
        print(f"  P{p}: OOD={pct:.1f}%  ({ood_raw_p.sum()}/{len(ll_raw_all)})")

    df_raw_summary = pd.DataFrame(rows_raw_summary)
    df_raw_summary.to_csv(OUTPUT_DIR / "raw_spectra_ood_sweep_summary.csv", index=False)

    # tabla per-species
    print("\n--- Raw spectra: % OOD per species per threshold ---")
    rows_raw_sp = []
    for sp in unique_raw_species:
        mask_sp = (labels_raw_all == sp)
        n_sp = mask_sp.sum()
        if n_sp == 0:
            continue
        row = {"species": sp, "n": n_sp, "known": sp in TARGET_SPECIES}
        for p in SWEEP_PERCENTILES:
            thr_p = np.percentile(ll_train, p)
            pct = 100 * (ll_raw_all[mask_sp] < thr_p).mean()
            row[f"P{p}"] = round(pct, 1)
        rows_raw_sp.append(row)

    df_raw_per_species = pd.DataFrame(rows_raw_sp).set_index("species")
    print(df_raw_per_species.to_string())
    df_raw_per_species.to_csv(OUTPUT_DIR / "raw_spectra_ood_per_species.csv")


############################################################
# CONFUSION MATRICES (percentil de referencia)
############################################################
print("\n===== CONFUSION MATRICES =====")

for ds_name, Z_te, y_te in [
    ("Test-source", Z_test,  y_test),
    ("DRIAMS-D",    Z_D,     labels_D),
    ("MS-UMG",      Z_msumg, labels_msumg),
]:
    y_te_enc = le.transform(y_te)
    loader   = make_loader(Z_te, y_te_enc)

    metrics = metrics_report_mlp(
        loader,
        mlp_lat,
        f"novelty-{ds_name}-latent",
        device=device,
        class_names=le.classes_,
    )

    print(f"\n{ds_name}")
    print(f"  Balanced Accuracy: {metrics['Balanced_Accuracy']:.3f}")
    print(f"  F1 Macro:          {metrics['F1_Macro']:.3f}")
    print(f"  ROC AUC Macro:     {metrics['ROC_AUC_Macro']:.3f}")

    cm = metrics["Confusion Matrix"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels_short = [s.split("_")[0] for s in le.classes_]
    ax.set_xticks(range(len(labels_short)))
    ax.set_yticks(range(len(labels_short)))
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels_short, fontsize=9)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(f"Confusion Matrix — {ds_name} (latent space)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fname = f"cm_{ds_name.replace('-','_')}_latent.png"
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {OUTPUT_DIR / fname}")


############################################################
# COVERAGE VS METRICS — curva continua
############################################################
print("\n===== COVERAGE VS ACCURACY =====")

preds_D = get_predictions(mlp_lat, Z_D, device)
preds_msumg = get_predictions(mlp_lat, Z_msumg, device)
y_D_enc = le.transform(labels_D)
y_msumg_enc = le.transform(labels_msumg)

probs_D = get_probs(mlp_lat, Z_D, device)
probs_msumg = get_probs(mlp_lat, Z_msumg, device)

percentiles_curve = np.array(SWEEP_PERCENTILES)
thresholds_range = np.percentile(ll_train, percentiles_curve)

metric_names = ["BA", "F1", "Precision", "Recall", "Specificity", "ROC AUC"]
colors_metrics = ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

for ax, ds_name, ll, y_enc, preds, probs in [
    (axes[0], "DRIAMS-D", ll_D,     y_D_enc,     preds_D,     probs_D),
    (axes[1], "MS-UMG",   ll_msumg, y_msumg_enc, preds_msumg, probs_msumg),
]:
    coverages = []
    metrics_all = {k: [] for k in metric_names}

    for thr in thresholds_range:
        accepted = (ll >= thr)
        coverage = 100 * accepted.mean()
        coverages.append(coverage)

        y_true = y_enc[accepted]
        y_pred = preds[accepted]
        y_prob = probs[accepted]
        n_classes = len(le.classes_)

        if accepted.sum() > 10 and len(np.unique(y_true)) >= 2:
            ba = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

            specs = []
            for c in range(n_classes):
                tn = ((y_true != c) & (y_pred != c)).sum()
                fp = ((y_true != c) & (y_pred == c)).sum()
                specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            spec = np.nanmean(specs)

            try:
                classes_present = np.unique(y_true)
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                y_bin_p = y_bin[:, classes_present]
                y_prob_p = y_prob[:, classes_present]
                roc = roc_auc_score(y_bin_p, y_prob_p, average="macro", multi_class="ovr")
            except Exception:
                roc = np.nan
        else:
            ba = f1 = prec = rec = spec = roc = np.nan

        metrics_all["BA"].append(ba)
        metrics_all["F1"].append(f1)
        metrics_all["Precision"].append(prec)
        metrics_all["Recall"].append(rec)
        metrics_all["Specificity"].append(spec)
        metrics_all["ROC AUC"].append(roc)

    for metric, color in zip(metric_names, colors_metrics):
        vals = np.array(metrics_all[metric], dtype=float)
        mask = ~np.isnan(vals)
        ax.plot(np.array(coverages)[mask], vals[mask], color=color, lw=1.8, label=metric)

    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Coverage vs Metrics — {ds_name}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(100, 0)
    if ds_name == "MS-UMG":
        ax.set_ylim(0.9, 1.05)
    else:
        ax.set_ylim(0.8, 1.05)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "coverage_vs_metrics.png", dpi=150, bbox_inches="tight")
plt.show()

############################################################
# COVERAGE VS METRICS — tabla con sweep de percentiles
############################################################
for ds_name, ll, y_enc, preds, probs in [
    ("DRIAMS-D", ll_D,     y_D_enc,     preds_D,     probs_D),
    ("MS-UMG",   ll_msumg, y_msumg_enc, preds_msumg, probs_msumg),
]:
    n_total = len(ll)
    rows = []
    for p in SWEEP_PERCENTILES:
        thr = np.percentile(ll_train, p)
        accepted = (ll >= thr)
        n_kept = int(accepted.sum())
        n_excluded = int((~accepted).sum())
        cov = 100 * accepted.mean()
        y_true = y_enc[accepted]
        y_pred = preds[accepted]
        y_prob = probs[accepted]
        n_classes = len(le.classes_)

        if accepted.sum() > 10 and len(np.unique(y_true)) >= 2:
            ba = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            specs = []
            for c in range(n_classes):
                tn = ((y_true != c) & (y_pred != c)).sum()
                fp = ((y_true != c) & (y_pred == c)).sum()
                specs.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            spec = np.nanmean(specs)
            try:
                classes_present = np.unique(y_true)
                y_bin = label_binarize(y_true, classes=list(range(n_classes)))
                y_bin_p = y_bin[:, classes_present]
                y_prob_p = y_prob[:, classes_present]
                roc = roc_auc_score(y_bin_p, y_prob_p, average="macro", multi_class="ovr")
            except Exception:
                roc = np.nan
        else:
            ba = f1 = prec = rec = spec = roc = np.nan

        rows.append({
            "percentile": p, "kept": n_kept, "excluded": n_excluded, "total": n_total,
            "coverage_%": round(cov, 1),
            "BA": round(ba, 3) if not np.isnan(ba) else ba,
            "F1": round(f1, 3) if not np.isnan(f1) else f1,
            "Prec": round(prec, 3) if not np.isnan(prec) else prec,
            "Recall": round(rec, 3) if not np.isnan(rec) else rec,
            "Spec": round(spec, 3) if not np.isnan(spec) else spec,
            "ROC_AUC": round(roc, 3) if not np.isnan(roc) else roc,
        })

    df_sweep = pd.DataFrame(rows).set_index("percentile")
    print(f"\n{ds_name} (n_total={n_total}):")
    print(df_sweep.to_string())
    df_sweep.to_csv(OUTPUT_DIR / f"coverage_sweep_{ds_name.replace('-','_')}.csv")


############################################################
# Plots
############################################################
if RUN_PLOTS:
    print("\n===== t-SNE VISUALIZATION =====")

    for ds_name, Z_ood, y_ood, ood_flags in [
        ("Test-source", Z_test, y_test,     ood_test),
        ("DRIAMS-D", Z_D,     labels_D,     ood_D),
        ("MS-UMG",   Z_msumg, labels_msumg, ood_msumg),
    ]:
        print(f"\nFitting t-SNE for {ds_name} "
            f"({len(Z_train)} train + {len(Z_ood)} OOD = {len(Z_train)+len(Z_ood)} points)...")

        Z_all = np.vstack([Z_train, Z_ood])
        y_all = np.concatenate([y_train, y_ood])

        tsne = TSNE(n_components=2)
        Z_2d = tsne.fit_transform(Z_all)

        Z_2d_train = Z_2d[:len(Z_train)]
        Z_2d_ood   = Z_2d[len(Z_train):]

        y_ood_enc   = le.transform(y_ood)
        preds_lat   = get_predictions(mlp_lat, Z_ood, device)
        correct_lat = (preds_lat == y_ood_enc)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        fig.suptitle(f"t-SNE Latent Space — {ds_name}", fontsize=13, fontweight="bold")

        for sp in TARGET_SPECIES:
            mask = (y_train == sp)
            ax.scatter(
                Z_2d_train[mask, 0], Z_2d_train[mask, 1],
                c=SPECIES_COLORS[sp], alpha=0.08, s=4,
                linewidths=0, zorder=1,
            )

        for sp in TARGET_SPECIES:
            mask_sp = (y_train == sp)
            if mask_sp.sum() < 3:
                continue

            pts  = Z_2d_train[mask_sp]
            mean = pts.mean(axis=0)
            cov  = np.cov(pts.T)

            vals, vecs = np.linalg.eigh(cov)
            order  = vals.argsort()[::-1]
            vals   = vals[order]
            vecs   = vecs[:, order]

            angle  = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            width  = 2 * np.sqrt(vals[0])
            height = 2 * np.sqrt(vals[1])

            for n_std, lw, ls in [(1, 2.0, "-"), (2, 1.2, "--")]:
                ellipse = Ellipse(
                    xy=mean,
                    width=n_std * width,
                    height=n_std * height,
                    angle=angle,
                    edgecolor=SPECIES_COLORS[sp],
                    facecolor="none",
                    linewidth=lw,
                    linestyle=ls,
                    zorder=4,
                )
                ax.add_patch(ellipse)

            ax.scatter(
                *mean,
                c=SPECIES_COLORS[sp],
                s=80,
                edgecolors="white",
                linewidths=1.5,
                zorder=5,
            )

        for sp in TARGET_SPECIES:
            mask = (y_ood == sp) & ood_flags & correct_lat
            if mask.sum() > 0:
                ax.scatter(
                    Z_2d_ood[mask, 0], Z_2d_ood[mask, 1],
                    c="black", marker="^",
                    s=40, alpha=0.7, linewidths=0.5,
                    edgecolors="white", zorder=8,
                )

        for sp in TARGET_SPECIES:
            mask = (y_ood == sp) & ~ood_flags & correct_lat
            if mask.sum() > 0:
                ax.scatter(
                    Z_2d_ood[mask, 0], Z_2d_ood[mask, 1],
                    c=SPECIES_COLORS[sp], marker="o",
                    s=35, alpha=1.0, linewidths=0.8,
                    edgecolors="white", zorder=7,
                )

        mask_wrong = ~correct_lat
        if mask_wrong.sum() > 0:
            ax.scatter(
                Z_2d_ood[mask_wrong, 0], Z_2d_ood[mask_wrong, 1],
                c="#EF4444", marker="X",
                s=80, alpha=0.9, linewidths=0.3,
                edgecolors="white", zorder=9,
            )

        legend_patches = [
            plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor=SPECIES_COLORS[sp], markersize=8,
                    label=sp.replace("_", " "))
            for sp in TARGET_SPECIES
        ]
        legend_patches += [
            plt.Line2D([0], [0], marker="o", color="w",
                    markerfacecolor="grey", markersize=8, label="In-dist correct"),
            plt.Line2D([0], [0], marker="^", color="w",
                    markerfacecolor="black", markersize=8, label=f"OOD correct (P{REFERENCE_PERCENTILE})"),
            plt.Line2D([0], [0], marker="X", color="w",
                    markerfacecolor="#EF4444", markersize=8, label="Misclassified"),
            plt.Line2D([0], [0], linestyle="-", color="grey",
                    linewidth=2, alpha=0.5, label="Gaussian cluster (1/2 std)"),
        ]
        ax.legend(handles=legend_patches, fontsize=8,
                loc="upper right", framealpha=0.8)

        plt.tight_layout()
        fname = f"tsne_{ds_name.replace('-','_').replace(' ','_')}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {OUTPUT_DIR / fname}")

print(f"\n===== DONE — results in {OUTPUT_DIR} =====")
