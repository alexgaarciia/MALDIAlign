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
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_rki, load_msumg
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.reconstruction_error import *
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


############################################################
# CONFIG
############################################################
cfg = load_config()

TARGET_SPECIES = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex",
]

EXPERIMENT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior/20260409_100009")
MODEL_PATH = EXPERIMENT_DIR / "model.pth"
SPLITS_PATH = EXPERIMENT_DIR / "data_splits.pkl"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/novelty_detection/reconstruction") / timestamp
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DOMAIN_IDS = {"DRIAMS_A": 0, "DRIAMS_B": 1, "DRIAMS_C": 2, "MARISMA": 3, "RKI": 4}
ID_TO_DOMAIN = {v: k for k, v in DOMAIN_IDS.items()}
RUN_RAW = False


############################################################
# LOAD MODEL
############################################################
print("\n===== LOADING MODEL =====")
state = torch.load(MODEL_PATH, map_location="cpu")
decoder_indices = {int(k.split(".")[2]) for k in state if k.startswith("decoder.net.")}
num_domains = max(decoder_indices) + 1

model = MultiVAE_Bernoulli_SpeciesPrior_Extended(
    input_dim=6000, latent_dim=64,
    num_domains=num_domains, n_species=len(TARGET_SPECIES),
)
model.load_state_dict(state)
model.to(device).eval()
print(f"Model loaded — num_domains={num_domains}")


############################################################
# LOAD DATA
############################################################
print("\n===== LOADING DATA =====")

driams_dict = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict = load_rki(cfg["data"]["RKI_FULL"])

mask_driams = np.isin(driams_dict["label"], TARGET_SPECIES)
mask_marisma = np.isin(marisma_dict["label"], TARGET_SPECIES)
mask_rki = np.isin(rki_dict["label"], TARGET_SPECIES)

data_driams = driams_dict["data"][mask_driams]
label_driams = driams_dict["label"][mask_driams]
meta_driams = driams_dict["meta"][mask_driams].reset_index(drop=True)

data_marisma = marisma_dict["data"][mask_marisma]
label_marisma = marisma_dict["label"][mask_marisma]

data_rki = rki_dict["data"][mask_rki]
label_rki = rki_dict["label"][mask_rki]

maskA = meta_driams["hospital"] == "DRIAMS_A"
maskB = meta_driams["hospital"] == "DRIAMS_B"
maskC = meta_driams["hospital"] == "DRIAMS_C"
maskD = meta_driams["hospital"] == "DRIAMS_D"

dataA = row_minmax_normalize(data_driams[maskA])
labelA = label_driams[maskA]
dataB = row_minmax_normalize(data_driams[maskB])
labelB = label_driams[maskB]
dataC = row_minmax_normalize(data_driams[maskC])
labelC = label_driams[maskC]
dataD = row_minmax_normalize(data_driams[maskD])
labelD = label_driams[maskD]

data_marisma = row_minmax_normalize(data_marisma)
data_rki = row_minmax_normalize(data_rki)

data_final = np.vstack([dataA, dataB, dataC, data_marisma, data_rki])
label_final = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki])
print(f"data_final: {data_final.shape}")

msumg_dict = load_msumg(cfg["data"]["MSUMG_FULL"])
mask_msumg = np.isin(msumg_dict["label"], TARGET_SPECIES)
data_msumg = row_minmax_normalize(msumg_dict["data"][mask_msumg])
labels_msumg = msumg_dict["label"][mask_msumg]
labels_D = labelD

with open(SPLITS_PATH, "rb") as f:
    splits = pickle.load(f)

domain_splits = splits.get("splits_per_domain", {})
SPLIT_NAMES = {"A": "DRIAMS_A", "B": "DRIAMS_B", "C": "DRIAMS_C", "MARISMA": "MARISMA", "RKI": "RKI"}

print(f"DRIAMS-D: {len(dataD)} | MS-UMG: {len(data_msumg)}")


############################################################
# STEP 1 — FIT LOG-NORMAL + UMBRAL POR DECODER
############################################################
print("\n===== FITTING LOG-NORMAL PER DECODER =====")

domain_lognormals = {}
errors_per_domain = {}
thr_per_domain = {}

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


############################################################
# PRECALCULAR SCORES OOD — una sola vez por decoder
############################################################
scores_test_by_decoder = {}
scores_D_by_decoder = {}
scores_msumg_by_decoder = {}
errors_D_by_decoder = {}
errors_msumg_by_decoder = {}

for k, sk in SPLIT_NAMES.items():
    domain_id = DOMAIN_IDS[sk]
    mu_log, sig_log = domain_lognormals[domain_id]

    te_idx = domain_splits[sk].get("test_idx", np.array([], dtype=int))
    if len(te_idx) > 0:
        errs_te = reconstruction_error(model, data_final[te_idx], domain_id, device)
        scores_test_by_decoder[k] = recon_ood_score(errs_te, mu_log, sig_log)

    errs_D = reconstruction_error(model, dataD, domain_id, device)
    errs_msumg = reconstruction_error(model, data_msumg, domain_id, device)

    scores_D_by_decoder[k] = recon_ood_score(errs_D, mu_log, sig_log)
    scores_msumg_by_decoder[k] = recon_ood_score(errs_msumg, mu_log, sig_log)
    errors_D_by_decoder[k] = errs_D
    errors_msumg_by_decoder[k] = errs_msumg


############################################################
# THRESHOLD SWEEP
############################################################
SWEEP_PERCENTILES = [99, 99.5, 99.9]
colors_domain = {
    "DRIAMS_A": "#90CAF9", "DRIAMS_B": "#A5D6A7", "DRIAMS_C": "#FFCC80",
    "MARISMA": "#F28B82", "RKI": "#B39DDB",
}

# precalcular umbrales para todos los percentiles
thr_sweep_all = {}
for p in SWEEP_PERCENTILES:
    thr_sweep_all[p] = {}
    for k, sk in SPLIT_NAMES.items():
        domain_id = DOMAIN_IDS[sk]
        scores_train = recon_ood_score(errors_per_domain[k], *domain_lognormals[domain_id])
        thr_sweep_all[p][domain_id] = np.percentile(scores_train, p)


############################################################
# SECCIÓN 1 — VALIDATION TEST SOURCE
############################################################
print("\n===== VALIDATION — TEST SOURCE =====")
print("Each domain's test samples passed through their own decoder. OOD% per threshold percentile.\n")

rows = []
for k, sk in SPLIT_NAMES.items():
    if k not in scores_test_by_decoder:
        continue
    domain_id = DOMAIN_IDS[sk]
    row = {"domain": sk}
    for p in SWEEP_PERCENTILES:
        pct = 100 * (scores_test_by_decoder[k] > thr_sweep_all[p][domain_id]).mean()
        row[f"P{p}"] = f"{pct:.1f}%"
    rows.append(row)
df_val = pd.DataFrame(rows).set_index("domain")
print(df_val.to_string())
df_val.to_csv(OUTPUT_DIR / "sweep_validation_test_source_all_percentiles.csv")


############################################################
# SECCIÓN 2 — MULTI-DECODER VALIDATION
############################################################
print("\n===== MULTI-DECODER VALIDATION =====")
print("% OOD when passing each source test domain through each decoder, per threshold percentile.\n")

for p in SWEEP_PERCENTILES:
    print(f"\n--- P{p} ---")
    cross_sweep = pd.DataFrame(index=SPLIT_NAMES.keys(), columns=SPLIT_NAMES.keys(), dtype=float)
    for k_data, sk_data in SPLIT_NAMES.items():
        te_idx = domain_splits[sk_data].get("test_idx", np.array([], dtype=int))
        if len(te_idx) == 0:
            continue
        X_data = data_final[te_idx]
        for k_dec, sk_dec in SPLIT_NAMES.items():
            domain_id = DOMAIN_IDS[sk_dec]
            errs = reconstruction_error(model, X_data, domain_id, device)
            mu_log, sig_log = domain_lognormals[domain_id]
            ood_pct = 100 * (recon_ood_score(errs, mu_log, sig_log) > thr_sweep_all[p][domain_id]).mean()
            cross_sweep.loc[k_data, k_dec] = round(ood_pct, 1)
    print(cross_sweep.to_string())
    cross_sweep.to_csv(OUTPUT_DIR / f"sweep_cross_domain_p{str(p).replace('.','')}.csv")


############################################################
# SECCIÓN 3 — PER-SPECIES OOD PER DECODER
############################################################
for ds_name, scores_by_dec, labels in [
    ("DRIAMS-D", scores_D_by_decoder, labels_D),
    ("MS-UMG", scores_msumg_by_decoder, labels_msumg),
]:
    print(f"\n===== {ds_name} — % OOD per species per decoder (all percentiles) =====")
    for k, sk in SPLIT_NAMES.items():
        domain_id = DOMAIN_IDS[sk]
        print(f"\n--- decoder {sk} ---")
        rows = []
        for sp in TARGET_SPECIES:
            mask = (labels == sp)
            if mask.sum() == 0:
                continue
            row = {"species": sp.split("_")[0], "n": mask.sum()}
            for p in SWEEP_PERCENTILES:
                pct = 100 * (scores_by_dec[k][mask] > thr_sweep_all[p][domain_id]).mean()
                row[f"P{p}"] = f"{pct:.1f}%"
            rows.append(row)
        df = pd.DataFrame(rows).set_index("species")
        print(df.to_string())
        df.to_csv(OUTPUT_DIR / f"sweep_per_species_{ds_name.replace('-','_')}_dec_{sk}.csv")


############################################################
# SECCIÓN 4 — RELIABLE DECODERS ONLY
############################################################
print("\n===== DECODER A vs MARISMA — RELIABLE DECODERS ONLY =====")
RELIABLE_DECODERS = {"A": "DRIAMS_A", "MARISMA": "MARISMA"}

for ds_name, scores_by_dec, labels in [
    ("DRIAMS-D", scores_D_by_decoder, labels_D),
    ("MS-UMG",   scores_msumg_by_decoder, labels_msumg),
]:
    print(f"\n{ds_name}:")
    rows = []
    for p in SWEEP_PERCENTILES:
        row = {"percentile": f"P{p}"}
        for k, sk in RELIABLE_DECODERS.items():
            domain_id = DOMAIN_IDS[sk]
            pct = 100 * (scores_by_dec[k] > thr_sweep_all[p][domain_id]).mean()
            row[f"dec_{k}"] = f"{pct:.1f}%"
        rows.append(row)
    df_summary = pd.DataFrame(rows).set_index("percentile")
    print(df_summary.to_string())

    print(f"\n  {ds_name} — per species:")
    for sp in TARGET_SPECIES:
        mask = (labels == sp)
        if mask.sum() == 0:
            continue
        sp_short = sp.split("_")[0]
        print(f"\n  --- {sp_short} ---")
        rows = []
        for p in SWEEP_PERCENTILES:
            row = {"percentile": f"P{p}"}
            for k, sk in RELIABLE_DECODERS.items():
                domain_id = DOMAIN_IDS[sk]
                pct = 100 * (scores_by_dec[k][mask] > thr_sweep_all[p][domain_id]).mean()
                row[f"dec_{k}"] = f"{pct:.1f}%"
            rows.append(row)
        df_sp = pd.DataFrame(rows).set_index("percentile")
        print(df_sp.to_string())


############################################################
# SECCIÓN 5 — UNSEEN SPECIES (reconstruction error)
############################################################
print("\n===== UNSEEN SPECIES — RECONSTRUCTION ERROR =====")

driams_dict_all  = load_driams(cfg["data"]["DRIAMS_FULL"])
marisma_dict_all = load_marisma(cfg["data"]["MARISMa_FULL"])
rki_dict_all     = load_rki(cfg["data"]["RKI_FULL"])
msumg_dict_all   = load_msumg(cfg["data"]["MSUMG_FULL"])

unseen_data_list   = []
unseen_label_list  = []
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
    print(f"  {sp}: n={(labels_unseen == sp).sum()}")

# error de reconstrucción y score OOD con cada decoder
errors_unseen_by_decoder = {}
scores_unseen_by_decoder = {}
for k, sk in SPLIT_NAMES.items():
    domain_id = DOMAIN_IDS[sk]
    mu_log, sig_log = domain_lognormals[domain_id]
    errs_unseen = reconstruction_error(model, data_unseen, domain_id, device)
    errors_unseen_by_decoder[k] = errs_unseen
    scores_unseen_by_decoder[k] = recon_ood_score(errs_unseen, mu_log, sig_log)

# tabla global — % OOD por decoder a cada percentil
print("\n--- Unseen species: % OOD per decoder, all percentiles ---")
rows = []
for k, sk in SPLIT_NAMES.items():
    domain_id = DOMAIN_IDS[sk]
    row = {"decoder": sk}
    for p in SWEEP_PERCENTILES:
        pct = 100 * (scores_unseen_by_decoder[k] > thr_sweep_all[p][domain_id]).mean()
        row[f"P{p}"] = f"{pct:.1f}%"
    rows.append(row)
df_unseen_dec = pd.DataFrame(rows).set_index("decoder")
print(df_unseen_dec.to_string())
df_unseen_dec.to_csv(OUTPUT_DIR / "unseen_species_ood_per_decoder.csv")

# tabla per-species — todos los decoders, desglosado por fuente
print("\n--- Unseen species: % OOD per species, by source, all decoders ---")
for k, sk in SPLIT_NAMES.items():
    domain_id = DOMAIN_IDS[sk]
    print(f"\n  --- decoder {sk} ---")
    rows_sp = []
    for sp in unique_unseen_species:
        for src in np.unique(source_unseen):
            mask_sp = (labels_unseen == sp) & (source_unseen == src)
            n_sp = mask_sp.sum()
            if n_sp == 0:
                continue
            row = {"species": sp, "source": src, "n": n_sp}
            for p in SWEEP_PERCENTILES:
                pct = 100 * (scores_unseen_by_decoder[k][mask_sp] > thr_sweep_all[p][domain_id]).mean()
                row[f"P{p}"] = round(pct, 1)
            rows_sp.append(row)
    df_sp_dec = pd.DataFrame(rows_sp).set_index(["species", "source"])
    print(df_sp_dec.to_string())
    df_sp_dec.to_csv(OUTPUT_DIR / f"unseen_species_ood_per_species_by_source_dec_{sk}.csv")


############################################################
# SECCIÓN 6 — RAW SPECTRA (todas las especies, solo binning)
############################################################
if RUN_RAW:
    print("\n===== RAW SPECTRA — RECONSTRUCTION ERROR (ALL SPECIES) =====")

    RAW_PKL = {
        "DRIAMS": cfg["data"]["DRIAMS_FULL_RAW"],
        "MARISMA": cfg["data"]["MARISMa_FULL_RAW"],
        "RKI": cfg["data"]["RKI_FULL_RAW"],
        "MS-UMG": cfg["data"]["MSUMG_FULL_RAW"],
    }

    raw_data_list  = []
    raw_label_list = []

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

    data_raw_all   = np.vstack(raw_data_list)
    labels_raw_all = np.concatenate(raw_label_list)
    unique_raw_species = np.unique(labels_raw_all)

    print(f"Found {len(unique_raw_species)} species in raw pkl, n={len(data_raw_all)} samples")
    for sp in unique_raw_species:
        print(f"  {sp}: n={(labels_raw_all == sp).sum()}")

    # error de reconstrucción y score OOD con cada decoder
    errors_raw_by_decoder = {}
    scores_raw_by_decoder = {}
    for k, sk in SPLIT_NAMES.items():
        domain_id = DOMAIN_IDS[sk]
        mu_log, sig_log = domain_lognormals[domain_id]
        errs_raw = reconstruction_error(model, data_raw_all, domain_id, device)
        errors_raw_by_decoder[k] = errs_raw
        scores_raw_by_decoder[k] = recon_ood_score(errs_raw, mu_log, sig_log)

    # tabla global
    print("\n--- Raw spectra: % OOD per decoder, all percentiles ---")
    rows = []
    for k, sk in SPLIT_NAMES.items():
        domain_id = DOMAIN_IDS[sk]
        row = {"decoder": sk}
        for p in SWEEP_PERCENTILES:
            pct = 100 * (scores_raw_by_decoder[k] > thr_sweep_all[p][domain_id]).mean()
            row[f"P{p}"] = f"{pct:.1f}%"
        rows.append(row)
    df_raw_dec = pd.DataFrame(rows).set_index("decoder")
    print(df_raw_dec.to_string())
    df_raw_dec.to_csv(OUTPUT_DIR / "raw_spectra_ood_per_decoder.csv")

    # tabla per-species
    print("\n--- Raw spectra: % OOD per species, all decoders (split by decoder) ---")
    for k, sk in SPLIT_NAMES.items():
        domain_id = DOMAIN_IDS[sk]
        print(f"\n  --- decoder {sk} ---")
        rows_sp = []
        for sp in unique_raw_species:
            mask_sp = (labels_raw_all == sp)
            n_sp = mask_sp.sum()
            if n_sp == 0:
                continue
            row = {"species": sp, "n": n_sp, "known": sp in TARGET_SPECIES}
            for p in SWEEP_PERCENTILES:
                pct = 100 * (scores_raw_by_decoder[k][mask_sp] > thr_sweep_all[p][domain_id]).mean()
                row[f"P{p}"] = round(pct, 1)
            rows_sp.append(row)
        df_sp_dec = pd.DataFrame(rows_sp).set_index("species")
        print(df_sp_dec.to_string())
        df_sp_dec.to_csv(OUTPUT_DIR / f"raw_spectra_ood_per_species_dec_{sk}.csv")


############################################################
# PLOTS DEL SWEEP
############################################################
for ds_name, errors_by_dec, ood_color in [
    ("DRIAMS-D", errors_D_by_decoder, "#3B82F6"),
    ("MS-UMG", errors_msumg_by_decoder, "#EF4444"),
]:
    for p in SWEEP_PERCENTILES:
        fig, axes = plt.subplots(1, num_domains, figsize=(4 * num_domains, 4), sharey=False)
        for ax, (k, sk) in zip(axes, SPLIT_NAMES.items()):
            domain_id = DOMAIN_IDS[sk]
            mu_log, sig_log = domain_lognormals[domain_id]
            color_train = colors_domain[sk]
            errs_train = errors_per_domain[k]
            errs_ood = errors_by_dec[k]
            ax.hist(errs_train, bins=50, density=True, alpha=0.5, color=color_train, label="Train")
            x_range = np.linspace(errs_train.min(), errs_train.max(), 300)
            ax.plot(x_range, lognorm.pdf(x_range, s=sig_log, scale=np.exp(mu_log)), color=color_train, lw=2.0)
            ax.axvline(np.percentile(errs_train, p), color="black", ls="--", lw=1.2, label=f"P{p} threshold")
            ax.hist(errs_ood, bins=50, density=True, alpha=0.4, color=ood_color, label=ds_name)
            ax.set_title(sk.replace("DRIAMS_", ""), fontsize=10, fontweight="bold")
            ax.set_xlabel("Reconstruction MSE")
            ax.grid(True, linestyle=":", alpha=0.4)
        axes[0].set_ylabel("Density")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Reconstruction Error — {ds_name} vs Train per Decoder (P{p})", fontsize=11, fontweight="bold")
        plt.tight_layout()
        fname = f"sweep_p{str(p).replace('.','')}_recon_ood_vs_train_{ds_name.replace('-','_')}.png"
        plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.show()


############################################################
# PLOTS FIJOS (sin sweep)
############################################################
# PLOT 1 — distribución de train en escala log
fig, axes = plt.subplots(1, num_domains, figsize=(4 * num_domains, 4), sharey=False)
for ax, (k, sk) in zip(axes, SPLIT_NAMES.items()):
    errs = errors_per_domain[k]
    log_errs = np.log(errs + 1e-10)
    mu_log, sig_log = domain_lognormals[DOMAIN_IDS[sk]]
    color = colors_domain[sk]
    ax.hist(log_errs, bins=50, density=True, alpha=0.6, color=color)
    x_range = np.linspace(log_errs.min(), log_errs.max(), 300)
    ax.plot(x_range, norm.pdf(x_range, mu_log, sig_log), color=color, lw=2.5)
    ax.axvline(np.percentile(log_errs, 99), color="black", ls="--", lw=1.2, label="P99")
    ax.set_title(sk.replace("DRIAMS_", ""), fontsize=10, fontweight="bold")
    ax.set_xlabel("log(Reconstruction MSE)")
    ax.grid(True, linestyle=":", alpha=0.4)
axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8)
fig.suptitle("Train Reconstruction Error per Decoder — log scale", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recon_train_distribution_log.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 2 — distribución de train en escala original
fig, axes = plt.subplots(1, num_domains, figsize=(4 * num_domains, 4), sharey=False)
for ax, (k, sk) in zip(axes, SPLIT_NAMES.items()):
    errs = errors_per_domain[k]
    mu_log, sig_log = domain_lognormals[DOMAIN_IDS[sk]]
    color = colors_domain[sk]
    ax.hist(errs, bins=50, density=True, alpha=0.6, color=color)
    x_range = np.linspace(errs.min(), errs.max(), 300)
    pdf_vals = lognorm.pdf(x_range, s=sig_log, scale=np.exp(mu_log))
    ax.plot(x_range, pdf_vals, color=color, lw=2.5)
    ax.axvline(np.percentile(errs, 99), color="black", ls="--", lw=1.2, label="P99")
    ax.set_title(sk.replace("DRIAMS_", ""), fontsize=10, fontweight="bold")
    ax.set_xlabel("Reconstruction MSE")
    ax.grid(True, linestyle=":", alpha=0.4)
axes[0].set_ylabel("Density")
axes[0].legend(fontsize=8)
fig.suptitle("Train Reconstruction Error per Decoder", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "recon_train_distribution_original.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\n===== DONE — results in {OUTPUT_DIR} =====")
