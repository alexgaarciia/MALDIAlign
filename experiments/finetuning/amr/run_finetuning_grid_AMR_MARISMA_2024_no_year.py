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
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data.io import load_pkl
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import evaluate_amr_head

from experiments.finetuning.amr.run_finetuning_amr import run_finetuning_amr, _count_species
from models.baselines.amr_mlp import SimpleAMRMLP_Extended
from models.deep.MultiVAEPriorAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended


############################################################
# CONFIGURATION
############################################################
DATASET_PATH = "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl"
TARGET_DOMAIN = "MARISMA_2024"
TARGET_HOSPITAL = "MARISMA"
TARGET_YEAR = "2024"
SPECIES = "Klebsiella_Pneumoniae"

PRETRAINED_MODEL_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260506_232838/model.pth")
SPLITS_PATH = Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_Klebsiella_Pneumoniae_MARISMA_2024_20260506_234415")
OUTPUT_PATH = Path("/export/usuarios_ml4ds/agnavarr/MALDIAlign/finetuning_amr_kpneu")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

ANTIBIOTICS =  ["Ceftazidime", "Ciprofloxacin", "Ertapenem", "Piperacillin-Tazobactam"]

FT_MODES = ["enc_dec_amr"]

GRID_PREV = [0]
GRID_NEW  = [50, 100, 250, 500, 1000, 1500]

RUN_EVAL = False

LATENT_DIM = 128
LAMBDA_AMR = 100
NUM_DOMAINS = 4 

N_PARTITIONS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


############################################################
# LOAD DATASET
############################################################
print("\n===== LOADING DATA =====")

dataset = load_pkl(DATASET_PATH)
data_raw = dataset["data"]
amr_raw  = dataset["amr"]
ab_list_raw = list(dataset["antibiotics"])
labels_raw = dataset["label"]
meta_raw = pd.DataFrame(dataset["meta"])
data_norm = row_minmax_normalize(data_raw)

# Filter antibiotics on full dataset
keep_idx = [ab_list_raw.index(n) for n in ANTIBIOTICS if n in ab_list_raw]
ab_list  = [n for n in ANTIBIOTICS if n in ab_list_raw]
amr_aligned = amr_raw[:, keep_idx]

# Build a target-only, species-filtered sub-dataset (MARISMA 2024)
target_dom_mask = (
    (meta_raw["hospital"] == TARGET_HOSPITAL) &
    (meta_raw["year"].astype(str) == TARGET_YEAR)
).values
species_mask = (labels_raw == SPECIES)
local_mask = target_dom_mask & species_mask

data_norm_D  = data_norm[local_mask]
amr_D = amr_aligned[local_mask]
labels_D = labels_raw[local_mask]
meta_D = meta_raw.loc[local_mask].reset_index(drop=True)

print(f"\n{TARGET_DOMAIN} ({TARGET_HOSPITAL} {TARGET_YEAR}): {local_mask.sum()} samples")
for j, atb in enumerate(ab_list):
    y = amr_D[:, j]
    valid = ~np.isnan(y)
    n_r = int((y[valid] == 1).sum())
    n_s = int((y[valid] == 0).sum())
    if (n_r + n_s) > 0:
        print(f"  {atb}: S={n_s}, R={n_r}, prev={100*n_r/(n_r+n_s):.1f}%")


############################################################
# LOAD PRETRAINED MODEL
############################################################
print("\n===== LOADING PRETRAINED MODEL =====")

state = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
n_species_pretrained = _count_species(PRETRAINED_MODEL_PATH)

vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
    input_dim=data_norm.shape[1],
    latent_dim=LATENT_DIM,
    num_domains=NUM_DOMAINS,
    n_species=n_species_pretrained,
    n_antibiotics=len(ab_list),
    lambda_amr=LAMBDA_AMR,
    antibiotic_names=ab_list,
)
vae_pretrained.load_state_dict(state)
vae_pretrained.to(device).eval()


############################################################
# GRID LOOP
############################################################
results = []
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
print(f"\n{'#'*70}")
print(f"{SPECIES} | {TARGET_DOMAIN} | {N_PARTITIONS} partitions")
print(f"Grid: {GRID_NEW} | Modes: {FT_MODES}")
print(f"{'#'*70}")
 
for i_part in range(N_PARTITIONS):
    print(f"\n{'*'*70}")
    print(f" PARTITION {i_part + 1}/{N_PARTITIONS}")
    print(f"{'*'*70}")
 
    for n_prev in GRID_PREV:
        for n_new in GRID_NEW:
 
            split_file = SPLITS_PATH / f"run_{i_part}" / f"prev_{n_prev}_new_{n_new}.pkl"
            if not split_file.exists():
                print(f"Split not found: {split_file}. Skipping.")
                continue
 
            print(f"\n{'='*60}")
            print(f"Partition={i_part} | N_PREV={n_prev} | N_NEW={n_new}")
            print(f"{'='*60}")
 
            splits   = load_pkl(split_file)
            idx_ft   = splits[TARGET_DOMAIN]["finetuning"]
            idx_test = splits[TARGET_DOMAIN]["test"]
 
            X_ft   = data_norm_D[idx_ft]
            amr_ft = amr_D[idx_ft]
            X_test   = data_norm_D[idx_test]
            amr_test = amr_D[idx_test]
 
            print(f"Finetuning: {len(idx_ft)} | Test: {len(idx_test)}")
 
            # ==========================================
            # ZERO-SHOT
            # ==========================================
            if n_prev == 0:
                print("Zero-shot AMR head...")
                metrics_zero = evaluate_amr_head(vae_pretrained, X_test, amr_test, ab_list, device)
                for atb, m in metrics_zero.items():
                    results.append({
                        "partition": i_part,
                        "n_prev": n_prev,
                        "n_new": n_new,
                        "model": "AMR_head_zero_shot",
                        "antibiotic": atb,
                        "auc": m["auc"],
                        "pr_auc": m["pr_auc"],
                        "n_test": m.get("n", len(idx_test)),
                    })
                    print(f"  {atb}: AUC={m['auc']:.3f}")
 
            # ==========================================
            # FINETUNING MODES
            # ==========================================
            for mode in FT_MODES:
                print(f"Finetuning mode={mode}...")
                try:
                    consider_prev = n_prev > 0
                    vae_ft, _, _ = run_finetuning_amr(
                        splits_path=split_file,
                        target_domain=TARGET_DOMAIN,
                        pretrained_model_path=PRETRAINED_MODEL_PATH,
                        finetuning_mode=mode,
                        n_prev=n_prev,
                        n_new=n_new,
                        output_dir=OUTPUT_PATH,
                        device=device,
                        target_antibiotics=ANTIBIOTICS,
                        species=SPECIES,
                        consider_prev_domains=consider_prev,
                        lambda_amr=LAMBDA_AMR,
                        run_latent_evaluation=RUN_EVAL,
                        ft_lr=1e-5,
                    )
                    vae_ft.eval()
 
                    metrics_ft = evaluate_amr_head(vae_ft, X_test, amr_test, ab_list, device)
                    for atb, m in metrics_ft.items():
                        results.append({
                            "partition": i_part,
                            "n_prev": n_prev,
                            "n_new": n_new,
                            "model": f"FT_{mode}",
                            "antibiotic": atb,
                            "auc": m["auc"],
                            "pr_auc": m["pr_auc"],
                            "n_test": m.get("n", len(idx_test)),
                        })
                        print(f"  {atb}: AUC={m['auc']:.3f} PR={m['pr_auc']:.3f}")
 
                    del vae_ft
                    torch.cuda.empty_cache()
 
                except Exception as e:
                    print(f"ERROR in mode={mode}: {e}")
                    import traceback
                    traceback.print_exc()
 
            # ==========================================
            # MLP FROM SCRATCH
            # ==========================================
            print("MLP from scratch...")
            if len(X_ft) >= 20:
                X_tr, X_val, a_tr, a_val = train_test_split(
                    X_ft, amr_ft, test_size=0.2, random_state=i_part
                )
 
                train_loader_mlp = DataLoader(
                    TensorDataset(
                        torch.tensor(X_tr, dtype=torch.float32),
                        torch.tensor(a_tr, dtype=torch.float32),
                    ),
                    batch_size=64, shuffle=True,
                )
                val_loader_mlp = DataLoader(
                    TensorDataset(
                        torch.tensor(X_val, dtype=torch.float32),
                        torch.tensor(a_val, dtype=torch.float32),
                    ),
                    batch_size=64, shuffle=False,
                )
 
                mlp = SimpleAMRMLP_Extended(
                    input_dim=X_tr.shape[1],
                    n_antibiotics=len(ab_list),
                    latent_dim=LATENT_DIM,
                    antibiotic_names=ab_list,
                    epochs=100,
                    lr=1e-3,
                    patience=15,
                )
                mlp.trainloop(train_loader_mlp, val_loader_mlp, device)
                metrics_mlp = mlp.evaluate(X_test, amr_test, device)
 
                for atb, m in metrics_mlp.items():
                    results.append({
                        "partition": i_part,
                        "n_prev": n_prev,
                        "n_new": n_new,
                        "model": "MLP_from_scratch",
                        "antibiotic": atb,
                        "auc": m["auc"],
                        "pr_auc": m["pr_auc"],
                        "n_test": m.get("n", len(idx_test)),
                    })
                    print(f"  {atb}: AUC={m['auc']:.3f} PR={m['pr_auc']:.3f}")
 
                del mlp
                torch.cuda.empty_cache()
            else:
                print("Too few samples for MLP.")
 
            pd.DataFrame(results).to_csv(
                OUTPUT_PATH / f"partial_results_{timestamp}.csv", index=False
            )
 
 
############################################################
# SAVE FINAL RESULTS
############################################################
df_results = pd.DataFrame(results)
csv_path = OUTPUT_PATH / f"amr_finetuning_{SPECIES}_{TARGET_DOMAIN}_{timestamp}.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nFull results saved: {csv_path}")
 
 
############################################################
# SUMMARY: MEDIA Y STD POR PARTICION
############################################################
print(f"\n{'='*70}")
print("SUMMARY (mean ± std across partitions)")
print(f"{'='*70}")

if len(df_results) > 0:
    col_order = (
        ["AMR_head_zero_shot"]
        + [f"FT_{m}" for m in FT_MODES]
        + ["MLP_from_scratch"]
    )

    per_partition = (
        df_results
        .groupby(["partition", "n_prev", "n_new", "model"])["auc"]
        .mean()
        .reset_index()
    )

    summary = (
        per_partition
        .groupby(["n_prev", "n_new", "model"])["auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["mean_std"] = (
        summary["mean"].round(3).astype(str) + " ± " +
        summary["std"].round(3).astype(str)
    )
    summary.to_csv(OUTPUT_PATH / f"summary_global_{timestamp}.csv", index=False)

    pivot = summary.pivot_table(
        index=["n_prev", "n_new"], columns="model", values="mean_std", aggfunc="first"
    )
    existing_cols = [c for c in col_order if c in pivot.columns]
    print("\nMean ± Std AUC (across partitions):")
    print(pivot[existing_cols].to_string())

    for atb in ab_list:
        sub = df_results[df_results["antibiotic"] == atb]
        if len(sub) == 0:
            continue
        print(f"\n--- {atb} ---")

        per_partition_atb = (
            sub
            .groupby(["partition", "n_prev", "n_new", "model"])["auc"]
            .mean()
            .reset_index()
        )
        atb_summary = (
            per_partition_atb
            .groupby(["n_prev", "n_new", "model"])["auc"]
            .agg(["mean", "std"])
            .reset_index()
        )
        atb_summary["mean_std"] = (
            atb_summary["mean"].round(3).astype(str) + " ± " +
            atb_summary["std"].round(3).astype(str)
        )
        atb_summary.to_csv(OUTPUT_PATH / f"summary_{atb}_{timestamp}.csv", index=False)

        pivot_atb = atb_summary.pivot_table(
            index=["n_prev", "n_new"], columns="model", values="mean_std", aggfunc="first"
        )
        existing = [c for c in col_order if c in pivot_atb.columns]
        print(pivot_atb[existing].to_string())

print(f"\n===== DONE =====")
