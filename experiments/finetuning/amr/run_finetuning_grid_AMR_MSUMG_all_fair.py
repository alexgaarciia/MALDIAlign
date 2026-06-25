############################################################
# PATH CONFIGURATION
############################################################
from pathlib import Path
import os
import sys
import pickle
import copy

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
SOURCE_DOMAINS = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
TARGET_DOMAIN  = "MS-UMG"
OUTPUT_PATH    = Path("/export/usuarios_ml4ds/agnavarr/MALDIAlign/finetuning_amr_all_species_msumg")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

FT_MODES     = ["enc_dec_amr"]
GRID_PREV    = [0]
GRID_NEW     = [50, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000]
N_PARTITIONS = 10
RUN_EVAL     = False
LATENT_DIM   = 128
LAMBDA_AMR   = 100
NUM_DOMAINS  = 4
MLP_POOLED_FT_LR = 1e-5
MLP_POOLED_FT_EPOCHS = 50    

SPECIES_CONFIG = {
    "Klebsiella_Pneumoniae": {
        "antibiotics":  ["Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl",
        "model_path":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260514_101045/model.pth"),
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260514_101045/data_splits.pkl"),
        "splits_dir":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_KlebsiellaPneumoniae_MSUMG_20260519_120221"),
    },
    "Escherichia_Coli": {
        "antibiotics":  ["Ceftazidime", "Ciprofloxacin", "Ampicillin", "Piperacillin-Tazobactam"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl",
        "model_path":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260513_152147/model.pth"),
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260513_152147/data_splits.pkl"),
        "splits_dir":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_EscherichiaColi_MSUMG_20260519_120221"),
    },
    "Staphylococcus_Aureus": {
        "antibiotics":  ["Oxacillin", "Clindamycin", "Erythromycin"],
        "dataset_path": "/export/usuarios01/agnavarr/MALDIAlign/amr_global_v2.pkl",
        "model_path":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260516_160321/model.pth"),
        "splits_path":  Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/results/vae_multidecoder_prior_multiamr/20260516_160321/data_splits.pkl"),
        "splits_dir":   Path("/export/usuarios01/agnavarr/MALDIAlign/experiments/finetuning/amr/output_data/splits_StaphylococcusAureus_MSUMG_20260519_120221"),
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


############################################################
# MAIN LOOP OVER SPECIES
############################################################
all_results = []
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")

for species, cfg in SPECIES_CONFIG.items():
    antibiotics  = cfg["antibiotics"]
    model_path   = cfg["model_path"]
    splits_dir   = cfg["splits_dir"]
    dataset_path = cfg["dataset_path"]

    print(f"\n{'#'*70}")
    print(f"  SPECIES: {species}")
    print(f"  Antibiotics: {antibiotics}")
    print(f"  Dataset: {dataset_path}")
    print(f"{'#'*70}")

    # ----------------------------------------------------------
    # Load dataset for this species
    # ----------------------------------------------------------
    dataset     = load_pkl(dataset_path)
    data_raw    = dataset["data"]
    amr_raw     = dataset["amr"]
    ab_list_raw = list(dataset["antibiotics"])
    labels_raw  = dataset["label"]
    meta_raw    = pd.DataFrame(dataset["meta"]) if not isinstance(dataset["meta"], pd.DataFrame) \
                  else dataset["meta"]
    data_norm   = row_minmax_normalize(data_raw)

    # Chrom-agar filter
    if "agar" in meta_raw.columns:
        chrom     = ((meta_raw["hospital"] == TARGET_DOMAIN) &
                     (meta_raw["agar"] == "chrom")).values
        keep      = ~chrom
        data_norm = data_norm[keep]
        amr_raw   = amr_raw[keep]
        labels_raw = labels_raw[keep]
        meta_raw  = meta_raw[keep].reset_index(drop=True)
        print(f"Removed {chrom.sum()} chrom-agar samples from MS-UMG")

    keep_idx = [ab_list_raw.index(n) for n in antibiotics if n in ab_list_raw]
    ab_list  = [n for n in antibiotics if n in ab_list_raw]

    # ----------------------------------------------------------
    # Build target sub-dataset (MS-UMG, species filtered)
    # ----------------------------------------------------------
    local_mask = (meta_raw["hospital"] == TARGET_DOMAIN).values & (labels_raw == species)
    data_t     = data_norm[local_mask]
    amr_t      = amr_raw[:, keep_idx][local_mask]

    print(f"\nMS-UMG ({species}): {len(data_t)} samples")
    for j, atb in enumerate(ab_list):
        y     = amr_t[:, j]
        valid = ~np.isnan(y)
        n_r   = int((y[valid] == 1).sum())
        n_s   = int((y[valid] == 0).sum())
        if (n_r + n_s) > 0:
            print(f"  {atb}: S={n_s}, R={n_r}, prev={100*n_r/(n_r+n_s):.1f}%")

    # ----------------------------------------------------------
    # Load pretrained DALMA model
    # ----------------------------------------------------------
    state = torch.load(model_path, map_location="cpu")
    n_species_pretrained = _count_species(model_path)

    vae_pretrained = MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_Extended(
        input_dim    = data_norm.shape[1],
        latent_dim   = LATENT_DIM,
        num_domains  = NUM_DOMAINS,
        n_species    = n_species_pretrained,
        n_antibiotics = len(ab_list),
        lambda_amr   = LAMBDA_AMR,
        antibiotic_names = ab_list,
    )
    vae_pretrained.load_state_dict(state)
    vae_pretrained.to(device).eval()
    print(f"Pretrained DALMA model loaded.")

    # ----------------------------------------------------------
    # Train pooled MLP baseline ONCE per species
    # Trained on all source domains pooled — fair comparison vs DALMA
    # ----------------------------------------------------------
    print(f"\nTraining pooled MLP baseline on {'+'.join(SOURCE_DOMAINS)}...")

    with open(cfg["splits_path"], "rb") as f:
        vae_splits = pickle.load(f)
    domain_splits = vae_splits.get("splits_per_domain", {})

    # Filter dataset to species for indexing
    sp_mask_src = (labels_raw == species)
    data_sp     = data_norm[sp_mask_src]
    amr_sp      = amr_raw[:, keep_idx][sp_mask_src]

    all_X_tr, all_y_tr = [], []
    all_X_vl, all_y_vl = [], []

    for dom in SOURCE_DOMAINS:
        if dom not in domain_splits:
            continue
        sp = domain_splits[dom]
        tr_idx = sp["train_idx"]
        vl_idx = sp.get("val_idx", np.array([], dtype=int))
        if len(tr_idx) > 0:
            all_X_tr.append(data_sp[tr_idx])
            all_y_tr.append(amr_sp[tr_idx].astype(float))
        if len(vl_idx) > 0:
            all_X_vl.append(data_sp[vl_idx])
            all_y_vl.append(amr_sp[vl_idx].astype(float))

    X_pool_tr = np.vstack(all_X_tr)
    y_pool_tr = np.vstack(all_y_tr)
    X_pool_vl = np.vstack(all_X_vl)
    y_pool_vl = np.vstack(all_y_vl)

    print(f"  Pooled train: n={len(X_pool_tr)}")
    for j, atb in enumerate(ab_list):
        col   = y_pool_tr[:, j]
        valid = ~np.isnan(col)
        print(f"    {atb}: n={valid.sum()}, "
              f"R={(col[valid]==1).sum():.0f}, S={(col[valid]==0).sum():.0f}")

    pool_tr_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_pool_tr, dtype=torch.float32),
            torch.tensor(y_pool_tr, dtype=torch.float32),
        ),
        batch_size=256, shuffle=True
    )
    pool_vl_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_pool_vl, dtype=torch.float32),
            torch.tensor(y_pool_vl, dtype=torch.float32),
        ),
        batch_size=256, shuffle=False
    )

    mlp_pooled = SimpleAMRMLP_Extended(
        input_dim        = X_pool_tr.shape[1],
        latent_dim       = LATENT_DIM,
        n_antibiotics    = len(ab_list),
        antibiotic_names = ab_list,
        epochs=100, lr=1e-3, patience=15,
    )
    mlp_pooled.trainloop(pool_tr_loader, pool_vl_loader, device)
    mlp_pooled.eval()
    print(f"  Pooled MLP trained.")

    # ----------------------------------------------------------
    # Grid loop
    # ----------------------------------------------------------
    for i_part in range(N_PARTITIONS):
        print(f"\n{'*'*60}")
        print(f" {species} | PARTITION {i_part + 1}/{N_PARTITIONS}")
        print(f"{'*'*60}")

        for n_prev in GRID_PREV:
            for n_new in GRID_NEW:

                split_file = splits_dir / f"run_{i_part}" / f"prev_{n_prev}_new_{n_new}.pkl"
                if not split_file.exists():
                    print(f"Split not found: {split_file}. Skipping.")
                    continue

                splits   = load_pkl(split_file)
                idx_ft   = splits[TARGET_DOMAIN]["finetuning"]
                idx_test = splits[TARGET_DOMAIN]["test"]

                X_ft     = data_t[idx_ft]
                amr_ft   = amr_t[idx_ft]
                X_test   = data_t[idx_test]
                amr_test = amr_t[idx_test]

                print(f"\nPartition={i_part} | N_PREV={n_prev} | N_NEW={n_new} "
                      f"| FT={len(idx_ft)} | Test={len(idx_test)}")

                # ── ZERO-SHOT ─────────────────────────────────────
                if n_prev == 0:
                    metrics_zero = evaluate_amr_head(
                        vae_pretrained, X_test, amr_test, ab_list, device
                    )
                    for atb, m in metrics_zero.items():
                        all_results.append({
                            "species":   species,
                            "partition": i_part,
                            "n_prev":    n_prev,
                            "n_new":     n_new,
                            "model":     "AMR_head_zero_shot",
                            "antibiotic": atb,
                            "auc":       m["auc"],
                            "pr_auc":    m["pr_auc"],
                            "n_test":    m.get("n", len(idx_test)),
                        })
                        print(f"  Zero-shot {atb}: AUC={m['auc']:.3f}")

                # ── FINETUNING ────────────────────────────────────
                for mode in FT_MODES:
                    try:
                        vae_ft, _, _ = run_finetuning_amr(
                            splits_path              = split_file,
                            target_domain            = TARGET_DOMAIN,
                            pretrained_model_path    = model_path,
                            finetuning_mode          = mode,
                            n_prev                   = n_prev,
                            n_new                    = n_new,
                            output_dir               = OUTPUT_PATH,
                            device                   = device,
                            target_antibiotics       = antibiotics,
                            species                  = species,
                            consider_prev_domains    = (n_prev > 0),
                            lambda_amr               = LAMBDA_AMR,
                            run_latent_evaluation    = RUN_EVAL,
                            ft_lr                    = 1e-5,
                            dataset_path             = dataset_path,
                        )
                        vae_ft.eval()
                        metrics_ft = evaluate_amr_head(
                            vae_ft, X_test, amr_test, ab_list, device
                        )
                        for atb, m in metrics_ft.items():
                            all_results.append({
                                "species":   species,
                                "partition": i_part,
                                "n_prev":    n_prev,
                                "n_new":     n_new,
                                "model":     f"FT_{mode}",
                                "antibiotic": atb,
                                "auc":       m["auc"],
                                "pr_auc":    m["pr_auc"],
                                "n_test":    m.get("n", len(idx_test)),
                            })
                            print(f"  FT_{mode} {atb}: AUC={m['auc']:.3f}")

                        del vae_ft
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"ERROR mode={mode}: {e}")
                        import traceback
                        traceback.print_exc()

                # ── MLP FROM SCRATCH ──────────────────────────────
                if len(X_ft) >= 20:
                    X_tr_s, X_val_s, a_tr_s, a_val_s = train_test_split(
                        X_ft, amr_ft, test_size=0.2, random_state=i_part
                    )
                    mlp_scratch = SimpleAMRMLP_Extended(
                        input_dim        = X_tr_s.shape[1],
                        n_antibiotics    = len(ab_list),
                        latent_dim       = LATENT_DIM,
                        antibiotic_names = ab_list,
                        epochs=100, lr=1e-3, patience=15,
                    )
                    mlp_scratch.trainloop(
                        DataLoader(
                            TensorDataset(
                                torch.tensor(X_tr_s, dtype=torch.float32),
                                torch.tensor(a_tr_s, dtype=torch.float32),
                            ),
                            batch_size=64, shuffle=True,
                        ),
                        DataLoader(
                            TensorDataset(
                                torch.tensor(X_val_s, dtype=torch.float32),
                                torch.tensor(a_val_s, dtype=torch.float32),
                            ),
                            batch_size=64, shuffle=False,
                        ),
                        device
                    )
                    metrics_scratch = mlp_scratch.evaluate(X_test, amr_test, device)
                    for atb, m in metrics_scratch.items():
                        all_results.append({
                            "species":   species,
                            "partition": i_part,
                            "n_prev":    n_prev,
                            "n_new":     n_new,
                            "model":     "MLP_from_scratch",
                            "antibiotic": atb,
                            "auc":       m["auc"],
                            "pr_auc":    m["pr_auc"],
                            "n_test":    m.get("n", len(idx_test)),
                        })
                        print(f"  MLP_scratch {atb}: AUC={m['auc']:.3f}")

                    del mlp_scratch
                    torch.cuda.empty_cache()

                # ── MLP POOLED (entrenado en source domains) ──────
                # Mismo modelo para todos los n_new — solo evaluamos
                metrics_pooled = mlp_pooled.evaluate(X_test, amr_test, device)
                for atb, m in metrics_pooled.items():
                    all_results.append({
                        "species":   species,
                        "partition": i_part,
                        "n_prev":    n_prev,
                        "n_new":     n_new,
                        "model":     "MLP_pooled_source",
                        "antibiotic": atb,
                        "auc":       m["auc"],
                        "pr_auc":    m["pr_auc"],
                        "n_test":    m.get("n", len(idx_test)),
                    })
                    print(f"  MLP_pooled {atb}: AUC={m['auc']:.3f}")

                # ── MLP POOLED FINETUNED ──────────────────────────
                if len(X_ft) >= 20:
                    X_tr_p, X_val_p, a_tr_p, a_val_p = train_test_split(
                        X_ft, amr_ft, test_size=0.2, random_state=42
                    )
                    # deepcopy preserves original weights for next iteration
                    mlp_pooled_ft = copy.deepcopy(mlp_pooled)

                    # Lower lr for finetuning
                    for param_group in mlp_pooled_ft.optimizer.param_groups:
                        param_group["lr"] = MLP_POOLED_FT_LR
                        
                    mlp_pooled_ft.epochs = MLP_POOLED_FT_EPOCHS

                    mlp_pooled_ft.trainloop(
                        DataLoader(
                            TensorDataset(
                                torch.tensor(X_tr_p, dtype=torch.float32),
                                torch.tensor(a_tr_p, dtype=torch.float32),
                            ),
                            batch_size=64, shuffle=True,
                        ),
                        DataLoader(
                            TensorDataset(
                                torch.tensor(X_val_p, dtype=torch.float32),
                                torch.tensor(a_val_p, dtype=torch.float32),
                            ),
                            batch_size=64, shuffle=False,
                        ),
                        device
                    )
                    metrics_pooled_ft = mlp_pooled_ft.evaluate(X_test, amr_test, device)
                    for atb, m in metrics_pooled_ft.items():
                        all_results.append({
                            "species":    species,
                            "partition":  i_part,
                            "n_prev":     n_prev,
                            "n_new":      n_new,
                            "model":      "MLP_pooled_finetuned",
                            "antibiotic": atb,
                            "auc":        m["auc"],
                            "pr_auc":     m["pr_auc"],
                            "n_test":     m.get("n", len(idx_test)),
                        })
                        print(f"  MLP_pooled_ft {atb}: AUC={m['auc']:.3f}")

                    del mlp_pooled_ft
                    torch.cuda.empty_cache()

    del vae_pretrained, mlp_pooled
    torch.cuda.empty_cache()

    # ==========================================================
    # SAVE INTERMEDIATE RESULTS AFTER EACH SPECIES
    # ==========================================================
    df_partial = pd.DataFrame(all_results)

    species_tag = species.replace("_", "")
    partial_csv = OUTPUT_PATH / f"partial_results_{species_tag}_{timestamp}.csv"

    df_partial.to_csv(partial_csv, index=False)

    print(f"\nIntermediate results saved: {partial_csv}")


############################################################
# SAVE FINAL RESULTS
############################################################
df_results = pd.DataFrame(all_results)
csv_path   = OUTPUT_PATH / f"amr_finetuning_all_species_MSUMG_{timestamp}.csv"
df_results.to_csv(csv_path, index=False)
print(f"\nFull results saved: {csv_path}")


############################################################
# SUMMARY
############################################################
col_order = (
    ["AMR_head_zero_shot"]
    + [f"FT_{m}" for m in FT_MODES]
    + ["MLP_pooled_source", "MLP_pooled_finetuned", "MLP_from_scratch"]
)

for species in SPECIES_CONFIG.keys():
    sub_sp  = df_results[df_results["species"] == species]
    ab_list = SPECIES_CONFIG[species]["antibiotics"]

    if len(sub_sp) == 0:
        continue

    print(f"\n{'='*70}")
    print(f"  {species}")
    print(f"{'='*70}")

    for atb in ab_list:
        sub_atb = sub_sp[sub_sp["antibiotic"] == atb]
        if len(sub_atb) == 0:
            continue
        print(f"\n--- {atb} ---")

        per_partition = (
            sub_atb
            .groupby(["partition", "n_prev", "n_new", "model"])["auc"]
            .mean()
            .reset_index()
        )
        atb_summary = (
            per_partition
            .groupby(["n_prev", "n_new", "model"])["auc"]
            .agg(["mean", "std"])
            .reset_index()
        )
        atb_summary["mean_std"] = (
            atb_summary["mean"].round(3).astype(str) + " ± " +
            atb_summary["std"].round(3).astype(str)
        )
        species_tag = species.replace("_", "")
        atb_summary.to_csv(
            OUTPUT_PATH / f"summary_{species_tag}_{atb}_{timestamp}.csv",
            index=False
        )

        pivot = atb_summary.pivot_table(
            index=["n_prev", "n_new"],
            columns="model",
            values="mean_std",
            aggfunc="first"
        )
        existing = [c for c in col_order if c in pivot.columns]
        print(pivot[existing].to_string())

print(f"\n===== DONE =====")
