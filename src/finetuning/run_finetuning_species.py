############################################################
# IMPORTS
############################################################
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config.loader import load_config
from src.data.datasets import load_driams, load_marisma, load_msumg, load_rki
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import eval_model, run_tsne_evaluation
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


############################################################
# CONFIG
############################################################
TARGET_SPECIES = [
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"
]   


############################################################
# UTILS
############################################################
def run_finetuning(splits_path, target_domain, pretrained_model_path, finetuning_mode, n_prev, n_new, output_dir, device, consider_prev_domains=True, run_latent_evaluation=True):
    """
    Executes finetuning of a pretrained MultiVAE model
    for a specific target domain using precomputed splits.

    Parameters
    ----------
    splits_path : str
        Path to pickle file containing finetuning indices.
    target_domain : str
        Domain to finetune (e.g. "DRIAMS_D").
    pretrained_model_path : str
        Path to pretrained model checkpoint.
    finetuning_mode : str
        Strategy for parameter freezing.
    output_dir : Path
        Directory to save model and logs.
    device : torch.device
        CPU or CUDA.
    """
    print("\n===== INITIALIZING EXPERIMENT =====")

    # Extract base model name automatically
    base_name = Path(pretrained_model_path).parents[1].name

    # Build structured experiment name
    exp_config_name = (
        f"base={base_name}_"
        f"target={target_domain}_"
        f"prev={n_prev}_new={n_new}_"
        f"mode={finetuning_mode}"
    )

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Final experiment directory
    experiment_dir = (output_dir / exp_config_name / timestamp)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Base model:", base_name)
    print("Target domain:", target_domain)
    print("n_prev:", n_prev)
    print("n_new:", n_new)
    print("Finetuning mode:", finetuning_mode)
    print("Output directory:", experiment_dir)
    print("Experiment directory created.")

    # Load data
    print("\n===== LOADING DATA =====")
    cfg = load_config()

    driams_pkl = cfg["data"]["DRIAMS_FULL"]
    driams_A = load_driams(driams_pkl, filter=["DRIAMS_A"])["DRIAMS_A"]
    driams_B = load_driams(driams_pkl, filter=["DRIAMS_B"])["DRIAMS_B"]
    driams_C = load_driams(driams_pkl, filter=["DRIAMS_C"])["DRIAMS_C"]
    driams_D = load_driams(driams_pkl, filter=["DRIAMS_D"])["DRIAMS_D"]
    marisma = load_marisma(cfg["data"]["MARISMa_FULL"])
    rki = load_rki(cfg["data"]["RKI_FULL"])
    msumg = load_msumg(cfg["data"]["MSUMG_FULL"])

    # Concatenate
    data_list = [
        driams_A["data"], driams_B["data"], driams_C["data"],
        marisma["data"], rki["data"]
    ]

    label_list = [
        driams_A["label"], driams_B["label"], driams_C["label"],
        marisma["label"], rki["label"]
    ]

    meta_list = [
        driams_A["meta"], driams_B["meta"], driams_C["meta"],
        marisma["meta"], rki["meta"]
    ]

    data_all_raw = np.vstack(data_list)
    label_all_raw = np.concatenate(label_list)
    meta_all_raw = pd.concat(meta_list, ignore_index=True)

    # Filter by species
    mask_sp = np.isin(label_all_raw, TARGET_SPECIES)
    data_all = data_all_raw[mask_sp]
    label_all = label_all_raw[mask_sp]
    meta_all = meta_all_raw.iloc[mask_sp].reset_index(drop=True)

    # Normalize
    data_all = row_minmax_normalize(data_all)

    # Load splits
    print("Loading finetuning indices...")
    with open(splits_path, "rb") as f:
        splits_idx = pickle.load(f)

    # Previous domains: indices are on data_all
    idxA = splits_idx["DRIAMS_A"]["finetuning"]
    idxB = splits_idx["DRIAMS_B"]["finetuning"]
    idxC = splits_idx["DRIAMS_C"]["finetuning"]
    idxM = splits_idx["MARISMA"]["finetuning"]
    idxR = splits_idx["RKI"]["finetuning"]

    dataA_ft = data_all[idxA] if len(idxA) > 0 else np.empty((0, data_all.shape[1]))
    labelA_ft = label_all[idxA] if len(idxA) > 0 else np.array([])
    metaA_ft = meta_all.iloc[idxA].reset_index(drop=True) if len(idxA) > 0 else pd.DataFrame()

    dataB_ft = data_all[idxB] if len(idxB) > 0 else np.empty((0, data_all.shape[1]))
    labelB_ft = label_all[idxB] if len(idxB) > 0 else np.array([])
    metaB_ft = meta_all.iloc[idxB].reset_index(drop=True) if len(idxB) > 0 else pd.DataFrame()

    dataC_ft = data_all[idxC] if len(idxC) > 0 else np.empty((0, data_all.shape[1]))
    labelC_ft = label_all[idxC] if len(idxC) > 0 else np.array([])
    metaC_ft = meta_all.iloc[idxC].reset_index(drop=True) if len(idxC) > 0 else pd.DataFrame()

    dataM_ft = data_all[idxM] if len(idxM) > 0 else np.empty((0, data_all.shape[1]))
    labelM_ft = label_all[idxM] if len(idxM) > 0 else np.array([])
    metaM_ft = meta_all.iloc[idxM].reset_index(drop=True) if len(idxM) > 0 else pd.DataFrame()

    dataR_ft = data_all[idxR] if len(idxR) > 0 else np.empty((0, data_all.shape[1]))
    labelR_ft = label_all[idxR] if len(idxR) > 0 else np.array([])
    metaR_ft = meta_all.iloc[idxR].reset_index(drop=True) if len(idxR) > 0 else pd.DataFrame()

    print(f"Finetuning samples: A={len(dataA_ft)}, B={len(dataB_ft)}, C={len(dataC_ft)}, M={len(dataM_ft)}, R={len(dataR_ft)}")

    # Target domain (indices are local to filtered dataset)
    mask_sp_D = np.isin(driams_D["label"], TARGET_SPECIES)
    dataD = row_minmax_normalize(driams_D["data"][mask_sp_D])
    labelD = driams_D["label"][mask_sp_D]
    metaD = driams_D["meta"].iloc[mask_sp_D].reset_index(drop=True)

    mask_sp_M = np.isin(msumg["label"], TARGET_SPECIES)
    data_msumg = row_minmax_normalize(msumg["data"][mask_sp_M])
    label_msumg = msumg["label"][mask_sp_M]
    meta_msumg = msumg["meta"].iloc[mask_sp_M].reset_index(drop=True)

    if target_domain == "DRIAMS_D":
        idx_target = splits_idx["DRIAMS_D"]["finetuning"]
        data_target = dataD[idx_target]
        label_target = labelD[idx_target]
        meta_target = metaD.iloc[idx_target].reset_index(drop=True)

        DOMAIN_MAP = {
            "DRIAMS_A": 0,
            "DRIAMS_B": 1,
            "DRIAMS_C": 2,
            "MARISMA": 3,
            "RKI": 4,
            "DRIAMS_D": 5
        }

    elif target_domain == "MS-UMG":
        idx_target = splits_idx["MS-UMG"]["finetuning"]
        data_target = data_msumg[idx_target]
        label_target = label_msumg[idx_target]
        meta_target = meta_msumg.iloc[idx_target].reset_index(drop=True)

        DOMAIN_MAP = {
            "DRIAMS_A": 0,
            "DRIAMS_B": 1,
            "DRIAMS_C": 2,
            "MARISMA": 3,
            "RKI": 4,
            "MS-UMG": 5
        }
    else:
        raise ValueError(f"Unknown target_domain: {target_domain}")
    
    # Build finetuning dataset
    print("Building finetuning dataset...")

    if consider_prev_domains:
        print("Using previous domains for finetuning...")
        X_ft = np.vstack([dataA_ft, dataB_ft, dataC_ft, dataM_ft, dataR_ft, data_target])
        y_ft = np.concatenate([labelA_ft, labelB_ft, labelC_ft, labelM_ft, labelR_ft, label_target])
        meta_ft = pd.concat([metaA_ft, metaB_ft, metaC_ft, metaM_ft, metaR_ft, meta_target],ignore_index=True)
    else:
        print("Running target-only finetuning (no previous domains)")
        X_ft = data_target
        y_ft = label_target
        meta_ft = meta_target.copy()

    domain_ft = meta_ft["hospital"].map(DOMAIN_MAP).values.astype(np.int64)

    # Robust label encoder
    species_encoder = LabelEncoder()
    species_encoder.fit(TARGET_SPECIES)
    species_ft = species_encoder.transform(y_ft)
    print(f"Finetuning samples: {len(X_ft)}")

    # Load base model
    print("Loading base pretrained model...")

    vae = MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=X_ft.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species = len(TARGET_SPECIES))

    vae.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    vae.to(device)

    # Add new decoder 
    print(f"Adding new decoder for {target_domain}...")

    old_decoders = vae.decoder.net
    latent_dim  = old_decoders[0][0].in_features
    output_dim  = old_decoders[0][-2].out_features

    new_decoder = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, output_dim),
        nn.Sigmoid()
    )

    vae.decoder.net = nn.ModuleList(list(old_decoders) + [new_decoder])
    vae.decoder.num_domains = len(vae.decoder.net)
    vae.to(device)

    # Train/val split
    print("Preparing training split...")
    counts = np.bincount(species_ft)
    can_stratify = (len(np.unique(species_ft)) > 1) and (counts.min() >= 2)

    stratify_vec = species_ft if can_stratify else None
    if not can_stratify:
        print("Too few samples per class to stratify train/val. Using random split.")

    X_tr, X_val, d_tr, d_val, s_tr, s_val = train_test_split(
        X_ft, domain_ft, species_ft,
        test_size=0.2,
        stratify=stratify_vec,
        random_state=42
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(d_tr, dtype=torch.long), torch.tensor(s_tr, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(d_val, dtype=torch.long), torch.tensor(s_val, dtype=torch.long)), batch_size=64, shuffle=False)

    # Finetuning
    print("\n===== STARTING FINETUNING =====")
    new_domain_idx = len(vae.decoder.net) - 1

    if finetuning_mode == "full":
        for p in vae.parameters():
            p.requires_grad = True

    elif finetuning_mode == "freeze_priors":
        for p in vae.parameters():
            p.requires_grad = True
        vae.prior.mu_embed.weight.requires_grad = False
        vae.prior.logvar_embed.weight.requires_grad = False

    elif finetuning_mode == "decoder_only":
        for p in vae.parameters():
            p.requires_grad = False
        for p in vae.decoder.net[new_domain_idx].parameters():
            p.requires_grad = True

    elif finetuning_mode == "partial_encoder":
        for p in vae.parameters():
            p.requires_grad = False
        for p in vae.decoder.net[new_domain_idx].parameters():
            p.requires_grad = True
        first_layer = vae.encoder.net[0]
        for p in first_layer.parameters():
            p.requires_grad = True
            
    else:
        raise ValueError(
            f"Unknown finetuning_mode: '{finetuning_mode}'. "
            "Supported modes are: ['full', 'freeze_priors', 'decoder_only', 'partial_encoder']"
        )
    
    vae.epochs = 30
    vae.lr = 1e-4
    vae.patience = 10

    vae.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=vae.lr,
        weight_decay=1e-5
    )

    vae.trainloop(
        trainloader=train_loader,
        validloader=val_loader,
        device=device
    )

    model_filename = (
        f"model_finetuned_"
        f"base={base_name}_"
        f"target={target_domain}_"
        f"prev={n_prev}_new={n_new}_"
        f"mode={finetuning_mode}.pth"
    )

    torch.save(vae.state_dict(), experiment_dir / model_filename)

    print("Finetuning complete. Model saved.")

    # Latent evaluation
    if run_latent_evaluation:
        print("\n===== RUNNING LATENT EVALUATION =====")

        # Structural latent evaluation (not predictive evaluation)
        X_eval = np.vstack([dataA_ft, dataB_ft, dataC_ft, dataM_ft, dataR_ft, data_target])
        y_eval = np.concatenate([labelA_ft, labelB_ft, labelC_ft, labelM_ft, labelR_ft, label_target])
        meta_eval = pd.concat([metaA_ft, metaB_ft, metaC_ft, metaM_ft, metaR_ft, meta_target], ignore_index=True)

        domain_eval = meta_eval["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
        species_eval = species_encoder.transform(y_eval) 
        
        eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_eval, dtype=torch.float32),
                torch.tensor(domain_eval, dtype=torch.long),
                torch.tensor(species_eval, dtype=torch.long),
            ),
            batch_size=256,
            shuffle=False
        )

        vae.eval()
        mus_all = eval_model(vae, eval_loader, device)

        figure_prefix = (
            f"FT_base={base_name}_"
            f"target={target_domain}_"
            f"prev={n_prev}_new={n_new}_"
            f"mode={finetuning_mode}"
        )

        run_tsne_evaluation(
            mus_all=mus_all,
            label_final=y_eval,
            meta_final=meta_eval,
            output_dir=experiment_dir,
            prefix=figure_prefix
        )

        print("Latent evaluation finished.")

    print("\n===== EXPERIMENT COMPLETE =====")

    return vae, species_encoder, DOMAIN_MAP
