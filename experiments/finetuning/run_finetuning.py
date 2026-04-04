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

    ####################
    # EXPERIMENT SETUP
    ####################
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


    ####################
    # LOAD DATA (CORREGIDO BIEN)
    ####################
    print("\n===== LOADING DATA =====")
    cfg = load_config()

    TARGET_SPECIES = [
    "Klebsiella_Pneumoniae","Escherichia_Coli","Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa","Enterococcus_Faecium", "Enterobacter_cloacae_complex"
    ]   

    # Load datasets
    driams_dict  = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
    marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
    rki_dict     = load_rki(cfg["data"]["RKI_PKL"])
    msumg_dict   = load_msumg(cfg["data"]["MSUMG_PKL"])

    # Normalize
    data_driams_norm = row_minmax_normalize(driams_dict["data"])
    data_marisma_norm = row_minmax_normalize(marisma_dict["data"])
    data_rki_norm = row_minmax_normalize(rki_dict["data"])
    data_msumg_norm = row_minmax_normalize(msumg_dict["data"])

    print("Loading finetuning indices...")
    with open(splits_path, "rb") as f:
        splits_idx = pickle.load(f)

    # ============================
    # RECONSTRUCT GLOBAL DATASET
    # ============================
    maskA = driams_dict["meta"]["hospital"] == "DRIAMS_A"
    maskB = driams_dict["meta"]["hospital"] == "DRIAMS_B"
    maskC = driams_dict["meta"]["hospital"] == "DRIAMS_C"

    data_list = [
        data_driams_norm[maskA],
        data_driams_norm[maskB],
        data_driams_norm[maskC],
        data_marisma_norm,
        data_rki_norm
    ]

    label_list = [
        driams_dict["label"][maskA],
        driams_dict["label"][maskB],
        driams_dict["label"][maskC],
        marisma_dict["label"],
        rki_dict["label"]
    ]

    meta_list = [
        driams_dict["meta"][maskA],
        driams_dict["meta"][maskB],
        driams_dict["meta"][maskC],
        marisma_dict["meta"],
        rki_dict["meta"]
    ]

    data_all = np.vstack(data_list)
    label_all = np.concatenate(label_list)
    meta_all = pd.concat(meta_list, ignore_index=True)

    idxA = splits_idx["DRIAMS_A"]["finetuning"]
    idxB = splits_idx["DRIAMS_B"]["finetuning"]
    idxC = splits_idx["DRIAMS_C"]["finetuning"]
    idxM = splits_idx["MARISMA"]["finetuning"]
    idxR = splits_idx["RKI"]["finetuning"]

    dataA_ft = data_all[idxA]
    labelA_ft = label_all[idxA]
    metaA_ft = meta_all.iloc[idxA].reset_index(drop=True)

    dataB_ft = data_all[idxB]
    labelB_ft = label_all[idxB]
    metaB_ft = meta_all.iloc[idxB].reset_index(drop=True)

    dataC_ft = data_all[idxC]
    labelC_ft = label_all[idxC]
    metaC_ft = meta_all.iloc[idxC].reset_index(drop=True)

    dataM_ft = data_all[idxM]
    labelM_ft = label_all[idxM]
    metaM_ft = meta_all.iloc[idxM].reset_index(drop=True)

    dataR_ft = data_all[idxR]
    labelR_ft = label_all[idxR]
    metaR_ft = meta_all.iloc[idxR].reset_index(drop=True)

    if target_domain == "DRIAMS_D":
        idx_target = splits_idx["DRIAMS_D"]["finetuning"]
        data_target = data_driams_norm[idx_target]
        label_target = driams_dict["label"][idx_target]
        meta_target = driams_dict["meta"].iloc[idx_target].reset_index(drop=True)

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
        data_target = data_msumg_norm[idx_target]
        label_target = msumg_dict["label"][idx_target]
        meta_target = msumg_dict["meta"].iloc[idx_target].reset_index(drop=True)

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
    
    ####################
    # BUILD FINETUNING DATASET
    ####################
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

    # Keep only species seen during pretraining
    mask_sp = np.isin(y_ft, TARGET_SPECIES)

    X_ft = X_ft[mask_sp]
    y_ft = y_ft[mask_sp]
    meta_ft = meta_ft.iloc[mask_sp].reset_index(drop=True)
    domain_ft = meta_ft["hospital"].map(DOMAIN_MAP).values.astype(np.int64)

    # Robust label encoder
    species_encoder = LabelEncoder()
    species_encoder.fit(TARGET_SPECIES)
    species_ft = species_encoder.transform(y_ft)


    ####################
    # LOAD BASE MODEL
    ####################
    print("Loading base pretrained model...")

    vae = MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=X_ft.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species = len(TARGET_SPECIES))

    vae.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    vae.to(device)


    ####################
    # ADD NEW DECODER 
    ####################
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


    ####################
    # TRAIN / VALID SPLIT
    ####################
    print("Preparing training split...")

    # Robust stratify for tiny few-shot
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


    ####################
    # FINETUNING
    ####################
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


    ####################
    # LATENT EVALUATION
    ####################
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
