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

from utils.config import load_config
from utils.data import load_driams, load_marisma, load_rki, load_msumg, row_minmax_normalize
from utils.eval import eval_model, run_tsne_evaluation
from models.deep.MultiVAEPrior import MultiVAE_Bernoulli_SpeciesPrior_Extended


def run_finetuning(splits_path, target_domain, pretrained_model_path, finetuning_mode, n_prev, n_new, output_dir, device):
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
    # LOAD DATA
    ####################
    print("\n===== LOADING DATA =====")

    cfg = load_config()

    driams_dict  = load_driams(cfg["data"]["DRIAMS_REDUCED_PKL"])
    marisma_dict = load_marisma(cfg["data"]["MARISMa_REDUCED_PKL"])
    msumg_dict   = load_msumg(cfg["data"]["MSUMG_PKL"])  # ya filtra agar
    rki_dict     = load_rki(cfg["data"]["RKI_PKL"])

    data_driams, label_driams, meta_driams = driams_dict["data"], driams_dict["label"], driams_dict["meta"]
    data_marisma, label_marisma, meta_marisma = marisma_dict["data"], marisma_dict["label"], marisma_dict["meta"]
    data_rki, label_rki, meta_rki = rki_dict["data"], rki_dict["label"], rki_dict["meta"]
    data_msumg, label_msumg, meta_msumg = msumg_dict["data"], msumg_dict["label"], msumg_dict["meta"]

    # Split DRIAMS
    maskA = meta_driams["hospital"] == "DRIAMS_A"
    maskB = meta_driams["hospital"] == "DRIAMS_B"
    maskC = meta_driams["hospital"] == "DRIAMS_C"
    maskD = meta_driams["hospital"] == "DRIAMS_D"

    dataA, labelA, metaA = data_driams[maskA], label_driams[maskA], meta_driams[maskA]
    dataB, labelB, metaB = data_driams[maskB], label_driams[maskB], meta_driams[maskB]
    dataC, labelC, metaC = data_driams[maskC], label_driams[maskC], meta_driams[maskC]
    dataD, labelD, metaD = data_driams[maskD], label_driams[maskD], meta_driams[maskD]

    # Normalize
    dataA = row_minmax_normalize(dataA)
    dataB = row_minmax_normalize(dataB)
    dataC = row_minmax_normalize(dataC)
    dataD = row_minmax_normalize(dataD)
    data_marisma = row_minmax_normalize(data_marisma)
    data_rki = row_minmax_normalize(data_rki)
    data_msumg = row_minmax_normalize(data_msumg)


    ####################
    # LOAD FINETUNING SPLITS
    ####################
    print("Loading finetuning indices...")

    with open(splits_path, "rb") as f:
        splits_idx = pickle.load(f)

    def select(data, label, meta, idx):
        return data[idx], label[idx], meta.iloc[idx].reset_index(drop=True)

    dataA_ft, labelA_ft, metaA_ft = select(dataA,labelA,metaA,splits_idx["DRIAMS_A"]["finetuning"])
    dataB_ft, labelB_ft, metaB_ft = select(dataB,labelB,metaB,splits_idx["DRIAMS_B"]["finetuning"])
    dataC_ft, labelC_ft, metaC_ft = select(dataC,labelC,metaC,splits_idx["DRIAMS_C"]["finetuning"])
    dataM_ft, labelM_ft, metaM_ft = select(data_marisma,label_marisma,meta_marisma,splits_idx["MARISMA"]["finetuning"])
    dataR_ft, labelR_ft, metaR_ft = select(data_rki,label_rki,meta_rki,splits_idx["RKI"]["finetuning"])

    if target_domain == "DRIAMS_D":
        data_target, label_target, meta_target = select(dataD, labelD ,metaD, splits_idx["DRIAMS_D"]["finetuning"])
        DOMAIN_MAP = {
            "DRIAMS_A": 0,
            "DRIAMS_B": 1,
            "DRIAMS_C": 2,
            "MARISMA":  3,
            "RKI":      4,
            "DRIAMS_D": 5,
        }
    elif target_domain == "MS-UMG":
        data_target, label_target, meta_target = select(data_msumg, label_msumg, meta_msumg, splits_idx["MS-UMG"]["finetuning"])
        DOMAIN_MAP = {
            "DRIAMS_A": 0,
            "DRIAMS_B": 1,
            "DRIAMS_C": 2,
            "MARISMA":  3,
            "RKI":      4,
            "MS-UMG":   5,  
        }
    else:
        raise ValueError(f"Unknown target_domain: {target_domain}")


    ####################
    # BUILD FINETUNING DATASET
    ####################
    print("Building finetuning dataset...")

    X_ft = np.vstack([dataA_ft,dataB_ft,dataC_ft,dataM_ft,dataR_ft,data_target])
    y_ft = np.concatenate([labelA_ft,labelB_ft,labelC_ft,labelM_ft,labelR_ft,label_target])
    meta_ft = pd.concat([metaA_ft,metaB_ft,metaC_ft,metaM_ft,metaR_ft,meta_target],ignore_index=True)

    domain_ft = meta_ft["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
    species_encoder = LabelEncoder()
    species_ft = species_encoder.fit_transform(y_ft).astype(np.int64)


    ####################
    # LOAD BASE MODEL
    ####################
    print("Loading base pretrained model...")

    vae = MultiVAE_Bernoulli_SpeciesPrior_Extended(
        input_dim=X_ft.shape[1],
        latent_dim=64,
        num_domains=5,
        n_species = len(np.unique(y_ft)))

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
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 1024),
        nn.ReLU(),
        nn.Linear(1024, output_dim),
        nn.Sigmoid(),
    )

    vae.decoder.net = nn.ModuleList(list(old_decoders) + [new_decoder])
    vae.decoder.num_domains = len(vae.decoder.net)
    vae.to(device)


    ####################
    # TRAIN / VALID SPLIT
    ####################
    print("Preparing training split...")

    X_tr, X_val, d_tr, d_val, s_tr, s_val = train_test_split(
        X_ft,
        domain_ft,
        species_ft,
        test_size=0.2,
        stratify=species_ft,
        random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(d_tr, dtype=torch.long),
            torch.tensor(s_tr, dtype=torch.long),
        ),
        batch_size=32,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(d_val, dtype=torch.long),
            torch.tensor(s_val, dtype=torch.long),
        ),
        batch_size=32,
        shuffle=False,
    )


    ####################
    # FINETUNING
    ####################
    print("\n===== STARTING FINETUNING =====")

    if finetuning_mode == "full":
        for p in vae.parameters():
            p.requires_grad = True
    elif finetuning_mode == "freeze_priors":
        for p in vae.parameters():
            p.requires_grad = True
        vae.prior.mu_embed.weight.requires_grad = False
        vae.prior.logvar_embed.weight.requires_grad = False
    else:
        raise ValueError(
            f"Unknown finetuning_mode: '{finetuning_mode}'. "
            "Supported modes are: ['full', 'freeze_priors']"
        )
    
    vae.epochs = 30
    vae.lr = 1e-5
    vae.annealing_epochs = 10
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
    print("\n===== RUNNING LATENT EVALUATION =====")

    # Structural latent evaluation (not predictive evaluation)
    X_eval = np.vstack([dataA, dataB, dataC, data_marisma, data_rki, data_target])
    y_eval = np.concatenate([labelA, labelB, labelC, label_marisma, label_rki, label_target])
    meta_eval = pd.concat([metaA, metaB, metaC, meta_marisma, meta_rki, meta_target],ignore_index=True)
    meta_eval["year"] = meta_eval["year"].fillna("unknown").astype(str)

    domain_eval = meta_eval["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
    species_eval = species_encoder.transform(y_eval).astype(np.int64)

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
