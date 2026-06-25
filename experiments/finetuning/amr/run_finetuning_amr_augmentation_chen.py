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

from src.data.io import load_pkl
from src.data.preprocessing import row_minmax_normalize
from src.evaluation.eval import run_tsne_evaluation
from models.deep.MultiVAEPriorAMRHeadZChen import MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZChinos
from models.deep.IWAEAMRHead import MultiVAE_Bernoulli_SpeciesPrior_AMR_IWAE_Extended


TARGET_SPECIES = [
    "Klebsiella_Pneumoniae", "Escherichia_Coli", "Staphylococcus_Aureus",
    "Pseudomonas_Aeruginosa", "Enterococcus_Faecium", "Enterobacter_cloacae_complex",
]

_AMR_FROZEN_MODES = {"enc_dec"}

# Maps logical domain keys to (hospital, year) when the key encodes a specific year.
# Used so that finetuning scripts can pass "MARISMA_2024" or "MS-UMG_2020" as
# target_domain without the raw hospital name.
_DOMAIN_FILTERS = {
    "MS-UMG_2020":  ("MS-UMG",  "2020"),
    "MS-UMG_2021":  ("MS-UMG",  "2021"),
    "MARISMA_2024": ("MARISMA", "2024"),
}

# These (hospital, year) combos are excluded from the SOURCE sub-dataset so
# that split indices generated from a model trained without them remain valid.
_OOD_HOLDOUT = [("MARISMA", "2024")]


def run_finetuning_amr(
    splits_path,
    target_domain,
    pretrained_model_path,
    finetuning_mode,
    n_prev,
    n_new,
    output_dir,
    device,
    target_antibiotics,
    dataset_path="/export/usuarios01/agnavarr/MALDIAlign/amr_global_final.pkl",
    species="Klebsiella_Pneumoniae",
    consider_prev_domains=True,
    run_latent_evaluation=False,
    latent_dim=128,
    lambda_amr=25,
    ft_epochs=50,
    ft_lr=1e-4,
    ft_patience=15,
    batch_size=64,
    n_amr_samples=1,
    random_state=42,
    model_type="vae_z",
    use_fixed_prior=False,
    annealing_epochs=None
):
    """
    Finetune a pretrained MultiVAE AMR model for a target domain.

    Parameters
    ----------
    splits_path : str or Path
        Path to pickle file with precomputed finetuning indices.
        Structure: {domain: {"finetuning": idx_array, "test": idx_array (for target)}}
        IMPORTANT — index spaces:
          - source domain indices  → local to (source domains + species) sub-dataset
          - target domain indices  → local to (target domain + species) sub-dataset
    target_domain : str
        Target hospital ("DRIAMS_D" or "MS-UMG").
    pretrained_model_path : str or Path
        Path to pretrained model.pth.
    finetuning_mode : str
        One of: "decoder_only", "enc_dec", "enc_dec_amr", "freeze_priors", "full".
    n_prev : int
        Number of source domain samples used (for logging).
    n_new : int
        Number of target domain samples used (for logging).
    output_dir : Path
        Where to save finetuned model.
    device : torch.device
    species : str
        Species label to filter (must match generate_splits).
    consider_prev_domains : bool
        If True, include source domain samples in finetuning data.
    lambda_amr : float
        AMR loss weight.  Automatically overridden to 0 for modes
        "decoder_only" and "enc_dec" (AMR head frozen → backward bug).

    Returns
    -------
    vae : finetuned model
    species_encoder : LabelEncoder
    domain_map : dict
    """

    ####################
    # EXPERIMENT SETUP
    ####################
    print("\n===== INITIALIZING AMR FINETUNING =====")

    base_name = Path(pretrained_model_path).parents[1].name
    exp_config_name = (
        f"amr_base={base_name}_"
        f"target={target_domain}_"
        f"prev={n_prev}_new={n_new}_"
        f"mode={finetuning_mode}"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(output_dir) / exp_config_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    lambda_amr_eff = lambda_amr

    print("Base model:", base_name)
    print("Target domain:", target_domain)
    print("Species:", species)
    print("n_prev:", n_prev)
    print("n_new:", n_new)
    print("Finetuning mode:", finetuning_mode)
    print(f"lambda_amr: {lambda_amr} → effective: {lambda_amr_eff}")
    print("Output directory:", experiment_dir)

    is_iwae = model_type.startswith("iwae")

    model_class = (
        MultiVAE_Bernoulli_SpeciesPrior_AMR_IWAE_Extended
        if is_iwae
        else MultiVAE_Bernoulli_SpeciesPrior_AMR_Head_ExtendedZChinos
    )

    ####################
    # LOAD FULL DATASET
    ####################
    print("\n===== LOADING DATA =====")

    dataset = load_pkl(dataset_path)
    data_raw = dataset["data"]
    amr_raw = dataset["amr"]
    ab_list_raw = list(dataset["antibiotics"])
    labels_raw = dataset["label"]
    raw_meta = dataset["meta"]
    meta_raw = (
        pd.DataFrame(list(raw_meta))
        if isinstance(raw_meta, (list, np.ndarray))
        else pd.DataFrame(raw_meta)
    )

    # Filter antibiotics
    keep_idx = [ab_list_raw.index(n) for n in target_antibiotics if n in ab_list_raw]
    keep_names = [n for n in target_antibiotics if n in ab_list_raw]
    amr_raw_filt = amr_raw[:, keep_idx]
    data_norm = row_minmax_normalize(data_raw)

    # Remove all-NaN antibiotic columns
    valid_ab = ~np.all(np.isnan(amr_raw_filt), axis=0)
    keep_names = [a for a, v in zip(keep_names, valid_ab) if v]
    amr_norm = amr_raw_filt[:, valid_ab]

    print(f"Data shape: {data_norm.shape} | Antibiotics: {keep_names}")
    SOURCE_DOMAINS_ALL = ["DRIAMS_A", "DRIAMS_B", "DRIAMS_C", "MARISMA"]
    # Resolve logical target key to actual hospital name for source exclusion
    _tgt_hospital = _DOMAIN_FILTERS.get(target_domain, (target_domain,))[0]
    source_domains = [d for d in SOURCE_DOMAINS_ALL if d != _tgt_hospital]

    src_mask = (
        meta_raw["hospital"].isin(source_domains).values
        & (labels_raw == species)
    )
    # Exclude OOD holdout from source (keeps split indices aligned with training)
    if "year" in meta_raw.columns:
        for (ood_hosp, ood_year) in _OOD_HOLDOUT:
            ood_rows = (
                (meta_raw["hospital"] == ood_hosp) &
                (meta_raw["year"].astype(str) == str(ood_year))
            ).values
            src_mask = src_mask & ~ood_rows
    data_src   = data_norm[src_mask]
    amr_src    = amr_norm[src_mask]
    labels_src = labels_raw[src_mask]
    meta_src   = meta_raw.loc[src_mask].reset_index(drop=True)

    print(f"Source sub-dataset ({species}): {len(data_src)} samples")

    # Build target mask — supports logical keys like "MARISMA_2024"
    if target_domain in _DOMAIN_FILTERS:
        tgt_hospital, tgt_year = _DOMAIN_FILTERS[target_domain]
        tgt_dom_mask = (
            (meta_raw["hospital"] == tgt_hospital) &
            (meta_raw["year"].astype(str) == str(tgt_year))
        ).values
    else:
        tgt_dom_mask = (meta_raw["hospital"] == target_domain).values

    tgt_hospital_name = _DOMAIN_FILTERS.get(target_domain, (target_domain,))[0]
    if tgt_hospital_name == "MS-UMG" and "agar" in meta_raw.columns:
        chrom = (meta_raw["agar"] == "chrom").values
        tgt_dom_mask = tgt_dom_mask & ~chrom
        print(f"  {target_domain}: removed {chrom.sum()} chrom-agar samples")

    tgt_mask = tgt_dom_mask & (labels_raw == species)
    data_tgt   = data_norm[tgt_mask]
    amr_tgt    = amr_norm[tgt_mask]
    labels_tgt = labels_raw[tgt_mask]
    meta_tgt   = meta_raw.loc[tgt_mask].reset_index(drop=True)

    print(f"Target sub-dataset ({target_domain}, {species}): {len(data_tgt)} samples")


    ####################
    # LOAD SPLITS
    ####################
    print("Loading finetuning indices...")
    with open(splits_path, "rb") as f:
        splits_idx = pickle.load(f)

    # Build domain map
    DOMAIN_MAP = {d: i for i, d in enumerate(source_domains)}
    if target_domain not in DOMAIN_MAP:
        DOMAIN_MAP[target_domain] = len(source_domains)
    # When target_domain is a logical key (e.g. "MARISMA_2024"), meta_ft["hospital"]
    # contains the actual hospital name ("MARISMA"). Add it as an alias so .map() works.
    if target_domain in _DOMAIN_FILTERS:
        actual_hospital = _DOMAIN_FILTERS[target_domain][0]
        if actual_hospital not in DOMAIN_MAP:
            DOMAIN_MAP[actual_hospital] = DOMAIN_MAP[target_domain]

    source_data_list  = []
    source_label_list = []
    source_meta_list  = []
    source_amr_list   = []

    for src_dom in source_domains:
        if src_dom in splits_idx:
            idx = splits_idx[src_dom]["finetuning"]
            if len(idx) > 0:
                source_data_list.append(data_src[idx])
                source_label_list.append(labels_src[idx])
                source_meta_list.append(meta_src.iloc[idx].reset_index(drop=True))
                source_amr_list.append(amr_src[idx])
                print(f"  {src_dom}: {len(idx)} finetuning samples")

    idx_target = splits_idx[target_domain]["finetuning"]
    idx_test   = splits_idx[target_domain]["test"]

    data_target  = data_tgt[idx_target]  if len(idx_target) > 0 else np.empty((0, data_tgt.shape[1]))
    label_target = labels_tgt[idx_target] if len(idx_target) > 0 else np.array([])
    meta_target  = meta_tgt.iloc[idx_target].reset_index(drop=True) if len(idx_target) > 0 else pd.DataFrame()
    amr_target   = amr_tgt[idx_target]   if len(idx_target) > 0 else np.empty((0, amr_tgt.shape[1]))
    print(f"  {target_domain}: {len(idx_target)} finetuning, {len(idx_test)} test")


    ####################
    # BUILD FINETUNING DATASET
    ####################
    print("Building finetuning dataset...")

    if consider_prev_domains and len(source_data_list) > 0:
        print("  Including source domain data for anchoring...")
        parts_X     = source_data_list  + ([data_target]  if len(data_target)  > 0 else [])
        parts_y     = source_label_list + ([label_target] if len(label_target) > 0 else [])
        parts_meta  = source_meta_list  + ([meta_target]  if len(meta_target)  > 0 else [])
        parts_amr   = source_amr_list   + ([amr_target]   if len(amr_target)   > 0 else [])
        X_ft    = np.vstack(parts_X)
        y_ft    = np.concatenate(parts_y)
        meta_ft = pd.concat(parts_meta, ignore_index=True)
        amr_ft  = np.vstack(parts_amr)
    else:
        print("  Target-only finetuning (no source anchoring)")
        X_ft    = data_target
        y_ft    = label_target
        meta_ft = meta_target.copy()
        amr_ft  = amr_target

    if len(X_ft) == 0:
        print("WARNING: No finetuning data available. Returning pretrained model.")
        n_pretrained_domains = _count_decoders(pretrained_model_path)
        n_species_pretrained = 1

        if is_iwae:
            vae = model_class(
                input_dim=data_norm.shape[1],
                latent_dim=latent_dim,
                num_domains=n_pretrained_domains,
                n_species=n_species_pretrained,
                n_antibiotics=len(keep_names),
                lambda_amr=lambda_amr_eff,
                antibiotic_names=keep_names,
                epochs=ft_epochs,
                lr=ft_lr,
                patience=ft_patience,
                use_fixed_prior=use_fixed_prior,
                n_iwae_samples=5,
            )
        else:
            vae = model_class(
                input_dim=data_norm.shape[1],
                latent_dim=latent_dim,
                num_domains=n_pretrained_domains,
                n_species=n_species_pretrained,
                n_antibiotics=len(keep_names),
                lambda_amr=lambda_amr_eff,
                antibiotic_names=keep_names,
                epochs=ft_epochs,
                lr=ft_lr,
                patience=ft_patience,
                use_fixed_prior=use_fixed_prior,
            )
            
        vae.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        vae.to(device).eval()
        species_encoder = LabelEncoder()
        species_encoder.fit(TARGET_SPECIES)
        return vae, species_encoder, DOMAIN_MAP

    domain_ft = meta_ft["hospital"].map(DOMAIN_MAP).values.astype(np.int64)
    n_species_pretrained = 1

    species_encoder = LabelEncoder()
    species_encoder.fit(TARGET_SPECIES[:n_species_pretrained])

    y_ft_known = np.where(
        np.isin(y_ft, species_encoder.classes_),
        y_ft,
        species_encoder.classes_[0],
    )
    species_ft = species_encoder.transform(y_ft_known)

    print(f"Total finetuning samples: {len(X_ft)}")

    ####################
    # LOAD PRETRAINED MODEL
    ####################
    print("Loading pretrained AMR model...")
    n_pretrained_domains = _count_decoders(pretrained_model_path)

    common_kwargs = dict(
        input_dim=X_ft.shape[1],
        latent_dim=latent_dim,
        num_domains=n_pretrained_domains,
        n_species=n_species_pretrained,
        n_antibiotics=len(keep_names),
        lambda_amr=lambda_amr_eff,
        antibiotic_names=keep_names,
        epochs=ft_epochs,
        lr=ft_lr,
        annealing_epochs=annealing_epochs,
        patience=ft_patience,
        use_fixed_prior=use_fixed_prior,
    )

    if is_iwae:
        vae = model_class(
            **common_kwargs,
            n_iwae_samples=5,
        )
    else:
        vae = model_class(
            **common_kwargs
        )

    vae.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    vae.to(device)


    ####################
    # ADD NEW DECODER IF NEEDED
    ####################
    n_needed = max(int(domain_ft.max()) + 1, n_pretrained_domains)

    if n_needed > n_pretrained_domains:
        print(f"Adding {n_needed - n_pretrained_domains} new decoder(s)...")
        old_decoders = vae.decoder.net
        lat_dim = old_decoders[0][0].in_features
        out_dim = old_decoders[0][-2].out_features

        for _ in range(n_needed - n_pretrained_domains):
            new_dec = nn.Sequential(
                nn.Linear(lat_dim, 256), nn.ReLU(),
                nn.Linear(256, 512),    nn.ReLU(),
                nn.Linear(512, out_dim), nn.Sigmoid(),
            )
            vae.decoder.net = nn.ModuleList(list(vae.decoder.net) + [new_dec])

        vae.decoder.num_domains = len(vae.decoder.net)
        vae.to(device)

    new_domain_idx = DOMAIN_MAP[target_domain]


    ####################
    # FREEZE / UNFREEZE
    ####################
    for p in vae.parameters():
        p.requires_grad = False
    if finetuning_mode == "amr_head":
        for head in vae.amr_heads:
            for p in head.parameters():
                p.requires_grad = True
    elif finetuning_mode == "enc_dec":
        for p in vae.encoder.parameters():
            p.requires_grad = True
        for p in vae.decoder.net[new_domain_idx].parameters():
            p.requires_grad = True
    elif finetuning_mode == "enc_dec_amr":
        for p in vae.encoder.parameters():
            p.requires_grad = True
        for p in vae.decoder.net[new_domain_idx].parameters():
            p.requires_grad = True
        for p in vae.amr_drop.parameters():  
            p.requires_grad = True
        for head in vae.amr_heads:
            for p in head.parameters():
                p.requires_grad = True
    elif finetuning_mode == "enc_dec_amr_prior":
        for p in vae.encoder.parameters():
            p.requires_grad = True
        for p in vae.decoder.net[new_domain_idx].parameters():
            p.requires_grad = True
        if hasattr(vae, "prior"):
            for p in vae.prior.parameters():
                p.requires_grad = True
        for p in vae.amr_trunk.parameters():
            p.requires_grad = True
        for head in vae.amr_heads:
            for p in head.parameters():
                p.requires_grad = True
    else:
        raise ValueError(
            f"Unknown finetuning_mode: '{finetuning_mode}'. "
            "Options: amr_head, enc_dec, enc_dec_amr, enc_dec_amr_prior")
    
    ####################
    # TRAIN / VAL SPLIT
    ####################
    counts = np.bincount(species_ft)
    can_stratify = (len(np.unique(species_ft)) > 1) and (counts[counts > 0].min() >= 2)
    stratify_vec = species_ft if can_stratify else None

    if not can_stratify:
        print("Too few samples per class for stratification. Using random split.")

    X_tr, X_val, d_tr, d_val, s_tr, s_val, a_tr, a_val = train_test_split(
        X_ft, domain_ft, species_ft, amr_ft,
        test_size=0.2,
        stratify=stratify_vec,
        random_state=random_state
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(d_tr, dtype=torch.long),
            torch.tensor(s_tr, dtype=torch.long),
            torch.tensor(a_tr, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(d_val, dtype=torch.long),
            torch.tensor(s_val, dtype=torch.long),
            torch.tensor(a_val, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=False,
    )

    print(f"Train: {len(X_tr)} | Val: {len(X_val)}")


    ####################
    # FINETUNING
    ####################
    print("\n===== STARTING FINETUNING =====")

    vae.optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, vae.parameters()),
        lr=ft_lr, weight_decay=1e-5)
    
    vae.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(vae.optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6, cooldown=5)
    vae.epochs = ft_epochs
    vae.patience = ft_patience

    vae.n_amr_samples = n_amr_samples
    print(f"AMR z-augmentation: {vae.n_amr_samples} samples per forward pass")

    vae.trainloop(
        trainloader=train_loader,
        validloader=val_loader,
        device=device,

    )

    # Go back to 1 for evaluation
    vae.n_amr_samples = 1

    # model_filename = (
    #     f"model_finetuned_amr_"
    #     f"target={target_domain}_"
    #     f"prev={n_prev}_new={n_new}_"
    #     f"mode={finetuning_mode}.pth"
    # )
    # torch.save(vae.state_dict(), experiment_dir / model_filename)
    # print(f"Model saved: {experiment_dir / model_filename}")

    ####################
    # LATENT EVALUATION 
    ####################
    if run_latent_evaluation:
        print("\n===== LATENT EVALUATION =====")
        from src.evaluation.eval import encode_latent

        if len(data_target) > 0:
            X_eval = np.vstack([data_src, data_target])
            y_eval = np.concatenate([labels_src, label_target])
            meta_eval = pd.concat([meta_src, meta_target], ignore_index=True)
            amr_eval = np.vstack([amr_src, amr_target])
        else:
            X_eval = data_src
            y_eval = labels_src
            meta_eval = meta_src.copy()
            amr_eval = amr_src

        Z_eval = encode_latent(vae, X_eval, device)
        
        figure_prefix = (
            f"AMR_FT_target={target_domain}_"
            f"prev={n_prev}_new={n_new}_"
            f"mode={finetuning_mode}"
        )
        
        if amr_eval is not None and len(keep_names) > 0:
            for i, atb_name in enumerate(keep_names):
                if amr_eval.ndim > 1:
                    meta_eval[atb_name] = amr_eval[:, i]
                else:
                    meta_eval[atb_name] = amr_eval

        run_tsne_evaluation(
            mus_all=Z_eval,
            label_final=y_eval,
            meta_final=meta_eval,
            output_dir=experiment_dir,
            prefix=figure_prefix,
            antibiotics_list=keep_names,
            target_domain_name=target_domain
        )
        print("Latent evaluation finished.")

    print("\n===== FINETUNING COMPLETE =====")
    return vae, species_encoder, DOMAIN_MAP


# ---------------------------------------------------------------------------
# Helpers to inspect a saved state dict
# ---------------------------------------------------------------------------

def _count_decoders(model_path):
    """Count number of decoders from a saved state dict."""
    state = torch.load(model_path, map_location="cpu")
    indices = {int(k.split(".")[2]) for k in state if k.startswith("decoder.net.")}
    return max(indices) + 1 if indices else 1
