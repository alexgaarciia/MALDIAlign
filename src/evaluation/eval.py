import torch
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

from torch.utils.data import TensorDataset, DataLoader

from src.visualization.viz import compute_tsne_df, compute_tsne_per_species, plot_tsne_global, plot_tsne_species
from src.visualization.amr_viz import plot_tsne_amr, plot_tsne_amr_split_by_hospital


def eval_model(model, dataloader, device, use_domain=False):
    """
    Extracts latent representations for all samples in a given dataloader.

    Supports probabilistic encoders (extracting the mean μ) and deterministic 
    encoders. The function dynamically handles batch sizes of 2, 3, or 4 elements 
    to accommodate domain, species, and AMR data. Conditioning is applied if 
    the model possesses a `domain_emb` or if `use_domain` is enabled via one-hot 
    encoding.

    Args:
        model (torch.nn.Module): Trained model exposing an `encoder` module.
        dataloader (torch.utils.data.DataLoader): Data provider for batches.
        device (torch.device): Device (CPU/CUDA) for evaluation.
        use_domain (bool): If True, provides one-hot domain conditioning to the 
            encoder (requires `model.cond_dim`).

    Returns:
        np.ndarray: Concatenated latent representations of shape (N, latent_dim).
    """
    
    model.eval()
    model.to(device)

    mus_all = []

    with torch.no_grad():

        for batch in dataloader:

            # ------------------------
            # Unpack batch
            # ------------------------
            if len(batch) == 4:
                x, domain_id, species_id, amr = batch
            elif len(batch) == 3:
                x, domain_id, species_id = batch
                amr = None
            elif len(batch) == 2:
                x, domain_id = batch
                species_id = None
                amr = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            x = x.to(device)

            # ------------------------
            # Conditional encoder (AMR)
            # ------------------------
            if hasattr(model, "domain_emb"):
                domain_id = domain_id.to(device)
                c = model.domain_emb(domain_id)
                mu, _ = model.encoder(x, c)

            # ------------------------
            # Domain conditioning
            # ------------------------
            elif use_domain and hasattr(model, "cond_dim"):
                domain_id = domain_id.to(device)
                c = torch.nn.functional.one_hot(
                    domain_id,
                    num_classes=model.cond_dim
                ).float().to(device)

                mu, _ = model.encoder(x, c)

            # ------------------------
            # Standard encoder
            # ------------------------
            else:
                out = model.encoder(x)
                if isinstance(out, tuple):
                    mu, _ = out
                else:
                    mu = out

            mus_all.append(mu.cpu().numpy())

    return np.concatenate(mus_all, axis=0)

def run_tsne_evaluation(mus_all, label_final, meta_final, output_dir, prefix, prior_samples=None, prior_labels=None, antibiotics_list=None, target_domain_name=None):
    """
    Executes a t-SNE visualization suite on latent representations.

    Computes t-SNE embeddings and generates multiple plots, including global 
    distributions, species-specific clusters, and hospital/year overlays. 
    Optionally incorporates prior distribution samples for comparison and 
    generates AMR-specific visualizations for listed antibiotics.

    Args:
        mus_all (np.ndarray): Latent representations of real samples.
        label_final (np.ndarray): Class/species labels for real samples.
        meta_final (pd.DataFrame): Metadata (hospital, year, AMR status).
        output_dir (Path): Directory where PNG plots will be saved.
        prefix (str): String prefix for identifying saved files.
        prior_samples (np.ndarray, optional): Latent samples from a prior.
        prior_labels (np.ndarray, optional): Labels for the prior samples.
        antibiotics_list (list of str, optional): Antibiotic columns to plot.

    Returns:
        None: Plots are saved directly to the specified directory.
    """

    if prior_samples is not None and prior_labels is not None:
        prior_meta = pd.DataFrame({
            "year": [np.nan] * len(prior_labels),
            "hospital": ["Prior"] * len(prior_labels)
        })

        X_total = np.vstack([mus_all, prior_samples])
        labels_total = np.concatenate([label_final, prior_labels])

        source = ["real"] * len(label_final) + ["prior"] * len(prior_labels)

        meta_total = pd.concat([meta_final, prior_meta], ignore_index=True)

    else:
        X_total = mus_all
        labels_total = label_final
        meta_total = meta_final
        source = None

    tsne_df = compute_tsne_df(X_total, labels_total, meta_total, source)  
    df_all, tsne_results = compute_tsne_per_species(X_total, labels_total, meta_total, source=source)  

    plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_global.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_global_species.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path=output_dir / f"{prefix}_tsne_global_species_overlay.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_year=True, save=True, path=output_dir / f"{prefix}_tsne_global_species_overlay_year.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_species.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=True, save=True, path=output_dir / f"{prefix}_tsne_species_overlay.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_year_per_species=True, save=True, path=output_dir / f"{prefix}_tsne_species_overlay_year.png")

    if antibiotics_list is not None:
        print(f"Generating AMR t-SNE plots for: {antibiotics_list}")
        for atb in antibiotics_list:
            if atb in df_all.columns:
                safe_atb = atb.replace("/", "-")
                out_path = output_dir / f"{prefix}_tsne_amr_{safe_atb}.png"
                plot_tsne_amr(df_all, tsne_results, antibiotic_col=atb, save=True, path=out_path)

                if target_domain_name:
                    out_path_split = output_dir / f"{prefix}_tsne_amr_{safe_atb}_SPLIT.png"
                    plot_tsne_amr_split_by_hospital(df_all, atb, target_domain_name, save=True, path=out_path_split)
    else:
        print("No AMR data provided. Skipping AMR t-SNE plots.")

def encode_latent(model, X, device, domain=None, amr=None, batch_size=256):
    """
    Encodes raw numpy arrays into latent space by wrapping them in a DataLoader.

    A utility for out-of-training inference. It handles optional conditioning 
    on domain embeddings or AMR status and extracts the deterministic latent 
    mean from the encoder.

    Args:
        model (torch.nn.Module): Model with an `encoder` module.
        X (np.ndarray): Feature matrix.
        device (torch.device): Device (CPU/CUDA) for computation.
        domain (np.ndarray, optional): Domain/hospital IDs for conditioning.
        amr (np.ndarray, optional): AMR status for conditional encoding.
        batch_size (int): Size of batches for processing.

    Returns:
        np.ndarray: Stacked latent representations for the input features.
    """

    model.eval()
    Z = []

    X_tensor = torch.tensor(X, dtype=torch.float32)

    tensors = [X_tensor]

    if domain is not None:
        domain_tensor = torch.tensor(domain, dtype=torch.long)
        tensors.append(domain_tensor)

    if amr is not None:
        amr_tensor = torch.tensor(amr, dtype=torch.float32)
        tensors.append(amr_tensor)

    dataset = torch.utils.data.TensorDataset(*tensors)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():

        for batch in loader:

            x = batch[0].to(device)

            domain_id = None
            a = None

            if domain is not None:
                domain_id = batch[1].to(device)

            if amr is not None:
                a = batch[-1].to(device).view(-1, 1)
 
            # -------------------------
            # Forward logic
            # -------------------------
            if hasattr(model, "domain_emb") and domain_id is not None:
                c = model.domain_emb(domain_id)
                out = model.encoder(x, c)

            elif a is not None:
                out = model.encoder(x, a)

            else:
                out = model.encoder(x)

            # -------------------------
            # VAE vs deterministic
            # -------------------------
            if isinstance(out, tuple):
                mu, _ = out
            else:
                mu = out

            Z.append(mu.cpu().numpy())

    return np.vstack(Z)

def make_loader(X, y, batch_size=256, shuffle=False):
    """
    Converts numpy arrays into a standard PyTorch DataLoader.

    Args:
        X (np.ndarray): Input feature arrays.
        y (np.ndarray): Target label arrays.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data every epoch.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing TensorDatasets.
    """

    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=shuffle
    )

def load_model(model, path):
    """
    Loads a model's state dictionary from disk and prepares it for inference.

    Automatically detects the available hardware (CUDA vs CPU) to map 
    storage, loads the weights, and sets the model to evaluation mode.

    Args:
        model (torch.nn.Module): The model architecture to populate.
        path (str): File path to the saved `.pt` or `.pth` state dictionary.

    Returns:
        torch.nn.Module: The model moved to the appropriate device in eval mode.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_amr_head(model, X, amr_labels, ab_list, device, batch_size=512, species=None, n_species=None):
    """
    Evaluates AMR prediction heads of a trained VAE model.
    Returns per-antibiotic AUROC, balanced accuracy (threshold=0.5), PR-AUC and n.
    """
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i:i + batch_size].to(device)
            mu, _ = model.encoder(batch)

            if hasattr(model, "amr_trunk"):
                h = model.amr_trunk(mu)
            elif hasattr(model, "amr_drop"):
                h = model.amr_drop(mu)
            else:
                h = mu

            if species is not None:
                sp_batch = torch.tensor(
                    species[i:i + batch_size], dtype=torch.long, device=device
                )
                if hasattr(model, "species_emb"):
                    u_s = model.species_emb(sp_batch)
                    h = torch.cat([h, u_s], dim=1)
                elif n_species is not None:
                    head_input_dim = model.amr_heads[0].in_features
                    if head_input_dim > h.shape[1]:
                        u_s = torch.nn.functional.one_hot(
                            sp_batch, num_classes=n_species
                        ).float()
                        h = torch.cat([h, u_s], dim=1)

            amr_logits = torch.cat([head(h) for head in model.amr_heads], dim=1)
            all_logits.append(amr_logits.cpu())

    all_logits = torch.cat(all_logits, dim=0).numpy()
    probs = 1 / (1 + np.exp(-all_logits))

    results = {}
    for j, atb_name in enumerate(ab_list):
        y_true = amr_labels[:, j]
        valid  = ~np.isnan(y_true)
        y_true_clean = y_true[valid].astype(int)
        y_prob       = probs[valid, j]
        y_pred       = (y_prob >= 0.5).astype(int)

        if len(y_true_clean) < 10 or len(np.unique(y_true_clean)) < 2:
            continue

        results[atb_name] = {
            "auc":      roc_auc_score(y_true_clean, y_prob),
            "bal_acc":  balanced_accuracy_score(y_true_clean, y_pred),
            "pr_auc":   average_precision_score(y_true_clean, y_prob),
            "n":        int(valid.sum()),
        }
    return results
