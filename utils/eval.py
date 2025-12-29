import torch
import numpy as np
from utils.viz import compute_tsne_df, compute_tsne_per_species, plot_tsne_global, plot_tsne_species


def eval_model(model, dataloader, device, use_domain=False):
    """
    Compute latent representations for all samples in a given dataloader.

    This function runs the encoder in evaluation mode and extracts the
    latent mean vectors (mu) for each input sample. It supports both
    standard VAEs and conditional/multi-domain VAEs.

    Parameters
    ----------
    model : torch.nn.Module
        Trained VAE-like model with an encoder that returns (mu, logvar).
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the input samples (and optionally domain IDs).
    device : torch.device
        Device on which the model and data should be evaluated.
    use_domain : bool, optional
        If True, domain information is used as a conditional input to the
        encoder (e.g., for conditional or multi-domain VAEs). Default is False.

    Returns
    -------
    mus_all : np.ndarray
        Array of shape (N, latent_dim) containing the latent mean vectors
        for all samples in the dataloader.
    """

    model.eval()
    model.to(device)
    mus_all = []

    with torch.no_grad():
        for batch in dataloader:

            if len(batch) == 3:
                x, domain_id, species_id = batch
            elif len(batch) == 2:
                x, domain_id = batch
                species_id = None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")

            x = x.to(device)

            # Species-conditioned encoder 
            if species_id is not None and hasattr(model, "n_species"):
                species_id = species_id.to(device)
                c = torch.nn.functional.one_hot(
                    species_id, num_classes=model.n_species
                ).float().to(device)

                mu, logvar = model.encoder(x, c)

            # Invariant encoder
            else:
                mu, logvar = model.encoder(x)

            mus_all.append(mu.cpu().numpy())

    return np.concatenate(mus_all, axis=0)


def run_tsne_evaluation(mus_all, label_final, meta_final, output_dir, prefix):
    """
    Run t-SNE analysis on latent representations and generate visualization plots.

    This function computes t-SNE embeddings from the provided latent vectors
    and generates multiple visualizations, including:
    - global t-SNE
    - per-species t-SNE
    - optional hospital/domain overlays

    All plots are saved to the specified output directory.

    Parameters
    ----------
    mus_all : np.ndarray
        Latent representations of shape (N, latent_dim) for all samples.
    label_final : np.ndarray
        Array of class labels corresponding to each sample.
    meta_final : pandas.DataFrame
        Metadata DataFrame containing sample-level information (e.g. hospital).
    output_dir : pathlib.Path
        Directory where all generated plots will be saved.

    Returns
    -------
    None
        The function produces and saves plots to disk but does not return values.
    """

    tsne_df = compute_tsne_df(mus_all, label_final, meta_final)
    df_all, tsne_results = compute_tsne_per_species(mus_all, label_final, meta_final)

    plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_global.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_global_species.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path=output_dir / f"{prefix}_tsne_global_species_overlay.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_year=True, save=True, path=output_dir / f"{prefix}_tsne_global_species_overlay_year.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, save=True, path=output_dir / f"{prefix}_tsne_species.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=True, save=True, path=output_dir / f"{prefix}_tsne_species_overlay.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_year_per_species=True, save=True, path=output_dir / f"{prefix}_tsne_species_overlay_year.png")
