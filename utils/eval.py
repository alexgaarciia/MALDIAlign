import torch
import numpy as np
from utils.viz import compute_tsne_df, compute_tsne_per_species, plot_tsne_global, plot_tsne_species


def eval_model(model, dataloader, device, use_domain=False):
    """
    Extract latent representations for all samples in a given dataloader.

    This function runs the model's encoder in evaluation mode and returns
    a deterministic latent representation for each input sample. It supports
    both probabilistic encoders (e.g. VAEs) and deterministic encoders
    (e.g. Domain-Adversarial Neural Networks).

    - For VAE-like models, the latent representation corresponds to the
      posterior mean μ of q(z | x).
    - For deterministic models (e.g. DANN), the latent representation
      corresponds directly to the encoder output z.

    Conditional encoders are also supported when `use_domain=True`.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model exposing an `encoder` module. The encoder may return
        either:
        - a tuple (mu, logvar) for probabilistic models, or
        - a single tensor z for deterministic models.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of input samples. Each batch is expected
        to be either:
        - (x, domain_id, species_id), or
        - (x, domain_id).
    device : torch.device
        Device on which the model and data are evaluated.
    use_domain : bool, optional
        If True and the model supports conditional encoding, domain information
        is provided to the encoder as a one-hot conditioning vector.
        Default is False.

    Returns
    -------
    mus_all : np.ndarray
        Array of shape (N, latent_dim) containing the latent representations
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

            # Conditional encoder (domain-conditioned)
            if use_domain and hasattr(model, "cond_dim"):
                domain_id = domain_id.to(device)
                c = torch.nn.functional.one_hot(
                    domain_id,
                    num_classes=model.cond_dim
                ).float().to(device)

                mu, _ = model.encoder(x, c)

            # Plain encoder
            else:
                enc_out = model.encoder(x)

                if isinstance(enc_out, tuple):
                    # VAE-style
                    mu, _ = enc_out
                else:
                    # DANN-style
                    mu = enc_out

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


def encode_latent(model, X, device, batch_size=256):
    """
    Encode input data into latent space using a trained VAE model.

    This function passes the input data through the model's encoder
    and extracts the posterior mean (μ) of the latent distribution
    for each sample. It assumes a VAE-style encoder returning (mu, logvar).

    Parameters
    ----------
    model : torch.nn.Module
        Trained model containing an `encoder` method that returns
        (mu, logvar).
    X : np.ndarray
        Input data of shape (N, input_dim), where N is the number
        of samples.
    device : torch.device
        Device on which computation will be performed (CPU or CUDA).
    batch_size : int, optional (default=256)
        Batch size used during encoding to avoid memory overflow.

    Returns
    -------
    Z : np.ndarray
        Latent representations of shape (N, latent_dim),
        corresponding to the posterior mean μ for each sample.
    """

    model.eval()
    Z = []

    X_tensor = torch.tensor(X, dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        X_tensor,
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu, _ = model.encoder(x)
            Z.append(mu.cpu().numpy())

    return np.vstack(Z)
