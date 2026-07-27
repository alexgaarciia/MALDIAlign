import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations_with_replacement
from sklearn.metrics.pairwise import cosine_similarity


def build_pike_gram_matrix(D, t=8, device=None):
    """
    Builds the fixed PIKE Gram matrix G (D x D) such that
    PIKE(x, y) = x @ G @ y / (4t*pi).

    This is the expensive part, but it only needs to be computed ONCE
    for the whole experiment.

    Parameters
    ----------
    D : int
        Spectrum dimensionality (size of G).
    t : float, default=8
        PIKE kernel bandwidth parameter.
    device : torch.device or str, optional
        Device to build the matrix on. Defaults to CUDA if available,
        otherwise CPU.

    Returns
    -------
    torch.Tensor
        Gram matrix G, shape (D, D), float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = torch.arange(D, device=device, dtype=torch.float32)
    diff_sq = (positions.unsqueeze(1) - positions.unsqueeze(0)) ** 2  # (D, D)
    G = torch.exp(-diff_sq / (4 * t)) / (4 * t * math.pi)
    return G  # (D, D), float32


def compute_all_domain_species_pike(data_all, meta_all, label_all, t=8, device=None, batch_size=2000):
    """
    Computes, for every species and every domain, the domain x domain
    mean PIKE similarity matrix, using the vectorized bilinear
    formulation (no Python loops over samples).

    Parameters
    ----------
    data_all : numpy.ndarray
        Spectra, shape (n_samples, D).
    meta_all : pandas.DataFrame
        Metadata with a "hospital" column, aligned with `data_all`.
    label_all : numpy.ndarray
        Species label per sample, aligned with `data_all`.
    t : float, default=8
        PIKE kernel bandwidth parameter.
    device : torch.device or str, optional
        Device to run the computation on. Defaults to CUDA if
        available, otherwise CPU.
    batch_size : int, default=2000
        Batch size used when computing cross-domain similarity, to
        bound memory usage for large domains.

    Returns
    -------
    dict of str -> pandas.DataFrame
        Per-species hospital x hospital mean PIKE similarity matrix.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = data_all.shape[1]
    G = build_pike_gram_matrix(D, t=t, device=device)  # (D, D), reused for everything

    species_list = sorted(np.unique(label_all))
    hospitals_all = sorted(meta_all["hospital"].unique())

    results = {}

    for sp in species_list:
        mask_sp = (label_all == sp)
        hospitals = [h for h in hospitals_all if ((meta_all["hospital"].values == h) & mask_sp).sum() > 0]

        # Preload and precompute self-similarity (K_ii) for each domain, once
        domain_data = {}
        domain_self_K = {}
        for h in hospitals:
            idx_h = np.where((meta_all["hospital"].values == h) & mask_sp)[0]
            X_h = torch.as_tensor(data_all[idx_h], dtype=torch.float32, device=device)  # (n_h, D)
            domain_data[h] = X_h
            # K_ii = diag(X @ G @ X.T), equivalent to sum((X @ G) * X, dim=1)
            XG = X_h @ G  # (n_h, D)
            K_ii = (XG * X_h).sum(dim=1)  # (n_h,)
            domain_self_K[h] = K_ii

        sim_matrix = pd.DataFrame(np.nan, index=hospitals, columns=hospitals)

        for h1, h2 in combinations_with_replacement(hospitals, 2):
            X1, X2 = domain_data[h1], domain_data[h2]
            K1, K2 = domain_self_K[h1], domain_self_K[h2]

            # K_cross = X1 @ G @ X2.T, batched in case X1/X2 are very large
            n1 = X1.shape[0]
            cross_vals = []
            for start in range(0, n1, batch_size):
                end = min(start + batch_size, n1)
                X1_batch = X1[start:end]  # (b, D)
                K1_batch = K1[start:end]  # (b,)
                cross = (X1_batch @ G) @ X2.T  # (b, n2)
                norm = torch.sqrt(K1_batch.unsqueeze(1) * K2.unsqueeze(0) + 1e-10)
                pike_norm = cross / norm  # (b, n2)
                cross_vals.append(pike_norm.mean().item() * pike_norm.numel())
            mean_sim = sum(cross_vals) / (n1 * X2.shape[0])

            sim_matrix.loc[h1, h2] = mean_sim
            sim_matrix.loc[h2, h1] = mean_sim

        results[sp] = sim_matrix

    return results


def compute_all_domain_species_cosine(X, meta_all, label_all):
    """
    Computes, for every species and every domain, the domain x domain
    mean cosine similarity matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Spectra or latent representations, shape (n_samples, D).
    meta_all : pandas.DataFrame
        Metadata with a "hospital" column, aligned with `X`.
    label_all : numpy.ndarray
        Species label per sample, aligned with `X`.

    Returns
    -------
    dict of str -> pandas.DataFrame
        Per-species hospital x hospital mean cosine similarity matrix.
    """
    species_list = sorted(np.unique(label_all))
    hospitals_all = sorted(meta_all["hospital"].unique())
    results = {}

    for sp in species_list:
        mask_sp = (label_all == sp)
        hospitals = [h for h in hospitals_all if ((meta_all["hospital"].values == h) & mask_sp).sum() > 0]

        domain_data = {h: X[np.where((meta_all["hospital"].values == h) & mask_sp)[0]] for h in hospitals}

        sim_matrix = pd.DataFrame(np.nan, index=hospitals, columns=hospitals)
        for h1, h2 in combinations_with_replacement(hospitals, 2):
            mean_sim = cosine_similarity(domain_data[h1], domain_data[h2]).mean()
            sim_matrix.loc[h1, h2] = mean_sim
            sim_matrix.loc[h2, h1] = mean_sim

        results[sp] = sim_matrix

    return results


def normalize_diagonal_to_one(sim_matrix):
    """
    Normalizes a similarity matrix so that its diagonal is exactly 1,
    using R_ij = S_ij / sqrt(S_ii * S_jj).

    Parameters
    ----------
    sim_matrix : pandas.DataFrame
        Square similarity matrix.

    Returns
    -------
    pandas.DataFrame
        Normalized similarity matrix, same index/columns as the input.
    """
    mat = sim_matrix.astype(float)
    diag = np.diag(mat.values)
    norm_factor = np.sqrt(np.outer(diag, diag))
    norm_mat = mat.values / norm_factor
    return pd.DataFrame(norm_mat, index=mat.index, columns=mat.columns)


def format_hospital_label(h):
    """
    Formats a hospital/domain identifier for display (e.g. plot ticks).

    Parameters
    ----------
    h : str
        Hospital identifier, e.g. "DRIAMS_A".

    Returns
    -------
    str
        Display-formatted label, e.g. "DRIAMS-A".
    """
    return h.replace("_", "-")


def plot_similarity_heatmap_combined(sim_raw, sim_latent, save_path=None, vmin=0.5, vmax=1, cmap="viridis", figsize=(7, 6)):
    """
    Plots a single heatmap combining two similarity matrices: raw-space
    similarity below the diagonal, latent-space similarity above it.

    Parameters
    ----------
    sim_raw : pandas.DataFrame
        Hospital x hospital similarity matrix in the original (raw)
        space.
    sim_latent : pandas.DataFrame
        Hospital x hospital similarity matrix in the latent space, same
        index/columns as `sim_raw`.
    save_path : str or pathlib.Path, optional
        If given, the figure is saved to this path.
    vmin : float, default=0.5
        Minimum value for the color scale.
    vmax : float, default=1
        Maximum value for the color scale.
    cmap : str, default="viridis"
        Matplotlib colormap name.
    figsize : tuple of float, default=(7, 6)
        Figure size in inches.

    Returns
    -------
    None
        Displays the figure and optionally saves it to `save_path`.
    """
    raw = sim_raw.astype(float).copy()
    lat = sim_latent.astype(float).copy()
    raw.index = [format_hospital_label(h) for h in raw.index]
    raw.columns = [format_hospital_label(h) for h in raw.columns]
    lat.index = [format_hospital_label(h) for h in lat.index]
    lat.columns = [format_hospital_label(h) for h in lat.columns]

    n = len(raw)
    combined = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if i > j:
                combined[i, j] = raw.values[i, j]
            elif i < j:
                combined[i, j] = lat.values[i, j]

    combined_df = pd.DataFrame(combined, index=raw.index, columns=raw.columns)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(combined_df, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, square=True, cbar=True, ax=ax, linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.8}, mask=np.isnan(combined))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    for k in range(n + 1):
        ax.plot([k, k+1], [k, k+1] if k < n else [n, n], color="white", lw=2)
    fig.subplots_adjust(right=0.82)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def aggregate_across_species(results_dict, all_hospitals):
    """
    Averages per-species similarity matrices into a single matrix.

    For each hospital pair, averages across species that have data for
    both hospitals (missing entries are excluded from the average
    rather than counted as zero).

    Parameters
    ----------
    results_dict : dict of str -> pandas.DataFrame
        Per-species hospital x hospital similarity matrices, as
        returned by `compute_all_domain_species_cosine` or
        `compute_all_domain_species_pike`.
    all_hospitals : list of str
        Full list of hospital identifiers to index the output on.

    Returns
    -------
    pandas.DataFrame
        Hospital x hospital similarity matrix averaged across species.
    """
    agg = pd.DataFrame(0.0, index=all_hospitals, columns=all_hospitals)
    counts = pd.DataFrame(0, index=all_hospitals, columns=all_hospitals)
    for sp, mat in results_dict.items():
        agg.loc[mat.index, mat.columns] += mat.fillna(0)
        counts.loc[mat.index, mat.columns] += mat.notna().astype(int)
    return agg / counts.replace(0, np.nan)


def off_diagonal_stats(sim_matrix):
    """
    Computes the mean, standard deviation and minimum of the
    off-diagonal values of a normalized similarity matrix.

    Parameters
    ----------
    sim_matrix : pandas.DataFrame
        Square similarity matrix.

    Returns
    -------
    dict of str -> float
        "mean_off_diag", "std_off_diag" and "min_off_diag" of the
        off-diagonal, non-NaN values.
    """
    mat = sim_matrix.astype(float).values
    n = mat.shape[0]
    off_diag_vals = mat[~np.eye(n, dtype=bool)]
    off_diag_vals = off_diag_vals[~np.isnan(off_diag_vals)]

    return {
        "mean_off_diag": np.mean(off_diag_vals),
        "std_off_diag": np.std(off_diag_vals),
        "min_off_diag": np.min(off_diag_vals),
    }
