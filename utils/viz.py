import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_model_metrics(model, model_name):
    """
    Plot training and validation losses for a VAE-like model.
    """
    train_losses = [t[0] for t in model.loss_during_training]
    train_recon = [t[0] for t in model.reconstruc_during_training]
    train_kl = [t[0] for t in model.KL_during_training]

    val_losses = [t[1] for t in model.loss_during_training]
    val_recon = [t[1] for t in model.reconstruc_during_training]
    val_kl = [t[1] for t in model.KL_during_training]

    epochs = range(1, len(train_losses) + 1)

    # Training 
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Total loss", linewidth=2)
    plt.plot(epochs, train_recon, label="Reconstruction (MSE)", linestyle="--")
    plt.plot(epochs, train_kl, label="KL divergence", linestyle=":")
    plt.title(f"{model_name} Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Validation
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label="Total loss", linewidth=2)
    plt.plot(epochs, val_recon, label="Reconstruction (MSE)", linestyle="--")
    plt.plot(epochs, val_kl, label="KL divergence", linestyle=":")
    plt.title(f"{model_name} Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_tsne_df(mus, labels, metadata):
    """
    Compute global t-SNE embeddings for all samples.
    """
    tsne = TSNE(n_components=2, random_state=42)
    Z_all = tsne.fit_transform(mus)

    tsne_df = pd.DataFrame({
        "x": Z_all[:, 0],
        "y": Z_all[:, 1],
        "species": labels,
        "hospital": metadata["hospital"].values
    })
    return tsne_df


def plot_tsne_global(tsne_df, per_species=False):
    """
    Plot t-SNE embeddings globally (colored by species)
    or per species (colored by hospital).
    """
    if not per_species:
        plt.figure(figsize=(8, 6))
        for sp in sorted(tsne_df["species"].unique()):
            subset = tsne_df[tsne_df["species"] == sp]
            plt.scatter(subset["x"], subset["y"], s=12, alpha=0.35, label=sp)

        plt.title("t-SNE (colored by species)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Species", markerscale=2, loc="upper left",
                   frameon=True, fontsize=8, title_fontsize=9)
        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, sp in enumerate(sorted(tsne_df["species"].unique())):
            subset = tsne_df[tsne_df["species"] == sp]
            for hosp in tsne_df["hospital"].unique():
                sub_h = subset[subset["hospital"] == hosp]
                axes[i].scatter(sub_h["x"], sub_h["y"], s=12, alpha=0.5,
                                label=hosp if i == 0 else None)
            axes[i].set_title(sp.replace("_", " "), fontsize=15)
            axes[i].set_xticks([]); axes[i].set_yticks([])

        axes[0].legend(title="Hospital", loc="upper left",
                       frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
        plt.suptitle("t-SNE embeddings per species (colored by hospital)", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def compute_tsne_per_species(mus, labels, metadata):
    """
    Compute independent t-SNEs per species.
    """
    df_all = pd.DataFrame({
        "species": labels,
        "hospital": metadata["hospital"].values
    })
    for d in range(mus.shape[1]):
        df_all[f"z{d}"] = mus[:, d]

    tsne_results = {}
    for sp in sorted(np.unique(labels)):
        mask = (df_all["species"] == sp)
        X_sp = mus[mask]
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sp)
        tsne_results[sp] = {
            "embedding": X_tsne,
            "hospital": df_all.loc[mask, "hospital"].values
        }

    return df_all, tsne_results


def plot_tsne_species(df_all, tsne_results):
    """
    Plot t-SNE embeddings computed independently for each species.
    """
    species_sorted = sorted(tsne_results.keys())
    n_species = len(species_sorted)
    n_rows = (n_species + 2) // 3 
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    for i, sp in enumerate(species_sorted):
        emb = tsne_results[sp]["embedding"]
        hosp = tsne_results[sp]["hospital"]
        for h in sorted(df_all["hospital"].unique()):
            idx = (hosp == h)
            axes[i].scatter(emb[idx, 0], emb[idx, 1], s=12, alpha=0.45,
                            label=h if i == 0 else None)
        axes[i].set_title(sp.replace("_", " "), fontsize=15)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    axes[0].legend(title="Hospital", loc="upper left",
                   frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
    plt.suptitle("Independent t-SNE embeddings per species (colored by hospital)",
                 fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
