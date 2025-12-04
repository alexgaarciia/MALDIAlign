import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_model_metrics(model, model_name, save=False, path=None):
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

    if save and path:
        plt.savefig(path + "_train.png")
        plt.close()
    else:
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
    if save and path:
        plt.savefig(path + "_val.png")
        plt.close()
    else:
        plt.show()
        

def compute_tsne_df(X, labels, metadata):
    """
    Compute global t-SNE embeddings for given samples.
    """
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(X)

    tsne_df = pd.DataFrame({
        "x": data_tsne[:, 0],
        "y": data_tsne[:, 1],
        "species": labels,
        "year": metadata["year"].values, 
        "hospital": metadata["hospital"].values
    })

    return tsne_df


def plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, save=False, path=None):
    """
    Plot t-SNE embeddings globally (colored by species)
    or per species (colored by hospital).
    """

    # ----------- 1) GLOBAL VIEW (colored by species) -----------
    if not per_species and not overlay_per_hospital:
        plt.figure(figsize=(8, 6))
        for sp in sorted(tsne_df["species"].unique()):
            subset = tsne_df[tsne_df["species"] == sp]
            plt.scatter(subset["x"], subset["y"], s=10, alpha=0.25, label=sp)

        plt.title("t-SNE (colored by species)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Species", markerscale=2, loc="upper left",
                frameon=True, fontsize=8, title_fontsize=9)
        plt.tight_layout()

        if save and path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    # ----------- 2) PER SPECIES (colored by hospital) -----------
    elif per_species and not overlay_per_hospital:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, sp in enumerate(sorted(tsne_df["species"].unique())):
            subset = tsne_df[tsne_df["species"] == sp]
            for hosp in sorted(tsne_df["hospital"].unique()):
                sub_h = subset[subset["hospital"] == hosp]
                axes[i].scatter(sub_h["x"], sub_h["y"], s=10, alpha=0.25,
                                label=hosp if i == 0 else None)
            axes[i].set_title(sp.replace("_", " "), fontsize=15)
            axes[i].set_xticks([]); axes[i].set_yticks([])

        axes[0].legend(title="Hospital", loc="upper left",
                    frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
        plt.suptitle("t-SNE embeddings per species (colored by hospital)", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save and path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    # ----------- 3) PER SPECIES OVERLAY PER HOSPITAL -----------
    elif overlay_per_hospital:
        hospitals = sorted(tsne_df["hospital"].unique())

        for hosp_focus in hospitals:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(sorted(tsne_df["species"].unique())):
                subset = tsne_df[tsne_df["species"] == sp]
                
                # background = all other hospitals
                background = subset[subset["hospital"] != hosp_focus]
                focus = subset[subset["hospital"] == hosp_focus]

                # plot background faintly
                axes[i].scatter(background["x"], background["y"], s=8, alpha=0.1, color="gray")

                # highlight focus hospital
                axes[i].scatter(focus["x"], focus["y"], s=12, alpha=0.5, color="red", label=hosp_focus)

                axes[i].set_title(sp.replace("_", " "), fontsize=15)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            axes[0].legend(loc="upper left", fontsize=12, frameon=True)
            plt.suptitle(f"t-SNE per species — Highlighting {hosp_focus}", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save and path:
                hosp_path = path.replace(".png", f"_{hosp_focus}.png")
                plt.savefig(hosp_path)
                plt.close()
            else:
                plt.show()


def compute_tsne_per_species(X, labels, metadata, prefix="z"):
    # Build dataframe
    df_all = pd.DataFrame({
        "species": labels,
        "year": metadata["year"].values,
        "hospital": metadata["hospital"].values
    })

    # Add feature columns generically
    for d in range(X.shape[1]):
        df_all[f"{prefix}{d}"] = X[:, d]

    # Compute t-SNE per species
    tsne_results = {}
    for sp in sorted(np.unique(labels)):
        mask = (df_all["species"] == sp)
        X_sp = X[mask]

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sp)

        tsne_results[sp] = {
            "embedding": X_tsne,
            "hospital": df_all.loc[mask, "hospital"].values
        }

    return df_all, tsne_results


def plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, save=False, path=None):
    """
    Plot t-SNE embeddings computed independently for each species.
    If overlay=True, also creates one plot per hospital highlighting its samples.
    """
    import os
    import matplotlib.pyplot as plt

    species_sorted = sorted(tsne_results.keys())
    n_species = len(species_sorted)
    n_rows = (n_species + 2) // 3 
    hospitals_sorted = sorted(df_all["hospital"].unique())

    # --- Plot per species (as before) ---
    if not overlay_per_hospital:
      fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows),
                              sharex=True, sharey=True)
      axes = axes.flatten()

      for i, sp in enumerate(species_sorted):
          emb = tsne_results[sp]["embedding"]
          hosp = tsne_results[sp]["hospital"]
          for h in hospitals_sorted:
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

      if save and path:
          plt.savefig(path)
          plt.close()
      else:
          plt.show()

    # --- Overlay plots (one per hospital) ---
    else:
        for h_focus in hospitals_sorted:
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows),
                                     sharex=True, sharey=True)
            axes = axes.flatten()
            for i, sp in enumerate(species_sorted):
                emb = tsne_results[sp]["embedding"]
                hosp = tsne_results[sp]["hospital"]
                idx_focus = (hosp == h_focus)
                idx_other = ~idx_focus

                # Background (other hospitals)
                axes[i].scatter(
                    emb[idx_other, 0], emb[idx_other, 1],
                    s=10, alpha=0.1, color="gray"
                )
                # Focus hospital (highlighted)
                axes[i].scatter(
                    emb[idx_focus, 0], emb[idx_focus, 1],
                    s=14, alpha=0.7, label=h_focus, color="red"
                )

                axes[i].set_title(sp.replace("_", " "), fontsize=15)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            axes[0].legend(title="Hospital", loc="upper left",
                           frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
            plt.suptitle(f"{h_focus} overlay across species", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save and path:
                hosp_path = path.replace(".png", f"_{h_focus}.png")
                plt.savefig(hosp_path)
                plt.close()
            else:
                plt.show()
