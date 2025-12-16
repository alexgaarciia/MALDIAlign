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


def plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, overlay_per_year=False, idx=None, filters=None, save=False, path=None):
    """
    Plot t-SNE embeddings globally (colored by species)
    or per species (colored by hospital).
    """

    df = tsne_df.copy()

    # Apply filters if provided
    if filters:
        for key, vals in filters.items():
            df = df[df[key].isin(vals)]

    # ----------- 1) GLOBAL VIEW (colored by species) -----------
    if not per_species and not overlay_per_hospital and not overlay_per_year:
        plt.figure(figsize=(12, 10))
        for sp in sorted(df["species"].unique()):
            subset = df[df["species"] == sp]
            plt.scatter(subset["x"], subset["y"], s=10, alpha=0.25, label=sp)

        # Overlay misclassified points
        if idx is not None and len(idx) > 0:
            valid_idx = idx[idx < len(df)]  
            plt.scatter(df.iloc[valid_idx]["x"], df.iloc[valid_idx]["y"], marker="x", s=45, color="purple", alpha=0.8, label="Misclassified")
        
        plt.title("t-SNE (colored by species)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(title="Species", markerscale=2, loc="upper left", frameon=True, fontsize=8, title_fontsize=9)
        plt.tight_layout()

        if save and path:
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

    # ----------- 2) PER SPECIES (colored by hospital) -----------
    elif per_species and not overlay_per_hospital and not overlay_per_year:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()


        for i, sp in enumerate(sorted(df["species"].unique())):
            subset = df[df["species"] == sp]
            for hosp in sorted(df["hospital"].unique()):
                sub_h = subset[subset["hospital"] == hosp]
                axes[i].scatter(
                    sub_h["x"], sub_h["y"],
                    s=10, alpha=0.25,
                    label=hosp if i == 0 else None
                )

            # --- Overlay misclassified points for this species ---
            if idx is not None and len(idx) > 0:
                valid_idx = idx[idx < len(df)]
                mis_points = df.iloc[valid_idx]
                mis_sp = mis_points[mis_points["species"] == sp]

                axes[i].scatter(
                    mis_sp["x"], mis_sp["y"],
                    marker="x", s=45, color="purple", alpha=0.8,
                    label="Misclassified" if i == 0 else None
                )

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
    elif per_species and overlay_per_hospital and not overlay_per_year:
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

                if idx is not None and len(idx) > 0:
                    valid_idx = idx[idx < len(df)] 
                    mis_points = df.iloc[valid_idx]
                    mis_sp = mis_points[mis_points["species"] == sp]
                    mis_sp_focus = mis_sp[mis_sp["hospital"] == hosp_focus]

                    axes[i].scatter(
                        mis_sp_focus["x"], mis_sp_focus["y"],
                        marker="x", s=45, color="purple", alpha=0.8,
                        label="Misclassified" if i == 0 else None
                    )

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

    # ----------- 4) OVERLAY PER YEAR (colored by year, one plot per hospital with all species) -----------
    elif per_species and overlay_per_year:
        hospitals = sorted(df["hospital"].unique())
        years = sorted(df["year"].unique())

        for hosp_focus in hospitals:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(sorted(df["species"].unique())):
                # subset of this species for current hospital and others
                subset_all = df[df["species"] == sp]
                subset_focus = subset_all[subset_all["hospital"] == hosp_focus]
                subset_other = subset_all[subset_all["hospital"] != hosp_focus]

                # --- Background: all other hospitals (gray, faint)
                axes[i].scatter(
                    subset_other["x"], subset_other["y"],
                    s=8, alpha=0.1, color="gray"
                )

                # --- Plot each year (colored)
                for year_focus in years:
                    sub_y = subset_focus[subset_focus["year"] == year_focus]
                    axes[i].scatter(
                        sub_y["x"], sub_y["y"],
                        s=12, alpha=0.5,
                        label=str(year_focus) if i == 0 else None
                    )

                # --- Overlay misclassified points (optional)
                if idx is not None and len(idx) > 0:
                    valid_idx = idx[idx < len(df)]
                    mis_points = df.iloc[valid_idx]
                    mis_sp = mis_points[
                        (mis_points["species"] == sp)
                        & (mis_points["hospital"] == hosp_focus)
                    ]
                    if len(mis_sp) > 0:
                        axes[i].scatter(
                            mis_sp["x"],
                            mis_sp["y"],
                            marker="x",
                            s=45,
                            color="purple",
                            alpha=0.8,
                            label="Misclassified" if i == 0 else None
                        )

                axes[i].set_title(sp.replace("_", " "), fontsize=14)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            axes[0].legend(
                title="Year",
                loc="upper left",
                frameon=True,
                fontsize=10,
                title_fontsize=11
            )
            plt.suptitle(f"t-SNE per species — {hosp_focus} (colored by year)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save and path:
                out_path = path.replace(".png", f"_{hosp_focus}_by_year.png")
                plt.savefig(out_path)
                plt.close()
            else:
                plt.show()



def compute_tsne_per_species(X, labels, metadata, prefix="z"):
    """
    Compute independent t-SNE embeddings for each species.
    Keeps track of global indices (mask) to maintain alignment with df_all.
    """

    df_all = pd.DataFrame({
        "species": labels,
        "year": metadata["year"].values,
        "hospital": metadata["hospital"].values
    })

    tsne_results = {}

    # Inicializamos columnas vacías
    df_all["x"] = np.nan
    df_all["y"] = np.nan

    for sp in sorted(np.unique(labels)):
        mask = np.where(labels == sp)[0]        
        X_sp = X[mask]
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sp)

        # Guardar resultados
        tsne_results[sp] = {
            "embedding": X_tsne,
            "hospital": metadata["hospital"].values[mask],
            "mask": mask,                      
        }

        # Rellenar df_all con coordenadas
        df_all.loc[mask, "x"] = X_tsne[:, 0]
        df_all.loc[mask, "y"] = X_tsne[:, 1]

    return df_all, tsne_results


def plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, overlay_per_year_per_species=False, idx=None, save=False, path=None):
    """
    Plot t-SNE embeddings computed independently for each species.
    If overlay=True, also creates one plot per hospital highlighting its samples.
    Optionally overlays misclassified points (idx: global indices of misclassified samples).
    """

    if idx is None:
        idx = np.array([], dtype=int)
        
    species_sorted = sorted(tsne_results.keys())
    n_species = len(species_sorted)
    n_rows = (n_species + 2) // 3 
    hospitals_sorted = sorted(df_all["hospital"].unique())

    # --- Plot per species (as before) ---
    if not overlay_per_hospital and not overlay_per_year_per_species:
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows),
                                 sharex=True, sharey=True)
        axes = axes.flatten()

        for i, sp in enumerate(species_sorted):
            emb = tsne_results[sp]["embedding"]
            hosp = tsne_results[sp]["hospital"]
            mask_global = tsne_results[sp]["mask"] 

            # --- Plot by hospital ---
            for h in hospitals_sorted:
                mask_hosp = (hosp == h)
                axes[i].scatter(emb[mask_hosp, 0], emb[mask_hosp, 1],
                                s=12, alpha=0.45, label=h if i == 0 else None)
              
            # --- Overlay misclassified points (if provided) ---
            if isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
                mis_mask = np.isin(mask_global, idx) 
                if np.any(mis_mask):
                    axes[i].scatter(
                        emb[mis_mask, 0], emb[mis_mask, 1],
                        marker="x", s=45, color="purple", alpha=0.8,
                        label="Misclassified" if i == 0 else None
                    )

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
    elif overlay_per_hospital:
        for h_focus in hospitals_sorted:
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows),
                                     sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(species_sorted):
                emb = tsne_results[sp]["embedding"]
                hosp = tsne_results[sp]["hospital"]
                mask_global = tsne_results[sp]["mask"]

                mask_focus = (hosp == h_focus)
                mask_other = ~mask_focus

                # Background (other hospitals)
                axes[i].scatter(
                    emb[mask_other, 0], emb[mask_other, 1],
                    s=10, alpha=0.1, color="gray"
                )

                # Focus hospital (highlighted)
                axes[i].scatter(
                    emb[mask_focus, 0], emb[mask_focus, 1],
                    s=14, alpha=0.7, label=h_focus, color="red"
                )

                # Overlay misclassified points for that hospital (using mask)
                if isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
                    mis_mask = np.isin(mask_global, idx)
                    if np.any(mis_mask):
                        mis_mask_focus = np.logical_and(mis_mask, mask_focus)
                        if np.any(mis_mask_focus):
                            axes[i].scatter(
                                emb[mis_mask_focus, 0], emb[mis_mask_focus, 1],
                                marker="x", s=45, color="purple", alpha=0.8,
                                label="Misclassified" if i == 0 else None
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

    # ----------- 5) OVERLAY PER YEAR (per species, with gray background for other hospitals) -----------
    elif overlay_per_year_per_species:
        years = sorted(df_all["year"].unique())
        hospitals_sorted = sorted(df_all["hospital"].unique())
        species_sorted = sorted(df_all["species"].unique())

        for h_focus in hospitals_sorted:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(species_sorted):
                subset_all = df_all[df_all["species"] == sp]
                subset_focus = subset_all[subset_all["hospital"] == h_focus]
                subset_other = subset_all[subset_all["hospital"] != h_focus]

                # --- Background: all other hospitals (gray)
                axes[i].scatter(
                    subset_other["x"], subset_other["y"],
                    s=8, alpha=0.1, color="gray"
                )

                # --- Plot each year for this hospital
                for year_focus in years:
                    sub_y = subset_focus[subset_focus["year"] == year_focus]
                    axes[i].scatter(
                        sub_y["x"], sub_y["y"],
                        s=12, alpha=0.5,
                        label=str(year_focus) if i == 0 else None
                    )

                # --- Overlay misclassified points (optional)
                if idx is not None and len(idx) > 0:
                    valid_idx = idx[idx < len(df_all)]
                    mis_points = df_all.iloc[valid_idx]
                    mis_sp = mis_points[
                        (mis_points["species"] == sp)
                        & (mis_points["hospital"] == h_focus)
                    ]
                    if len(mis_sp) > 0:
                        axes[i].scatter(
                            mis_sp["x"], mis_sp["y"],
                            marker="x", s=45, color="purple", alpha=0.8,
                            label="Misclassified" if i == 0 else None
                        )

                axes[i].set_title(sp.replace("_", " "), fontsize=14)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            axes[0].legend(
                title="Year",
                loc="upper left",
                frameon=True,
                fontsize=10,
                title_fontsize=11
            )
            plt.suptitle(f"t-SNE per species — {h_focus} (colored by year)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            if save and path:
                out_path = path.replace(".png", f"_{h_focus}_by_year.png")
                plt.savefig(out_path)
                plt.close()
            else:
                plt.show()
