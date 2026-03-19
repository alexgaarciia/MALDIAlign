import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from src.visualization.utils import handle_save_show, plot_misclassified, plot_prior_star


def plot_model_metrics(model, model_name, save=False, path=None):
    """
    Plot training and validation losses for a VAE-like model.
    Supports optional AMR classification loss if present.
    """

    if hasattr(model, "reconstruc_during_training") and hasattr(model, "KL_during_training"):
        train_losses = [t[0] for t in model.loss_during_training]
        train_recon = [t[0] for t in model.reconstruc_during_training]
        train_kl = [t[0] for t in model.KL_during_training]

        val_losses = [t[1] for t in model.loss_during_training]
        val_recon = [t[1] for t in model.reconstruc_during_training]
        val_kl = [t[1] for t in model.KL_during_training]

        # Optional AMR loss
        train_amr, val_amr = None, None
        if hasattr(model, "AMR_during_training"):
            train_amr = [t[0] for t in model.AMR_during_training]
            val_amr = [t[1] for t in model.AMR_during_training]

        epochs = range(1, len(train_losses) + 1)

        # ======================
        # Training losses
        # ======================
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label="Total loss", linewidth=2)
        plt.plot(epochs, train_recon, label="Reconstruction", linestyle="--")
        plt.plot(epochs, train_kl, label="KL divergence", linestyle=":")

        if train_amr is not None:
            plt.plot(epochs, train_amr, label="AMR BCE", linestyle="-.")

        plt.title(f"{model_name} Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save and path:
            plt.savefig(path.parent / f"{path.name}_train.png", dpi=300)
            plt.close()
        else:
            plt.show()
 
        # ======================
        # Validation losses
        # ======================
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, val_losses, label="Total loss", linewidth=2)
        plt.plot(epochs, val_recon, label="Reconstruction", linestyle="--")
        plt.plot(epochs, val_kl, label="KL divergence", linestyle=":")

        if val_amr is not None:
            plt.plot(epochs, val_amr, label="AMR BCE", linestyle="-.")

        plt.title(f"{model_name} Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save and path:
            plt.savefig(path.parent / f"{path.name}_val.png", dpi=300)
            plt.close()
        else:
            plt.show()

    elif hasattr(model, "species_loss_during_training") and hasattr(model, "domain_loss_during_training"):

        tr_loss = [t[0] for t in model.loss_during_training]
        tr_sp = [t[0] for t in model.species_loss_during_training]
        tr_dom = [t[0] for t in model.domain_loss_during_training]

        va_loss = [t[1] for t in model.loss_during_training]
        va_sp = [t[1] for t in model.species_loss_during_training]
        va_dom = [t[1] for t in model.domain_loss_during_training]

        epochs = range(1, len(model.loss_during_training) + 1)

        # ======================
        # Training losses
        # ======================
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, tr_loss, label="Total loss", linewidth=2)
        plt.plot(epochs, tr_sp, "--", label="Species loss")
        plt.plot(epochs, tr_dom, ":", label="Domain loss")

        plt.title(f"{model_name} Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        if save and path:
            plt.savefig(path.parent / f"{path.name}_train.png", dpi=300)
            plt.close()
        else:
            plt.show()

        # ======================
        # Validation losses
        # ======================
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, va_loss, label="Total loss", linewidth=2)
        plt.plot(epochs, va_sp, "--", label="Species loss")
        plt.plot(epochs, va_dom, ":", label="Domain loss")

        plt.title(f"{model_name} Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        if save and path:
            plt.savefig(path.parent / f"{path.name}_val.png", dpi=300)
            plt.close()
        else:
            plt.show()


def compute_tsne_df(X, labels, metadata, source=None):
    """
    Compute global t-SNE embeddings for given samples.
    """
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(X)

    if source is None:
        source = ["real"] * len(labels)

    # Copiamos todo el metadata para no perder los datos de AMR
    tsne_df = metadata.copy()
    
    tsne_df["x"] = data_tsne[:, 0]
    tsne_df["y"] = data_tsne[:, 1]
    tsne_df["species"] = labels
    tsne_df["source"] = source

    return tsne_df


def plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, overlay_per_year=False, idx=None, filters=None, save=False, path=None):
    """
    Plot t-SNE embeddings globally (colored by species)
    or per species (colored by hospital).
    """

    df = tsne_df.copy()

    # -------------------------
    # Apply filters
    # -------------------------
    if filters:
        for key, vals in filters.items():
            df = df[df[key].isin(vals)]

    species_list = sorted(df["species"].unique())
    hospitals = sorted(df["hospital"].unique()) if "hospital" in df.columns else []

    # -------------------------
    # Misclassified indices
    # -------------------------
    if idx is not None and len(idx) > 0:
        idx = idx[idx < len(df)]
        mis_points = df.iloc[idx]
    else:
        mis_points = None


    # ============================================================
    # 1) GLOBAL VIEW — COLORED BY SPECIES
    # ============================================================
    if not per_species and not overlay_per_hospital and not overlay_per_year:
        plt.figure(figsize=(12, 10))

        for i, sp in enumerate(species_list):
            subset_real = df[(df["species"] == sp) & (df["source"] == "real")]
            subset_prior = df[(df["species"] == sp) & (df["source"] == "prior")]

            # real samples
            plt.scatter(
                subset_real["x"],
                subset_real["y"],
                s=10,
                alpha=0.25,
                label=sp
            )

            # prior samples
            plot_prior_star(plt.gca(), subset_prior, i, label="Prior samples" if i == 0 else None)

        # Overlay misclassified points
        plot_misclassified(plt.gca(), mis_points)

        plt.title("t-SNE (colored by species)")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

        # species legend (colors)
        species_handles = [
            Line2D([0], [0],
                marker='o',
                color='w',
                label=sp,
                markerfacecolor=plt.cm.tab10(i),
                markersize=8)
            for i, sp in enumerate(species_list)
        ]

        # marker type legend 
        has_prior = (df["source"] == "prior").any()

        marker_handles = [
            Line2D([0], [0], marker='o', color='k',
                linestyle='None', markersize=6,
                label="Real samples")
        ]

        if has_prior:
            marker_handles.append(
                Line2D([0], [0], marker='*', color='k',
                    linestyle='None', markersize=12,
                    label="Prior samples")
            )

        if mis_points is not None:
            marker_handles.append(
                Line2D([0], [0], marker='x', color='purple',
                    linestyle='None', markersize=8,
                    label="Misclassified samples")
            )

        legend1 = plt.legend(
            handles=species_handles,
            title="Species",
            loc="upper left",
            frameon=True,
            fontsize=8,
            title_fontsize=9
        )

        plt.gca().add_artist(legend1)
        plt.legend(handles=marker_handles, title="Marker type", loc="upper right", frameon=True, fontsize=8)

        handle_save_show(save, path)


    # ============================================================
    # 2) PER-SPECIES VIEW — COLORED BY HOSPITAL
    # ============================================================
    elif per_species and not overlay_per_hospital and not overlay_per_year:
        n_species = len(species_list)
        n_cols = 3
        n_rows = int(np.ceil(n_species / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(18, 5 * n_rows),
            sharex=True,
            sharey=True
        )

        axes = axes.flatten()

        for i, sp in enumerate(species_list):
            subset_real = df[(df["species"] == sp) & (df["source"] == "real")]
            subset_prior = df[(df["species"] == sp) & (df["source"] == "prior")]

            for hosp in sorted(df["hospital"].unique()):
                if hosp == "Prior":
                    sub_h = subset_prior
                else:
                    sub_h = subset_real[subset_real["hospital"] == hosp]

                axes[i].scatter(
                    sub_h["x"],
                    sub_h["y"],
                    s=10,
                    alpha=0.25,
                    label=hosp
                )

            # Overlay misclassified points for this species
            plot_misclassified(axes[i], mis_points, sp=sp)

            axes[i].set_title(sp.replace("_", " "), fontsize=15)
            axes[i].set_xticks([]); axes[i].set_yticks([])

        # Recopilar etiquetas sin duplicados
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique = dict(zip(labels, handles))

        axes[0].legend(unique.values(), unique.keys(), title="Hospital", loc="upper left",
                    frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
        
        plt.suptitle("t-SNE embeddings per species (colored by hospital)", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        handle_save_show(save, path)

    # ============================================================
    # 3) PER-SPECIES — HIGHLIGHT ONE HOSPITAL
    # ============================================================
    elif per_species and overlay_per_hospital and not overlay_per_year:
        for hosp_focus in hospitals:
            n_species = len(species_list)
            n_cols = 3
            n_rows = int(np.ceil(n_species / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(18, 5 * n_rows),
                sharex=True,
                sharey=True
            )

            axes = axes.flatten()

            for i, sp in enumerate(species_list):
                subset_real = df[(df["species"]==sp)&(df["source"]=="real")]
                subset_prior = df[(df["species"]==sp)&(df["source"]=="prior")]

                if hosp_focus == "Prior":
                    subset_focus = subset_prior
                    subset_other = subset_real
                else:
                    subset_focus = subset_real[subset_real["hospital"] == hosp_focus]
                    subset_other = subset_real[subset_real["hospital"] != hosp_focus]

                axes[i].scatter(subset_other["x"],subset_other["y"],s=8,alpha=0.1,color="gray")

                if len(subset_focus) > 0:
                    axes[i].scatter(
                        subset_focus["x"],
                        subset_focus["y"],
                        s=12,
                        alpha=0.6,
                        color="red",
                        label=hosp_focus
                    )

                # Misclassified
                plot_misclassified(axes[i], mis_points, sp=sp, hosp=hosp_focus)

                axes[i].set_title(sp.replace("_", " "), fontsize=15)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            # Recopilar etiquetas sin duplicados
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            axes[0].legend(unique.values(), unique.keys(), loc="upper left", fontsize=12, frameon=True)
            plt.suptitle(f"t-SNE per species — Highlighting {hosp_focus}", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = path.parent / f"{path.stem}_{hosp_focus}.png" if (save and path) else None
            handle_save_show(save, out_path)    


    # ============================================================
    # 4) OVERLAY PER YEAR (colored by year, one plot per hospital with all species)
    # ============================================================
    elif per_species and overlay_per_year:
        for hosp_focus in hospitals:
            if hosp_focus == "Prior":
                continue

            focus_years_raw = df[df["hospital"] == hosp_focus]["year"].fillna("Unknown").unique()
            years = sorted([y for y in focus_years_raw if str(y) != "Unknown"])
            if "Unknown" in focus_years_raw:
                years.append("Unknown")

            n_species = len(species_list)
            n_cols = 3
            n_rows = int(np.ceil(n_species / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(18, 5 * n_rows),
                sharex=True,
                sharey=True
            )

            axes = axes.flatten()

            for i, sp in enumerate(species_list):
                subset_real = df[(df["species"] == sp) & (df["source"] == "real")]
                
                subset_focus = subset_real[subset_real["hospital"] == hosp_focus].copy()
                subset_focus["year"] = subset_focus["year"].fillna("Unknown")                
                
                subset_other = subset_real[subset_real["hospital"] != hosp_focus]

                axes[i].scatter(
                    subset_other["x"], subset_other["y"],
                    s=8, alpha=0.1, color="gray"
                )

                for year_focus in years:
                    sub_y = subset_focus[subset_focus["year"] == year_focus]
                    axes[i].scatter(
                        sub_y["x"],
                        sub_y["y"],
                        s=12,
                        alpha=0.5,
                        label=str(year_focus)
                    )

                # Misclassified
                plot_misclassified(axes[i], mis_points, sp=sp, hosp=hosp_focus)

                axes[i].set_title(sp.replace("_", " "), fontsize=14)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            # Recopilar etiquetas sin duplicados
            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            axes[0].legend(
                unique.values(), unique.keys(),
                title="Year",
                loc="upper left",
                frameon=True,
                fontsize=10,
                title_fontsize=11
            )
            plt.suptitle(f"t-SNE per species — {hosp_focus} (colored by year)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = path.parent / f"{path.stem}_{hosp_focus}_by_year.png" if (save and path) else None
            handle_save_show(save, out_path)


def compute_tsne_per_species(X, labels, metadata, prefix="z", source=None):
    """
    Compute independent t-SNE embeddings for each species.
    Keeps track of global indices (mask) to maintain alignment with df_all.
    """
    if source is None:
        source = ["real"] * len(labels)

    # Copiamos todo el metadata para arrastrar las columnas de antibióticos
    df_all = metadata.copy()
    
    df_all["species"] = labels
    df_all["source"] = source
    df_all["x"] = np.nan
    df_all["y"] = np.nan

    tsne_results = {}

    for sp in sorted(np.unique(labels)):
        mask = np.where(labels == sp)[0]        
        X_sp = X[mask]
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_sp)

        tsne_results[sp] = {
            "embedding": X_tsne,
            "mask": mask,
        }

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
        
    if isinstance(idx, (list, np.ndarray)) and len(idx) > 0:
        valid_idx = idx[idx < len(df_all)]
        mis_points = df_all.iloc[valid_idx]
    else:
        mis_points = None

    species_sorted = sorted(tsne_results.keys())
    n_species = len(species_sorted)
    n_rows = (n_species + 2) // 3 
    hospitals_sorted = sorted(df_all["hospital"].unique())


    # ============================================================
    # 1) PER-SPECIES — COLORED BY HOSPITAL
    # ============================================================
    if not overlay_per_hospital and not overlay_per_year_per_species:
        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, sp in enumerate(species_sorted):
            subset = df_all[df_all["species"] == sp]
            subset_real = subset[subset["source"] == "real"]
            subset_prior = subset[subset["source"] == "prior"]

            for h in hospitals_sorted:
                if h == "Prior":
                    sub_h = subset_prior
                else:
                    sub_h = subset_real[subset_real["hospital"] == h]
                            
                axes[i].scatter(
                    sub_h["x"],
                    sub_h["y"],
                    s=12,
                    alpha=0.45,
                    label=h)

            plot_misclassified(axes[i], mis_points, sp)

            axes[i].set_title(sp.replace("_", " "), fontsize=15)
            axes[i].set_xticks([]); axes[i].set_yticks([])

        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique = dict(zip(labels, handles))

        # CORREGIDO: De vuelta a axes[0]
        axes[0].legend(unique.values(), unique.keys(),
                title="Hospital",
                loc="upper left",
                frameon=True,
                fontsize=12,
                title_fontsize=13)

        plt.suptitle("Independent t-SNE embeddings per species (colored by hospital)",
                     fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        handle_save_show(save, path)

    # ============================================================
    # 2) PER-SPECIES — HIGHLIGHT ONE HOSPITAL
    # ============================================================
    elif overlay_per_hospital:
        for h_focus in hospitals_sorted:
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(species_sorted):
                subset = df_all[df_all["species"] == sp]
                subset_real = subset[subset["source"] == "real"]
                subset_prior = subset[subset["source"] == "prior"]

                if h_focus == "Prior":
                    subset_focus = subset_prior
                    subset_other = subset_real
                else:
                    subset_focus = subset_real[subset_real["hospital"] == h_focus]
                    subset_other = subset_real[subset_real["hospital"] != h_focus]

                axes[i].scatter(
                    subset_other["x"],
                    subset_other["y"],
                    s=10,
                    alpha=0.1,
                    color="gray"
                )

                if len(subset_focus) > 0:
                    axes[i].scatter(
                        subset_focus["x"],
                        subset_focus["y"],
                        s=14,
                        alpha=0.7,
                        color="red",
                        label=h_focus
                    )

                plot_misclassified(axes[i], mis_points, sp=sp, hosp=h_focus)

                axes[i].set_title(sp.replace("_", " "), fontsize=15)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            axes[0].legend(unique.values(), unique.keys(), title="Hospital", loc="upper left",
                           frameon=True, fontsize=12, title_fontsize=13, markerscale=2.0)
            plt.suptitle(f"{h_focus} overlay across species", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])


            hosp_path = path.parent / f"{path.stem}_{h_focus}.png" if (save and path) else None
            handle_save_show(save, hosp_path)


    # ============================================================
    # 3) PER-SPECIES — OVERLAY PER YEAR (FOCUS HOSPITAL)
    # ============================================================
    elif overlay_per_year_per_species:
        for h_focus in hospitals_sorted:
            if h_focus == "Prior":
                continue

            # Obtenemos los años, rellenando los vacíos (NaN) con "Unknown"
            focus_years_raw = df_all[df_all["hospital"] == h_focus]["year"].fillna("Unknown").unique()
                        
            # Ordenamos los años numéricos, pero dejamos "Unknown" para el final
            years = sorted([y for y in focus_years_raw if str(y) != "Unknown"])
            if "Unknown" in focus_years_raw:
                years.append("Unknown")

            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True, sharey=True)
            axes = axes.flatten()

            for i, sp in enumerate(species_sorted):
                subset = df_all[df_all["species"] == sp]

                subset_real = subset[subset["source"] == "real"]
                subset_prior = subset[subset["source"] == "prior"]

                subset_focus = subset_real[subset_real["hospital"] == h_focus].copy()
                subset_focus["year"] = subset_focus["year"].fillna("Unknown")
                subset_other = subset_real[subset_real["hospital"] != h_focus]

                axes[i].scatter(
                    subset_other["x"],
                    subset_other["y"],
                    s=8,
                    alpha=0.1,
                    color="gray"
                )

                for year_focus in years:
                    sub_y = subset_focus[subset_focus["year"] == year_focus]
                    axes[i].scatter(
                        sub_y["x"],
                        sub_y["y"],
                        s=12,
                        alpha=0.5,
                        label=str(year_focus)
                    )

                plot_misclassified(axes[i], mis_points, sp=sp, hosp=h_focus)

                axes[i].set_title(sp.replace("_", " "), fontsize=14)
                axes[i].set_xticks([]); axes[i].set_yticks([])

            handles, labels = [], []
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)
            unique = dict(zip(labels, handles))

            axes[0].legend(unique.values(), unique.keys(),
                    title="Year",
                    loc="upper left",
                    frameon=True,
                    fontsize=10,
                    title_fontsize=11)
            
            plt.suptitle(f"t-SNE per species — {h_focus} (colored by year)", fontsize=18)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = path.parent / f"{path.stem}_{h_focus}_by_year.png" if (save and path) else None
            handle_save_show(save, out_path)

