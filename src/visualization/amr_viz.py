import matplotlib.pyplot as plt

from src.visualization.utils import handle_save_show


def plot_tsne_amr(df_all, tsne_results, antibiotic_col="amr", save=False, path=None):
    """
    Plot t-SNE embeddings per species, colored by AMR status.
    """
    if antibiotic_col not in df_all.columns:
        print(f"Warning: '{antibiotic_col}' column not found in metadata. Skipping AMR plot.")
        return

    species_sorted = sorted(tsne_results.keys())
    n_species = len(species_sorted)
    n_rows = (n_species + 2) // 3 
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    amr_colors = {1.0: "red", 0.0: "blue", "Unknown": "gray"}
    amr_labels = {1.0: "Resistant", 0.0: "Susceptible", "Unknown": "Unknown"}

    for i, sp in enumerate(species_sorted):
        subset = df_all[df_all["species"] == sp].copy()
        subset[antibiotic_col] = subset[antibiotic_col].fillna("Unknown")
        sub_unk = subset[subset[antibiotic_col] == "Unknown"]

        axes[i].scatter(sub_unk["x"], sub_unk["y"], s=8, alpha=0.1, color=amr_colors["Unknown"], label="Unknown")

        for status in [0.0, 1.0]: 
            sub_status = subset[subset[antibiotic_col] == status]
            if len(sub_status) > 0:
                axes[i].scatter(
                    sub_status["x"], sub_status["y"], 
                    s=14, alpha=0.7, 
                    color=amr_colors[status], 
                    label=amr_labels[status]
                )

        axes[i].set_title(sp.replace("_", " "), fontsize=14)
        axes[i].set_xticks([]); axes[i].set_yticks([])

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))

    order = ["Resistant", "Susceptible", "Unknown"]
    ordered_handles = [unique[label] for label in order if label in unique]
    ordered_labels = [label for label in order if label in unique]

    axes[0].legend(ordered_handles, ordered_labels, title=f"AMR: {antibiotic_col}", 
                   loc="upper left", frameon=True, fontsize=11, title_fontsize=12)
    
    plt.suptitle(f"t-SNE per species — AMR ({antibiotic_col})", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    handle_save_show(save, path)
    