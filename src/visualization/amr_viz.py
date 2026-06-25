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
    

def plot_tsne_amr_split_by_hospital(df_all, antibiotic, target_domain, save=False, path=None):
    """
    Genera un plot comparativo: Columna 1 (Source Domains) vs Columna 2 (Target Domain).
    """
    df_valid = df_all.dropna(subset=[antibiotic]).copy()
    if df_valid.empty:
        return

    # Definimos quién es "Source" y quién es "Target"
    # Asumimos que lo que no es el target_domain actual es 'Source'
    df_valid['group'] = df_valid['hospital'].apply(
        lambda x: 'Target: ' + x if x == target_domain else 'Source Domains'
    )
    
    species_list = sorted(df_valid["species"].unique())
    groups = ['Source Domains', f'Target: {target_domain}']
    
    n_rows = len(species_list)
    n_cols = len(groups)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), 
                             sharex='row', sharey='row', squeeze=False)

    amr_colors = {1.0: "crimson", 0.0: "royalblue", "Unknown": "lightgray"}

    for i, sp in enumerate(species_list):
        df_sp = df_valid[df_valid["species"] == sp]
        for j, gp in enumerate(groups):
            ax = axes[i, j]
            
            # Datos del grupo actual y fondo (el otro grupo)
            df_g = df_sp[df_sp['group'] == gp]
            df_bg = df_sp[df_sp['group'] != gp]

            # Dibujar fondo en gris muy suave
            ax.scatter(df_bg['x'], df_bg['y'], color='lightgray', s=5, alpha=0.1)

            # Dibujar Sensibles y Resistentes
            for val, color in [(0.0, "royalblue"), (1.0, "crimson")]:
                sub = df_g[df_g[antibiotic] == val]
                label = "Resistant" if val == 1.0 else "Susceptible"
                if not sub.empty:
                    ax.scatter(sub['x'], sub['y'], color=color, s=15, alpha=0.7, label=label)

            ax.set_title(f"{sp} | {gp}", fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
            
            if i == 0 and j == n_cols - 1:
                ax.legend(loc='upper right', fontsize=10)

    plt.suptitle(f"Finetuning Analysis: {antibiotic} (Target: {target_domain})", fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save and path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        