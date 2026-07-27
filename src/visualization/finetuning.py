import numpy as np
import matplotlib.pyplot as plt

def plot_grid_species(df, title, ylim):
    models = [
        "MLP_original",
        "MLP_latent_zero",
        "MLP_few",
        "FT_full_MLP",
    ]
    
    label_map = {
        "MLP_original":    "Baseline: Trained only on other hospitals",
        "MLP_latent_zero": "Baseline: Using learned representations",
        "MLP_few":         "Trained from scratch on MS-UMG samples",
        "FT_full_MLP":     "Transfer learning: Adapted to MS-UMG",
    }
    
    style_dict = {
        "MLP_original":    {"color": "#1f77b4", "ls": "-", "lw": 1.5},   
        "MLP_latent_zero": {"color": "#ff7f0e", "ls": "-", "lw": 1.5},  
        "MLP_few":         {"color": "#2ca02c", "ls": "--", "lw": 2.0}, 
        "FT_full_MLP":     {"color": "#d62728", "ls": "-", "lw": 2.5},  
    }
    
    n_prev_values = sorted(df["n_prev"].unique())
    n_new_values  = sorted(df["n_new"].unique())
    n_plots = len(n_prev_values)
    
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (16, 6)
    elif n_plots <= 3:
        nrows, ncols = 1, 3
        figsize = (18, 6)
    elif n_plots <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    elif n_plots <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 11)
    else:
        nrows = int(np.ceil(n_plots / 3))
        ncols = 3
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, n_prev in enumerate(n_prev_values):
        ax = axes[i]
        df_row = df[df["n_prev"] == n_prev]
        
        for model in models:
            df_model = df_row[df_row["model"] == model]
            if df_model.empty:
                continue
            
            stats = df_model.groupby("n_new")["balanced_accuracy"].agg(["mean", "std"]).reset_index()
            stats["std"] = stats["std"].fillna(0)
            
            s = style_dict.get(model, {"color": "gray", "ls": "-", "lw": 1})
            line, = ax.plot(
                stats["n_new"],
                stats["mean"],
                marker="o" if "-" in s["ls"] else "x",
                linestyle=s["ls"],
                linewidth=s["lw"],
                color=s["color"],
                label=label_map[model],
                markersize=4
            )
            ax.fill_between(
                stats["n_new"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=line.get_color(),
                alpha=0.1 
            )
        
        ax.set_xticks(n_new_values)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    if n_plots < len(axes):
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')
    
    for i in range(n_plots):
        axes[i].set_ylim(*ylim)
    
    fig.supxlabel("Number of labeled samples from target hospital", fontsize=14, y=0.02)
    fig.supylabel("Species identification accuracy", fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.965)
    
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [0, 1, 2, 3] 
    handles_ordered = [handles[i] for i in legend_order]
    labels_ordered = [labels[i] for i in legend_order]
    
    fig.legend(
        handles_ordered,
        labels_ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89), 
        ncol=2,
        fontsize=10,
        frameon=True,
        edgecolor='#cccccc'
    )
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.84])
    plt.show()
