import numpy as np
import matplotlib.pyplot as plt


def plot_grid(df, title, ylim):
    # Lista ampliada de modelos
    models = [
        "RF_original", "MLP_original",
        "RF_latent_zero", "MLP_latent_zero",
        "RF_few", "MLP_few",
        "FT_full_RF", "FT_full_MLP",
        "FT_freeze_RF", "FT_freeze_MLP",
    ]
    
    # Mapeo de nombres para la leyenda
    label_map = {
        "RF_original": "Zero-Shot (Orig) - RF",
        "MLP_original": "Zero-Shot (Orig) - MLP",
        "RF_latent_zero": "Zero-Shot (Lat) - RF",
        "MLP_latent_zero": "Zero-Shot (Lat) - MLP",
        "RF_few": "Few-Shot (Orig) - RF",
        "MLP_few": "Few-Shot (Orig) - MLP",
        "FT_full_RF": "Full FT - RF",
        "FT_full_MLP": "Full FT - MLP",
        "FT_freeze_RF": "Freeze Priors - RF",
        "FT_freeze_MLP": "Freeze Priors - MLP",
    }
    
    # Definición de estilos por familia de experimento
    style_dict = {
        "RF_original":     {"color": "#1f77b4", "ls": "-",  "lw": 1.5},
        "MLP_original":    {"color": "#1f77b4", "ls": "--", "lw": 1.5},
        "RF_latent_zero":  {"color": "#ff7f0e", "ls": "-",  "lw": 1.5},
        "MLP_latent_zero": {"color": "#ff7f0e", "ls": "--", "lw": 1.5},
        "RF_few":          {"color": "#2ca02c", "ls": "-",  "lw": 1.5},
        "MLP_few":         {"color": "#2ca02c", "ls": "--", "lw": 1.5},
        "FT_full_RF":      {"color": "#d62728", "ls": "-",  "lw": 2.0},
        "FT_full_MLP":     {"color": "#d62728", "ls": "--", "lw": 2.0},
        "FT_freeze_RF":    {"color": "#9467bd", "ls": "-",  "lw": 2.0},
        "FT_freeze_MLP":   {"color": "#9467bd", "ls": "--", "lw": 2.0},
    }
    
    n_prev_values = sorted(df["n_prev"].unique())
    n_new_values  = sorted(df["n_new"].unique())
    n_plots = len(n_prev_values)
    
    # Calcular layout dinámicamente según número de n_prev
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
        # Para más de 6, usar 3 columnas
        nrows = int(np.ceil(n_plots / 3))
        ncols = 3
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    
    # Manejar caso de un solo subplot
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
        
        ax.set_title(f"Source samples: {n_prev}", fontsize=12, fontweight="bold")
        ax.set_xticks(n_new_values)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Ocultar subplots vacíos si hay más axes que n_plots
    if n_plots < len(axes):
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')
    
    # Establecer límites
    for i in range(n_plots):
        axes[i].set_ylim(*ylim)
    
    # Labels
    fig.supxlabel("Number of Target Samples (Few-Shot)", fontsize=14)
    fig.supylabel("Balanced Accuracy", fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    
    # Leyenda ajustada
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=5,
        fontsize=10,
        frameon=True
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.show()

def plot_grid_v2(df, title, ylim):
    # Lista ampliada de modelos
    models = [
        "MLP_original",
        "MLP_latent_zero",
        "MLP_few",
        "FT_full_MLP",
    ]
    
    # Mapeo de nombres para la leyenda
    label_map = {
        "MLP_original":    "Baseline: Trained only on other hospitals",
        "MLP_latent_zero": "Baseline: Using learned representations",
        "MLP_few":         "Trained from scratch on MS-UMG samples",
        "FT_full_MLP":     "Transfer learning: Adapted to MS-UMG",
    }
    
    # Definición de estilos por familia de experimento
    style_dict = {
        "MLP_original":    {"color": "#1f77b4", "ls": "-", "lw": 1.5},   # Solid para baseline
        "MLP_latent_zero": {"color": "#ff7f0e", "ls": "-", "lw": 1.5},   # Solid para baseline
        "MLP_few":         {"color": "#2ca02c", "ls": "--", "lw": 2.0},  # Dashed para entrenado
        "FT_full_MLP":     {"color": "#d62728", "ls": "-", "lw": 2.5},   # Más grueso para destacar
    }
    
    n_prev_values = sorted(df["n_prev"].unique())
    n_new_values  = sorted(df["n_new"].unique())
    n_plots = len(n_prev_values)
    
    # Calcular layout dinámicamente según número de n_prev
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
        # Para más de 6, usar 3 columnas
        nrows = int(np.ceil(n_plots / 3))
        ncols = 3
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    
    # Manejar caso de un solo subplot
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
    
    # Ocultar subplots vacíos si hay más axes que n_plots
    if n_plots < len(axes):
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')
    
    # Establecer límites
    for i in range(n_plots):
        axes[i].set_ylim(*ylim)
    
    # Labels
    fig.supxlabel("Number of labeled samples from target hospital", fontsize=14, y=0.02)
    fig.supylabel("Species identification accuracy", fontsize=14)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.965)
    
    # Leyenda ajustada
    handles, labels = axes[0].get_legend_handles_labels()
    legend_order = [0, 1, 2, 3]  # Baselines primero, luego entrenados
    handles_ordered = [handles[i] for i in legend_order]
    labels_ordered = [labels[i] for i in legend_order]
    
    fig.legend(
        handles_ordered,
        labels_ordered,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),  # Más abajo
        ncol=2,
        fontsize=10,
        frameon=True,
        edgecolor='#cccccc'
    )
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.84])
    plt.show()

def plot_grid_amr(df, title, metric="auc", ylim=(0.5, 1.0)):
    models = [
        "LGBM_original_zero", "LGBM_latent_zero", "LGBM_few_shot_orig",
        "FT_decoder_only_Align", "FT_enc_dec_Align", "FT_enc_dec_amr_Align", 
        "FT_freeze_priors_Align", "FT_full_Align"
    ]

    label_map = {
        "LGBM_original_zero": "Zero-Shot (Orig)",
        "LGBM_latent_zero": "Zero-Shot (Lat)",
        "LGBM_few_shot_orig": "Few-Shot (Orig)",
        "FT_decoder_only_Align": "FT: Decoder Only",
        "FT_enc_dec_Align": "FT: Enc+Dec",
        "FT_enc_dec_amr_Align": "FT: Enc+Dec+AMR",
        "FT_freeze_priors_Align": "FT: Freeze Priors",
        "FT_full_Align": "FT: Full"
    }
    
    style_dict = {
        "LGBM_original_zero":    {"color": "#1f77b4", "ls": "--", "lw": 1.5},
        "LGBM_latent_zero":      {"color": "#ff7f0e", "ls": "--", "lw": 1.5},
        "LGBM_few_shot_orig":    {"color": "#2ca02c", "ls": "-",  "lw": 1.5},
        "FT_decoder_only_Align": {"color": "#d62728", "ls": "-",  "lw": 2.0},
        "FT_enc_dec_Align":      {"color": "#9467bd", "ls": "-",  "lw": 2.0},
        "FT_enc_dec_amr_Align":  {"color": "#8c564b", "ls": "-",  "lw": 2.0},
        "FT_freeze_priors_Align":{"color": "#e377c2", "ls": "-",  "lw": 2.0},
        "FT_full_Align":         {"color": "#7f7f7f", "ls": "-",  "lw": 2.0},
    }

    n_prev_values = sorted(df["n_prev"].unique())
    n_new_values  = sorted(df["n_new"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, n_prev in enumerate(n_prev_values):
        ax = axes[i]
        df_prev = df[df["n_prev"] == n_prev]

        for model in models:
            df_model = df_prev[df_prev["model"] == model]
            if df_model.empty: 
                continue

            stats = df_model.groupby("n_new")[metric].agg(["mean", "std"]).reset_index()
            
            s = style_dict.get(model, {"color": "gray", "ls": "-", "lw": 1})

            line, = ax.plot(
                stats["n_new"],
                stats["mean"],
                marker="o",
                linestyle=s["ls"],
                linewidth=s["lw"],
                color=s["color"],
                label=label_map[model],
                markersize=5
            )

            ax.fill_between(
                stats["n_new"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=line.get_color(),
                alpha=0.1 
            )

        ax.set_title(f"Source samples (Anchoring): {n_prev}", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle='--', alpha=0.4)

    metric_name = "ROC-AUC" if metric == "auc" else "Precision-Recall"
    fig.supxlabel("Number of Target Samples (n_new) - Log Scale", fontsize=15)
    fig.supylabel(f"Macro-Average {metric_name}", fontsize=15)
    fig.suptitle(f"AMR Prediction Performance: {title}", fontsize=18, fontweight="bold", y=0.98)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
        ncol=4, fontsize=11, frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig
