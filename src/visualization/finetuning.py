import matplotlib.pyplot as plt


def plot_grid(df, title, ylim):
    models = [
        "RF_original",
        "RF_latent_zero",
        "RF_few",
        "FT_full_RF",
        "FT_freeze_RF",
    ]

    label_map = {
        "RF_original": "Zero-Shot (Original)",
        "RF_latent_zero": "Zero-Shot (Latent)",
        "RF_few": "Few-Shot (Original)",
        "FT_full_RF": "Full Finetuning",
        "FT_freeze_RF": "Freeze Priors",
    }
    
    n_prev_values = sorted(df["n_prev"].unique())
    n_new_values  = sorted(df["n_new"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
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

            style_dict = {
                "RF_original": {"ls": "--", "lw": 2},
                "RF_latent_zero": {"ls": ":", "lw": 2},
                "RF_few": {"ls": "-.", "lw": 2},
                "FT_full_RF": {"ls": "-", "lw": 2},
                "FT_freeze_RF": {"ls": "-", "lw": 2.5},
            }
            
            s = style_dict.get(model, {"ls": "-", "lw": 2})

            line, = ax.plot(
                stats["n_new"],
                stats["mean"],
                marker="o",
                linestyle=s["ls"],
                linewidth=s["lw"],
                label=label_map[model],
                markersize=4
            )

            ax.fill_between(
                stats["n_new"],
                stats["mean"] - stats["std"],
                stats["mean"] + stats["std"],
                color=line.get_color(),
                alpha=0.15 
            )

        ax.set_title(f"Source samples: {n_prev}", fontsize=12, fontweight="bold")
        ax.set_xticks(n_new_values)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Configuración de ejes
    for ax in axes:
        ax.set_ylim(*ylim)

    fig.supxlabel("Number of Target Samples (Few-Shot)", fontsize=14)
    fig.supylabel("Balanced Accuracy", fontsize=14)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    # Leyenda única
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=5,
        fontsize=11,
        frameon=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()
