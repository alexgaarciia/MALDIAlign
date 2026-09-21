# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ============================================================
# CONFIG
# ============================================================
PLOT_FGSM = True

MODEL_STYLES = {
    "MLP raw":                   {"color": "#2563EB", "ls": "-",  "marker": "o", "lw": 2.0, "label": "MLP raw (standard)"},
    "MLP raw (adversarial)":     {"color": "#2563EB", "ls": "--", "marker": "s", "lw": 2.0, "label": "MLP raw (adversarial)"},
    "VAE+Probe":                 {"color": "#DC2626", "ls": "-",  "marker": "o", "lw": 2.5, "label": "DALMA + Probe (standard)"},
    "VAE+Probe adv / probe adv": {"color": "#DC2626", "ls": "--", "marker": "s", "lw": 2.5, "label": "DALMA + Probe (adversarial)"},
}

ATTACKS = ["FGSM", "PGD-10", "PGD-40"]
ATTACK_TITLES = {
    "FGSM":   "FGSM (1 step)",
    "PGD-10": "PGD (10 steps)",
    "PGD-40": "PGD (40 steps)",
}

MAX_EPS = 0.05


# ============================================================
# DATA LOADING
# ============================================================
if not PLOT_FGSM:
    df_adv = pd.read_csv("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/20260921_063738/adversarial_robustness_adv.csv")
else:
    df_adv = pd.read_csv("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/20260718_103659/adversarial_robustness_adv.csv")

df_std = pd.read_csv("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/20260706_121949/adversarial_robustness.csv")
df_all = pd.concat([df_std, df_adv], ignore_index=True)



# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
if PLOT_FGSM:
    fig.suptitle("Adversarial Robustness on MS-UMG (OOD)\nAdv. training: ε=0.02, λ=0.5, FGSM", fontsize=11, fontweight="bold", y=1.04)
else:
    fig.suptitle("Adversarial Robustness on MS-UMG (OOD)\nAdv. training: ε=0.05, λ=0.5, PGD-10", fontsize=11, fontweight="bold", y=1.04)
    
for ax, attack in zip(axes, ATTACKS):
    sub = df_all[(df_all["attack"] == attack) & (df_all["epsilon"] <= MAX_EPS)]

    for model_name, style in MODEL_STYLES.items():
        sub_m = sub[sub["model"] == model_name].sort_values("epsilon")
        if len(sub_m) == 0:
            continue

        if attack == "FGSM":
            clean = df_all[(df_all["model"] == model_name) & (df_all["attack"] == "Clean")]
            if len(clean) > 0:
                ax.plot(0, clean["balanced_accuracy"].values[0],
                        marker="o", color=style["color"],
                        markersize=10, zorder=6, linestyle="None",
                        markeredgecolor="white", markeredgewidth=1.5)

        ax.plot(sub_m["epsilon"], sub_m["balanced_accuracy"],
                color=style["color"], ls=style["ls"],
                marker=style["marker"], markersize=5,
                lw=style["lw"], label=style["label"])

    ax.set_title(ATTACK_TITLES[attack], fontsize=11, fontweight="bold")
    ax.set_xlabel("ε (L∞ perturbation)", fontsize=9)
    ax.set_xlim(-0.002, MAX_EPS + 0.002)
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.tick_params(labelsize=8)
    if attack == "FGSM":
        ax.set_ylabel("Balanced Accuracy", fontsize=9)

handles = []
for model_name, style in MODEL_STYLES.items():
    handles.append(mlines.Line2D(
        [], [], color=style["color"], ls=style["ls"],
        marker=style["marker"], markersize=6, lw=style["lw"],
        label=style["label"],
    ))
handles.append(mlines.Line2D(
    [], [], color="gray", lw=0, marker="o", markersize=8,
    markeredgecolor="white", markeredgewidth=1.5,
    label="Clean accuracy (●)",
))

fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.95), fontsize=8, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.88])

if PLOT_FGSM:
    plt.savefig("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/adversarial_robustness_comparison_fgsm.png", dpi=150, bbox_inches="tight")
else:
    plt.savefig("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/adversarial_robustness_comparison_pgd.png", dpi=150, bbox_inches="tight")

plt.show()
