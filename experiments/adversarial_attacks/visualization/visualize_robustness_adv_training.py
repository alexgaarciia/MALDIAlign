# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# DATA LOADING
# ============================================================
df_adv = pd.read_csv("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/20260921_063738/adversarial_robustness_adv.csv")
df_adv = df_adv[df_adv["epsilon"] <= 0.05]


# ============================================================
# PLOT
# ============================================================
attacks_to_plot = ["Clean", "FGSM", "PGD-10", "PGD-40"]
colors = {"MLP raw (adversarial)": "#2c7bb6", "VAE+Probe adv / probe adv": "#d7191c"}
linestyle = {"Clean": "-", "FGSM": "--", "PGD-10": "-.", "PGD-40": ":"}
markers = {"Clean": "o", "FGSM": "s", "PGD-10": "^", "PGD-40": "D"}

fig, ax = plt.subplots(figsize=(11, 6))

for model_name in df_adv["model"].unique():
    sub = df_adv[df_adv["model"] == model_name]
    for attack in attacks_to_plot:
        sub2 = sub[sub["attack"] == attack].sort_values("epsilon")
        if sub2.empty:
            continue
        ax.plot(sub2["epsilon"], sub2["balanced_accuracy"], color=colors.get(model_name, "black"), linestyle=linestyle.get(attack, "-"), marker=markers.get(attack, "o"), markersize=6, linewidth=2, label=f"{model_name} — {attack}")

ax.set_xlabel("ε (perturbation budget)", fontsize=12)
ax.set_ylabel("Balanced Accuracy", fontsize=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.35)

plt.title("Adversarial Robustness — MS-UMG OOD\n(Models trained with Adversarial Training)", fontsize=13, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("/export/usuarios01/agnavarr/MALDIAlign/experiments/adversarial_attacks/results/20260921_063738/adversarial_robustness_curve.png", dpi=150, bbox_inches="tight")
plt.show()
