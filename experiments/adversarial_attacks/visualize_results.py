# ============================================================
# PATH CONFIGURATION
# ============================================================
from pathlib import Path
import os
PROJECT_NAME = "MALDIAlign"
cwd = Path().resolve()
target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break
if target is not None and target != cwd:
    os.chdir(target)
print("Working directory:", os.getcwd())


# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PLOT
# ============================================================
df_adv = pd.read_csv("/Users/agnavarr/Downloads/MALDIAlign_final/experiments/adversarial_attacks/results/20260706_121949/adversarial_robustness.csv")
df_adv = df_adv[df_adv["epsilon"] <= 0.05]

attacks_to_plot = ["Clean", "FGSM", "PGD-10", "PGD-40"]
colors    = {"MLP raw": "#2c7bb6", "VAE+Probe": "#d7191c"}
linestyle = {"Clean": "-", "FGSM": "--", "PGD-10": "-.", "PGD-40": ":"}
markers   = {"Clean": "o", "FGSM": "s", "PGD-10": "^", "PGD-40": "D"}

model_names = df_adv["model"].unique()
fig, axes = plt.subplots(1, len(model_names), figsize=(12 * len(model_names), 5), squeeze=False)
axes = axes[0]

fig, ax = plt.subplots(figsize=(10, 5))

for model_name in model_names:
    sub = df_adv[df_adv["model"] == model_name]
    for attack in attacks_to_plot:
        sub2 = sub[sub["attack"] == attack].sort_values("epsilon")
        if sub2.empty:
            continue
        ax.plot(
            sub2["epsilon"], sub2["balanced_accuracy"],
            color=colors.get(model_name, "black"),
            linestyle=linestyle.get(attack, "-"),
            marker=markers.get(attack, "o"),
            markersize=5,
            label=f"{model_name} — {attack}",
        )

ax.set_xlabel("ε", fontsize=11)
ax.set_ylabel("Balanced Accuracy", fontsize=11)
ax.set_ylim(-0.1, 1.05)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3)
ax.set_xticks(df_adv["epsilon"].unique())
ax.set_xticklabels([str(e) for e in sorted(df_adv["epsilon"].unique())], rotation=45, ha="right", fontsize=8)

plt.suptitle(
    "Adversarial Robustness — MS-UMG OOD",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("/Users/agnavarr/Downloads/MALDIAlign_final/experiments/adversarial_attacks/results/20260706_121949/adversarial_robustness_curve.png",
            dpi=150, bbox_inches="tight")
plt.close()
