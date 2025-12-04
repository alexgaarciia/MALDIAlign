#################################
# Path configuration
#################################
from pathlib import Path
import os
import sys

PROJECT_NAME = "MALDIAlign"

cwd = Path().resolve()

# Walk upwards until we find the project folder
target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break

# If the project folder is found and we are not already there, then change cwd
if target is not None and target != cwd:
    os.chdir(target)

sys.path.append(str(target))
print("Working directory:", os.getcwd())


#################################
# Imports
#################################
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.load_config import load_config
from utils.load_data import load_pkl
from utils.viz import compute_tsne_df, plot_tsne_global, compute_tsne_per_species, plot_tsne_species


#################################
# Configurable options
#################################

# Which hospitals to include (None = all)
HOSPITALS_TO_INCLUDE = None
#HOSPITALS_TO_INCLUDE = ["DRIAMS_A", "DRIAMS_D"]   # or None for all


#################################
# Load data
#################################

cfg = load_config()
driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
driams = load_pkl(driams_pkl)
data, label, meta = driams["data"], driams["label"], driams["meta"]
meta = pd.DataFrame.from_records(list(meta))

# Filter hospitals if requested
if HOSPITALS_TO_INCLUDE is not None:
    selected_idx = meta["hospital"].isin(HOSPITALS_TO_INCLUDE).values
    data = data[selected_idx]
    label = label[selected_idx]
    meta = meta.iloc[selected_idx].reset_index(drop=True)

print(f"Selected hospitals: {HOSPITALS_TO_INCLUDE if HOSPITALS_TO_INCLUDE else 'ALL'}")
print(f"Data shape: {data.shape}")


#################################
# PCA
#################################

print("====== Computing PCA... ======\n")

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

pca = PCA(n_components=100)
data_pca = pca.fit_transform(data_scaled)


#################################
# t-SNE
#################################

print("====== Computing t-SNE... ======", "\n")
tsne_df = compute_tsne_df(data_pca, label, meta)
df_all, tsne_results = compute_tsne_per_species(data_pca, label, meta, prefix="f")

print("====== Saving plots... ======", "\n")
plot_tsne_global(tsne_df, per_species=False, save=True, path='exploration/output_plots/DRIAMS_FULL/driams_reduced_tsne_global.png')
plot_tsne_global(tsne_df, per_species=True, save=True, path='exploration/output_plots/DRIAMS_FULL/driams_reduced_tsne_global_species.png')
plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path='exploration/output_plots/DRIAMS_FULL/driams_reduced_tsne_global_species_overlay.png')
plot_tsne_species(df_all, tsne_results, save=True, path='exploration/output_plots/DRIAMS_FULL/driams_reduced_tsne_species.png')
plot_tsne_species(df_all, tsne_results, overlay_per_hospital=True, save=True, path='exploration/output_plots/DRIAMS_FULL/driams_reduced_tsne_species_overlay.png')

print("====== Done! ======")
