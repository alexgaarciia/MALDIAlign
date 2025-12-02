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
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils.load_config import load_config
from utils.load_data import load_pkl
from utils.viz import compute_tsne_df, plot_tsne_global



#################################
# Load data
#################################

cfg = load_config()
driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
driams = load_pkl(driams_pkl)
data, label, meta = driams["data"], driams["label"], driams["meta"]
meta = pd.DataFrame.from_records(list(meta))

filtered_data = {}
for hosp in meta["hospital"].unique():
    idx = np.where(meta["hospital"].values == hosp)[0]
    filtered_data[hosp] = {
        "data": data[idx],
        "label": label[idx],
        "meta": meta.iloc[idx]
    }

# Individual hospital datasets
dataA, labelA, metaA = filtered_data["DRIAMS_A"]["data"], filtered_data["DRIAMS_A"]["label"], filtered_data["DRIAMS_A"]["meta"]
dataB, labelB, metaB = filtered_data["DRIAMS_B"]["data"], filtered_data["DRIAMS_B"]["label"], filtered_data["DRIAMS_B"]["meta"]
dataC, labelC, metaC = filtered_data["DRIAMS_C"]["data"], filtered_data["DRIAMS_C"]["label"], filtered_data["DRIAMS_C"]["meta"]
dataD, labelD, metaD = filtered_data["DRIAMS_D"]["data"], filtered_data["DRIAMS_D"]["label"], filtered_data["DRIAMS_D"]["meta"]

# Concatenate data and labels
data_final = np.vstack([dataA, dataD])
label_final = np.concatenate([labelA, labelD])
meta_final  = pd.concat([metaA, metaD], ignore_index=True)



#################################
# PCA
#################################

# Normalize each spectrum
data_norm = (data_final - data_final.min(axis=1, keepdims=True)) / (
    data_final.max(axis=1, keepdims=True) - data_final.min(axis=1, keepdims=True) + 1e-8)

# PCA
print("====== Computing PCA... ======", "\n")
data_norm_pca = PCA(n_components=100).fit_transform(data_norm)



#################################
# t-SNE
#################################

print("====== Computing t-SNE... ======", "\n")
tsne_df = compute_tsne_df(data_norm_pca, label_final, meta_final)

print("====== Saving plots... ======", "\n")
plot_tsne_global(tsne_df, per_species=False, save=True, path='01_exploration/output_plots/driams_reduced_tsne_global.png')
plot_tsne_global(tsne_df, per_species=True, save=True, path='01_exploration/output_plots/driams_reduced_tsne_species.png')
plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path='01_exploration/output_plots/driams_reduced_tsne_species_overlay.png')

print("====== Done! ======")
