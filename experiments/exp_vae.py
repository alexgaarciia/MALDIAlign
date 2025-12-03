# ---------------------------
# Add project root to PYTHONPATH
# ---------------------------
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# ---------------------------
# Imports
# ---------------------------
import pickle
import torch
from models.deep.VAE import VAE_Extended
from utils.load_config import load_config
from utils.load_data import load_driams, map_domains, scale_data, construct_dataloaders
from utils.viz import *
from utils.eval import eval_model

from sklearn.model_selection import train_test_split


# ---------------------------
# Experiment
# ---------------------------
def main():
    # ---------------------------
    # Create results folder
    # ---------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results_vae")
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------
    # Data preparation
    # ---------------------------
    cfg = load_config()
    driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL"]
    driams_filtered= load_driams(driams_pkl, filter=["DRIAMS_A", "DRIAMS_D"])
    dataA, labelA, metaA = driams_filtered["DRIAMS_A"]["data"], driams_filtered["DRIAMS_A"]["label"], driams_filtered["DRIAMS_A"]["meta"]
    dataD, labelD, metaD = driams_filtered["DRIAMS_D"]["data"], driams_filtered["DRIAMS_D"]["label"], driams_filtered["DRIAMS_D"]["meta"]

    # Downsample DRIAMS-A
    _, dataA_sub, _, labelA_sub, _, metaA_sub = train_test_split(
        dataA,
        labelA,
        metaA,
        test_size=len(dataD),
        random_state=42,
        stratify=labelA)

    # Concatenate data and labels
    data_final = np.vstack([dataA_sub, dataD])
    label_final = np.concatenate([labelA_sub, labelD])
    meta_final  = pd.concat([metaA_sub, metaD], ignore_index=True)

    # Construct dataloaders
    domain_ids = map_domains(meta_final)
    X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(data_final, label_final, domain_ids, test_size=0.2, random_state=42, stratify=label_final)

    # ---------------------------
    # Scaling
    # ---------------------------
    scaler, X_train_scaled, X_val_scaled = scale_data(X_train, X_val)

    with open(os.path.join(results_dir, "scaler.pkl"), 'wb') as file:
        pickle.dump(scaler, file)

    # ---------------------------
    # Dataloaders
    # ---------------------------
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32)

    domain_train_tensor = torch.tensor(domain_train, dtype=torch.long)
    domain_val_tensor   = torch.tensor(domain_val, dtype=torch.long)

    train_loader, val_loader = construct_dataloaders(X_train_tensor, X_val_tensor, domain_train_tensor, domain_val_tensor, batch_size=64)

    # ---------------------------
    # Model
    # ---------------------------
    model = VAE_Extended(X_train.shape[1], latent_dim=64, epochs=200, annealing_epochs=100, patience=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.trainloop(train_loader, val_loader, device=device)

    # ---------------------------
    # Store results
    # ---------------------------
    torch.save(model.state_dict(), os.path.join(results_dir, "vae.pth"))
    plot_model_metrics(model, "VAE", save=True, path=results_dir+"/vae_loss")

    # ---------------------------
    # t-SNEs
    # ---------------------------
    X_all = scaler.transform(data_final)
    X_all_tensor = torch.tensor(X_all, dtype=torch.float32)
    domain_all_tensor = torch.tensor(domain_ids, dtype=torch.long)

    all_dataset = TensorDataset(X_all_tensor, domain_all_tensor)
    all_loader  = DataLoader(all_dataset, batch_size=256, shuffle=False)

    mus_all = eval_model(model, all_loader, device, use_domain=False)

    tsne_df = compute_tsne_df(mus_all, label_final, meta_final)
    plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, save=True, path=results_dir + "/vae_tsne_global.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=False, save=True, path=results_dir + "/vae_tsne_global_species.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path=results_dir + "/vae_tsne_global_species_overlay.png")

    # t-SNE for each species
    df_all, tsne_results = compute_tsne_per_species(mus_all, label_final, meta_final)
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, save=True, path=results_dir + "/vae_tsne_species.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=True, save=True, path=results_dir + "/vae_tsne_species_overlay.png")
