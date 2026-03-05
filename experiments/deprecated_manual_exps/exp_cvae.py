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
import torch
from models.deep.cVAE import ConditionalVAE_Bernoulli_Extended
from src.config.loader import load_config
from src.data.data import load_driams, map_domains, construct_dataloaders
from src.visualization.viz import *
from src.evaluation.eval import eval_model
from sklearn.model_selection import train_test_split


# ---------------------------
# Experiment
# ---------------------------
def main():
    # ---------------------------
    # Create results folder
    # ---------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results_cvae")
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------
    # Data preparation
    # ---------------------------
    print("===== Loading and preparing data... =====\n")
    cfg = load_config()
    driams_pkl = cfg["data"]["DRIAMS_REDUCED_PKL2"]
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

    # Row-wise min-max normalization
    data_norm = (data_final - data_final.min(axis=1, keepdims=True)) / (
        data_final.max(axis=1, keepdims=True) - data_final.min(axis=1, keepdims=True) + 1e-8
    )

    # Construct dataloaders
    domain_ids = map_domains(meta_final)
    X_train, X_val, y_train, y_val, domain_train, domain_val = train_test_split(data_norm, label_final, domain_ids, test_size=0.2, random_state=42, stratify=label_final)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
    X_all_tensor = torch.tensor(data_norm, dtype=torch.float32)

    domain_train_tensor = torch.tensor(domain_train, dtype=torch.long)
    domain_val_tensor   = torch.tensor(domain_val, dtype=torch.long)
    domain_all_tensor = torch.tensor(domain_ids, dtype=torch.long)

    train_loader, val_loader, all_loader = construct_dataloaders(X_train_tensor, X_val_tensor, X_all_tensor, domain_train_tensor, domain_val_tensor, domain_all_tensor, batch_size=64)

    # ---------------------------
    # Model
    # ---------------------------
    print("===== Instantiating model.. =====\n")
    model = ConditionalVAE_Bernoulli_Extended(X_train.shape[1], latent_dim=64, cond_dim=2, epochs=200, annealing_epochs=100, patience=40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"===== Training model with {device} =====\n")
    model.trainloop(train_loader, val_loader, device=device)

    # ---------------------------
    # Store results
    # ---------------------------
    print("===== Finished training, saving model and metrics... =====\n")
    torch.save(model.state_dict(), os.path.join(results_dir, "cvae.pth"))
    plot_model_metrics(model, "cVAE", save=True, path=results_dir+"/cvae_loss")

    # ---------------------------
    # t-SNEs
    # ---------------------------
    print("====== Computing t-SNE... ======\n")
    mus_all = eval_model(model, all_loader, device, use_domain=True)
    tsne_df = compute_tsne_df(mus_all, label_final, meta_final)
    df_all, tsne_results = compute_tsne_per_species(mus_all, label_final, meta_final)

    print("====== Saving plots... ======\n")
    plot_tsne_global(tsne_df, per_species=False, overlay_per_hospital=False, save=True, path=results_dir + "/cvae_tsne_global.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=False, save=True, path=results_dir + "/cvae_tsne_global_species.png")
    plot_tsne_global(tsne_df, per_species=True, overlay_per_hospital=True, save=True, path=results_dir + "/cvae_tsne_global_species_overlay.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=False, save=True, path=results_dir + "/cvae_tsne_species.png")
    plot_tsne_species(df_all, tsne_results, overlay_per_hospital=True, save=True, path=results_dir + "/cvae_tsne_species_overlay.png")

if __name__=="__main__":
    main()
