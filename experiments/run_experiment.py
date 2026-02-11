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


#################################
# CLI
#################################
import argparse

def parse_args():
    """
    Parse command-line arguments for running an experiment.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments containing:
        - config (str): Path to the YAML configuration file.
    """

    parser = argparse.ArgumentParser(
        description="Run a MALDIAlign experiment from a YAML config file"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment YAML file (e.g. configs/vae.yaml)"
    )

    args = parser.parse_args()
    return args


#################################
# Imports
#################################
import torch
from utils.config import load_config
from utils.experiment import init_experiment
from utils.data import prepare_data
from models.build_model import build_model
from utils.training import train_model
from utils.viz import plot_model_metrics
from utils.eval import eval_model, run_tsne_evaluation


#################################
# Main
#################################
def main():
    # Parse arguments
    args = parse_args()
    print("===== Running experiment =====")
    print("Using config:", args.config)

    # Initialize experiment
    print("\n===== Initializing experiment =====")
    cfg = load_config(args.config)
    experiment_dir = init_experiment(cfg["experiment"])
    print("Experiment directory:", experiment_dir)

    # Load data
    print("\n===== Loading and preparing data =====")
    data_cfg = cfg["data"]
    data = prepare_data(domains=data_cfg["domains"], normalization=data_cfg.get("normalization", "row_minmax"), test_size=data_cfg.get("test_size", 0.2), batch_size=data_cfg.get("batch_size", 64), use_species_weight=data_cfg.get("use_species_weights", False))
    data_final, label_final, meta_final, train_loader, val_loader, all_loader, in_dim, species_weights = data["data_final"], data["label_final"], data["meta_final"], data["train_loader"], data["val_loader"], data["all_loader"], data["input_dim"], data["species_weights"]

    # Build model
    print("\n===== Instantiating model =====")
    model = build_model(cfg, data["input_dim"])

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Training model on {device} =====")
    trained_model = train_model(model, train_loader, val_loader, device, species_weights=species_weights)
    print("\n===== Finished training =====")
    torch.save(trained_model.state_dict(), experiment_dir / "model.pth")
    print("Model saved to:", experiment_dir / "model.pth")
    plot_model_metrics(trained_model, cfg["training"]["metrics_plot_title"], save=True, path=experiment_dir / "vae_loss")
    print("Training metrics saved")

    # t-SNEs
    print("\n===== Computing latent representations =====")
    mus_all = eval_model(trained_model, all_loader, device, use_domain=cfg["evaluation"]["use_domain"])

    print("\n===== Running t-SNE evaluation =====")
    run_tsne_evaluation(mus_all, label_final, meta_final, experiment_dir, cfg["model"]["type"])

    print("\n===== Experiment completed successfully =====")

if __name__ == "__main__":
    main()
