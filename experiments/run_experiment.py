#################################
# Path configuration
#################################
from pathlib import Path
import os
import sys
import json

PROJECT_NAME = "MALDIAlign"

cwd = Path().resolve()

target = None
for parent in [cwd] + list(cwd.parents):
    if parent.name == PROJECT_NAME:
        target = parent
        break

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
import numpy as np

import torch

from src.config.loader import load_config
from src.experiments.experiment import init_experiment
from src.training.data_pipeline import prepare_data
from models.build_model import build_model
from src.training.training import train_model
from src.visualization.viz import plot_model_metrics
from src.evaluation.eval import eval_model, run_tsne_evaluation, evaluate_amr_head
from src.evaluation.prior_sampling import sample_all_species_priors


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
    data = prepare_data(
        domains=data_cfg["domains"],
        experiment_dir=experiment_dir,
        target_domain=data_cfg.get("target_domain", None),
        n_target_samples=data_cfg.get("n_target_samples", None),
        pkl_path=data_cfg.get("pkl_path"),
        species_list=data_cfg.get("species_list", None),
        antibiotics_filter=data_cfg.get("antibiotics_filter", None),
        normalization=data_cfg.get("normalization", "row_minmax"),
        test_size=data_cfg.get("test_size", 0.2),
        batch_size=data_cfg.get("batch_size", 64),
        use_species_weight=data_cfg.get("use_species_weights", False),
        use_year_domains=data_cfg.get("use_year_domains", False),
        ood_holdout=data_cfg.get("ood_holdout", None),
    )
 
    data_final = data["data_final"]
    label_final = data["label_final"]
    meta_final = data["meta_final"]
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    all_loader = data["all_loader"]
    in_dim = data["input_dim"]
    species_weights = data["species_weights"]
    pos_weight = data["pos_weight"]
    species_names = data["species_names"]
    antibiotics_list = data["antibiotics"]
    
    # Build model
    print("\n===== Instantiating model =====")
    model = build_model(cfg, data["input_dim"], antibiotic_names=antibiotics_list,
                        num_domains=data.get("num_domains"))

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n===== Training model on {device} =====")
    trained_model = train_model(model, train_loader, val_loader, device, species_weights=species_weights, pos_weight=pos_weight)
    
    print("\n===== Finished training =====")
    torch.save(trained_model.state_dict(), experiment_dir / "model.pth")
    print("Model saved to:", experiment_dir / "model.pth")
    plot_model_metrics(trained_model, cfg["training"]["metrics_plot_title"], save=True, path=experiment_dir / "vae_loss")
    print("Training metrics saved")

    # AMR test evaluation
    if antibiotics_list is not None and hasattr(model, "amr_heads"):
        print("\n===== AMR evaluation on test set =====")
        amr_results = evaluate_amr_head(
            trained_model,
            data["test_data_norm"],
            data["amr_test"],
            antibiotics_list,
            device,
            species=data.get("test_species_encoded"),
            n_species=cfg["model"].get("n_species"),
        )
        print(f"\n{'Antibiotic':<30} {'AUC':>8} {'PR-AUC':>8} {'N':>6}")
        print("-" * 56)
        for atb, metrics in amr_results.items():
            print(f"{atb:<30} {metrics['auc']:>8.4f} {metrics['pr_auc']:>8.4f} {metrics['n']:>6}")

        with open(experiment_dir / "amr_test_results.json", "w") as f:
            json.dump(amr_results, f, indent=2)
        print(f"\nAMR test results saved to {experiment_dir / 'amr_test_results.json'}")

    # t-SNE evaluation
    if cfg["evaluation"]["compute_tsne"]:

        print("\n===== Computing latent representations =====")
        mus_all = eval_model(trained_model, all_loader, device, use_domain=cfg["evaluation"]["use_domain"])

        Z_prior = None
        prior_labels = None

        # optional prior sampling
        if cfg["evaluation"].get("prior_sampling", False):
            print("\n===== Sampling from priors =====")
            Z_prior, prior_ids = sample_all_species_priors(trained_model, cfg["model"]["n_species"], n_samples=cfg["evaluation"].get("prior_samples_per_species", 200), device=device)
            prior_labels = np.array([species_names[i] for i in prior_ids])

        print("\n===== Running t-SNE evaluation =====")
        run_tsne_evaluation(mus_all, label_final, meta_final, experiment_dir, cfg["model"]["type"], prior_samples=Z_prior, prior_labels=prior_labels, antibiotics_list=antibiotics_list)
        
    print("\n===== Experiment completed successfully =====")

if __name__ == "__main__":
    main()
