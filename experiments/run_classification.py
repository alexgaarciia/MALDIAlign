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
        description="Run a classification model experiment from a YAML config file"
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
from utils.config import load_config
from utils.experiment import init_experiment
from utils.data import prepare_data
from models.build_classifier import build_classifier
from sklearn.model_selection import train_test_split
from utils.metrics import metrics_report, print_metrics



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

    print("\n===== Loading data =====")
    data = prepare_data(
        domains=None,
        normalization=cfg["data"]["normalization"],
        seed=cfg["experiment"]["seed"],
        classification=True
    )

    X, y, meta, class_names, = data["X"], data["y"], data["meta"], data["class_names"]

    # Create directories for each classifiers
    print("\n===== Setting up classification directories =====")
    classification_dir = experiment_dir / "classification"
    classification_dir.mkdir(exist_ok=True)

    for space in cfg["classification"]:
        print(f"\n--- Space: {space} ---")
        train_domain = cfg["classification"][space]["train_domain"]
        test_domains = cfg["classification"][space]["test_domains"]
        val_size = cfg["classification"][space]["val_size"]

        space_dir = classification_dir / space
        space_dir.mkdir(parents=True, exist_ok=True)

        for clf in cfg["classification"][space]:
            # Skip non-classifier keys
            if clf in ["train_domain", "test_domains", "val_size"]:
                continue

            # Create directory for each classifier in each space
            clf_dir = space_dir / clf
            clf_dir.mkdir(parents=True, exist_ok=True)

            # Instantiate classifier
            print(f"\nInstantiating classifier: {clf}")
            model = build_classifier(
                classifier_type=clf,
                classifier_cfg=cfg["classification"][space][clf]
            )

            if space == "original":
                print("Using original feature space")

                # ---------------------------
                # Train / validation split
                # ---------------------------
                train_mask = meta["hospital"] == train_domain
                X_train_full = X[train_mask]
                y_train_full = y[train_mask]

                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=val_size,
                    random_state=cfg["experiment"]["seed"],
                    stratify=y_train_full
                )

                print("Training classifier...")
                model.fit(X_train, y_train)

                print("\n===== Validation on train domain =====")

                metrics_val = metrics_report(
                    X_val,
                    y_val,
                    model,
                    domain_name=f"{train_domain} (val)",
                    class_names=class_names
                )

                print_metrics(metrics_val, logs=True, save=True, path=clf_dir / f"metrics_{train_domain}")

                # ---------------------------
                # Test per domain (SEPARATE)
                # ---------------------------
                for test_domain in test_domains:
                    print(f"\n===== Testing on domain: {test_domain} =====")

                    test_mask = meta["hospital"] == test_domain
                    X_test = X[test_mask]
                    y_test = y[test_mask]

                    metrics = metrics_report(
                        X_test,
                        y_test,
                        model,
                        domain_name=test_domain,
                        class_names=class_names
                    )

                    print_metrics(metrics, logs=True, save=True, path=clf_dir / f"metrics_{test_domain}")
            else:
                print("Skipping (latent space not implemented yet)")
                pass


if __name__ == "__main__":
    main()