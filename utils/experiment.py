from pathlib import Path
from datetime import datetime
import random
import numpy as np
import torch

def init_experiment(experiment_cfg: dict):
    """
    Initialize an experiment: validate config, set seeds, create result directory,
    and store a copy of the YAML configuration.

    Parameters
    ----------
    experiment_cfg : dict
        Dictionary from cfg["experiment"] containing at least:
        - name : str
        - seed : int
        - output_dir : str

    Returns
    -------
    experiment_dir : pathlib.Path
        Path to the directory where all experiment outputs should be saved.
    """
    
    # Validate required keys
    required_keys = {"name", "seed", "output_dir"}
    missing = required_keys - experiment_cfg.keys()

    if missing:
        raise KeyError(f"Missing required experiment keys: {missing}")

    name = experiment_cfg["name"]
    seed = experiment_cfg["seed"]
    output_dir = Path(experiment_cfg["output_dir"])

    # Set global seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=False)

    return experiment_dir
