import os
import joblib
from pathlib import Path


def save_model(model, path):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : object
        Trained model to be saved (e.g., sklearn model, dictionary
        containing model and scaler, etc.).
    path : str or Path
        Full file path (including filename) where the model will be stored.
        Example: "models/rf_latent.joblib"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model successfully saved to {path}")


def load_model(path):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : str or Path
        File path of the saved model.

    Returns
    -------
    object
        The loaded model.
    """
    return joblib.load(path)
