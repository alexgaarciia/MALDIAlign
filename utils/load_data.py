import pickle
from pathlib import Path

def verify_data_path(data_dir):
    """
    Check whether a given path exists.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Path to check.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    data_path = Path(data_dir)
    exists = data_path.exists()

    if exists:
        print(f"Path exists: {data_path}")
    else:
        print(f"Path does not exist: {data_dir}")
    

def load_pkl(pkl_file):
    """
    Load and deserialize a Python object from a pickle file.

    This function provides a safe and readable interface for loading
    `.pkl` files. It performs basic validation, including checking 
    for file existence and extension consistency, and raises informative
    errors if deserialization fails.

    Parameters
    ----------
    pkl_file : str or pathlib.Path
        Path to the pickle file to load.

    Returns
    -------
    object
        The Python object stored inside the pickle file.
    """
    pkl_path = Path(pkl_file)

    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    if pkl_path.suffix != ".pkl":
        print("The input file does not have a .pkl extension")
    
    try:
        with open(pkl_path, "rb") as pkl:
            return pickle.load(pkl)
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Error unpickling {pkl_path}: {e}")
    