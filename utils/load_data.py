import os 

def verify_data_path(data_dir):
    """
    Verify that the given data directory exists and contains readable files.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the dataset is expected to be located.

    Returns
    -------
    bool
        True if the directory exists and contains files, False otherwise.
    """
    if not os.path.exists(data_dir):
        print(f"Path does not exist: {data_dir}")
        return False

    if not os.path.isdir(data_dir):
        print(f"Path exists but is not a directory: {data_dir}")
        return False

    files = os.listdir(data_dir)
    if not files:
        print(f"The directory {data_dir} is empty.")
        return False

    print(f"Valid path. Found {len(files)} items.")
    return True
