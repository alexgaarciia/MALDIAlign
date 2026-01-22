import yaml


def load_config(path: str = "configs/config.yaml"):
    """
    Load configuration parameters from a YAML file.

    Parameters
    ----------
    path : str, optional
        Path to the YAML configuration file. 
        Defaults to "config.yaml" in the current working directory.

    Returns
    -------
    dict
        A dictionary containing all configuration parameters parsed 
        from the YAML file.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file cannot be parsed properly.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg
