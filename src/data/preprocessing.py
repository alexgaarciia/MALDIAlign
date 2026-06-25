import numpy as np

def map_domains(meta):
    """
    Map each hospital to a unique integer domain ID.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame containing a column 'hospital'.

    Returns
    -------
    domain_ids : np.ndarray
        Array of integers representing the domain of each sample.
    """

    unique_hosp = np.unique(meta["hospital"])
    domain_map = {h: i for i, h in enumerate(unique_hosp)}
    domain_ids = meta["hospital"].map(domain_map).values

    return domain_ids


def map_domains_by_year(meta):
    """
    Map each (hospital, year) pair to a unique integer domain ID.

    Parameters
    ----------
    meta : pd.DataFrame
        Metadata DataFrame containing columns 'hospital' and 'year'.

    Returns
    -------
    domain_ids : np.ndarray
        Integer domain ID for each sample.
    domain_map : dict
        Mapping from (hospital, year) tuple to integer ID, sorted
        deterministically so IDs are stable across runs.
    """
    pairs = list(zip(meta["hospital"], meta["year"].astype(str)))
    unique_pairs = sorted(set(pairs))
    domain_map = {p: i for i, p in enumerate(unique_pairs)}
    domain_ids = np.array([domain_map[p] for p in pairs])
    return domain_ids, domain_map


def row_minmax_normalize(X):
    """
    Apply row-wise min-max normalization to a 2D array.

    Each row (sample) is independently scaled to the range [0, 1] using:

        X_norm = (X - min_row) / (max_row - min_row)

    A small epsilon (1e-8) is added to the denominator to prevent division
    by zero in case a row has constant values.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Input array where each row corresponds to a sample (e.g., a MALDI
        or FTIR spectrum) and each column to a feature (e.g., m/z or
        wavenumber intensity).

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        Row-wise min-max normalized array with values scaled to [0, 1].
    """
    X_min = X.min(axis=1, keepdims=True)
    X_max = X.max(axis=1, keepdims=True)
    return (X - X_min) / (X_max - X_min + 1e-8)
