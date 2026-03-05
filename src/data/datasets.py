import numpy as np
import pandas as pd
from src.data.io import load_pkl


def load_driams(driams_pkl, filter=None):
    """
    Load the DRIAMS dataset from a PKL file, optionally filtering by hospital.

    Parameters
    ----------
    driams_pkl : str or pathlib.Path
        Path to the PKL file containing the DRIAMS dataset.
    filter : list of str or None, optional
        List of hospital identifiers to keep (e.g. ["DRIAMS_A", "DRIAMS_D"]).
        If None, the full dataset is returned.

    Returns
    -------
    dict
        If filter is None:
            {
                "data": np.ndarray,
                "label": np.ndarray,
                "meta": pd.DataFrame
            }

        If filter is provided:
            Keys are hospital names and values are dictionaries with:
                - "data": np.ndarray
                - "label": np.ndarray
                - "meta": pd.DataFrame
    """

    driams = load_pkl(driams_pkl)
    data, label, meta = driams["data"], driams["label"], pd.DataFrame.from_records(list(driams["meta"]))

    if filter is None:
        return {
            "data": data,
            "label": label,
            "meta": meta 
        }
    
    result = {}
    for hosp in filter:
        mask = np.where(meta["hospital"].values == hosp)[0]
        subdata, sublabel, submeta = data[mask], label[mask], meta.iloc[mask]

        result[hosp] = {
            "data": subdata,
            "label": sublabel,
            "meta": submeta
        }
        
    return result


def load_marisma(marisma_pkl):
    """
    Load the MARISMA dataset from a PKL file.

    Parameters
    ----------
    marisma_pkl : str or pathlib.Path
        Path to the PKL file containing the MARISMA dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "MARISMA".
    """

    marisma = load_pkl(marisma_pkl)
    data, label, meta = marisma["data"], marisma["label"], pd.DataFrame.from_records(list(marisma["meta"]))

    meta.insert(
        loc=0,
        column="hospital",
        value="MARISMA"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }


def load_msumg(msumg_pkl):
    """
    Load the MS-UMG dataset from a PKL file.

    Only samples grown on standard agar are retained.

    Parameters
    ----------
    msumg_pkl : str or pathlib.Path
        Path to the PKL file containing the MS-UMG dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "MS-UMG".
    """

    msumg = load_pkl(msumg_pkl)
    data, label, meta = msumg["data"], msumg["label"], pd.DataFrame.from_records(list(msumg["meta"]))

    mask_agar = meta["agar"] == "regular"
    data = data[mask_agar.values]
    label = label[mask_agar.values]
    meta = meta.loc[mask_agar].reset_index(drop=True)

    meta.insert(
        loc=0,
        column="hospital",
        value="MS-UMG"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }


def load_rki(rki_pkl):
    """
    Load the RKI dataset from a PKL file.

    Parameters
    ----------
    rki_pkl : str or pathlib.Path
        Path to the PKL file containing the RKI dataset.

    Returns
    -------
    dict
        {
            "data": np.ndarray,
            "label": np.ndarray,
            "meta": pd.DataFrame
        }
        The metadata includes a fixed 'hospital' column set to "RKI".
    """

    rki = load_pkl(rki_pkl)
    data, label, meta = rki["data"], rki["label"], pd.DataFrame.from_records(list(rki["meta"]))

    meta.insert(
        loc=0,
        column="hospital",
        value="RKI"
    )

    return {
        "data": data, 
        "label": label, 
        "meta": meta
    }
