from __future__ import annotations

import numpy as np
import pandas as pd

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def hdf_rule(X: pd.DataFrame) -> pd.Series:
    temp_diff = X["Process temperature [K]"] - X["Air temperature [K]"]
    return (temp_diff < 8.6) & (X["Rotational speed [rpm]"] < 1380)


def pwf_rule(X: pd.DataFrame) -> pd.Series:
    power_w = X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
    return (power_w < 3500) | (power_w > 9000)


def osf_rule(X: pd.DataFrame) -> pd.Series:
    limit = X["Type"].map(OSF_LIMIT)
    wear_x_torque = X["Tool wear [min]"] * X["Torque [Nm]"]
    return wear_x_torque > limit


def oracle_predict(X: pd.DataFrame) -> np.ndarray:
    triggered = hdf_rule(X) | pwf_rule(X) | osf_rule(X)
    return triggered.astype(int).to_numpy()


def oracle_rule_breakdown(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "HDF": hdf_rule(X).to_numpy(),
            "PWF": pwf_rule(X).to_numpy(),
            "OSF": osf_rule(X).to_numpy(),
        },
        index=X.index,
    )
