"""
features.py
------------
Feature engineering for the three experiment scenarios:

  A - Raw:              the five numeric columns + Type, untouched.
  B - Generic automatic: A + PolynomialFeatures degree 2 (interactions/squares).
  C - Physics-informed:  A + temp_diff, power_W, wear_x_torque.

Scenario C's transformer, PhysicsFeatures, is the important one to keep here
(rather than inline in a notebook): it is saved inside the final joblib
pipeline, so the demo notebook (05_demo_gradio) must be able to import this
exact class to unpickle the model. Do not rename this file or this class
without re-saving the models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# UCI-documented overstrain thresholds (tool wear x torque, in minNm), by Type
OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """
    Adds three physics-informed columns used in Scenario C:

      temp_diff     = Process temperature [K] - Air temperature [K]
                       (compare against the HDF threshold: 8.6 K)
      power_W       = Torque [Nm] * Rotational speed [rpm] * 2*pi / 60
                       (compare against the PWF thresholds: 3500 W / 9000 W)
      wear_x_torque = Tool wear [min] * Torque [Nm]
                       (compare against the OSF thresholds: 11000/12000/13000 minNm,
                        which depend on Type — see osf_margin below)

    Input: a DataFrame with at least the NUMERIC_COLS and, for osf_margin,
    the "Type" column. Output: the same DataFrame with the new columns
    appended (original columns are kept, so this can feed straight into the
    same ColumnTransformer used for Scenario A, with the new columns added to
    its numeric list).
    """

    def __init__(self, add_osf_margin: bool = True):
        self.add_osf_margin = add_osf_margin

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["temp_diff"] = X["Process temperature [K]"] - X["Air temperature [K]"]
        X["power_W"] = (
            X["Torque [Nm]"] * X["Rotational speed [rpm]"] * 2 * np.pi / 60
        )
        X["wear_x_torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]

        if self.add_osf_margin:
            if "Type" not in X.columns:
                raise ValueError(
                    "PhysicsFeatures(add_osf_margin=True) needs a 'Type' column "
                    "to know the OSF threshold (11000/12000/13000 differs by type)."
                )
            limit = X["Type"].map(OSF_LIMIT)
            if limit.isna().any():
                bad = X.loc[limit.isna(), "Type"].unique()
                raise ValueError(f"Unknown Type value(s) for OSF limit: {bad}")
            # negative = still under the documented overstrain limit
            X["osf_margin"] = X["wear_x_torque"] - limit

        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["temp_diff", "power_W", "wear_x_torque"]
        if self.add_osf_margin:
            extra.append("osf_margin")
        base = list(input_features) if input_features is not None else []
        return np.array(base + extra)

    @property
    def engineered_columns(self) -> list[str]:
        cols = ["temp_diff", "power_W", "wear_x_torque"]
        if self.add_osf_margin:
            cols.append("osf_margin")
        return cols


def numeric_cols_for_scenario_c(add_osf_margin: bool = True) -> list[str]:
    """Numeric column list to pass to ColumnTransformer after PhysicsFeatures."""
    extra = ["temp_diff", "power_W", "wear_x_torque"] + (["osf_margin"] if add_osf_margin else [])
    return NUMERIC_COLS + extra
