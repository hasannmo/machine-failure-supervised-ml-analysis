"""
preprocessing.py
-----------------
Loading, cleaning, and the train/test split for the AI4I 2020 dataset.

Rules this module exists to enforce (see README / context.txt):
  - The five failure-mode flags (TWF, HDF, PWF, OSF, RNF) are target leakage
    and must NEVER be used as model inputs. They are returned separately so
    they can be used for analysis only (e.g. recall per failure mode).
  - The split must be stratified and done ONCE, before any scaling/encoding
    is fit, so nothing about the test set leaks into training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_ID_COLS = ["UDI", "Product ID"]
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
TARGET_COL = "Machine failure"

NUMERIC_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COLS = ["Type"]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Load the raw AI4I 2020 CSV as downloaded from UCI."""
    df = pd.read_csv(csv_path)
    missing = [c for c in RAW_ID_COLS + FAILURE_MODE_COLS + [TARGET_COL] + NUMERIC_COLS + CATEGORICAL_COLS
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}. "
            "Check that this is the original AI4I 2020 CSV (column names must match exactly)."
        )
    return df


def split_inputs_target_modes(df: pd.DataFrame):
    """
    Split a raw dataframe into:
      X       - model inputs only (numeric + categorical features, NO ids, NO mode flags)
      y       - the target, Machine failure (0/1)
      modes   - the five failure-mode flags, for analysis only (never fed to a model)
    """
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[TARGET_COL].copy()
    modes = df[FAILURE_MODE_COLS].copy()
    return X, y, modes


def make_train_test_split(X: pd.DataFrame, y: pd.Series, modes: pd.DataFrame,
                           test_size: float = 0.2, random_state: int = 42):
    """
    One stratified split shared by every scenario (A/B/C) and every model,
    so results across scenarios/models are comparable. `modes` is split with
    the same indices so failure-mode analysis lines up with the test set.
    """
    idx_train, idx_test = train_test_split(
        X.index, test_size=test_size, random_state=random_state, stratify=y
    )
    return (
        X.loc[idx_train], X.loc[idx_test],
        y.loc[idx_train], y.loc[idx_test],
        modes.loc[idx_train], modes.loc[idx_test],
    )


def make_base_column_transformer(scale_numeric: bool) -> ColumnTransformer:
    """
    Base preprocessing: one-hot encode Type; optionally scale numeric columns.
    scale_numeric=True for Logistic Regression and KNN.
    scale_numeric=False for Decision Tree (scaling doesn't matter for trees,
    and leaving raw values makes the tree's learned thresholds directly
    comparable to the true physical thresholds in error analysis).

    NOTE: for scenarios B and C, the numeric feature list changes (extra
    engineered columns). Build the transformer AFTER adding those columns —
    see notebooks/02_features.ipynb — passing the updated numeric column list
    via the `numeric_cols` argument on FeatureAdderX transformers, or by
    constructing a fresh ColumnTransformer with the new numeric column names.
    This function covers Scenario A (raw features) directly.
    """
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
