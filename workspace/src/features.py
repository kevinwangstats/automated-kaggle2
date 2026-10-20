"""
Feature engineering pipeline for tabular modeling.
Stateless and leak-free transformations.
"""

import pandas as pd
import numpy as np


def extract_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Transforms raw features into model-ready numerical columns.
    """
    out = pd.DataFrame(index=df.index)

    # 1. Direct numeric columns
    numeric_cols = [c for c in ["Pclass", "Age", "SibSp", "Parch", "Fare"] if c in df.columns]
    for col in numeric_cols:
        s = df[col]
        # Missing value imputation
        if col == "Age":
            out[col] = s.fillna(28.0)
        elif col == "Fare":
            out[col] = s.fillna(14.45)
        else:
            out[col] = s.fillna(0)

    # 2. Categorical frequency / binary encoding
    if "Sex" in df.columns:
        out["is_female"] = (df["Sex"] == "female").astype(float)

    if "Embarked" in df.columns:
        for val in ["S", "C", "Q"]:
            out[f"embarked_{val}"] = (df["Embarked"] == val).astype(float)

    # 3. Family / Group Size basic feature
    if "SibSp" in df.columns and "Parch" in df.columns:
        out["family_size"] = df["SibSp"] + df["Parch"] + 1


    # Applied hypothesis: h1_fare_sex_pclass_interaction
    if 'Age' in df.columns and 'Pclass' in df.columns:
        out['age_pclass_interaction'] = out['Age'] * out['Pclass']
    if 'Fare' in df.columns and 'family_size' in out.columns:
        out['fare_per_person'] = out['Fare'] / (out['family_size'] + 1e-5)

    return out
