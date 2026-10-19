"""
Empirical Coder Agent.

Responsible for writing and modifying:
- workspace/src/features.py
- workspace/src/train.py

Enforces strict input/output contracts and manages the inner self-healing loop.
"""

from pathlib import Path
import os
import re
import sys
from typing import Any, Dict, Optional

from orchestrator.state import AgentState, HypothesisSpec


BASELINE_FEATURES_TEMPLATE = '''"""
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

    return out
'''

BASELINE_TRAIN_TEMPLATE = '''"""
Cross-validation training pipeline.
Trains a model family across 5 folds and outputs out-of-fold and test predictions.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import extract_features


def main():
    root = Path(__file__).parent.parent
    folds_path = root / "data" / "train_folds.parquet"
    test_path = root / "data" / "raw" / "test.csv"
    oof_out_path = root / "artifacts" / "oof_preds.npy"
    test_out_path = root / "artifacts" / "test_preds.npy"

    train_df = pd.read_parquet(folds_path)
    test_df = pd.read_csv(test_path)

    # Detect target
    target_col = "Survived" if "Survived" in train_df.columns else [c for c in train_df.columns if c not in ["fold", "PassengerId", "id"]][-1]
    y = train_df[target_col].to_numpy()

    X_train = extract_features(train_df, is_train=True)
    X_test = extract_features(test_df, is_train=False)

    oof_preds = np.zeros(len(train_df), dtype=float)
    test_fold_preds = np.zeros((len(test_df), 5), dtype=float)

    folds = sorted(train_df["fold"].unique())
    for fold in folds:
        train_idx = (train_df["fold"] != fold).to_numpy()
        val_idx = (train_df["fold"] == fold).to_numpy()

        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            random_state=42 + fold,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)

        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_probs
        test_fold_preds[:, fold] = model.predict_proba(X_test)[:, 1]

    oof_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(oof_out_path, oof_preds)
    np.save(test_out_path, np.mean(test_fold_preds, axis=1))
    print(f"Training completed successfully. OOF size: {len(oof_preds)}, Test size: {len(test_df)}")


if __name__ == "__main__":
    main()
'''


class EmpiricalCoder:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.src_dir = self.workspace_dir / "src"

    def write_initial_baseline(self) -> None:
        """Writes the clean, modular starting baseline for features and training."""
        self.src_dir.mkdir(parents=True, exist_ok=True)
        features_file = self.src_dir / "features.py"
        train_file = self.src_dir / "train.py"

        with open(features_file, "w") as f:
            f.write(BASELINE_FEATURES_TEMPLATE)
        with open(train_file, "w") as f:
            f.write(BASELINE_TRAIN_TEMPLATE)

        print("Baseline features.py and train.py successfully created.")

    def apply_hypothesis(self, hypothesis: HypothesisSpec) -> None:
        """
        Translates a formal HypothesisSpec into a code modification in features.py or train.py.
        """
        features_path = self.src_dir / "features.py"
        train_path = self.src_dir / "train.py"

        with open(features_path) as f:
            code = f.read()

        category = hypothesis.intervention_category
        spec_text = hypothesis.proposed_intervention

        if category == "feature_engineering":
            # Synthesize custom feature code into extract_features
            feature_snippet = f"\n    # Applied hypothesis: {hypothesis.hypothesis_id}\n"
            if "interaction" in spec_text.lower() or "ratio" in spec_text.lower():
                feature_snippet += (
                    "    if 'Age' in df.columns and 'Pclass' in df.columns:\n"
                    "        out['age_pclass_interaction'] = out['Age'] * out['Pclass']\n"
                    "    if 'Fare' in df.columns and 'family_size' in out.columns:\n"
                    "        out['fare_per_person'] = out['Fare'] / (out['family_size'] + 1e-5)\n"
                )
            elif "family" in spec_text.lower():
                feature_snippet += (
                    "    if 'family_size' in out.columns:\n"
                    "        out['is_alone'] = (out['family_size'] == 1).astype(float)\n"
                    "        out['large_family'] = (out['family_size'] > 4).astype(float)\n"
                )
            elif "title" in spec_text.lower() or "name" in spec_text.lower():
                feature_snippet += (
                    "    if 'Name' in df.columns:\n"
                    "        titles = df['Name'].str.extract(' ([A-Za-z]+)\\\\.', expand=False)\n"
                    "        out['is_mr'] = (titles == 'Mr').astype(float)\n"
                    "        out['is_mrs'] = (titles == 'Mrs').astype(float)\n"
                    "        out['is_miss'] = (titles == 'Miss').astype(float)\n"
                    "        out['is_master'] = (titles == 'Master').astype(float)\n"
                )
            elif "deck" in spec_text.lower() or "cabin" in spec_text.lower():
                feature_snippet += (
                    "    if 'Cabin' in df.columns:\n"
                    "        out['has_cabin'] = df['Cabin'].notna().astype(float)\n"
                )
            else:
                feature_snippet += (
                    "    if 'Fare' in out.columns:\n"
                    "        out['log_fare'] = np.log1p(np.maximum(out['Fare'], 0))\n"
                )

            # Insert before the return out statement
            code = code.replace("    return out", f"{feature_snippet}\n    return out")
            with open(features_path, "w") as f:
                f.write(code)
            print(f"Applied feature engineering hypothesis to {features_path}")

        elif category == "hyperparameter":
            with open(train_path) as f:
                train_code = f.read()
            # Tune learning rate or num_leaves
            if "learning_rate" in spec_text:
                train_code = re.sub(r"learning_rate=[\d\.]+", "learning_rate=0.03", train_code)
                train_code = re.sub(r"n_estimators=\d+", "n_estimators=150", train_code)
            elif "num_leaves" in spec_text:
                train_code = re.sub(r"num_leaves=\d+", "num_leaves=25", train_code)
                train_code = re.sub(r"max_depth=\d+", "max_depth=5", train_code)
            with open(train_path, "w") as f:
                f.write(train_code)
            print(f"Applied hyperparameter hypothesis to {train_path}")

    def heal_traceback(self, traceback_str: str) -> bool:
        """
        Self-healing logic: analyzes execution errors and repairs features.py or train.py.
        """
        features_path = self.src_dir / "features.py"
        with open(features_path) as f:
            code = f.read()

        # Fix KeyError or division by zero
        if "KeyError" in traceback_str:
            match = re.search(r"KeyError: ['\"](.*?)['\"]", traceback_str)
            if match:
                missing_key = match.group(1)
                print(f"Self-healing: fixing KeyError for column '{missing_key}'")
                # Wrap access in conditional check
                code = code.replace(f"df['{missing_key}']", f"df.get('{missing_key}', pd.Series(0, index=df.index))")
                with open(features_path, "w") as f:
                    f.write(code)
                return True
        elif "ZeroDivisionError" in traceback_str or "division by zero" in traceback_str:
            code = code.replace("/ out['family_size']", "/ (out['family_size'] + 1e-5)")
            with open(features_path, "w") as f:
                f.write(code)
            return True
        elif "SyntaxError" in traceback_str:
            # Revert to clean baseline features
            print("Self-healing: repairing SyntaxError by restoring safe feature template.")
            self.write_initial_baseline()
            return True

        return False
