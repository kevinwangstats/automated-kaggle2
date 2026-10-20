"""
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
