"""
Adversarial Validation Module for Detecting Train vs. Test Covariate Shift.

Trains a discriminator to separate train samples (label 0) from test samples (label 1).
If ROC-AUC > 0.70, significant covariate shift exists between train and test distributions,
and drifting features are highlighted for regularization or pruning.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    sample_size: int = 5000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Computes Adversarial Validation ROC-AUC between train and test.
    """
    train_clean = train_df.copy()
    test_clean = test_df.copy()

    # Drop target and id columns if present
    cols_to_drop = [c for c in [target_col, id_col, "fold"] if c and c in train_clean.columns]
    train_clean = train_clean.drop(columns=cols_to_drop, errors="ignore")
    test_clean = test_clean.drop(columns=[c for c in [id_col] if c in test_clean.columns], errors="ignore")

    # Align columns
    common_cols = [c for c in train_clean.columns if c in test_clean.columns]
    if not common_cols:
        return {
            "adversarial_auc": 0.5,
            "covariate_shift_detected": False,
            "drifting_features": [],
            "error": "No common columns between train and test",
        }

    train_clean = train_clean[common_cols]
    test_clean = test_clean[common_cols]

    # Sample if datasets are large for fast sub-5-second execution
    if len(train_clean) > sample_size:
        train_clean = train_clean.sample(n=sample_size, random_state=random_state)
    if len(test_clean) > sample_size:
        test_clean = test_clean.sample(n=sample_size, random_state=random_state)

    train_clean["_is_test"] = 0
    test_clean["_is_test"] = 1

    combined = pd.concat([train_clean, test_clean], ignore_index=True)
    y_adv = combined["_is_test"].to_numpy()
    X_adv = combined.drop(columns=["_is_test"])

    # Preprocess categorical / text features into numeric codes
    X_proc = pd.DataFrame(index=X_adv.index)
    for col in X_adv.columns:
        s = X_adv[col]
        if pd.api.types.is_numeric_dtype(s):
            X_proc[col] = s.fillna(s.median())
        else:
            # Frequency encoding for categoricals/strings
            freq = s.value_counts(normalize=True).to_dict()
            X_proc[col] = s.map(freq).fillna(0.0)

    # Use fast LightGBM if installed, otherwise fast ExtraTreesClassifier
    feature_importances: Dict[str, float] = {}
    oof_preds = np.zeros(len(y_adv))

    try:
        import lightgbm as lgb
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        for tr_idx, val_idx in cv.split(X_proc, y_adv):
            X_tr, y_tr = X_proc.iloc[tr_idx], y_adv[tr_idx]
            X_val, y_val = X_proc.iloc[val_idx], y_adv[val_idx]

            clf = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=4,
                num_leaves=15,
                random_state=random_state,
                verbosity=-1,
                n_jobs=-1,
            )
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]

            # Accumulate importances
            for col, imp in zip(X_proc.columns, clf.feature_importances_):
                feature_importances[col] = feature_importances.get(col, 0.0) + float(imp)

    except ImportError:
        from sklearn.ensemble import ExtraTreesClassifier
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        for tr_idx, val_idx in cv.split(X_proc, y_adv):
            X_tr, y_tr = X_proc.iloc[tr_idx], y_adv[tr_idx]
            X_val = X_proc.iloc[val_idx]
            clf = ExtraTreesClassifier(
                n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1
            )
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
            for col, imp in zip(X_proc.columns, clf.feature_importances_):
                feature_importances[col] = feature_importances.get(col, 0.0) + float(imp)

    auc_score = float(roc_auc_score(y_adv, oof_preds))

    # Normalize feature importances
    total_imp = sum(feature_importances.values()) or 1.0
    sorted_features = sorted(
        [{"name": k, "drift_importance": round(v / total_imp, 4)} for k, v in feature_importances.items()],
        key=lambda x: x["drift_importance"],
        reverse=True,
    )

    top_drifting = [f["name"] for f in sorted_features[:3] if f["drift_importance"] > 0.05]
    shift_detected = auc_score >= 0.70

    return {
        "adversarial_auc": round(auc_score, 4),
        "covariate_shift_detected": shift_detected,
        "drifting_features": top_drifting,
    }
