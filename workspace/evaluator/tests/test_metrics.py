"""
Unit tests for the Universal Metric Registry using standard library unittest.
"""

from pathlib import Path
import sys
import unittest
import numpy as np

repo_root = Path(__file__).parent.parent.parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from workspace.evaluator.metrics import MetricRegistry


class TestMetricRegistry(unittest.TestCase):
    def test_rmse(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        self.assertEqual(MetricRegistry.evaluate("rmse", y_true, y_pred), 0.0)

        y_pred_err = np.array([2.0, 3.0, 4.0])
        self.assertEqual(MetricRegistry.evaluate("rmse", y_true, y_pred_err), 1.0)

    def test_rmsle(self):
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        self.assertEqual(MetricRegistry.evaluate("rmsle", y_true, y_pred), 0.0)

        # Test negative clipping
        y_pred_neg = np.array([-5.0, 20.0, 30.0])
        score = MetricRegistry.evaluate("rmsle", y_true, y_pred_neg)
        self.assertGreater(score, 0.0)

    def test_roc_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(MetricRegistry.evaluate("roc_auc", y_true, y_pred_perfect), 1.0)

        y_pred_inverted = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertEqual(MetricRegistry.evaluate("roc_auc", y_true, y_pred_inverted), 0.0)

    def test_qwk(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        self.assertEqual(MetricRegistry.evaluate("qwk", y_true, y_pred), 1.0)

        y_pred_partial = np.array([0, 1, 2, 4, 3])
        score = MetricRegistry.evaluate("qwk", y_true, y_pred_partial)
        self.assertTrue(0.8 < score < 1.0)

    def test_nan_inf_validation(self):
        y_true = np.array([0, 1, 1])
        y_pred_nan = np.array([0.1, np.nan, 0.9])
        with self.assertRaises(ValueError):
            MetricRegistry.evaluate("roc_auc", y_true, y_pred_nan)

        y_pred_inf = np.array([0.1, np.inf, 0.9])
        with self.assertRaises(ValueError):
            MetricRegistry.evaluate("roc_auc", y_true, y_pred_inf)

    def test_is_improvement(self):
        # Lower is better (RMSE)
        self.assertTrue(MetricRegistry.is_improvement("rmse", new_score=0.45, baseline_score=0.50, epsilon=0.01))
        self.assertFalse(MetricRegistry.is_improvement("rmse", new_score=0.495, baseline_score=0.50, epsilon=0.01))

        # Higher is better (ROC-AUC)
        self.assertTrue(MetricRegistry.is_improvement("roc_auc", new_score=0.88, baseline_score=0.85, epsilon=0.01))
        self.assertFalse(MetricRegistry.is_improvement("roc_auc", new_score=0.855, baseline_score=0.85, epsilon=0.01))


if __name__ == "__main__":
    unittest.main()
