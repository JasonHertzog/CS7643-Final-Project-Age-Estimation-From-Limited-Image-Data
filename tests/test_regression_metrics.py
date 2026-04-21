import unittest
import torch

from src.utils.regression import compute_regression_metrics

class TestRegressionMetrics(unittest.TestCase):
    def test_metrics(self):
        outputs = torch.tensor([10.0, 20.0, 30.0, 40.0])
        targets = torch.tensor([12.0, 19.0, 35.0, 47.0])

        metrics = compute_regression_metrics(outputs, targets)

        self.assertAlmostEqual(metrics["mae"], 3.75)
        self.assertAlmostEqual(metrics["mse"], 19.75)
        self.assertAlmostEqual(metrics["acc_at_3"], 0.5)
        self.assertAlmostEqual(metrics["acc_at_5"], 0.75)

    def test_perfect_predictions(self):
        outputs = torch.tensor([5., 10.0, 15.0])
        targets = torch.tensor([5.0, 10.0, 15.0])

        metrics = compute_regression_metrics(outputs, targets)

        self.assertAlmostEqual(metrics["mae"], 0.0)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["acc_at_3"], 1.0)
        self.assertAlmostEqual(metrics["acc_at_5"], 1.0)

if __name__ == "__main__":
    unittest.main()
