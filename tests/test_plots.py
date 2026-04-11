import unittest
from unittest.mock import patch, MagicMock
from src.utils.plots import plot_curves

class TestPlotCurves(unittest.TestCase):
    
    def test_plot_curves_basic(self):
        """Test that plot_curves executes without errors with valid inputs."""
        train_loss = [0.5, 0.4, 0.33, 0.3, 0.28, 0.25]
        train_acc = [0.7, 0.8, 0.9, 0.92, 0.94, 0.95]
        valid_loss = [0.6, 0.5, 0.45, 0.38, 0.35, 0.38]
        valid_acc = [0.65, 0.75, 0.85, 0.88, 0.90, 0.92]

        plot_curves(train_loss, train_acc, valid_loss, valid_acc)
    
    
if __name__ == '__main__':
    unittest.main()