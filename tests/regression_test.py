import unittest
import numpy as np
from src.methods_opt import kde, adaptive_kde, plugin_kde
from src.methods import kde as kde_slow, adaptive_kde as akde_slow, plugin_kde as plugin_slow

class TestOptimizations(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.randn(100)
        self.h = 0.4
        self.eval_points = np.linspace(-3, 3, 50)

    def test_kde_consistency(self):
        """Verify vectorized KDE matches loop-based KDE"""
        f_slow = kde_slow(self.h, self.data)
        f_fast = kde(self.h, self.data)
        
        # Test vector output
        y_slow = f_slow(self.eval_points)
        y_fast = f_fast(self.eval_points)
        
        np.testing.assert_allclose(y_slow, y_fast, rtol=1e-10, err_msg="KDE outputs mismatch!")

    def test_akde_consistency(self):
        """Verify vectorized Adaptive KDE matches loop-based AKDE"""
        f_slow = akde_slow(self.h, self.data)
        f_fast = adaptive_kde(self.h, self.data)
        
        y_slow = f_slow(self.eval_points)
        y_fast = f_fast(self.eval_points)
        
        np.testing.assert_allclose(y_slow, y_fast, rtol=1e-10, err_msg="AKDE outputs mismatch!")

    def test_plugin_consistency(self):
        """Verify vectorized Plugin KDE matches loop-based Plugin"""
        f_slow = plugin_slow(self.h, self.data)
        f_fast = plugin_kde(self.h, self.data)
        
        y_slow = f_slow(self.eval_points)
        y_fast = f_fast(self.eval_points)
        
        np.testing.assert_allclose(y_slow, y_fast, rtol=1e-10, err_msg="Plugin outputs mismatch!")

if __name__ == '__main__':
    unittest.main()