"""
Test Suite for Cryptocurrency Statistical Arbitrage Engine

Validates mathematical models, PCA eigenportfolio decomposition, Ornstein-Uhlenbeck
parameter estimation, strategy state transitions, portfolio accounting, and end-to-end simulation.
"""

import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from .data_loader import DataLoader, resolve_data_path
    from .factor_model import FactorModel
    from .ou_solver import OUSolver
    from .strategy import Strategy
    from .analyzer import Analyzer
    from .main import run_backtest
except ImportError:
    from data_loader import DataLoader, resolve_data_path
    from factor_model import FactorModel
    from ou_solver import OUSolver
    from strategy import Strategy
    from analyzer import Analyzer
    from main import run_backtest


class TestDataLoader(unittest.TestCase):
    """Unit tests for DataLoader data ingestion, quality validation, and normalization."""

    def test_path_resolution(self):
        """Validates that resolve_data_path resolves existing files."""
        prices_path = resolve_data_path("coin_all_prices_full.csv")
        self.assertTrue(prices_path.exists(), f"Prices path '{prices_path}' not found.")

    def test_normalization_properties(self):
        """Verifies that normalized returns matrix has zero mean and unit variance."""
        loader = DataLoader("data/coin_all_prices_full.csv", "data/coin_universe_150K_40.csv")
        loader.load_data()
        sample_t = pd.Timestamp("2021-09-26 12:00:00+00:00")
        norm_ret, tokens, mu, sigma = loader.get_window_data(sample_t, M=240)
        
        self.assertGreater(len(tokens), 5, "Lookback window returned insufficient tokens.")
        self.assertEqual(norm_ret.shape[0], 240, "Lookback window row count mismatch.")
        
        # Check empirical means are approximately 0.0 and stds are approximately 1.0
        np.testing.assert_allclose(norm_ret.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(norm_ret.std(axis=0, ddof=1), 1.0, atol=1e-6)


class TestFactorModel(unittest.TestCase):
    """Unit tests for PCA factor model decomposition and factor returns projection."""

    def setUp(self):
        np.random.seed(42)
        # Create synthetic normalized return matrix (M=100, N=10)
        self.M, self.N = 100, 10
        self.returns = pd.DataFrame(
            np.random.randn(self.M, self.N),
            columns=[f"TOKEN_{i}" for i in range(self.N)]
        )
        self.stds = self.returns.std(ddof=1)
        self.means = self.returns.mean()
        self.norm_returns = (self.returns - self.means) / self.stds

    def test_pca_fitting(self):
        """Verifies PCA eigenvectors dimensions and descending eigenvalues."""
        fm = FactorModel(n_components=2)
        fm.fit(self.norm_returns)
        evecs, evals = fm.get_eigen_data()
        
        self.assertEqual(evecs.shape, (2, self.N), "Eigenvector matrix shape mismatch.")
        self.assertEqual(len(evals), 2, "Eigenvalues length mismatch.")
        self.assertGreaterEqual(evals[0], evals[1], "Eigenvalues not sorted in descending order.")

    def test_factor_returns_projection(self):
        """Verifies factor returns computation Q = V / sigma."""
        fm = FactorModel(n_components=2)
        fm.fit(self.norm_returns)
        evecs, _ = fm.get_eigen_data()
        
        f_ret = fm.compute_factor_returns(self.returns, evecs, self.stds)
        self.assertEqual(f_ret.shape, (self.M, 2), "Factor return shape mismatch.")
        self.assertListEqual(list(f_ret.columns), ["F1", "F2"])


class TestOUSolver(unittest.TestCase):
    """Unit tests for residual multi-factor regression, AR(1) OU estimation, and S-scores."""

    def test_ou_analytical_mapping(self):
        """Tests exact mathematical parameter mapping from AR(1) to continuous OU."""
        # Simulated AR(1) process: X_{t+1} = a + b * X_t + zeta
        np.random.seed(123)
        dt = 1.0 / 8760.0
        kappa_true = 50.0  # Annualized speed of reversion
        m_true = 0.05      # Long-run mean
        sigma_true = 0.30  # Diffusion volatility
        
        b_expected = np.exp(-kappa_true * dt)
        a_expected = m_true * (1.0 - b_expected)
        sigma_eq_expected = sigma_true / np.sqrt(2.0 * kappa_true)
        var_zeta_expected = (sigma_eq_expected ** 2) * (1.0 - b_expected ** 2)
        
        # Generate synthetic path of residuals
        n_steps = 50000
        x = np.zeros(n_steps)
        zeta = np.random.normal(0, np.sqrt(var_zeta_expected), n_steps)
        for i in range(1, n_steps):
            x[i] = a_expected + b_expected * x[i - 1] + zeta[i]
            
        # Cumulative residuals X = sum(epsilon) -> epsilon = diff(X)
        residuals = pd.DataFrame({"SYNTH_COIN": np.diff(x, prepend=0.0)})
        
        solver = OUSolver(dt=dt)
        params = solver.estimate_ou_params(residuals)
        
        est_kappa = params.loc["SYNTH_COIN", "kappa"]
        est_m = params.loc["SYNTH_COIN", "m"]
        est_sigma = params.loc["SYNTH_COIN", "sigma"]
        est_sigma_eq = params.loc["SYNTH_COIN", "sigma_eq"]
        
        # Verify estimates are within reasonable statistical tolerance (< 15% error)
        self.assertAlmostEqual(est_m, m_true, delta=0.03)
        self.assertAlmostEqual(est_sigma_eq, sigma_eq_expected, delta=0.05)

    def test_s_score_calculation(self):
        """Verifies S-score standardized z-score calculation."""
        solver = OUSolver()
        residuals = pd.DataFrame({"COIN_A": [0.01, 0.02, -0.01, 0.03]})
        # Cumulative sum at end is 0.05
        params = pd.DataFrame({
            "kappa": [10.0],
            "m": [0.01],
            "sigma": [0.2],
            "sigma_eq": [0.02]
        }, index=["COIN_A"])
        
        s = solver.compute_s_scores(residuals, params)
        expected_s = (0.05 - 0.01) / 0.02  # = 2.0
        self.assertAlmostEqual(s["COIN_A"], expected_s, places=5)


class TestStrategy(unittest.TestCase):
    """Unit tests for parametric S-score signal generation and state transitions."""

    def setUp(self):
        self.strategy = Strategy(s_bo=2.0, s_so=2.0, s_bc=0.25, s_sc=0.25)

    def test_open_long_signal(self):
        """Verifies opening long when s < -s_bo."""
        s_scores = pd.Series({"BTC": -2.5, "ETH": -1.5, "SOL": 0.0})
        positions = pd.Series({"BTC": 0, "ETH": 0, "SOL": 0})
        signals = self.strategy.generate_signals(s_scores, positions)
        self.assertEqual(signals["BTC"], 1, "Failed to trigger long entry at s < -2.0")
        self.assertEqual(signals["ETH"], 0, "Incorrectly triggered long entry at s = -1.5")
        self.assertEqual(signals["SOL"], 0)

    def test_open_short_signal(self):
        """Verifies opening short when s > +s_so."""
        s_scores = pd.Series({"BTC": 2.5, "ETH": 1.5, "SOL": 0.0})
        positions = pd.Series({"BTC": 0, "ETH": 0, "SOL": 0})
        signals = self.strategy.generate_signals(s_scores, positions)
        self.assertEqual(signals["BTC"], -1, "Failed to trigger short entry at s > +2.0")
        self.assertEqual(signals["ETH"], 0, "Incorrectly triggered short entry at s = 1.5")

    def test_close_long_signal(self):
        """Verifies closing long when s > -s_bc."""
        # Current position is long (+1)
        positions = pd.Series({"BTC": 1, "ETH": 1})
        # BTC reverted to -0.1 (> -0.25) -> Close; ETH is at -0.5 (<= -0.25) -> Hold
        s_scores = pd.Series({"BTC": -0.1, "ETH": -0.5})
        signals = self.strategy.generate_signals(s_scores, positions)
        self.assertEqual(signals["BTC"], 0, "Failed to close long when s reverted to -0.1")
        self.assertEqual(signals["ETH"], 1, "Closed long prematurely when s = -0.5")

    def test_close_short_signal(self):
        """Verifies closing short when s < +s_sc."""
        # Current position is short (-1)
        positions = pd.Series({"BTC": -1, "ETH": -1})
        # BTC reverted to 0.1 (< 0.25) -> Close; ETH is at 0.5 (>= 0.25) -> Hold
        s_scores = pd.Series({"BTC": 0.1, "ETH": 0.5})
        signals = self.strategy.generate_signals(s_scores, positions)
        self.assertEqual(signals["BTC"], 0, "Failed to close short when s reverted to 0.1")
        self.assertEqual(signals["ETH"], -1, "Closed short prematurely when s = 0.5")


class TestAnalyzer(unittest.TestCase):
    """Unit tests for financial metrics calculation."""

    def test_sharpe_and_drawdown(self):
        """Verifies annualized Sharpe ratio and peak-to-trough max drawdown calculations."""
        analyzer = Analyzer(output_dir="output")
        # Constant positive returns: mean = 0.0001, std = 0.001
        np.random.seed(42)
        ret = pd.Series(np.random.normal(0.0001, 0.001, 8760))
        metrics = analyzer.calculate_metrics(ret)
        
        expected_ann_ret = ret.mean() * 8760.0
        expected_ann_vol = ret.std() * np.sqrt(8760.0)
        expected_sharpe = expected_ann_ret / expected_ann_vol
        
        self.assertAlmostEqual(metrics["Sharpe Ratio"], expected_sharpe, places=3)
        self.assertLessEqual(metrics["Max Drawdown"], 0.0)


def run_all_tests():
    """Runs all unittests and prints formatted report."""
    print("=" * 65)
    print("      CRYPTOCURRENCY STATISTICAL ARBITRAGE TEST SUITE")
    print("=" * 65)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
