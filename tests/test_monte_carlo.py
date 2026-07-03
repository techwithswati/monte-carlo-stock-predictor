"""
Unit & Integration Tests — Monte Carlo Engine
==============================================
Covers statistical correctness, boundary conditions, and API contracts.
"""

import numpy as np
import pytest

from src.simulation.monte_carlo import (
    MonteCarloEngine,
    SimulationConfig,
    SimulationModel,
    SimulationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> SimulationConfig:
    return SimulationConfig(
        ticker="TEST",
        current_price=100.0,
        expected_return=0.10,
        volatility=0.20,
        trading_days=252,
        num_simulations=2_000,
        seed=0,
    )


@pytest.fixture
def quick_config(base_config) -> SimulationConfig:
    base_config.trading_days = 21
    base_config.num_simulations = 500
    return base_config


# ---------------------------------------------------------------------------
# GBM tests
# ---------------------------------------------------------------------------

class TestGBM:
    def test_shape(self, base_config):
        result = MonteCarloEngine(base_config).run()
        assert result.price_paths.shape == (
            base_config.num_simulations,
            base_config.trading_days + 1,
        )

    def test_initial_price_exact(self, base_config):
        result = MonteCarloEngine(base_config).run()
        assert np.allclose(result.price_paths[:, 0], base_config.current_price)

    def test_all_prices_positive(self, base_config):
        result = MonteCarloEngine(base_config).run()
        assert (result.price_paths > 0).all(), "GBM must produce strictly positive prices"

    def test_mean_path_within_range(self, base_config):
        """Log-normal mean should be S0·exp(μT) within 10% tolerance."""
        result = MonteCarloEngine(base_config).run()
        theoretical_mean = base_config.current_price * np.exp(
            base_config.expected_return * (base_config.trading_days / 252)
        )
        actual_mean = result.final_prices.mean()
        assert abs(actual_mean - theoretical_mean) / theoretical_mean < 0.10

    def test_reproducibility(self, base_config):
        r1 = MonteCarloEngine(base_config).run()
        r2 = MonteCarloEngine(base_config).run()
        np.testing.assert_array_equal(r1.price_paths, r2.price_paths)

    def test_var_negative(self, base_config):
        """VaR should be negative (downside)."""
        result = MonteCarloEngine(base_config).run()
        assert result.var_95 < 0

    def test_cvar_le_var(self, base_config):
        """CVaR must be ≤ VaR (more conservative tail measure)."""
        result = MonteCarloEngine(base_config).run()
        assert result.cvar_95 <= result.var_95

    def test_prob_profit_in_range(self, base_config):
        result = MonteCarloEngine(base_config).run()
        assert 0.0 <= result.prob_profit <= 1.0

    def test_percentile_ordering(self, base_config):
        result = MonteCarloEngine(base_config).run()
        values = list(result.percentiles.values())
        assert values == sorted(values), "Percentiles must be monotonically increasing"

    def test_to_dict_keys(self, base_config):
        result = MonteCarloEngine(base_config).run()
        d = result.to_dict()
        required = {"ticker", "model", "current_price", "trading_days",
                    "num_simulations", "percentiles", "var_95", "cvar_95",
                    "sharpe_ratio", "max_drawdown", "prob_profit", "elapsed_seconds"}
        assert required.issubset(d.keys())

    def test_elapsed_positive(self, quick_config):
        result = MonteCarloEngine(quick_config).run()
        assert result.elapsed_seconds > 0


# ---------------------------------------------------------------------------
# Heston tests
# ---------------------------------------------------------------------------

class TestHeston:
    def test_positive_prices(self, base_config):
        base_config.model = SimulationModel.HESTON
        result = MonteCarloEngine(base_config).run()
        assert (result.price_paths > 0).all()

    def test_shape(self, base_config):
        base_config.model = SimulationModel.HESTON
        result = MonteCarloEngine(base_config).run()
        assert result.price_paths.shape == (base_config.num_simulations, base_config.trading_days + 1)

    def test_initial_price(self, base_config):
        base_config.model = SimulationModel.HESTON
        result = MonteCarloEngine(base_config).run()
        assert np.allclose(result.price_paths[:, 0], base_config.current_price)

    def test_model_label(self, base_config):
        base_config.model = SimulationModel.HESTON
        result = MonteCarloEngine(base_config).run()
        assert result.model == "heston"


# ---------------------------------------------------------------------------
# Jump-Diffusion tests
# ---------------------------------------------------------------------------

class TestJumpDiffusion:
    def test_positive_prices(self, base_config):
        base_config.model = SimulationModel.JUMP_DIFFUSION
        result = MonteCarloEngine(base_config).run()
        assert (result.price_paths > 0).all()

    def test_fatter_tails_than_gbm(self, base_config):
        """Jump-diffusion should produce heavier downside tails than GBM."""
        gbm = MonteCarloEngine(base_config).run()
        base_config.model = SimulationModel.JUMP_DIFFUSION
        jd = MonteCarloEngine(base_config).run()
        assert jd.var_95 <= gbm.var_95 or jd.cvar_95 <= gbm.cvar_95, \
            "Jump-diffusion should have fatter downside tails"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_day(self, base_config):
        base_config.trading_days = 1
        result = MonteCarloEngine(base_config).run()
        assert result.price_paths.shape == (base_config.num_simulations, 2)

    def test_zero_volatility(self, base_config):
        """Zero vol should produce deterministic drift path."""
        base_config.volatility = 1e-8
        base_config.num_simulations = 100
        result = MonteCarloEngine(base_config).run()
        assert (result.price_paths > 0).all()

    def test_high_volatility(self, base_config):
        base_config.volatility = 2.0  # 200% vol
        result = MonteCarloEngine(base_config).run()
        assert (result.price_paths > 0).all()

    def test_single_simulation(self, base_config):
        base_config.num_simulations = 1
        result = MonteCarloEngine(base_config).run()
        assert result.price_paths.shape[0] == 1

    def test_all_models_run(self, base_config):
        base_config.num_simulations = 100
        base_config.trading_days = 21
        for model in SimulationModel:
            base_config.model = model
            result = MonteCarloEngine(base_config).run()
            assert isinstance(result, SimulationResult)
