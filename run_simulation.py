#!/usr/bin/env python3
"""
Monte Carlo CLI Runner
=======================
Standalone entry-point for running simulations from the command line
and generating publication-quality charts.

Usage:
    python run_simulation.py --ticker AAPL --model gbm --sims 10000
    python run_simulation.py --ticker NVDA --model heston --days 126
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")  # MUST come before pyplot import
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import fetch_market_data
from src.simulation.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationModel
from src.utils.logger import setup_logging

setup_logging(level="INFO")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Stock Price Simulation — PE-Grade Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker")
    parser.add_argument("--days", type=int, default=252, help="Trading days to simulate")
    parser.add_argument("--sims", type=int, default=10_000, help="Number of MC paths")
    parser.add_argument(
        "--model", choices=["gbm", "heston", "jump_diffusion"], default="gbm",
        help="Stochastic model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs", help="Directory for charts & JSON")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

PALETTE = {
    "bg": "#0d1117",
    "grid": "#21262d",
    "text": "#e6edf3",
    "accent": "#58a6ff",
    "green": "#3fb950",
    "red": "#f85149",
    "orange": "#d29922",
    "purple": "#bc8cff",
}


def plot_simulation(result, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    paths = result.price_paths
    n_paths_plot = min(200, paths.shape[0])
    x = np.arange(paths.shape[1])

    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── 1. Price Paths ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    sample_idx = np.random.choice(paths.shape[0], n_paths_plot, replace=False)
    for idx in sample_idx:
        ax1.plot(x, paths[idx], alpha=0.04, lw=0.5, color=PALETTE["accent"])

    pcts = [5, 25, 50, 75, 95]
    colours = [
        PALETTE["red"],
        PALETTE["orange"],
        PALETTE["accent"],
        PALETTE["orange"],
        PALETTE["green"],
    ]
    for pct, col in zip(pcts, colours):
        pct_path = np.percentile(paths, pct, axis=0)
        ax1.plot(x, pct_path, lw=2, color=col, label=f"P{pct}")

    ax1.axhline(
        result.current_price,
        color=PALETTE["text"],
        lw=1.2,
        ls="--",
        alpha=0.6,
        label="Current",
    )
    _style_ax(
        ax1,
        f"{result.ticker} — Monte Carlo Price Paths ({result.model.upper()})",
        "Trading Days",
        "Price (USD)",
    )
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.3)

    # ── 2. Final Price Distribution ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(
        result.final_prices,
        bins=80,
        color=PALETTE["accent"],
        alpha=0.8,
        orientation="horizontal",
        density=True
    )
    ax2.axhline(result.current_price, color=PALETTE["red"], lw=2, ls="--", label="Current")
    ax2.axhline(np.percentile(result.final_prices, 5), color=PALETTE["orange"],
                lw=1.5, ls=":", label="P5 / P95")
    ax2.axhline(np.percentile(result.final_prices, 95), color=PALETTE["green"], lw=1.5, ls=":")
    _style_ax(ax2, "Final Price Distribution", "Density", "Price (USD)")
    ax2.legend(fontsize=7, framealpha=0.3)

    # ── 3. Returns Distribution ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    returns = (result.final_prices - result.current_price) / result.current_price * 100
    ax3.hist(returns, bins=80, color=PALETTE["purple"], alpha=0.8, density=True)
    var_pct = result.var_95 * 100
    ax3.axvline(var_pct, color=PALETTE["red"], lw=2, label=f"VaR 95%: {var_pct:.1f}%")
    ax3.axvline(0, color=PALETTE["text"], lw=1.2, ls="--", alpha=0.5)
    _style_ax(ax3, "Return Distribution", "Return (%)", "Density")
    ax3.legend(fontsize=8, framealpha=0.3)

    # ── 4. Risk Metrics ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    metrics = {
        "VaR 95%": f"{result.var_95*100:.2f}%",
        "CVaR 95%": f"{result.cvar_95*100:.2f}%",
        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
        "Max Drawdown": f"{result.max_drawdown*100:.2f}%",
        "Prob. Profit": f"{result.prob_profit*100:.1f}%",
        "Simulations": f"{result.num_simulations:,}",
        "Horizon": f"{result.trading_days}d",
        "Elapsed": f"{result.elapsed_seconds:.2f}s",
    }
    ax4.axis("off")
    y = 0.95
    ax4.text(0.02, y, "Risk Metrics", fontsize=13, fontweight="bold",
             color=PALETTE["accent"], transform=ax4.transAxes)
    y -= 0.12
    for label, val in metrics.items():
        color = (
            PALETTE["red"]
            if "VaR" in label or "Drawdown" in label
            else PALETTE["green"]
            if "Profit" in label
            else PALETTE["text"]
        )
        ax4.text(
            0.02,
            y,
            label,
            fontsize=10,
            color=PALETTE["text"],
            transform=ax4.transAxes
        )
        ax4.text(
            0.65, y, val, fontsize=10, fontweight="bold", color=color, transform=ax4.transAxes
        )
        y -= 0.1
    ax4.set_facecolor(PALETTE["bg"])

    # ── 5. Percentile Table ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    ax5.text(0.02, 0.95, "Price Targets", fontsize=13, fontweight="bold",
             color=PALETTE["accent"], transform=ax5.transAxes)
    ax5.text(0.02, 0.84, "Percentile", fontsize=9, color=PALETTE["text"], transform=ax5.transAxes)
    ax5.text(0.55, 0.84, "Price (USD)", fontsize=9, color=PALETTE["text"], transform=ax5.transAxes)
    y = 0.73
    for lvl, price in result.percentiles.items():
        pct = float(lvl) * 100
        color = PALETTE["green"] if pct > 50 else PALETTE["red"] if pct < 25 else PALETTE["text"]
        ax5.text(0.02, y, f"P{pct:.0f}", fontsize=10, color=PALETTE["text"], transform=ax5.transAxes)
        ax5.text(0.55, y, f"${price:,.2f}", fontsize=10, fontweight="bold",
                 color=color, transform=ax5.transAxes)
        y -= 0.1
    ax5.set_facecolor(PALETTE["bg"])

    fig.text(
        0.5,
        0.01,
        "⚠ For educational and research purposes only. Not financial advice.",
        ha="center",
        fontsize=8,
        color="#6e7681",
        style="italic"
    )

    path = os.path.join(output_dir, f"{result.ticker}_{result.model}_simulation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    logger.info("Chart saved → %s", path)
    return path


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=PALETTE["text"], fontsize=9)
    ax.set_ylabel(ylabel, color=PALETTE["text"], fontsize=9)
    ax.tick_params(colors=PALETTE["text"], labelsize=8)
    ax.spines[:].set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    logger.info("=== Monte Carlo Stock Predictor ===")
    logger.info(
        "Ticker=%s | Model=%s | Days=%d | Sims=%d", args.ticker, args.model, args.days, args.sims
    )

    # Fetch market data
    market = fetch_market_data(args.ticker)
    logger.info("Current price: $%.2f | μ=%.4f | σ=%.4f",
                market.current_price, market.expected_return, market.volatility)

    # Configure & run
    config = SimulationConfig(
        ticker=args.ticker,
        current_price=market.current_price,
        expected_return=market.expected_return,
        volatility=market.volatility,
        trading_days=args.days,
        num_simulations=args.sims,
        model=SimulationModel(args.model),
        seed=args.seed,
    )
    engine = MonteCarloEngine(config)
    result = engine.run()

    # Print summary
    summary = result.to_dict()
    print("\n" + "═" * 60)
    print(f"  SIMULATION RESULTS — {args.ticker.upper()} ({args.model.upper()})")
    print("═" * 60)
    print(f"  Current Price  : ${result.current_price:,.2f}")
    print(f"  Horizon        : {result.trading_days} trading days")
    print(f"  Simulations    : {result.num_simulations:,}")
    print()
    for lvl, price in result.percentiles.items():
        pct = float(lvl) * 100
        change = (price - result.current_price) / result.current_price * 100
        arrow = "▲" if change >= 0 else "▼"
        print(f"  P{pct:4.0f}           : ${price:>10,.2f}  {arrow} {abs(change):.1f}%")
    print()
    print(f"  VaR 95%        : {result.var_95*100:.2f}%")
    print(f"  CVaR 95%       : {result.cvar_95*100:.2f}%")
    print(f"  Sharpe Ratio   : {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown   : {result.max_drawdown*100:.2f}%")
    print(f"  Prob. Profit   : {result.prob_profit*100:.1f}%")
    print(f"  Elapsed        : {result.elapsed_seconds:.3f}s")
    print("═" * 60 + "\n")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"{args.ticker}_{args.model}_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved → %s", json_path)

    # Generate chart
    if not args.no_chart:
        plot_simulation(result, args.output_dir)


if __name__ == "__main__":
    main()
