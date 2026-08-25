"""
Analyzer Module

Computes quantitative performance metrics (Sharpe Ratio, Max Drawdown, Total Return,
Annualized Volatility, Win Rate) and generates financial visualizations.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Analyzer:
    """
    Quantitative performance analytics and visualization engine for crypto trading strategies.
    """

    def __init__(self, output_dir: Union[str, Path] = "output"):
        """
        Initializes the Analyzer with the target output directory.
        
        Args:
            output_dir: Directory path where output figures and reports are saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_metrics(self, returns: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculates annualized financial KPIs for hourly strategy return series.
        
        Hourly Annualization:
            - Observations per year: 24 * 365 = 8,760
            - Annualized Return = mean_return * 8,760
            - Annualized Volatility = std_return * sqrt(8,760)
            - Sharpe Ratio = (Annualized Return - risk_free_rate) / Annualized Volatility
            - Maximum Drawdown = min_t (CumProd_t - Peak_t) / Peak_t
            
        Args:
            returns: Time series of portfolio period returns.
            risk_free_rate: Annualized risk-free benchmark rate (default: 0.0).
            
        Returns:
            Dict[str, float]: Calculated metrics dictionary.
        """
        clean_returns = returns.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        mean_ret = float(clean_returns.mean())
        std_ret = float(clean_returns.std())
        
        ann_return = mean_ret * 8760.0
        ann_vol = std_ret * np.sqrt(8760.0)
        
        if ann_vol == 0.0 or np.isnan(ann_vol):
            sharpe = 0.0
        else:
            sharpe = (ann_return - risk_free_rate) / ann_vol
            
        # Cumulative wealth curve and peak tracking
        cum_ret = (1.0 + clean_returns).cumprod()
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        max_drawdown = float(drawdown.min())
        total_return = float(cum_ret.iloc[-1] - 1.0) if len(cum_ret) > 0 else 0.0
        
        # Win rate and profit metrics
        positive_trades = clean_returns[clean_returns > 0]
        negative_trades = clean_returns[clean_returns < 0]
        win_rate = len(positive_trades) / max(1, (len(positive_trades) + len(negative_trades)))
        
        # Sortino Ratio (Downside deviation)
        downside_std = clean_returns[clean_returns < 0].std() * np.sqrt(8760.0)
        sortino = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0

        return {
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown,
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sortino Ratio": sortino,
            "Win Rate": win_rate,
        }

    def plot_cumulative_returns(self, returns_dict: Dict[str, pd.Series], title: str, filename: str) -> None:
        """
        Plots comparative cumulative growth curves for strategy and benchmark assets.
        
        Args:
            returns_dict: Mapping of asset / factor label to period return series.
            title: Figure title.
            filename: Destination image filename within output_dir.
        """
        plt.figure(figsize=(12, 6), dpi=150)
        for label, returns in returns_dict.items():
            cum_ret = (1.0 + returns).cumprod()
            plt.plot(cum_ret, label=label, linewidth=1.5)
            
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Cumulative Growth Factor (Base = 1.0)", fontsize=11)
        plt.legend(loc="best", framealpha=0.9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_eigen_weights(self, weights: pd.Series, title: str, filename: str) -> None:
        """
        Plots sorted eigenportfolio weight distribution across the cryptocurrency universe.
        
        Args:
            weights: Series of asset weights in the eigenportfolio.
            title: Figure title.
            filename: Destination image filename.
        """
        sorted_weights = weights.sort_values(ascending=False)
        
        plt.figure(figsize=(12, 6), dpi=150)
        plt.plot(range(len(sorted_weights)), sorted_weights.values, marker="o", markersize=4, linewidth=1.5, color="#1f77b4")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Asset Rank", fontsize=11)
        plt.ylabel("Portfolio Weight", fontsize=11)
        plt.axhline(0, color="gray", linestyle=":", alpha=0.7)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_s_score_evolution(self, s_scores_df: pd.DataFrame, title: str, filename: str,
                               s_bo: float = 2.0, s_so: float = 2.0,
                               s_bc: float = 0.25, s_sc: float = 0.25) -> None:
        """
        Plots historical S-score evolution with active entry/exit threshold lines.
        
        Args:
            s_scores_df: S-score time series DataFrame (Time x Assets).
            title: Figure title.
            filename: Destination image filename.
            s_bo: Long entry threshold.
            s_so: Short entry threshold.
            s_bc: Long exit threshold.
            s_sc: Short exit threshold.
        """
        plt.figure(figsize=(12, 6), dpi=150)
        for col in s_scores_df.columns:
            plt.plot(s_scores_df[col], label=f"{col} S-Score", linewidth=1.2)
            
        plt.axhline(y=s_so, color="crimson", linestyle="--", linewidth=1.2, label=f"Open Short (s > +{s_so})")
        plt.axhline(y=-s_bo, color="forestgreen", linestyle="--", linewidth=1.2, label=f"Open Long (s < -{s_bo})")
        plt.axhline(y=s_sc, color="orange", linestyle=":", linewidth=1.1, label=f"Close Short (s < +{s_sc})")
        plt.axhline(y=-s_bc, color="purple", linestyle=":", linewidth=1.1, label=f"Close Long (s > -{s_bc})")
        
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=11)
        plt.ylabel("Standardized S-Score", fontsize=11)
        plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def plot_histogram(self, returns: pd.Series, title: str, filename: str) -> None:
        """
        Plots the return empirical frequency distribution.
        
        Args:
            returns: Series of period returns.
            title: Figure title.
            filename: Destination image filename.
        """
        clean_returns = returns.dropna()
        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(clean_returns, bins=60, alpha=0.75, color="#2ca02c", edgecolor="black", linewidth=0.5)
        plt.axvline(clean_returns.mean(), color="red", linestyle="--", linewidth=1.5,
                    label=f"Mean: {clean_returns.mean():.6f}")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Hourly Return", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

