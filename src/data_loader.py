"""
DataLoader Module

Handles loading, cleaning, aligning, and normalizing cryptocurrency price
and universe data for rolling-window quantitative factor models.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import os
import pandas as pd
import numpy as np


def resolve_data_path(file_path: Union[str, Path]) -> Path:
    """
    Resolves data file path using command-line arguments and relative search fallbacks.
    
    Args:
        file_path: Target filename or path.
        
    Returns:
        Path: Existing absolute or relative Path object.
    """
    p = Path(file_path)
    if p.exists():
        return p

    search_paths = [
        Path("data") / p.name,
        Path(__file__).parent.parent / "data" / p.name,
        Path("..") / "data" / p.name,
        Path(p.name),
    ]
    for sp in search_paths:
        if sp.exists():
            return sp

    return p


class DataLoader:
    """
    Loads and processes cryptocurrency price matrices and dynamic asset universe definitions.
    """

    def __init__(self, prices_path: Union[str, Path] = "data/coin_all_prices_full.csv",
                 universe_path: Union[str, Path] = "data/coin_universe_150K_40.csv"):
        self.prices_path = resolve_data_path(prices_path)
        self.universe_path = resolve_data_path(universe_path)
        self.prices: Optional[pd.DataFrame] = None
        self.universe: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Loads price and universe data from CSV files and parses timestamps."""
        if not self.prices_path.exists():
            raise FileNotFoundError(f"Prices file not found: {self.prices_path}")
        if not self.universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {self.universe_path}")

        self.prices = pd.read_csv(self.prices_path)
        if "startTime" in self.prices.columns:
            self.prices["startTime"] = pd.to_datetime(self.prices["startTime"])
            self.prices.set_index("startTime", inplace=True)

        self.universe = pd.read_csv(self.universe_path)
        if "startTime" in self.universe.columns:
            self.universe["startTime"] = pd.to_datetime(self.universe["startTime"])
            self.universe.set_index("startTime", inplace=True)

    def get_window_data(self, t: pd.Timestamp, M: int = 240) -> Tuple[pd.DataFrame, List[str], pd.Series, pd.Series]:
        """
        Retrieves and aligns data for a specific historical lookback window [t-M, t-1].
        
        Args:
            t: Current timestamp (end of lookback window, exclusive of test step).
            M: Lookback window size in hours (default: 240 hours / 10 days).
            
        Returns:
            Tuple containing:
                - pd.DataFrame: Normalized returns for valid tokens (M x N).
                - List[str]: List of valid token symbols present in the window.
                - pd.Series: Asset return empirical means over the window.
                - pd.Series: Asset return empirical standard deviations over the window.
        """
        if self.prices is None or self.universe is None:
            self.load_data()

        start_time = t - pd.Timedelta(hours=M)
        end_time = t - pd.Timedelta(hours=1)

        # Filter universe to tokens present in the top-cap list throughout the window
        window_universe = self.universe.loc[start_time:end_time]

        if window_universe.empty:
            return pd.DataFrame(), [], pd.Series(dtype=float), pd.Series(dtype=float)

        # Find the intersection of tokens across all hours in the window
        common_tokens = set(window_universe.iloc[0].values)
        for i in range(1, len(window_universe)):
            common_tokens = common_tokens.intersection(set(window_universe.iloc[i].values))

        common_tokens = list(common_tokens)

        price_start_time = start_time - pd.Timedelta(hours=1)
        window_prices = self.prices.loc[price_start_time:end_time, common_tokens].copy()

        window_prices = window_prices.ffill()

        # Validate data quality: Ensure at least 80% valid price points in the window
        valid_tokens = [
            token for token in common_tokens
            if window_prices[token].count() >= 0.8 * (M + 1)
        ]

        if not valid_tokens:
            return pd.DataFrame(), [], pd.Series(dtype=float), pd.Series(dtype=float)

        window_prices = window_prices[valid_tokens]

        # Calculate hourly returns: R_k = (P_k - P_{k-1}) / P_{k-1}
        returns = window_prices.pct_change().dropna()

        # Normalize returns: Y_{ik} = (R_{ik} - mean_i) / std_i
        means = returns.mean()
        stds = returns.std()
        stds = stds.replace(0, 1e-8)

        normalized_returns = (returns - means) / stds
        normalized_returns = normalized_returns.fillna(0.0)

        return normalized_returns, valid_tokens, means, stds


if __name__ == "__main__":
    loader = DataLoader("data/coin_all_prices_full.csv", "data/coin_universe_150K_40.csv")
    loader.load_data()
    print("Data loaded successfully.")
    sample_t = pd.Timestamp("2021-03-08 05:00:00+00:00")
    norm_ret, tokens, mu, sigma = loader.get_window_data(sample_t)
    print(f"Window ending {sample_t}: {len(tokens)} valid tokens identified.")
    print(norm_ret.head())

