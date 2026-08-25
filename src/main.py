"""
Main Execution Module

Orchestrates the cryptocurrency statistical arbitrage simulation:
1. Data Ingestion & Universe Alignment
2. Rolling-Window PCA Factor Model Decomposition
3. Multi-Factor Residual Regression & Ornstein-Uhlenbeck S-Score Estimation
4. Mean-Reversion Signal Generation & Portfolio Mark-to-Market Accounting
5. Performance Analysis, Metric Reporting, and Visualization Exports
"""

import argparse
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .data_loader import DataLoader, resolve_data_path
    from .factor_model import FactorModel
    from .ou_solver import OUSolver
    from .strategy import Strategy
    from .analyzer import Analyzer
except ImportError:
    from data_loader import DataLoader, resolve_data_path
    from factor_model import FactorModel
    from ou_solver import OUSolver
    from strategy import Strategy
    from analyzer import Analyzer


def run_backtest(prices_path: str = "data/coin_all_prices_full.csv",
                 universe_path: str = "data/coin_universe_150K_40.csv",
                 output_dir: str = "output",
                 window_m: int = 240,
                 start_date: str = "2021-09-26 00:00:00+00:00",
                 end_date: str = "2022-09-25 23:00:00+00:00",
                 s_bo: float = 2.0,
                 s_so: float = 2.0,
                 s_bc: float = 0.25,
                 s_sc: float = 0.25,
                 generate_plots: bool = True) -> dict:
    """
    Executes the end-to-end rolling-window statistical arbitrage backtest.
    
    Args:
        prices_path: Path to cryptocurrency historical prices CSV.
        universe_path: Path to top-cap universe definitions CSV.
        output_dir: Directory to save generated CSV artifacts and plots.
        window_m: Lookback training window size in hours (default: 240 hours).
        start_date: Backtest start timestamp string.
        end_date: Backtest end timestamp string.
        s_bo: Long entry S-score threshold.
        s_so: Short entry S-score threshold.
        s_bc: Long exit S-score threshold.
        s_sc: Short exit S-score threshold.
        generate_plots: Whether to generate and save Matplotlib visualization plots.
        
    Returns:
        dict: Performance metrics summary dictionary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulation components
    loader = DataLoader(prices_path, universe_path)
    factor_model = FactorModel(n_components=2)
    ou_solver = OUSolver(dt=1.0 / 8760.0)
    strategy = Strategy(s_bo=s_bo, s_so=s_so, s_bc=s_bc, s_sc=s_sc)
    analyzer = Analyzer(output_dir=out_path)
    
    print(f"Loading market datasets from '{prices_path}' and '{universe_path}'...")
    loader.load_data()
    
    # Storage structures for time series results
    eigenvectors_list_1 = []
    eigenvectors_list_2 = []
    signals_list = []
    portfolio_returns = []
    f1_returns = []
    f2_returns = []
    s_scores_list = []
    
    current_positions = {}
    
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    time_range = pd.date_range(start=start_ts, end=end_ts, freq="h")
    
    print(f"Starting rolling-window backtest ({start_ts.strftime('%Y-%m-%d %H:%M')} to "
          f"{end_ts.strftime('%Y-%m-%d %H:%M')}, {len(time_range)} hourly periods)...")
    
    for t in tqdm(time_range, desc="Simulating"):
        # 1. Historical Window Ingestion [t-M, t-1]
        normalized_returns, valid_tokens, means, stds = loader.get_window_data(t, window_m)
        
        if normalized_returns.empty or len(valid_tokens) < 2:
            portfolio_returns.append(0.0)
            continue
            
        # 2. PCA Factor Model Decomposition
        factor_model.fit(normalized_returns)
        eigenvectors, eigenvalues = factor_model.get_eigen_data()
        
        ev1 = pd.Series(eigenvectors[0], index=valid_tokens, name=t)
        ev2 = pd.Series(eigenvectors[1], index=valid_tokens, name=t)
        eigenvectors_list_1.append(ev1)
        eigenvectors_list_2.append(ev2)
        
        # Reconstruct returns to compute factor returns
        returns_window = normalized_returns * stds + means
        returns_window = returns_window.fillna(0.0)
        
        factor_returns = factor_model.compute_factor_returns(returns_window, eigenvectors, stds)
        factor_returns = factor_returns.fillna(0.0)
        
        # 3. Residual Regression and OU Parameter Estimation
        residuals = ou_solver.compute_residuals(returns_window, factor_returns)
        ou_params = ou_solver.estimate_ou_params(residuals)
        s_scores = ou_solver.compute_s_scores(residuals, ou_params)
        
        # Explicit tracking for benchmark tokens (BTC, ETH)
        tracking_tokens = ["BTC", "ETH"]
        extra_s_scores = {}
        
        for token in tracking_tokens:
            if token not in valid_tokens:
                start_time = t - pd.Timedelta(hours=window_m)
                end_time = t - pd.Timedelta(hours=1)
                price_start = start_time - pd.Timedelta(hours=1)
                
                if token in loader.prices.columns:
                    token_prices = loader.prices.loc[price_start:end_time, token]
                    token_prices = token_prices.ffill()
                    
                    if len(token_prices) > 1:
                        token_returns = token_prices.pct_change().dropna()
                        common_idx = token_returns.index.intersection(factor_returns.index)
                        if len(common_idx) > 0:
                            token_returns = token_returns.loc[common_idx]
                            token_ret_df = pd.DataFrame({token: token_returns})
                            
                            res = ou_solver.compute_residuals(token_ret_df, factor_returns.loc[common_idx])
                            params = ou_solver.estimate_ou_params(res)
                            s = ou_solver.compute_s_scores(res, params)
                            
                            extra_s_scores[token] = s[token]
        
        if extra_s_scores:
            s_scores = pd.concat([s_scores, pd.Series(extra_s_scores)])
        
        s_scores_with_time = s_scores.copy()
        s_scores_with_time.name = t
        s_scores_list.append(s_scores_with_time)
        
        # 4. Strategy Signal Generation
        active_positions = pd.Series(0, index=valid_tokens)
        for token in valid_tokens:
            active_positions[token] = current_positions.get(token, 0)
            
        new_signals = strategy.generate_signals(s_scores, active_positions)
        
        for token, sig in new_signals.items():
            current_positions[token] = sig
            
        signals_series = pd.Series(current_positions)
        signals_series.name = t
        signals_list.append(signals_series)
        
        # 5. Mark-to-Market Portfolio PnL Accounting [t, t+1]
        try:
            prices_t = loader.prices.loc[t]
            prices_next = loader.prices.loc[t + pd.Timedelta(hours=1)]
        except KeyError:
            portfolio_returns.append(0.0)
            continue
            
        pnl = 0.0
        gross_market_value = 0.0
        f1_ret = 0.0
        f2_ret = 0.0
        
        for i, token in enumerate(valid_tokens):
            if token in prices_t and token in prices_next:
                p0 = prices_t[token]
                p1 = prices_next[token]
                if p0 > 0:
                    r_next = (p1 - p0) / p0
                    w1 = eigenvectors[0][i] / stds[token]
                    w2 = eigenvectors[1][i] / stds[token]
                    f1_ret += w1 * r_next
                    f2_ret += w2 * r_next
                    
        for token, pos in current_positions.items():
            if pos != 0 and token in prices_t and token in prices_next:
                p0 = prices_t[token]
                p1 = prices_next[token]
                pnl += pos * (p1 - p0)
                gross_market_value += abs(pos) * p0
                
        if gross_market_value > 0:
            ret = pnl / gross_market_value
        else:
            ret = 0.0
            
        portfolio_returns.append(ret)
        f1_returns.append(f1_ret)
        f2_returns.append(f2_ret)

    # --- Data Export & KPI Metrics ---
    print("\nExporting backtest artifacts to disk...")
    
    ev1_df = pd.DataFrame(eigenvectors_list_1)
    ev2_df = pd.DataFrame(eigenvectors_list_2)
    ev1_df.to_csv(out_path / "task1a_1.csv")
    ev2_df.to_csv(out_path / "task1a_2.csv")
    
    signals_df = pd.DataFrame(signals_list)
    signals_df.to_csv(out_path / "trading_signal.csv")
    
    s_scores_df = pd.DataFrame(s_scores_list)
    s_scores_df.to_csv(out_path / "s_scores.csv")
    
    returns_df = pd.DataFrame({
        "Strategy": portfolio_returns,
        "F1": f1_returns,
        "F2": f2_returns
    }, index=time_range[:len(portfolio_returns)])
    returns_df.to_csv(out_path / "returns.csv")
    
    returns_series = returns_df["Strategy"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
    metrics = analyzer.calculate_metrics(returns_series)
    
    # --- Visualization Generation ---
    if generate_plots:
        print("Generating performance visualizations...")
        btc_prices = loader.prices.loc[time_range, "BTC"]
        eth_prices = loader.prices.loc[time_range, "ETH"]
        btc_ret = btc_prices.pct_change().fillna(0.0)
        eth_ret = eth_prices.pct_change().fillna(0.0)
        
        f1_series = returns_df["F1"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        f2_series = returns_df["F2"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
        f1_clipped = f1_series.clip(-10, 10)
        f2_clipped = f2_series.clip(-10, 10)
        
        try:
            analyzer.plot_cumulative_returns({
                "Strategy": returns_series,
                "BTC": btc_ret,
                "ETH": eth_ret,
                "Eigenportfolio 1": f1_clipped,
                "Eigenportfolio 2": f2_clipped
            }, "Cumulative Returns: Statistical Arbitrage vs Benchmarks", "cumulative_return.png")
        except Exception as e:
            print(f"Warning: Error plotting cumulative returns: {e}")
            
        analyzer.plot_histogram(returns_series, "Return Histogram: Hourly Strategy Returns", "hist_return.png")
        
        target_times = [
            pd.Timestamp("2021-09-26 12:00:00+00:00"),
            pd.Timestamp("2022-04-15 20:00:00+00:00")
        ]
        for target in target_times:
            if target in ev1_df.index:
                w1 = ev1_df.loc[target].dropna()
                w2 = ev2_df.loc[target].dropna()
                analyzer.plot_eigen_weights(w1, f"Eigenportfolio 1 Weights ({target.strftime('%Y-%m-%d %H:%M')})",
                                            f"weights_1_{target.date()}.png")
                analyzer.plot_eigen_weights(w2, f"Eigenportfolio 2 Weights ({target.strftime('%Y-%m-%d %H:%M')})",
                                            f"weights_2_{target.date()}.png")
                
        subset_s = s_scores_df.loc[pd.Timestamp("2021-09-26 00:00:00+00:00"):pd.Timestamp("2021-10-25 23:00:00+00:00")]
        if "BTC" in subset_s.columns:
            analyzer.plot_s_score_evolution(subset_s[["BTC"]], "S-Score Evolution: BTC (Sep-Oct 2021)",
                                            "s_score_btc.png", s_bo=s_bo, s_so=s_so, s_bc=s_bc, s_sc=s_sc)
        if "ETH" in subset_s.columns:
            analyzer.plot_s_score_evolution(subset_s[["ETH"]], "S-Score Evolution: ETH (Sep-Oct 2021)",
                                            "s_score_eth.png", s_bo=s_bo, s_so=s_so, s_bc=s_bc, s_sc=s_sc)

    # Print Formatted Results
    print("\n" + "=" * 65)
    print("      QUANTITATIVE STATISTICAL ARBITRAGE BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Simulation Window:        {start_ts.strftime('%Y-%m-%d')} to {end_ts.strftime('%Y-%m-%d')}")
    print(f"  Lookback Parameter (M):   {window_m} hours (10 days)")
    print(f"  Entry Thresholds:         s_bo = {s_bo}, s_so = {s_so}")
    print(f"  Exit Thresholds:          s_bc = {s_bc}, s_sc = {s_sc}")
    print("-" * 65)
    print(f"  Annualized Sharpe Ratio:  {metrics['Sharpe Ratio']:>12.4f}")
    print(f"  Maximum Drawdown:         {metrics['Max Drawdown'] * 100:>11.2f}%")
    print(f"  Total Strategy Return:    {metrics['Total Return'] * 100:>11.2f}%")
    print(f"  Annualized Return:        {metrics['Annualized Return'] * 100:>11.2f}%")
    print(f"  Annualized Volatility:    {metrics['Annualized Volatility'] * 100:>11.2f}%")
    print(f"  Sortino Ratio:            {metrics['Sortino Ratio']:>12.4f}")
    print(f"  Trade Win Rate:           {metrics['Win Rate'] * 100:>11.2f}%")
    print("=" * 65 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Cryptocurrency Statistical Arbitrage Engine (PCA & Ornstein-Uhlenbeck Mean Reversion)"
    )
    parser.add_argument("--prices", type=str, default="data/coin_all_prices_full.csv",
                        help="Path to cryptocurrency historical prices CSV")
    parser.add_argument("--universe", type=str, default="data/coin_universe_150K_40.csv",
                        help="Path to dynamic asset universe CSV")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Destination directory for output CSV artifacts and plots")
    parser.add_argument("--window", "-M", type=int, default=240,
                        help="Rolling lookback window size in hours (default: 240)")
    parser.add_argument("--start-date", type=str, default="2021-09-26 00:00:00+00:00",
                        help="Simulation start timestamp (YYYY-MM-DD HH:MM:SS+00:00)")
    parser.add_argument("--end-date", type=str, default="2022-09-25 23:00:00+00:00",
                        help="Simulation end timestamp (YYYY-MM-DD HH:MM:SS+00:00)")
    parser.add_argument("--s-bo", type=float, default=2.0,
                        help="Long entry S-score threshold (default: 2.0)")
    parser.add_argument("--s-so", type=float, default=2.0,
                        help="Short entry S-score threshold (default: 2.0)")
    parser.add_argument("--s-bc", type=float, default=0.25,
                        help="Long exit S-score threshold (default: 0.25)")
    parser.add_argument("--s-sc", type=float, default=0.25,
                        help="Short exit S-score threshold (default: 0.25)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plot generation")

    args = parser.parse_args()
    
    run_backtest(
        prices_path=args.prices,
        universe_path=args.universe,
        output_dir=args.output_dir,
        window_m=args.window,
        start_date=args.start_date,
        end_date=args.end_date,
        s_bo=args.s_bo,
        s_so=args.s_so,
        s_bc=args.s_bc,
        s_sc=args.s_sc,
        generate_plots=not args.no_plot
    )


if __name__ == "__main__":
    main()

