"""
OUSolver Module

Implements residual regression on systematic risk factors, Ornstein-Uhlenbeck (OU)
parameter estimation via discrete AR(1) Maximum Likelihood regression, and standardized
S-score generation.
"""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class OUSolver:
    """
    Solves continuous Ornstein-Uhlenbeck process parameters from discrete residual time series:
        dX_t = kappa * (m - X_t) dt + sigma * dW_t
    """

    def __init__(self, dt: float = 1.0 / 8760.0):
        """
        Initialize the OU Solver.
        
        Args:
            dt: Time step size in years (default: 1/8760 for hourly frequency).
        """
        self.dt = dt

    def compute_residuals(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Regresses asset returns against systematic factor returns to isolate idiosyncratic residuals.
        
        Model:
            R_{i,k} = beta_{i,0} + sum_{j=1}^K beta_{i,j} * F_{j,k} + epsilon_{i,k}
            
        Args:
            asset_returns: Asset returns DataFrame (M x N).
            factor_returns: Systematic factor returns DataFrame (M x K).
            
        Returns:
            pd.DataFrame: Matrix of idiosyncratic residuals epsilon_{i,k} (M x N).
        """
        residuals = pd.DataFrame(index=asset_returns.index, columns=asset_returns.columns, dtype=float)
        
        common_index = asset_returns.index.intersection(factor_returns.index)
        if len(common_index) == 0:
            return residuals
            
        y_df = asset_returns.loc[common_index]
        X = factor_returns.loc[common_index].values
        
        # Fit multi-factor OLS regression for each asset
        for col in y_df.columns:
            y = y_df[col].values
            reg = LinearRegression().fit(X, y)
            residuals.loc[common_index, col] = y - reg.predict(X)
            
        return residuals

    def estimate_ou_params(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates continuous OU parameters (kappa, m, sigma, sigma_eq) via discrete AR(1) regression:
            X_{l+1} = a + b * X_l + zeta_{l+1}
            
        Mapping formulas:
            kappa = -ln(b) / dt              (Speed of mean reversion)
            m = a / (1 - b)                  (Long-run equilibrium mean)
            sigma_eq = sqrt(Var(zeta) / (1 - b^2)) (Equilibrium standard deviation)
            sigma = sigma_eq * sqrt(2 * kappa)     (Diffusion coefficient)
            
        Args:
            residuals: Residual return matrix (M x N).
            
        Returns:
            pd.DataFrame: Estimated parameters for each asset indexed by asset symbol.
        """
        params = pd.DataFrame(index=residuals.columns, columns=["kappa", "m", "sigma", "sigma_eq"], dtype=float)
        
        # Auxiliary process: cumulative sum of idiosyncratic residuals
        X = residuals.cumsum()
        
        for col in residuals.columns:
            X_series = X[col].dropna().values
            if len(X_series) < 3:
                continue
                
            X_lag = X_series[:-1].reshape(-1, 1)
            X_curr = X_series[1:]
            
            reg = LinearRegression().fit(X_lag, X_curr)
            a = reg.intercept_
            b = reg.coef_[0]
            zeta = X_curr - reg.predict(X_lag)
            
            # Verify stability condition for mean-reverting stationary process: 0 < b < 1
            if b <= 0.0 or b >= 1.0:
                kappa = np.nan
                m = np.nan
                sigma_eq = np.nan
                sigma = np.nan
            else:
                kappa = -np.log(b) / self.dt
                m = a / (1.0 - b)
                var_zeta = float(np.var(zeta))
                sigma_eq = np.sqrt(var_zeta / (1.0 - b**2))
                sigma = sigma_eq * np.sqrt(2.0 * kappa)
            
            params.loc[col] = [kappa, m, sigma, sigma_eq]
            
        return params

    def compute_s_scores(self, residuals: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
        """
        Computes standardized dimensionless S-scores for each asset at current time t:
            s_t = (X_t - m) / sigma_eq
            
        Args:
            residuals: Historical window residuals (M x N).
            params: Estimated OU parameter DataFrame.
            
        Returns:
            pd.Series: S-score per asset.
        """
        X_t = residuals.cumsum().iloc[-1]
        s_scores = (X_t - params["m"]) / params["sigma_eq"]
        return s_scores

