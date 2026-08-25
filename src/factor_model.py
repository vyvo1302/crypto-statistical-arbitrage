"""
FactorModel Module

Implements Principal Component Analysis (PCA) on normalized asset returns to extract
systematic risk factors and construct eigenportfolios according to Avellaneda & Lee (2010).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class FactorModel:
    """
    Constructs statistical risk factors from the empirical correlation matrix of normalized returns.
    """

    def __init__(self, n_components: int = 2):
        """
        Initialize the factor model.
        
        Args:
            n_components: Number of principal systematic factors to retain (default: 2).
        """
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.eigenvectors: Optional[np.ndarray] = None
        self.eigenvalues: Optional[np.ndarray] = None

    def fit(self, normalized_returns: pd.DataFrame) -> None:
        """
        Fits PCA on the normalized returns matrix Y (M x N).
        
        Args:
            normalized_returns: Standardized returns (M observations x N assets).
        """
        n_comp = min(self.n_components, normalized_returns.shape[1], normalized_returns.shape[0])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(normalized_returns)
        
        self.eigenvectors = self.pca.components_
        self.eigenvalues = self.pca.explained_variance_

    def get_eigen_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns the top eigenvectors (eigenportfolio weights) and corresponding eigenvalues.
        
        Returns:
            Tuple of (eigenvectors, eigenvalues).
        """
        return self.eigenvectors, self.eigenvalues

    def compute_factor_returns(self, returns: pd.DataFrame, eigenvectors: np.ndarray, stds: pd.Series) -> pd.DataFrame:
        """
        Computes the factor returns F_{j,k} by projecting asset returns onto normalized eigenportfolios.
        
        Formula:
            Q_{ji} = v_i^{(j)} / sigma_i
            F = R @ Q^T
            
        Args:
            returns: Original asset returns R_{i,k} (M x N).
            eigenvectors: Top eigenvectors v^{(j)} (K x N).
            stds: Asset return volatilities sigma_i (N,).
            
        Returns:
            pd.DataFrame: Factor returns matrix (M x K) with columns ['F1', 'F2', ...].
        """
        # Calculate projection matrix Q: Q_{ji} = v_i^{(j)} / sigma_i
        Q = eigenvectors / stds.values
        
        # Factor returns F = R @ Q^T
        factor_returns = returns.dot(Q.T)
        factor_returns.columns = [f"F{i+1}" for i in range(eigenvectors.shape[0])]
        
        return factor_returns

