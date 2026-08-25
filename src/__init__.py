"""
Cryptocurrency Statistical Arbitrage Package

PCA factor model and Ornstein-Uhlenbeck mean-reversion trading strategy.
"""

from .data_loader import DataLoader
from .factor_model import FactorModel
from .ou_solver import OUSolver
from .strategy import Strategy
from .analyzer import Analyzer

__all__ = [
    "DataLoader",
    "FactorModel",
    "OUSolver",
    "Strategy",
    "Analyzer",
]
