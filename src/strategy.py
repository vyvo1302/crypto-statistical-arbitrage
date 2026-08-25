"""
Strategy Module

Generates quantitative mean-reversion trading signals (Open Long, Open Short, Close)
based on parametric S-score thresholds.
"""

from typing import Dict, Union
import numpy as np
import pandas as pd


class Strategy:
    """
    Parametric statistical arbitrage decision engine based on Ornstein-Uhlenbeck S-scores.
    """

    def __init__(self, s_bo: float = 2.0, s_so: float = 2.0, s_bc: float = 0.25, s_sc: float = 0.25):
        """
        Initializes trading thresholds.
        
        Args:
            s_bo: Buy-open (Long entry) threshold multiplier. Open long when s < -s_bo.
            s_so: Sell-open (Short entry) threshold multiplier. Open short when s > s_so.
            s_bc: Buy-close (Long exit) threshold multiplier. Close long when s > -s_bc.
            s_sc: Sell-close (Short exit) threshold multiplier. Close short when s < s_sc.
        """
        self.s_bo = s_bo
        self.s_so = s_so
        self.s_bc = s_bc
        self.s_sc = s_sc

    def generate_signals(self, s_scores: pd.Series, current_positions: Union[pd.Series, Dict[str, int]]) -> pd.Series:
        """
        Generates updated target portfolio positions (-1, 0, +1) based on current S-scores.
        
        Signal Rules:
            - Flat (pos == 0):
                - s < -s_bo -> Open Long (+1)
                - s > +s_so -> Open Short (-1)
            - Long (pos == +1):
                - s > -s_bc -> Close Long (0)
            - Short (pos == -1):
                - s < +s_sc -> Close Short (0)
                
        Args:
            s_scores: Standardized S-scores for active assets.
            current_positions: Currently held unit positions.
            
        Returns:
            pd.Series: Updated target positions (-1, 0, +1).
        """
        if isinstance(current_positions, dict):
            signals = pd.Series(current_positions, dtype=int).copy()
        else:
            signals = current_positions.copy()

        for asset, s in s_scores.items():
            if pd.isna(s):
                continue

            curr_pos = current_positions.get(asset, 0) if isinstance(current_positions, dict) else current_positions.get(asset, 0)

            if curr_pos == 0:
                if s < -self.s_bo:
                    signals[asset] = 1
                elif s > self.s_so:
                    signals[asset] = -1
            elif curr_pos == 1:
                if s > -self.s_bc:
                    signals[asset] = 0
            elif curr_pos == -1:
                if s < self.s_sc:
                    signals[asset] = 0

        return signals

