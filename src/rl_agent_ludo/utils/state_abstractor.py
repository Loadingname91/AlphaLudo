"""
Pillar 2: State Abstraction (The Translator)

Converts raw Ludo board states into the 'Context-Aware Potential-Based' tuple.
Implements the logic for Weighted Equity, Potentials, and Move Simulation.
"""

import numpy as np
from typing import List, Tuple, Dict
from ..utils.state import State
from .board_analyser import LudoBoardAnalyser
from rl_agent_ludo.utils import board_analyser

# --- Ludo Board Constants (Matching ludopy.player.py) ---
# Ludopy uses these exact constants from player.py
HOME_INDEX = board_analyser.HOME_INDEX  # Home position (ludopy uses 0, not -1)
START_INDEX = board_analyser.START_INDEX
GOAL_INDEX = board_analyser.GOAL_INDEX
HOME_CORRIDOR_INDEXS = board_analyser.HOME_STRETCH_INDEXS
HOME_CORRIDOR_START = 52
SAFE_GLOBES = {1, 9, 22, 35, 48}  # Start + Globes (ludopy GLOB_INDEXS + START_INDEX)
STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]  # Stars that trigger jumps

# --- Context Definitions ---
CONTEXT_TRAILING = 0  # Panic Mode
CONTEXT_NEUTRAL = 1   # Balanced Race
CONTEXT_LEADING = 2   # Lockdown Mode

# --- Potential Definitions ---
POT_NULL = 0
POT_NEUTRAL = 1
POT_RISK = 2
POT_BOOST = 3
POT_SAFETY = 4
POT_KILL = 5
POT_GOAL = 6

class LudoStateAbstractor:
    """
    Abstracts raw board state into a tactical tuple: (P1, P2, P3, P4, Context).
    """

    def __init__(self, player_id: int = 0):
        self.player_id = player_id
    
    def _calculate_context(self, state: State) -> int:
        """
        Determines if we are Trailing, Neutral, or Leading based on Weighted Equity.
        """
        my_score = self._get_weighted_score(state.player_pieces)
        
        opponent_scores = []
        for enemy_list in state.enemy_pieces:
            opponent_scores.append(self._get_weighted_score(enemy_list))
        
        max_opponent_score = max(opponent_scores) if opponent_scores else 0
        gap = my_score - max_opponent_score

        # Thresholds from Theory Plan
        if gap < -20:
            return CONTEXT_TRAILING
        elif gap > 20:
            return CONTEXT_LEADING
        else:
            return CONTEXT_NEUTRAL

    def _get_weighted_score(self, pieces: List[int]) -> int:
        """
        Score = (Goal*100) + (Corridor*50) + (Safe*10) + (Distance).
        """
        score = 0
        for pos in pieces:
            if pos == GOAL_INDEX:  # Finished
                score += 100
            elif pos == HOME_INDEX:  # Home (ludopy uses 0, not -1)
                score += 0
            elif pos in HOME_CORRIDOR_INDEXS:  # Corridor (52-56)
                score += 50
                score += pos  # Distance bonus
            elif pos in SAFE_GLOBES:  # Globe or Start (safe positions)
                score += 10
                score += pos  # Distance
            else:  # Raw Distance on main board
                score += pos
        return score

    def _analyze_move_potential(self, current_pos: int, dice: int, 
                              all_enemy_pieces: List[List[int]]) -> int:
        """
        Simulates a move and classifies the outcome into a Potential Category.
        Uses board_analyser for accurate move simulation and collision detection.
        """
        # Use board_analyser to simulate the move correctly (handles star jumps, goal bounce, etc.)
        next_pos = LudoBoardAnalyser.simulate_move(current_pos, dice)
        
        # Track if we jumped via star (for Boost detection)
        original_pos = current_pos + dice
        is_boost = (original_pos in STAR_INDEXS and next_pos != original_pos)

        # 1. Check Goal (highest priority)
        if next_pos == GOAL_INDEX:
            return POT_GOAL

        # 2. Check Kill (capture) - uses ludopy coordinate conversion
        if LudoBoardAnalyser.can_capture(current_pos, dice, all_enemy_pieces):
            return POT_KILL

        # 3. Check Safety (Home Corridor or Globe/Start)
        if next_pos in HOME_CORRIDOR_INDEXS or next_pos in SAFE_GLOBES:
            return POT_SAFETY

        # 4. Check Boost (Speed via Star Jump)
        # If we jumped via star, and it wasn't a kill or safe (already checked), it's a boost.
        if is_boost:
            return POT_BOOST
        
        # 5. Check Risk (Threatened by enemy 1-6 steps behind)
        if LudoBoardAnalyser.is_threatened(next_pos, all_enemy_pieces):
            return POT_RISK

        # 6. Default: Neutral move
        return POT_NEUTRAL


    def get_abstract_state(self, state: State) -> Tuple[int, int, int, int, int]:
        """
        Main entry point: Converts State object -> 5-Int Tuple.
        """
        # 1. Calculate Global Context (Winning/Losing)
        context = self._calculate_context(state)

        # 2. Calculate Potentials for each piece (P1, P2, P3, P4)
        potentials = []
        # ludopy gives us 4 pieces. We process them in order.
        # Action index i corresponds to piece i (action 0 -> piece 0, etc.)
        
        player_pieces = state.player_pieces
        dice = state.dice_roll
        
        # Enemy pieces are provided in their own coordinate systems (relative to each enemy player)
        # Coordinate conversion is handled by ludopy's get_enemy_at_pos() via board_analyser
        
        for i, pos in enumerate(player_pieces):
            # Check if piece is movable using ludopy rules
            # In ludopy: HOME_INDEX (0) = home, GOAL_INDEX (57) = goal
            # Piece at home can only move with dice == 6
            # Piece at goal cannot move
            
            if pos == HOME_INDEX and dice != 6:
                # Stuck at home (need 6 to exit)
                # ROBUSTNESS: If ludopy says it's movable, trust it (override abstractor logic)
                if state.movable_pieces is not None and i in state.movable_pieces:
                     # Treat as if it's on the board (force simulation)
                     # We bypass simulate_move's home check by passing a dummy start pos if needed, 
                     # or rely on simulate_move to handle it if we modified it.
                     # Actually, if we pass 0, simulate_move will return 0.
                     # We want to simulate a move from 0 to 0+dice.
                     # So we calculate next_pos manually for this edge case.
                     next_pos = pos + dice # 0 + dice
                     # Analyze manually
                     # 1. Check Goal
                     if next_pos == 57: pot_val = POT_GOAL
                     # 2. Check Safety (1 is start)
                     elif next_pos in SAFE_GLOBES: pot_val = POT_SAFETY
                     else: pot_val = POT_NEUTRAL
                     
                     potentials.append(pot_val)
                     continue

                potentials.append(POT_NULL)
                continue
            
            if pos == GOAL_INDEX:
                # Already finished (cannot move)
                # ROBUSTNESS: If ludopy says movable
                if state.movable_pieces is not None and i in state.movable_pieces:
                    potentials.append(POT_GOAL) # Assume it stays at goal or re-enters
                    continue

                potentials.append(POT_NULL)
                continue
            
            # Check if piece is in movable_pieces list (from ludopy)
            if state.movable_pieces is not None and i not in state.movable_pieces:
                potentials.append(POT_NULL)
                continue
            
            # If movable, analyze what happens
            pot_val = self._analyze_move_potential(pos, dice, state.enemy_pieces)
            potentials.append(pot_val)

        # Ensure we have exactly 4 potentials
        while len(potentials) < 4:
            potentials.append(POT_NULL)

        return tuple(potentials + [context])

