"""
State representation for Ludo environment.

This file defines the State dataclass that encapsulates all state information
passed from the environment to the agent.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class State:
    """
    State object containing all information about the current game state.
    
    Attributes:
        player_pieces: List of 4 piece positions for the learning agent
        enemy_pieces: List of 3 lists, each containing 4 enemy piece positions
        current_player: ID of player whose turn it is (0-3)
        dice_roll: Current dice roll (1-6)
        valid_moves: List of valid action indices
        movable_pieces: Optional list of piece indices that can move this turn
    """
    player_pieces: List[int]
    enemy_pieces: List[List[int]]
    current_player: int
    dice_roll: int
    valid_moves: List[int]
    movable_pieces: Optional[List[int]] = None

