"""
Pillar 1.5: State DTO (Data Transfer Object)

Immutable dataclass that passes all necessary state information from LudoEnv to Agent.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass(frozen=True)
class State:
    """
    Immutable state representation.
    
    Attributes:
        full_vector: NumPy array for neural network input (continuous features)
        abstract_state: Hashable tuple for tabular methods (discrete state)
        valid_moves: List of valid action indices
        dice_roll: Current dice roll value (1-6)
    """
    full_vector: np.ndarray  # For neural networks
    abstract_state: tuple    # For tabular methods (hashable)
    valid_moves: List[int]   # List of valid action indices
    dice_roll: int           # Current dice roll (1-6)

    player_pieces : List[int] 
    enemy_pieces : List[int]
    movable_pieces : Optional[List[int]] = None

    def __post_init__(self):
        """Validate state after initialization."""
        if self.dice_roll < 1 or self.dice_roll > 6:
            raise ValueError(f"dice_roll must be between 1 and 6, got {self.dice_roll}")
        
        if len(self.valid_moves) == 0:
            raise ValueError("valid_moves cannot be empty (always at least one valid action: pass)")
        
        if not isinstance(self.full_vector, np.ndarray):
            raise TypeError(f"full_vector must be numpy.ndarray, got {type(self.full_vector)}")
        
        if not isinstance(self.abstract_state, tuple):
            raise TypeError(f"abstract_state must be tuple, got {type(self.abstract_state)}")