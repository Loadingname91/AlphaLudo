"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from rl_agent_ludo.utils.state import State


@pytest.fixture
def sample_state():
    """Create a sample State object for testing."""
    full_vector = np.array([0.0] * 234, dtype=np.float32)  # Simplified state vector
    abstract_state = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 1, 0)
    valid_moves = [0, 1, 2]
    dice_roll = 1
    player_pieces = [0, 1, 2, 3]
    enemy_pieces = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    
    return State(
        full_vector=full_vector,
        abstract_state=abstract_state,
        valid_moves=valid_moves,
        dice_roll=dice_roll,
        player_pieces=player_pieces,
        enemy_pieces=enemy_pieces
    )
