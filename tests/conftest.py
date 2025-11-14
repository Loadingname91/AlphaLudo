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
    
    return State(
        full_vector=full_vector,
        abstract_state=abstract_state,
        valid_moves=valid_moves,
        dice_roll=dice_roll
    )
