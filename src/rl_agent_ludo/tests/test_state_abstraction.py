"""
Pillar 6: Test State Abstraction

Tests for State DTO to ensure correct vector/tuple formats.
"""

import pytest
import numpy as np
from rl_agent_ludo.utils.state import State


def test_state_creation(sample_state):
    """Test that State can be created with valid inputs."""
    assert isinstance(sample_state, State)
    assert isinstance(sample_state.full_vector, np.ndarray)
    assert isinstance(sample_state.abstract_state, tuple)
    assert isinstance(sample_state.valid_moves, list)
    assert isinstance(sample_state.dice_roll, int)


def test_state_immutability(sample_state):
    """Test that State is immutable (frozen dataclass)."""
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        sample_state.dice_roll = 5


def test_state_validation_dice_roll():
    """Test that State validates dice_roll range."""
    full_vector = np.array([0.0] * 234, dtype=np.float32)
    abstract_state = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 1, 0)
    valid_moves = [0]
    
    # Valid dice roll (1-6)
    state = State(
        full_vector=full_vector,
        abstract_state=abstract_state,
        valid_moves=valid_moves,
        dice_roll=3,
        player_pieces=[0, 1, 2, 3],
        enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    )
    assert state.dice_roll == 3
    
    # Invalid dice roll (< 1)
    with pytest.raises(ValueError):
        State(
            full_vector=full_vector,
            abstract_state=abstract_state,
            valid_moves=valid_moves,
            dice_roll=0,
            player_pieces=[0, 1, 2, 3],
            enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
    
    # Invalid dice roll (> 6)
    with pytest.raises(ValueError):
        State(
            full_vector=full_vector,
            abstract_state=abstract_state,
            valid_moves=valid_moves,
            dice_roll=7,
            player_pieces=[0, 1, 2, 3],
            enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )


def test_state_validation_valid_moves():
    """Test that State validates valid_moves is not empty."""
    full_vector = np.array([0.0] * 234, dtype=np.float32)
    abstract_state = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 1, 0)
    
    # Valid: non-empty valid_moves
    state = State(
        full_vector=full_vector,
        abstract_state=abstract_state,
        valid_moves=[0],
        dice_roll=1,
        player_pieces=[0, 1, 2, 3],
        enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    )
    assert len(state.valid_moves) > 0
    
    # Invalid: empty valid_moves
    with pytest.raises(ValueError):
        State(
            full_vector=full_vector,
            abstract_state=abstract_state,
            valid_moves=[],
            dice_roll=1,
            player_pieces=[0, 1, 2, 3],
            enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )


def test_state_types():
    """Test that State enforces correct types."""
    full_vector = np.array([0.0] * 234, dtype=np.float32)
    abstract_state = ((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 1, 0)
    valid_moves = [0, 1, 2]
    
    # Invalid full_vector type
    with pytest.raises(TypeError):
        State(
            full_vector=[0.0] * 234,  # Should be np.ndarray
            abstract_state=abstract_state,
            valid_moves=valid_moves,
            dice_roll=1,
            player_pieces=[0, 1, 2, 3],
            enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
    
    # Invalid abstract_state type
    with pytest.raises(TypeError):
        State(
            full_vector=full_vector,
            abstract_state=[0, 0, 0, 0],  # Should be tuple
            valid_moves=valid_moves,
            dice_roll=1,
            player_pieces=[0, 1, 2, 3],
            enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
