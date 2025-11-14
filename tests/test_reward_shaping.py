"""
Pillar 6: Test Reward Shaping

Tests for RewardShaper to ensure correct rewards for mock events.
"""

import pytest
from rl_agent_ludo.environment.reward_shaper import RewardShaper, SparseReward, create_reward_shaper


def test_sparse_reward_win():
    """Test that SparseReward returns +100 for win."""
    reward_shaper = SparseReward()
    game_events = {'player_won': True}
    
    reward, ila_components = reward_shaper.get_reward(game_events)
    
    assert reward == 100.0
    assert ila_components == {}


def test_sparse_reward_loss():
    """Test that SparseReward returns -100 for loss."""
    reward_shaper = SparseReward()
    game_events = {'opponent_won': True}
    
    reward, ila_components = reward_shaper.get_reward(game_events)
    
    assert reward == -100.0
    assert ila_components == {}


def test_sparse_reward_ongoing():
    """Test that SparseReward returns 0 for ongoing game."""
    reward_shaper = SparseReward()
    game_events = {}  # No win/loss events
    
    reward, ila_components = reward_shaper.get_reward(game_events)
    
    assert reward == 0.0
    assert ila_components == {}


def test_sparse_reward_with_other_events():
    """Test that SparseReward ignores other events."""
    reward_shaper = SparseReward()
    game_events = {
        'piece_moved': True,
        'piece_captured': True,
        'piece_entered_home': True,
    }
    
    reward, ila_components = reward_shaper.get_reward(game_events)
    
    # Should return 0.0 since no win/loss
    assert reward == 0.0
    assert ila_components == {}


def test_reward_shaper_factory():
    """Test that RewardShaper factory creates correct instances."""
    # Valid schema
    reward_shaper = create_reward_shaper('sparse')
    assert isinstance(reward_shaper, SparseReward)
    assert reward_shaper.schema == 'sparse'
    
    # Invalid schema
    with pytest.raises(ValueError):
        create_reward_shaper('invalid_schema')
    
    # Case insensitive (should work)
    reward_shaper2 = create_reward_shaper('SPARSE')
    assert isinstance(reward_shaper2, SparseReward)
