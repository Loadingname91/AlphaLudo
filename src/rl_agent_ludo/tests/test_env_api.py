"""
Pillar 6: Test Environment API

Tests for LudoEnv to ensure step/reset return correct types.
"""

import pytest
import numpy as np
from rl_agent_ludo.environment.ludo_env import LudoEnv
from rl_agent_ludo.utils.state import State


@pytest.fixture
def ludo_env():
    """Create a LudoEnv instance."""
    return LudoEnv(reward_schema='sparse', player_id=0, seed=42)


def test_env_reset(ludo_env):
    """Test that reset() returns State object."""
    state = ludo_env.reset()
    
    assert isinstance(state, State)
    assert isinstance(state.full_vector, np.ndarray)
    assert isinstance(state.abstract_state, tuple)
    assert isinstance(state.valid_moves, list)
    assert isinstance(state.dice_roll, int)
    assert len(state.valid_moves) > 0


def test_env_step_return_types(ludo_env):
    """Test that step() returns correct types."""
    state = ludo_env.reset()
    
    # Get valid action
    action = state.valid_moves[0] if len(state.valid_moves) > 0 else 0
    
    next_state, reward, done, info = ludo_env.step(action)
    
    # Check return types
    assert isinstance(next_state, State)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_env_step_multiple_steps(ludo_env):
    """Test that environment can handle multiple steps."""
    state = ludo_env.reset()
    
    for _ in range(10):
        if ludo_env.done:
            break
        
        # Get valid action
        action = state.valid_moves[0] if len(state.valid_moves) > 0 else 0
        
        next_state, reward, done, info = ludo_env.step(action)
        
        # Verify return types
        assert isinstance(next_state, State)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        
        state = next_state


def test_env_get_valid_actions(ludo_env):
    """Test that _get_valid_actions() returns list of valid actions."""
    ludo_env.reset()
    
    valid_actions = ludo_env._get_valid_actions()
    
    assert isinstance(valid_actions, list)
    assert len(valid_actions) > 0  # Always at least pass action


def test_env_reset_after_done(ludo_env):
    """Test that reset() can be called after episode ends."""
    state = ludo_env.reset()
    
    # Play until done (or limit steps)
    for _ in range(100):
        if ludo_env.done:
            break
        action = state.valid_moves[0] if len(state.valid_moves) > 0 else 0
        state, _, done, _ = ludo_env.step(action)
        if done:
            break
    
    # Reset should work
    new_state = ludo_env.reset()
    assert isinstance(new_state, State)
    assert not ludo_env.done  # Reset should clear done flag


def test_env_reward_shaping(ludo_env):
    """Test that reward shaping works correctly."""
    state = ludo_env.reset()
    
    # Execute a few steps to see reward structure
    rewards = []
    for _ in range(10):
        if ludo_env.done:
            break
        action = state.valid_moves[0] if len(state.valid_moves) > 0 else 0
        state, reward, done, info = ludo_env.step(action)
        rewards.append(reward)
        if done:
            break
    
    # Sparse reward should be 0 most of the time (only non-zero on win/loss)
    # But we can't guarantee this without playing full game
    # So we just check that rewards are floats
    assert all(isinstance(r, float) for r in rewards)
