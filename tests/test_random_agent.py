"""
Test RandomAgent

Tests for RandomAgent to ensure correct behavior.
"""

import pytest
from rl_agent_ludo.agents.random_agent import RandomAgent
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from rl_agent_ludo.utils.state import State
import numpy as np


@pytest.fixture
def random_agent():
    """Create a RandomAgent instance."""
    return RandomAgent(seed=42)


def test_random_agent_properties(random_agent):
    """Test RandomAgent properties."""
    assert random_agent.is_on_policy == False
    assert random_agent.needs_replay_learning == False


def test_random_agent_act(sample_state, random_agent):
    """Test that RandomAgent selects actions from valid_moves."""
    action = random_agent.act(sample_state)
    
    assert action in sample_state.valid_moves


def test_random_agent_act_multiple_times(sample_state, random_agent):
    """Test that RandomAgent can act multiple times."""
    actions = [random_agent.act(sample_state) for _ in range(10)]
    
    # All actions should be valid
    for action in actions:
        assert action in sample_state.valid_moves


def test_random_agent_seed():
    """Test that RandomAgent respects seed."""
    agent1 = RandomAgent(seed=42)
    agent2 = RandomAgent(seed=42)
    
    state = State(
        full_vector=np.array([0.0] * 234, dtype=np.float32),
        abstract_state=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), 1, 0),
        valid_moves=[0, 1, 2, 3],
        dice_roll=1
    )
    
    # With same seed, should produce same sequence (for deterministic randomness)
    actions1 = [agent1.act(state) for _ in range(10)]
    actions2 = [agent2.act(state) for _ in range(10)]
    
    # Note: This test may be flaky if random behavior differs
    # In practice, with same seed, Python's random should be deterministic
    assert actions1 == actions2


def test_random_agent_learning_methods(random_agent, sample_state):
    """Test that RandomAgent learning methods are no-ops."""
    # These should not raise errors
    random_agent.learn_from_replay()
    random_agent.learn_from_rollout([])
    random_agent.push_to_replay_buffer(
        sample_state, 0, 0.0, sample_state, False
    )


def test_agent_registry_create_random():
    """Test that AgentRegistry can create RandomAgent."""
    config = {'type': 'random', 'seed': 42}
    agent = AgentRegistry.create_agent(config)
    
    assert isinstance(agent, RandomAgent)
    assert agent.seed == 42


def test_agent_registry_invalid_type():
    """Test that AgentRegistry raises error for invalid type."""
    config = {'type': 'invalid_agent_type'}
    
    with pytest.raises(ValueError):
        AgentRegistry.create_agent(config)
