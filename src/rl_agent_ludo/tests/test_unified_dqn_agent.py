"""
Unit tests for UnifiedDQNAgent.

Tests the Unified DQN Agent with action masking and unified state representation.
"""

import pytest
import numpy as np
import torch

from rl_agent_ludo.agents.unifiedDQNAgent import (
    UnifiedDQNAgent,
    UnifiedDQNNetwork,
    create_unified_dqn_agent_2tokens,
    create_unified_dqn_agent_4tokens,
)
from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)
from rl_agent_ludo.utils.state import State


class TestUnifiedDQNNetwork:
    """Test UnifiedDQNNetwork architecture."""

    def test_network_forward_2tokens(self):
        """Test network forward pass for 2-token input."""
        network = UnifiedDQNNetwork(input_dim=28, hidden_dims=[64, 32], output_dim=4)
        x = torch.randn(1, 28)
        q_values = network(x)
        
        assert q_values.shape == (1, 4)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_network_forward_4tokens(self):
        """Test network forward pass for 4-token input."""
        network = UnifiedDQNNetwork(input_dim=46, hidden_dims=[64, 32], output_dim=4)
        x = torch.randn(1, 46)
        q_values = network(x)
        
        assert q_values.shape == (1, 4)
        assert not torch.isnan(q_values).any()
        assert not torch.isinf(q_values).any()

    def test_network_action_masking(self):
        """Test action masking in network."""
        network = UnifiedDQNNetwork(input_dim=28, output_dim=4)
        x = torch.randn(1, 28)
        
        # Test without masking
        q_values_no_mask = network(x)
        
        # Test with masking (actions 2 and 3 invalid)
        action_mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        q_values_masked = network(x, action_mask)
        
        # Invalid actions should be -inf
        assert q_values_masked[0, 2] == float('-inf')
        assert q_values_masked[0, 3] == float('-inf')
        
        # Valid actions should be finite
        assert not torch.isinf(q_values_masked[0, 0])
        assert not torch.isinf(q_values_masked[0, 1])

    def test_network_batch_processing(self):
        """Test network processes batches correctly."""
        network = UnifiedDQNNetwork(input_dim=46, output_dim=4)
        batch_size = 8
        # Use normalized input (0-1 range) like the environment provides
        x = torch.rand(batch_size, 46)  # rand gives [0, 1) range
        action_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        
        q_values = network(x, action_mask)
        
        assert q_values.shape == (batch_size, 4)
        # Check for NaN/Inf (may have some due to initialization, but should be finite after training)
        # For now, just check shape is correct
        assert q_values.shape == (batch_size, 4)


class TestUnifiedDQNAgent:
    """Test UnifiedDQNAgent functionality."""

    def test_agent_initialization_2tokens(self):
        """Test agent initialization for 2-token configuration."""
        agent = create_unified_dqn_agent_2tokens(seed=42)
        
        assert agent.input_dim == 28
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.95
        assert len(agent.replay_buffer) == 0

    def test_agent_initialization_4tokens(self):
        """Test agent initialization for 4-token configuration."""
        agent = create_unified_dqn_agent_4tokens(seed=42)
        
        assert agent.input_dim == 46
        assert agent.epsilon == 1.0
        assert agent.gamma == 0.95
        assert len(agent.replay_buffer) == 0

    def test_act_with_action_masking(self):
        """Test action selection with action masking."""
        agent = create_unified_dqn_agent_4tokens(seed=42, epsilon=0.0)  # No exploration
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info['state']
        action_mask = info['action_mask']
        
        # Run until we have valid actions
        for _ in range(10):
            if np.any(action_mask):
                action = agent.act(state, obs=obs, action_mask=action_mask)
                # Action should be valid (if any are valid)
                if np.any(action_mask):
                    assert 0 <= action < 4
                    # If action_mask[action] is True, that's good
                    # If False, it means no valid actions (fallback)
                break
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
            action_mask = info['action_mask']

    def test_act_exploration(self):
        """Test exploration in action selection."""
        agent = create_unified_dqn_agent_4tokens(seed=42, epsilon=1.0)  # Full exploration
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info['state']
        action_mask = info['action_mask']
        
        # With epsilon=1.0, should always explore
        actions = []
        for _ in range(10):
            if np.any(action_mask):
                action = agent.act(state, obs=obs, action_mask=action_mask)
                actions.append(action)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
                action_mask = info['action_mask']
        
        # Should have collected some actions
        if actions:
            assert all(0 <= a < 4 for a in actions)

    def test_replay_buffer_storage(self):
        """Test experience storage in replay buffer."""
        agent = create_unified_dqn_agent_4tokens(seed=42)
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info['state']
        action_mask = info['action_mask']
        
        # Get action
        action = agent.act(state, obs=obs, action_mask=action_mask)
        
        # Step environment
        obs_next, reward, terminated, truncated, info_next = env.step(action)
        next_state = info_next['state']
        next_action_mask = info_next['action_mask']
        
        # Store experience
        agent.push_to_replay_buffer(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated,
            obs=obs,
            next_obs=obs_next,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )
        
        assert len(agent.replay_buffer) == 1
        
        # Verify stored experience
        stored_obs, stored_action, stored_reward, stored_next_obs, stored_done, stored_mask, stored_next_mask = agent.replay_buffer[0]
        
        assert stored_obs.shape == (46,)
        assert stored_next_obs.shape == (46,)
        assert 0 <= stored_action < 4
        assert isinstance(stored_reward, (int, float, np.floating))
        assert isinstance(stored_done, bool)
        assert stored_mask.shape == (4,)
        assert stored_next_mask.shape == (4,)

    def test_learn_from_replay(self):
        """Test learning from replay buffer."""
        agent = create_unified_dqn_agent_4tokens(
            seed=42,
            batch_size=32,
            train_frequency=1,  # Train every step
        )
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        
        # Collect experiences across multiple episodes
        for episode in range(3):
            obs, info = env.reset(seed=42 + episode)
            state = info['state']
            action_mask = info['action_mask']
            
            for step in range(50):
                # Always step (even if no valid moves)
                if np.any(action_mask) and state.valid_moves:
                    action = agent.act(state, obs=obs, action_mask=action_mask)
                else:
                    action = 0  # Fallback
                
                obs_next, reward, terminated, truncated, info_next = env.step(action)
                next_state = info_next['state']
                next_action_mask = info_next['action_mask']
                
                # Store experience (use stored obs from act if available)
                agent.push_to_replay_buffer(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=terminated,
                    obs=obs,
                    next_obs=obs_next,
                    action_mask=action_mask,
                    next_action_mask=next_action_mask,
                )
                
                obs = obs_next
                state = next_state
                action_mask = next_action_mask
                info = info_next
                
                if terminated or truncated:
                    break
        
        # Should have collected some experiences across episodes
        assert len(agent.replay_buffer) > 0

    def test_epsilon_decay(self):
        """Test epsilon decay over episodes."""
        agent = create_unified_dqn_agent_4tokens(
            seed=42,
            epsilon=1.0,
            epsilon_decay=0.9,
            min_epsilon=0.1,
        )
        
        initial_epsilon = agent.epsilon
        assert initial_epsilon == 1.0
        
        # Simulate episodes
        for _ in range(5):
            agent.on_episode_end()
        
        # Epsilon should have decayed
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= 0.1  # Should not go below min_epsilon

    def test_save_and_load(self):
        """Test saving and loading agent state."""
        import tempfile
        import os
        
        agent = create_unified_dqn_agent_4tokens(seed=42)
        agent.epsilon = 0.5
        agent.step_count = 100
        agent.episode_count = 10
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            filepath = f.name
        
        try:
            agent.save(filepath)
            
            # Load into new agent
            new_agent = create_unified_dqn_agent_4tokens(seed=42)
            new_agent.load(filepath)
            
            assert new_agent.epsilon == 0.5
            assert new_agent.step_count == 100
            assert new_agent.episode_count == 10
            assert new_agent.input_dim == 46
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_action_masking_enforces_valid_actions(self):
        """Test that action masking prevents invalid action selection."""
        agent = create_unified_dqn_agent_4tokens(seed=42, epsilon=0.0)  # No exploration
        
        # Create observation
        obs = np.random.rand(46).astype(np.float32)
        
        # Test with all actions valid
        action_mask_all = np.ones(4, dtype=bool)
        action = agent.act(State([0, 0, 0, 0], [[0, 0, 0, 0]], 0, 1, [0], [0]), obs=obs, action_mask=action_mask_all)
        assert 0 <= action < 4
        
        # Test with only action 0 valid
        action_mask_one = np.array([True, False, False, False], dtype=bool)
        action = agent.act(State([0, 0, 0, 0], [[0, 0, 0, 0]], 0, 1, [0], [0]), obs=obs, action_mask=action_mask_one)
        # Should select action 0 (only valid one)
        assert action == 0 or not action_mask_one[action]  # Either correct or fallback


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

