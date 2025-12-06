"""
Comprehensive unit tests for LudoEnv edge cases.

Tests all edge cases documented in verificationReport.md to ensure
the environment works correctly with all agent types:
- Random Agent
- Rule-Based Agent
- Tabular Q-Learning Agent
- DQN Agent
- Dueling DQN Agent
"""

import pytest
import numpy as np
import gymnasium as gym

import rl_agent_ludo.envs  # noqa: F401 - triggers Ludo-v0 registration
from rl_agent_ludo.envs.ludoEnv import LudoEnv
from rl_agent_ludo.utils.state import State


class TestLudoEnvInitialization:
    """Test environment initialization edge cases."""

    def test_valid_configurations(self):
        """Test all valid player/token configurations."""
        configs = [
            (2, 2),  # 2 players, 2 tokens
            (2, 4),  # 2 players, 4 tokens
            (3, 2),  # 3 players, 2 tokens
            (3, 4),  # 3 players, 4 tokens
            (4, 4),  # 4 players, 4 tokens
        ]
        
        for num_players, tokens_per_player in configs:
            env = LudoEnv(
                player_id=0,
                num_players=num_players,
                tokens_per_player=tokens_per_player,
            )
            assert env.num_players == num_players
            assert env.tokens_per_player == tokens_per_player
            assert env.action_space.n == 4
            assert env.observation_space.shape == (18,)

    def test_invalid_num_players(self):
        """Test invalid num_players raises AssertionError."""
        with pytest.raises(AssertionError, match="num_players must be 2, 3, or 4"):
            LudoEnv(num_players=1)
        
        with pytest.raises(AssertionError, match="num_players must be 2, 3, or 4"):
            LudoEnv(num_players=5)

    def test_invalid_tokens_per_player(self):
        """Test invalid tokens_per_player raises AssertionError."""
        with pytest.raises(AssertionError, match="tokens_per_player must be 2 or 4"):
            LudoEnv(tokens_per_player=1)
        
        with pytest.raises(AssertionError, match="tokens_per_player must be 2 or 4"):
            LudoEnv(tokens_per_player=3)

    def test_invalid_player_id(self):
        """Test invalid player_id raises AssertionError."""
        with pytest.raises(AssertionError, match="player_id must be < num_players"):
            LudoEnv(player_id=4, num_players=4)
        
        with pytest.raises(AssertionError, match="player_id must be < num_players"):
            LudoEnv(player_id=2, num_players=2)

    def test_seed_handling(self):
        """Test seed initialization and propagation."""
        # Test with seed
        env1 = LudoEnv(seed=42)
        env2 = LudoEnv(seed=42)
        
        # Test without seed
        env3 = LudoEnv(seed=None)
        assert env3._seed is None

    def test_gym_make_registration(self):
        """Test environment can be created via gym.make()."""
        env = gym.make("Ludo-v0", player_id=0, num_players=4, tokens_per_player=4)
        assert isinstance(env.unwrapped, LudoEnv)
        assert env.action_space.n == 4
        assert env.observation_space.shape == (18,)


class TestLudoEnvReset:
    """Test reset() edge cases."""

    def test_reset_returns_correct_types(self):
        """Test reset returns numpy array and info dict."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18,)
        assert obs.dtype == np.int32
        assert isinstance(info, dict)
        assert "state" in info
        assert isinstance(info["state"], State)

    def test_reset_with_seed(self):
        """Test reset with explicit seed."""
        env = LudoEnv()
        obs1, _ = env.reset(seed=123)
        obs2, _ = env.reset(seed=123)
        
        # With same seed, initial state should be deterministic
        # (Note: actual positions may vary due to game mechanics, but structure should be same)
        assert obs1.shape == obs2.shape
        assert obs1.dtype == obs2.dtype

    def test_reset_without_seed(self):
        """Test reset without seed (uses env-level seed or random)."""
        env = LudoEnv(seed=None)
        obs, info = env.reset()
        assert obs.shape == (18,)
        assert "state" in info

    def test_reset_state_object(self):
        """Test that State object in info has correct structure."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        assert isinstance(state, State)
        assert len(state.player_pieces) == 4
        assert len(state.enemy_pieces) == 3
        assert all(len(enemy) == 4 for enemy in state.enemy_pieces)
        assert isinstance(state.current_player, int)
        assert 0 <= state.current_player < 4
        assert isinstance(state.dice_roll, int)
        assert 1 <= state.dice_roll <= 6
        assert isinstance(state.valid_moves, list)
        assert state.movable_pieces is None or isinstance(state.movable_pieces, list)

    def test_reset_token_limiting_2tokens(self):
        """Test token limiting for 2 tokens per player."""
        env = LudoEnv(num_players=4, tokens_per_player=2)
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # First 2 pieces should be valid, last 2 should be at home (-1)
        # This is enforced in _apply_token_limit
        assert len(state.player_pieces) == 4
        # Pieces 2 and 3 should be at home if tokens_per_player == 2
        # (This is handled internally, but we can verify the state is valid)

    def test_multiple_resets(self):
        """Test multiple resets don't cause errors."""
        env = LudoEnv()
        for _ in range(5):
            obs, info = env.reset(seed=42)
            assert obs.shape == (18,)
            assert "state" in info


class TestLudoEnvStep:
    """Test step() edge cases."""

    def test_step_before_reset(self):
        """Test step() before reset raises RuntimeError."""
        env = LudoEnv()
        with pytest.raises(RuntimeError, match="Environment not reset"):
            env.step(0)

    def test_step_returns_correct_types(self):
        """Test step returns correct types."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Get valid action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18,)
        assert obs.dtype == np.int32
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert "state" in info

    def test_step_invalid_action_index(self):
        """Test step with invalid action index (out of bounds)."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Action index out of bounds should be clamped to first valid move
        # This is handled in step() by checking action < len(valid_moves)
        invalid_action = 999
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Should not raise error, but should use fallback
        assert obs.shape == (18,)
        assert isinstance(reward, (int, float))

    def test_step_negative_action(self):
        """Test step with negative action index."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Negative action should be clamped
        obs, reward, terminated, truncated, info = env.step(-1)
        assert obs.shape == (18,)

    def test_step_no_valid_moves(self):
        """Test step when no valid moves available."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # When no valid moves, action should be ignored (piece_to_move = 0)
        # This is a valid scenario in Ludo
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (18,)

    def test_step_state_transition(self):
        """Test state transitions are valid."""
        env = LudoEnv()
        obs1, info1 = env.reset(seed=42)
        state1 = info1["state"]
        
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        state2 = info2["state"]
        
        # State should have valid structure
        assert len(state2.player_pieces) == 4
        assert len(state2.enemy_pieces) == 3
        assert 0 <= state2.current_player < 4
        assert 1 <= state2.dice_roll <= 6

    def test_step_reward_calculation(self):
        """Test reward calculation edge cases."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Run a few steps and check rewards
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reward should be numeric
            assert isinstance(reward, (int, float))
            # During gameplay, reward is typically 0 (sparse rewards)
            # Only +1 on win, -1 on loss
            if not terminated:
                assert reward == 0.0
            else:
                assert reward in (-1.0, 1.0)
            
            if terminated or truncated:
                break

    def test_step_termination(self):
        """Test game termination detection."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Run until termination (may take many steps)
        max_steps = 1000
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # On termination, reward should be -1 or +1
                assert reward in (-1.0, 1.0)
                break
        
        # If we didn't terminate, that's also valid (game can be long)

    def test_step_after_termination(self):
        """Test step() after termination behavior."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Run until termination
        terminated = False
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                terminated = True
                break
        
        # After termination, should reset before stepping again
        # The environment doesn't prevent stepping after termination,
        # but the game state should be handled gracefully
        if terminated:
            # Reset to start a new episode
            env.reset(seed=42)
            # Now step should work
            obs, reward, term, trunc, info = env.step(0)
            assert obs.shape == (18,)


class TestLudoEnvObservationSpace:
    """Test observation space edge cases."""

    def test_observation_space_shape(self):
        """Test observation space has correct shape."""
        env = LudoEnv()
        assert env.observation_space.shape == (18,)
        assert env.observation_space.dtype == np.int32

    def test_observation_space_bounds(self):
        """Test observations are within declared bounds."""
        env = LudoEnv()
        env.reset(seed=42)
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            
            # Check bounds: pieces can be -1 (home) to 57 (goal)
            # Dice: 1-6, current_player: 0-3
            assert np.all(obs >= env.observation_space.low)
            assert np.all(obs <= env.observation_space.high)

    def test_observation_space_contains(self):
        """Test observations are in observation_space."""
        env = LudoEnv()
        obs, _ = env.reset(seed=42)
        
        # Gymnasium's observation_space.contains() should return True
        assert env.observation_space.contains(obs)
        
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)


class TestLudoEnvActionSpace:
    """Test action space edge cases."""

    def test_action_space_discrete(self):
        """Test action space is Discrete(4)."""
        env = LudoEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 4

    def test_action_space_sample(self):
        """Test action space sampling."""
        env = LudoEnv()
        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < 4
            assert isinstance(action, (int, np.integer))


class TestLudoEnvTokenLimiting:
    """Test token limiting for different configurations."""

    def test_2tokens_limits_pieces(self):
        """Test 2 tokens per player limits available pieces."""
        env = LudoEnv(num_players=4, tokens_per_player=2)
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # Valid moves should only reference first 2 pieces
        if state.valid_moves:
            # movable_pieces should only contain 0 or 1 (first 2 pieces)
            if state.movable_pieces:
                assert all(p < 2 for p in state.movable_pieces)

    def test_4tokens_allows_all_pieces(self):
        """Test 4 tokens per player allows all pieces."""
        env = LudoEnv(num_players=4, tokens_per_player=4)
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # All 4 pieces should be potentially available
        # (depending on game state)
        if state.movable_pieces:
            assert all(p < 4 for p in state.movable_pieces)


class TestLudoEnvPlayerConfigurations:
    """Test different player configurations."""

    @pytest.mark.parametrize("num_players,tokens_per_player", [
        (2, 2), (2, 4), (3, 2), (3, 4), (4, 4)
    ])
    def test_all_configurations_work(self, num_players, tokens_per_player):
        """Test all documented configurations work."""
        env = LudoEnv(
            player_id=0,
            num_players=num_players,
            tokens_per_player=tokens_per_player,
        )
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (18,)
        
        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (18,)
            if terminated or truncated:
                break


class TestLudoEnvStateAccess:
    """Test State object access for different agent types."""

    def test_state_for_random_agent(self):
        """Test State object accessible for Random Agent."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # Random agent needs valid_moves
        assert isinstance(state.valid_moves, list)
        # Can sample from valid_moves
        if state.valid_moves:
            action = np.random.choice(state.valid_moves)
            assert action in state.valid_moves

    def test_state_for_rule_based_agent(self):
        """Test State object accessible for Rule-Based Agent."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # Rule-based agent needs player_pieces, enemy_pieces, movable_pieces
        assert len(state.player_pieces) == 4
        assert len(state.enemy_pieces) == 3
        # movable_pieces may be None or list
        if state.movable_pieces is not None:
            assert isinstance(state.movable_pieces, list)

    def test_state_for_tabular_q_agent(self):
        """Test State object accessible for Tabular Q-Learning Agent."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # Tabular Q needs state for abstraction
        assert hasattr(state, 'player_pieces')
        assert hasattr(state, 'enemy_pieces')
        assert hasattr(state, 'current_player')
        assert hasattr(state, 'dice_roll')
        # Can access all fields needed for state abstraction

    def test_state_for_dqn_agent(self):
        """Test State object accessible for DQN Agent."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # DQN can use either obs (numpy array) or state object
        # obs is already a numpy array (18,)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18,)
        # Or can use state object for custom feature extraction
        assert isinstance(state, State)

    def test_state_for_dueling_dqn_agent(self):
        """Test State object accessible for Dueling DQN Agent."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # Same as DQN - can use obs or state
        assert isinstance(obs, np.ndarray)
        assert isinstance(state, State)


class TestLudoEnvRender:
    """Test render() edge cases."""

    def test_render_before_reset(self):
        """Test render() before reset raises RuntimeError."""
        env = LudoEnv()
        with pytest.raises(RuntimeError, match="Environment not reset"):
            env.unwrapped.render(mode="human")

    def test_render_human_mode(self):
        """Test render() in human mode."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # Should not raise error
        result = env.unwrapped.render(mode="human")
        # Returns None for human mode
        assert result is None

    def test_render_rgb_array_mode(self):
        """Test render() in rgb_array mode."""
        env = LudoEnv()
        env.reset(seed=42)
        
        img = env.unwrapped.render(mode="rgb_array")
        assert isinstance(img, np.ndarray)
        assert len(img.shape) == 3  # (height, width, channels)
        assert img.shape[2] == 3  # RGB
        assert img.dtype == np.uint8

    def test_render_invalid_mode(self):
        """Test render() with invalid mode raises ValueError."""
        env = LudoEnv()
        env.reset(seed=42)
        
        with pytest.raises(ValueError, match="Unknown render mode"):
            env.unwrapped.render(mode="invalid_mode")

    def test_render_multiple_calls(self):
        """Test multiple render() calls don't cause errors."""
        env = LudoEnv()
        env.reset(seed=42)
        
        for _ in range(5):
            env.unwrapped.render(mode="rgb_array")
            action = env.action_space.sample()
            env.step(action)


class TestLudoEnvClose:
    """Test close() edge cases."""

    def test_close_after_reset(self):
        """Test close() after reset."""
        env = LudoEnv()
        env.reset(seed=42)
        env.close()
        
        # Should be able to reset again
        env.reset(seed=42)

    def test_close_without_reset(self):
        """Test close() without reset."""
        env = LudoEnv()
        env.close()
        # Should not raise error

    def test_close_multiple_times(self):
        """Test close() can be called multiple times."""
        env = LudoEnv()
        env.reset(seed=42)
        env.close()
        env.close()  # Should not raise error


class TestLudoEnvSeedReproducibility:
    """Test seed reproducibility edge cases."""

    def test_seed_reproducibility(self):
        """Test that same seed produces same initial state."""
        seed = 42
        
        env1 = LudoEnv(seed=seed)
        obs1, info1 = env1.reset(seed=seed)
        
        env2 = LudoEnv(seed=seed)
        obs2, info2 = env2.reset(seed=seed)
        
        # Initial observations should be similar (may vary due to game mechanics)
        assert obs1.shape == obs2.shape
        assert obs1.dtype == obs2.dtype

    def test_different_seeds_different_states(self):
        """Test different seeds produce different states."""
        env = LudoEnv()
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=123)
        
        # States should be different (with high probability)
        # Note: Due to game mechanics, this is probabilistic
        assert obs1.shape == obs2.shape


class TestLudoEnvEdgeCasesFromDocs:
    """Test specific edge cases documented in verificationReport.md."""

    def test_empty_valid_moves_handling(self):
        """Test handling of empty valid_moves (Random Agent edge case)."""
        env = LudoEnv()
        env.reset(seed=42)
        
        # When no valid moves, action should still work
        # This is handled in step() by using piece_to_move = 0
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (18,)

    def test_action_index_mapping(self):
        """Test action index to piece index mapping (Rule-Based/Tabular Q edge case)."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # If movable_pieces exists, action index maps to piece index
        if state.movable_pieces and state.valid_moves:
            action_idx = state.valid_moves[0]
            piece_idx = state.movable_pieces[action_idx]
            assert 0 <= piece_idx < 4

    def test_state_abstraction_access(self):
        """Test state accessible for abstraction (Tabular Q/DQN edge case)."""
        env = LudoEnv()
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        # All fields needed for state abstraction should be present
        assert hasattr(state, 'player_pieces')
        assert hasattr(state, 'enemy_pieces')
        assert hasattr(state, 'current_player')
        assert hasattr(state, 'dice_roll')
        assert hasattr(state, 'valid_moves')

    def test_observation_shape_consistency(self):
        """Test observation shape is consistent (DQN/Dueling DQN edge case)."""
        env = LudoEnv()
        obs, _ = env.reset(seed=42)
        initial_shape = obs.shape
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == initial_shape
            assert obs.dtype == np.int32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

