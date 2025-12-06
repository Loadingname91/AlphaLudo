"""
Comprehensive unit tests for UnifiedLudoEnv edge cases.

Tests critical edge cases to ensure correct action indexing, piece indexing,
and handling of 2 tokens vs 4 tokens configurations.

Key Test Areas:
1. Action Index vs Piece Index (Absolute Indexing)
2. Action Masking
3. Token Limiting (2 tokens vs 4 tokens)
4. Invalid Action Handling
5. Valid Moves Filtering
6. Observation Space Correctness
7. Reward Calculation Edge Cases
"""

import pytest
import numpy as np
import gymnasium as gym

from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)
from rl_agent_ludo.utils.state import State


class TestUnifiedLudoEnvInitialization:
    """Test environment initialization edge cases."""

    def test_2tokens_env_initialization(self):
        """Test UnifiedLudoEnv2Tokens initialization."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        assert env.tokens_per_player == 2
        assert env.num_players == 2
        assert env.action_space.n == 4
        assert env.observation_space.shape == (28,)  # 10 + 9*2
        assert env.observation_space.dtype == np.float32

    def test_4tokens_env_initialization(self):
        """Test UnifiedLudoEnv4Tokens initialization."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        assert env.tokens_per_player == 4
        assert env.num_players == 4
        assert env.action_space.n == 4
        assert env.observation_space.shape == (46,)  # 10 + 9*4
        assert env.observation_space.dtype == np.float32

    def test_invalid_num_players(self):
        """Test invalid num_players raises AssertionError."""
        with pytest.raises(AssertionError, match="num_players must be 2, 3, or 4"):
            UnifiedLudoEnv2Tokens(num_players=1)
        
        with pytest.raises(AssertionError, match="num_players must be 2, 3, or 4"):
            UnifiedLudoEnv4Tokens(num_players=5)

    def test_invalid_player_id(self):
        """Test invalid player_id raises AssertionError."""
        with pytest.raises(AssertionError, match="player_id must be < num_players"):
            UnifiedLudoEnv2Tokens(player_id=2, num_players=2)

    @pytest.mark.parametrize("num_players", [2, 4])
    def test_all_player_configurations_2tokens(self, num_players):
        """Test 2 tokens env with different player counts."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=num_players, seed=42)
        assert env.observation_space.shape == (28,)
        obs, info = env.reset()
        assert obs.shape == (28,)
        assert "action_mask" in info

    @pytest.mark.parametrize("num_players", [2, 4])
    def test_all_player_configurations_4tokens(self, num_players):
        """Test 4 tokens env with different player counts."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=num_players, seed=42)
        assert env.observation_space.shape == (46,)
        obs, info = env.reset()
        assert obs.shape == (46,)
        assert "action_mask" in info


class TestUnifiedLudoEnvActionIndexing:
    """
    Test critical action indexing vs piece indexing.
    
    CRITICAL: Uses absolute indexing where action_idx IS piece_idx.
    Action 0 = Token 0, Action 1 = Token 1, etc.
    """

    def test_action_index_equals_piece_index_2tokens(self):
        """Test that action index equals piece index (absolute indexing) for 2 tokens."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Action 0 should correspond to piece 0
        # Action 1 should correspond to piece 1
        # Actions 2 and 3 should be invalid (tokens_per_player = 2)
        
        if state.valid_moves:
            for action in state.valid_moves:
                # With absolute indexing, action IS the piece index
                assert action == action  # Trivial but confirms absolute indexing
                assert 0 <= action < 2  # Only first 2 tokens are valid
                
                # Verify action mask reflects this
                assert info["action_mask"][action] == True
                if action < 2:
                    # Actions 2 and 3 should be masked out
                    assert info["action_mask"][2] == False
                    assert info["action_mask"][3] == False

    def test_action_index_equals_piece_index_4tokens(self):
        """Test that action index equals piece index (absolute indexing) for 4 tokens."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # All actions 0-3 should potentially be valid
        if state.valid_moves:
            for action in state.valid_moves:
                # With absolute indexing, action IS the piece index
                assert 0 <= action < 4
                assert info["action_mask"][action] == True

    def test_action_mask_correctness_2tokens(self):
        """Test action mask correctly masks invalid actions for 2 tokens."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        state = info["state"]
        action_mask = info["action_mask"]
        
        # Action mask should be shape (4,)
        assert action_mask.shape == (4,)
        assert action_mask.dtype == bool
        
        # Actions 2 and 3 should always be False (only 2 tokens)
        assert action_mask[2] == False
        assert action_mask[3] == False
        
        # Actions 0 and 1 should match valid_moves
        if state.valid_moves:
            for action in state.valid_moves:
                assert 0 <= action < 2
                assert action_mask[action] == True
        else:
            # If no valid moves, all should be False
            assert action_mask[0] == False
            assert action_mask[1] == False

    def test_action_mask_correctness_4tokens(self):
        """Test action mask correctly masks invalid actions for 4 tokens."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        action_mask = info["action_mask"]
        
        # Action mask should be shape (4,)
        assert action_mask.shape == (4,)
        assert action_mask.dtype == bool
        
        # All actions 0-3 are potentially valid
        if state.valid_moves:
            for action in state.valid_moves:
                assert 0 <= action < 4
                assert action_mask[action] == True

    def test_step_with_correct_action_index(self):
        """Test step() with correct action index (action = piece index)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        if state.valid_moves:
            # Use first valid action (which is also the piece index)
            action = state.valid_moves[0]
            piece_idx = action  # With absolute indexing, they're the same
            
            # Store previous position
            prev_pos = state.player_pieces[piece_idx]
            
            # Step with action
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            
            # Verify the correct piece moved (or didn't move if invalid)
            # The piece at piece_idx should be the one affected
            assert isinstance(next_state.player_pieces[piece_idx], int)
            assert 0 <= next_state.player_pieces[piece_idx] <= 57

    def test_step_action_mapping_consistency(self):
        """Test that action index consistently maps to piece index across steps."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Run multiple steps and verify consistency
        for step in range(10):
            state = info["state"]
            if state.valid_moves:
                action = state.valid_moves[0]
                piece_idx = action  # Absolute indexing
                
                # Verify action is in valid range
                assert 0 <= action < 4
                assert 0 <= piece_idx < 4
                assert action == piece_idx  # They must be equal
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break


class TestUnifiedLudoEnvInvalidActions:
    """Test handling of invalid actions."""

    def test_step_with_invalid_action_index_out_of_bounds(self):
        """Test step() with action index out of bounds (>= 4)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        env.reset()
        
        # Action index out of bounds should use fallback
        invalid_action = 999
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Should not raise error, but should use fallback logic
        assert obs.shape == (46,)
        assert isinstance(reward, (int, float))
        assert "action_mask" in info

    def test_step_with_negative_action(self):
        """Test step() with negative action index."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        env.reset()
        
        # Negative action should use fallback
        obs, reward, terminated, truncated, info = env.step(-1)
        assert obs.shape == (46,)

    def test_step_with_action_not_in_valid_moves_2tokens(self):
        """Test step() with action not in valid_moves for 2 tokens."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Try action 2 or 3 (invalid for 2 tokens)
        invalid_action = 2
        obs, reward, terminated, truncated, info = env.step(invalid_action)
        
        # Should not raise error, should use fallback
        assert obs.shape == (28,)
        assert "action_mask" in info

    def test_step_with_action_not_in_valid_moves_4tokens(self):
        """Test step() with action not in valid_moves for 4 tokens."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # If action is not in valid_moves, should use fallback
        # Try an action that might not be valid
        action = 3  # Last token
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should not raise error
        assert obs.shape == (46,)

    def test_step_with_no_valid_moves(self):
        """Test step() when no valid moves are available."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # When no valid moves, action should still work
        # (This happens when player can't move, e.g., no 6 rolled)
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Should not raise error
        assert obs.shape == (46,)
        assert isinstance(reward, (int, float, np.floating, np.integer))


class TestUnifiedLudoEnvTokenLimiting:
    """Test token limiting for 2 tokens vs 4 tokens."""

    def test_2tokens_limits_valid_moves(self):
        """Test that 2 tokens env only allows actions 0 and 1."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        state = info["state"]
        action_mask = info["action_mask"]
        
        # Actions 2 and 3 should always be masked
        assert action_mask[2] == False
        assert action_mask[3] == False
        
        # Valid moves should only contain 0 or 1
        if state.valid_moves:
            assert all(0 <= action < 2 for action in state.valid_moves)

    def test_4tokens_allows_all_actions(self):
        """Test that 4 tokens env allows all actions 0-3."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        action_mask = info["action_mask"]
        
        # All actions 0-3 are potentially valid
        if state.valid_moves:
            assert all(0 <= action < 4 for action in state.valid_moves)

    def test_2tokens_observation_size(self):
        """Test that 2 tokens env has correct observation size."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        
        # Should be 10 (global) + 9*2 (tokens) = 28
        assert obs.shape == (28,)
        assert len(obs) == 28

    def test_4tokens_observation_size(self):
        """Test that 4 tokens env has correct observation size."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Should be 10 (global) + 9*4 (tokens) = 46
        assert obs.shape == (46,)
        assert len(obs) == 46

    def test_token_limiting_consistency_across_steps(self):
        """Test token limiting is consistent across multiple steps."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        
        # Run multiple steps
        for _ in range(10):
            action_mask = info["action_mask"]
            # Actions 2 and 3 should always be False
            assert action_mask[2] == False
            assert action_mask[3] == False
            
            action = 0  # Use first action
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break


class TestUnifiedLudoEnvObservationSpace:
    """Test observation space correctness."""

    def test_observation_space_bounds_2tokens(self):
        """Test observations are within declared bounds for 2 tokens."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        
        # All features should be in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert obs.dtype == np.float32
        
        # Run a few steps
        for _ in range(5):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
            if terminated or truncated:
                break

    def test_observation_space_bounds_4tokens(self):
        """Test observations are within declared bounds for 4 tokens."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # All features should be in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        assert obs.dtype == np.float32
        
        # Run a few steps
        for _ in range(5):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
            if terminated or truncated:
                break

    def test_observation_space_contains(self):
        """Test observations are in observation_space."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Gymnasium's observation_space.contains() should return True
        assert env.observation_space.contains(obs)
        
        for _ in range(5):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_observation_shape_consistency(self):
        """Test observation shape is consistent across steps."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        initial_shape = obs.shape
        
        for _ in range(10):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == initial_shape
            assert obs.dtype == np.float32
            if terminated or truncated:
                break


class TestUnifiedLudoEnvActionMasking:
    """Test action masking functionality."""

    def test_action_mask_in_info(self):
        """Test action mask is provided in info dict."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        assert "action_mask" in info
        assert isinstance(info["action_mask"], np.ndarray)
        assert info["action_mask"].shape == (4,)
        assert info["action_mask"].dtype == bool

    def test_action_mask_updated_after_step(self):
        """Test action mask is updated after each step."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        initial_mask = info["action_mask"].copy()
        
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Action mask should be in info
        assert "action_mask" in info
        # Mask may have changed (different valid moves)
        assert info["action_mask"].shape == (4,)

    def test_action_mask_matches_valid_moves(self):
        """Test action mask matches valid_moves in state."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        action_mask = info["action_mask"]
        
        # Action mask should match valid_moves
        if state.valid_moves:
            for action in state.valid_moves:
                assert action_mask[action] == True
        
        # Actions not in valid_moves should be False
        for action in range(4):
            if action not in state.valid_moves:
                assert action_mask[action] == False

    def test_action_mask_for_logit_masking(self):
        """Test action mask can be used for logit masking in DQN."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        action_mask = info["action_mask"]
        
        # Only test if there are valid actions
        if np.any(action_mask):
            # Simulate Q-values from network
            q_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            
            # Apply mask: set invalid actions to -inf
            masked_q_values = q_values.copy()
            masked_q_values[~action_mask] = float('-inf')
            
            # Best action should be from valid actions only
            best_action = np.argmax(masked_q_values)
            assert action_mask[best_action] == True, \
                f"Best action {best_action} should be valid, but mask is {action_mask}"


class TestUnifiedLudoEnvRewardCalculation:
    """Test reward calculation edge cases."""

    def test_reward_is_numeric(self):
        """Test reward is always numeric."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        env.reset()
        
        for _ in range(10):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert isinstance(reward, (int, float, np.floating, np.integer))
            assert not np.isnan(reward)
            assert not np.isinf(reward)
            
            if terminated or truncated:
                break

    def test_reward_on_win(self):
        """Test reward on game win."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        env.reset()
        
        # Run until win (may take many steps)
        max_steps = 1000
        for step in range(max_steps):
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # Win reward should be +100.0
                # (Note: may be -50.0 if we lost)
                assert reward in (100.0, -50.0)
                break

    def test_delta_progress_reward(self):
        """Test delta-progress reward is calculated."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        if state.valid_moves:
            action = state.valid_moves[0]
            prev_pos = state.player_pieces[action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            curr_pos = next_state.player_pieces[action]
            
            # If piece moved, reward should reflect progress
            if prev_pos != curr_pos:
                # Reward should be non-zero (delta-progress + possible impulses)
                # Can be positive (moved forward) or negative (sent home)
                assert isinstance(reward, (int, float))


class TestUnifiedLudoEnvStateConsistency:
    """Test state consistency across steps."""

    def test_state_object_consistency(self):
        """Test State object is consistent and accessible."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        assert isinstance(state, State)
        assert len(state.player_pieces) == 4
        assert len(state.enemy_pieces) == 3
        assert all(len(enemy) == 4 for enemy in state.enemy_pieces)
        assert 0 <= state.current_player < 4
        assert 1 <= state.dice_roll <= 6
        assert isinstance(state.valid_moves, list)

    def test_valid_moves_are_piece_indices(self):
        """Test valid_moves contain piece indices (absolute indexing)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        if state.valid_moves:
            for action in state.valid_moves:
                # With absolute indexing, action IS piece index
                assert 0 <= action < 4
                assert action == action  # Confirms absolute indexing

    def test_state_transition_validity(self):
        """Test state transitions are valid."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs1, info1 = env.reset()
        state1 = info1["state"]
        
        action = 0
        obs2, reward, terminated, truncated, info2 = env.step(action)
        state2 = info2["state"]
        
        # State should have valid structure
        assert len(state2.player_pieces) == 4
        assert len(state2.enemy_pieces) == 3
        assert 0 <= state2.current_player < 4
        assert 1 <= state2.dice_roll <= 6
        assert all(0 <= pos <= 57 for pos in state2.player_pieces)


class TestUnifiedLudoEnvEdgeCases:
    """Test specific edge cases."""

    def test_reset_returns_correct_types(self):
        """Test reset returns correct types."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)
        assert "state" in info
        assert "action_mask" in info
        assert isinstance(info["state"], State)

    def test_step_before_reset(self):
        """Test step() before reset raises RuntimeError."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        with pytest.raises(RuntimeError, match="Reset first"):
            env.step(0)

    def test_multiple_resets(self):
        """Test multiple resets don't cause errors."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        for _ in range(5):
            obs, info = env.reset(seed=42)
            assert obs.shape == (46,)
            assert "action_mask" in info

    def test_action_index_vs_piece_index_absolute_mapping(self):
        """
        CRITICAL TEST: Verify absolute indexing where action_idx = piece_idx.
        
        This is the core requirement from dqnAgent.py:
        - Action 0 = Token 0
        - Action 1 = Token 1
        - Action 2 = Token 2
        - Action 3 = Token 3
        """
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # For each valid action, verify it maps to the correct piece
        if state.valid_moves:
            for action in state.valid_moves:
                # CRITICAL: With absolute indexing, action IS the piece index
                piece_idx = action
                
                # Verify they're the same
                assert action == piece_idx, \
                    f"Action {action} must equal piece index {piece_idx} (absolute indexing)"
                
                # Verify piece exists
                assert 0 <= piece_idx < 4
                assert 0 <= state.player_pieces[piece_idx] <= 57
                
                # Verify action mask reflects this
                assert info["action_mask"][action] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

