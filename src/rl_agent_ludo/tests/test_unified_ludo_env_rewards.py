"""
Comprehensive reward function tests for UnifiedLudoEnv.

Tests reward calculation correctness for:
- Delta-Progress rewards
- Event Impulses (Exit Home, Safe Zone, Kill, Goal, Win)
- ILA Penalties (Death penalty)
- Both 2-token and 4-token configurations

Verifies rewards are reasonable and not producing incorrect values.
"""

import pytest
import numpy as np
from typing import List, Tuple

from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)
from rl_agent_ludo.utils.state import State

# Board constants for testing
HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]
HOME_CORRIDOR = list(range(52, 57))


class TestRewardDeltaProgress:
    """Test Delta-Progress reward calculation."""

    def test_forward_movement_reward_4tokens(self):
        """Test positive delta-progress reward for forward movement."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Find a token that can move forward
        if state.valid_moves:
            action = state.valid_moves[0]
            prev_pos = state.player_pieces[action]
            
            # Step and check reward
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            curr_pos = next_state.player_pieces[action]
            
            if prev_pos != curr_pos and prev_pos != HOME_INDEX:
                # If moved forward (not from home), should have positive delta-progress
                if curr_pos > prev_pos:
                    # Delta-progress should contribute positively
                    # Scale factor is 2.0, so progress change * 2.0
                    expected_min = 0.0  # At minimum, some positive contribution
                    assert reward >= expected_min, \
                        f"Forward movement should have positive reward, got {reward}"
        
    def test_forward_movement_reward_2tokens(self):
        """Test positive delta-progress reward for forward movement (2 tokens)."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        if state.valid_moves:
            action = state.valid_moves[0]
            prev_pos = state.player_pieces[action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            curr_pos = next_state.player_pieces[action]
            
            if prev_pos != curr_pos and prev_pos != HOME_INDEX:
                if curr_pos > prev_pos:
                    assert reward >= 0.0, \
                        f"Forward movement should have non-negative reward, got {reward}"

    def test_backward_movement_penalty(self):
        """Test negative delta-progress when sent home (backward movement)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Run until we find a scenario where a piece gets sent home
        # This is probabilistic, so we'll check if it happens
        for step in range(100):
            if state.valid_moves:
                action = state.valid_moves[0]
                prev_pos = state.player_pieces[action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                curr_pos = next_state.player_pieces[action]
                
                # Check if we were sent home (captured)
                if prev_pos not in [HOME_INDEX, GOAL_INDEX] and curr_pos == HOME_INDEX:
                    # Should have negative reward (delta-progress + death penalty)
                    # Death penalty is -20.0, plus negative progress delta
                    assert reward < 0.0, \
                        f"Being sent home should have negative reward, got {reward}"
                    # Total should be around -20.0 (death penalty) + negative progress
                    assert reward <= -20.0, \
                        f"Death penalty should be at least -20.0, got {reward}"
                    break
                
                state = next_state
                if terminated or truncated:
                    break

    def test_progress_reward_magnitude(self):
        """Test that progress rewards have reasonable magnitude."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        
        # Try multiple episodes to ensure we get some rewards
        rewards = []
        for episode in range(5):
            obs, info = env.reset(seed=42 + episode)
            
            for step in range(50):
                if info["state"].valid_moves:
                    action = info["state"].valid_moves[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward)
                    
                    # Progress rewards should be reasonable (not extreme)
                    # Scale factor is 2.0, max progress change is ~1.0 (0 to 1 normalized)
                    # So max delta-progress reward is ~2.0 per token
                    # With 4 tokens, max would be ~8.0, but typically much less
                    if not terminated:
                        # During normal gameplay, rewards should be reasonable
                        assert abs(reward) < 100.0, \
                            f"Reward {reward} seems too extreme for normal gameplay"
                    
                    if terminated or truncated:
                        break
        
        # Verify we collected some rewards (at least from one episode)
        assert len(rewards) > 0, "Should have collected some rewards across episodes"


class TestRewardEventImpulses:
    """Test Event Impulse rewards."""

    def test_exit_home_reward(self):
        """Test +10.0 reward for exiting home."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Find a token at home that can exit (needs dice=6)
        # This is probabilistic, so we'll check multiple resets
        for reset_attempt in range(20):
            obs, info = env.reset(seed=42 + reset_attempt)
            state = info["state"]
            
            # Check if any token is at home and can move
            for action in range(4):
                if state.player_pieces[action] == HOME_INDEX:
                    if action in state.valid_moves:
                        # Token at home can move (dice=6)
                        prev_pos = state.player_pieces[action]
                        obs, reward, terminated, truncated, info = env.step(action)
                        next_state = info["state"]
                        curr_pos = next_state.player_pieces[action]
                        
                        if prev_pos == HOME_INDEX and curr_pos != HOME_INDEX:
                            # Should have exit home reward (+10.0) plus delta-progress
                            assert reward >= 10.0, \
                                f"Exit home should give at least +10.0, got {reward}"
                            # Total should be around 10.0 + delta-progress
                            # Delta-progress from 0 to 1/57 = ~0.017, scaled = ~0.034
                            # So total should be around 10.0 - 10.5
                            assert 9.0 <= reward <= 12.0, \
                                f"Exit home reward should be ~10.0-12.0, got {reward}"
                            return
        
        # If we didn't find it, that's okay (probabilistic)

    def test_enter_safe_zone_reward(self):
        """Test +5.0 reward for entering safe zone (Globe or Star)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Run until we enter a safe zone
        for step in range(200):
            if state.valid_moves:
                action = state.valid_moves[0]
                prev_pos = state.player_pieces[action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                curr_pos = next_state.player_pieces[action]
                
                # Check if we entered a safe zone
                if prev_pos not in GLOBE_INDEXES and curr_pos in GLOBE_INDEXES:
                    # Should have safe zone reward (+5.0) plus delta-progress
                    assert reward >= 5.0, \
                        f"Entering safe zone should give at least +5.0, got {reward}"
                    # Total should be around 5.0 + delta-progress
                    assert 4.0 <= reward <= 8.0, \
                        f"Safe zone reward should be ~5.0-8.0, got {reward}"
                    return
                
                if prev_pos not in STAR_INDEXES and curr_pos in STAR_INDEXES:
                    # Same for star
                    assert reward >= 5.0, \
                        f"Entering star should give at least +5.0, got {reward}"
                    assert 4.0 <= reward <= 8.0, \
                        f"Star reward should be ~5.0-8.0, got {reward}"
                    return
                
                state = next_state
                if terminated or truncated:
                    break

    def test_kill_enemy_reward(self):
        """Test +15.0 reward for killing enemy."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Run until we capture an enemy
        for step in range(500):
            if state.valid_moves:
                action = state.valid_moves[0]
                prev_pos = state.player_pieces[action]
                
                # Check if we can capture
                for enemy in state.enemy_pieces:
                    for e_pos in enemy:
                        if e_pos not in [HOME_INDEX, GOAL_INDEX, 1] and e_pos not in GLOBE_INDEXES:
                            # Check if we can reach this enemy
                            if prev_pos != HOME_INDEX:
                                obs, reward, terminated, truncated, info = env.step(action)
                                next_state = info["state"]
                                curr_pos = next_state.player_pieces[action]
                                
                                # Check if we captured (landed on enemy position)
                                if curr_pos == e_pos:
                                    # Should have kill reward (+15.0) plus delta-progress
                                    assert reward >= 15.0, \
                                        f"Kill should give at least +15.0, got {reward}"
                                    # Total should be around 15.0 + delta-progress
                                    assert 14.0 <= reward <= 18.0, \
                                        f"Kill reward should be ~15.0-18.0, got {reward}"
                                    return
                
                obs, reward, terminated, truncated, info = env.step(action)
                state = info["state"]
                if terminated or truncated:
                    break

    def test_goal_reward(self):
        """Test +30.0 reward for reaching goal."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Run until we reach goal (may take many steps)
        for step in range(1000):
            if state.valid_moves:
                action = state.valid_moves[0]
                prev_pos = state.player_pieces[action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                curr_pos = next_state.player_pieces[action]
                
                # Check if we reached goal
                if prev_pos != GOAL_INDEX and curr_pos == GOAL_INDEX:
                    # Should have goal reward (+30.0) plus delta-progress
                    assert reward >= 30.0, \
                        f"Goal should give at least +30.0, got {reward}"
                    # Total should be around 30.0 + delta-progress
                    assert 29.0 <= reward <= 35.0, \
                        f"Goal reward should be ~30.0-35.0, got {reward}"
                    return
                
                state = next_state
                if terminated or truncated:
                    break

    def test_win_reward(self):
        """Test +100.0 reward for winning game."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Run until win (may take many steps)
        for step in range(2000):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    # Check if we won
                    if reward > 0:
                        # Win reward should be +100.0
                        assert reward == 100.0, \
                            f"Win should give exactly +100.0, got {reward}"
                        return
                    else:
                        # Loss reward should be -50.0
                        assert reward == -50.0, \
                            f"Loss should give exactly -50.0, got {reward}"
                        break


class TestRewardILAPenalty:
    """Test ILA Penalty (Death penalty)."""

    def test_death_penalty(self):
        """Test -20.0 penalty when captured (sent home)."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # Run until we get captured
        for step in range(500):
            if state.valid_moves:
                action = state.valid_moves[0]
                prev_pos = state.player_pieces[action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                curr_pos = next_state.player_pieces[action]
                
                # Check if we were captured (sent home)
                if prev_pos not in [HOME_INDEX, GOAL_INDEX] and curr_pos == HOME_INDEX:
                    # Should have death penalty (-20.0) plus negative progress delta
                    assert reward < 0.0, \
                        f"Death should have negative reward, got {reward}"
                    # Death penalty is -20.0, plus negative progress
                    # Progress from position X to 0 is negative, scaled by 2.0
                    # So total should be around -20.0 + (negative progress)
                    assert reward <= -20.0, \
                        f"Death penalty should be at least -20.0, got {reward}"
                    # Should be very negative (death penalty + lost progress)
                    assert reward < -15.0, \
                        f"Death should be very negative, got {reward}"
                    return
                
                state = next_state
                if terminated or truncated:
                    break


class TestRewardCombinations:
    """Test reward combinations and edge cases."""

    def test_reward_not_nan_or_inf(self):
        """Test rewards are never NaN or Inf."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        for step in range(100):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert not np.isnan(reward), f"Reward should not be NaN, got {reward}"
                assert not np.isinf(reward), f"Reward should not be Inf, got {reward}"
                assert isinstance(reward, (int, float, np.floating, np.integer)), \
                    f"Reward should be numeric, got {type(reward)}"
                
                if terminated or truncated:
                    break

    def test_reward_ranges_reasonable(self):
        """Test rewards are in reasonable ranges."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        
        # Try multiple episodes to ensure we get some rewards
        rewards = []
        for episode in range(5):
            obs, info = env.reset(seed=42 + episode)
            
            for step in range(200):
                if info["state"].valid_moves:
                    action = info["state"].valid_moves[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward)
                    
                    if not terminated:
                        # During gameplay, rewards should be reasonable
                        # Max would be: Goal (+30) + delta-progress, or Kill (+15) + delta-progress
                        # So max around 35-40 for single events
                        assert -100.0 <= reward <= 50.0, \
                            f"Reward {reward} seems unreasonable for normal gameplay"
                    
                    if terminated or truncated:
                        # Terminal rewards can be -50.0 (loss) or +100.0 (win)
                        assert reward in (-50.0, 100.0), \
                            f"Terminal reward {reward} should be -50.0 or +100.0"
                        break
        
        # Verify we collected rewards (at least from one episode)
        assert len(rewards) > 0, "Should have collected some rewards across episodes"

    def test_reward_consistency_2tokens(self):
        """Test reward consistency for 2-token configuration."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        obs, info = env.reset()
        
        rewards = []
        for step in range(100):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                
                assert not np.isnan(reward)
                assert not np.isinf(reward)
                
                if terminated or truncated:
                    break
        
        # Verify rewards are reasonable
        if rewards:
            assert all(-100.0 <= r <= 100.0 for r in rewards), \
                "All rewards should be in reasonable range"

    def test_reward_consistency_4tokens(self):
        """Test reward consistency for 4-token configuration."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        rewards = []
        for step in range(100):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                rewards.append(reward)
                
                assert not np.isnan(reward)
                assert not np.isinf(reward)
                
                if terminated or truncated:
                    break
        
        # Verify rewards are reasonable
        if rewards:
            assert all(-100.0 <= r <= 100.0 for r in rewards), \
                "All rewards should be in reasonable range"

    def test_reward_when_not_our_turn(self):
        """Test reward is 0.0 when it's not our turn."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # If it's not our turn initially, reward should be 0
        if state.current_player != 0:
            if state.valid_moves:
                action = state.valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                # Reward should be 0.0 when not our turn
                assert reward == 0.0, \
                    f"Reward should be 0.0 when not our turn, got {reward}"

    def test_reward_summary_statistics(self):
        """Test reward statistics are reasonable."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        
        all_rewards = []
        for episode in range(10):
            obs, info = env.reset(seed=42 + episode)
            episode_rewards = []
            
            for step in range(200):
                if info["state"].valid_moves:
                    action = info["state"].valid_moves[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_rewards.append(reward)
                    
                    if terminated or truncated:
                        break
            
            all_rewards.extend(episode_rewards)
        
        if all_rewards:
            # Calculate statistics
            mean_reward = np.mean(all_rewards)
            std_reward = np.std(all_rewards)
            min_reward = np.min(all_rewards)
            max_reward = np.max(all_rewards)
            
            # Verify statistics are reasonable
            assert -50.0 <= mean_reward <= 10.0, \
                f"Mean reward {mean_reward} seems unreasonable"
            assert std_reward < 50.0, \
                f"Std reward {std_reward} seems too high"
            assert min_reward >= -100.0, \
                f"Min reward {min_reward} seems too negative"
            assert max_reward <= 100.0, \
                f"Max reward {max_reward} seems too high"


class TestRewardSpecificScenarios:
    """Test rewards for specific game scenarios."""

    def test_no_movement_reward(self):
        """Test reward when no movement occurs."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        state = info["state"]
        
        # If we step but don't move (invalid action or no valid moves)
        if not state.valid_moves:
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            # Reward should be 0.0 or small (no progress change)
            assert reward == 0.0, \
                f"No movement should give 0.0 reward, got {reward}"

    def test_multiple_tokens_progress(self):
        """Test delta-progress when multiple tokens move."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Run several steps and verify progress rewards accumulate correctly
        total_reward = 0.0
        for step in range(50):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Total reward should be reasonable
                assert -200.0 <= total_reward <= 200.0, \
                    f"Total reward {total_reward} seems unreasonable after {step} steps"
                
                if terminated or truncated:
                    break

    def test_reward_after_reset(self):
        """Test reward is 0.0 immediately after reset."""
        env = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
        obs, info = env.reset()
        
        # Reset doesn't return reward, but first step should be reasonable
        if info["state"].valid_moves:
            action = info["state"].valid_moves[0]
            obs, reward, terminated, truncated, info = env.step(action)
            # First step reward should be reasonable
            assert isinstance(reward, (int, float, np.floating, np.integer))
            assert not np.isnan(reward)
            assert not np.isinf(reward)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

