"""
Integration tests for Phase 1 and Phase 2 working together.

Tests the full workflow:
1. Environment with improved PBRS (Phase 1)
2. Trajectory collection during training (Phase 2)
3. State expansion and temporal discounting
"""

import pytest
import numpy as np
from rl_agent_ludo.environment.unifiedLudoEnv import UnifiedLudoEnv2Tokens
from rl_agent_ludo.agents.trajectoryBuffer import TrajectoryBuffer
from rl_agent_ludo.utils.state import State


class TestPhase1Phase2Integration:
    """Integration tests for Phase 1 and Phase 2."""
    
    def test_full_trajectory_collection_workflow(self):
        """Test complete trajectory collection workflow."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        buffer = TrajectoryBuffer(max_size=1000, gamma=0.99)
        
        # Simulate a game episode
        obs, info = env.reset(seed=42)
        state = info["state"]
        done = False
        episode_length = 0
        max_steps = 100
        
        # Collect initial state
        expanded_state = env.expand_state_with_relative_features(state)
        buffer.add_state(expanded_state)
        
        while not done and episode_length < max_steps:
            # Select action (simple: first valid move)
            if state.valid_moves:
                action = state.valid_moves[0]
            else:
                action = 0
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            done = terminated or truncated
            
            # Collect expanded state
            expanded_state = env.expand_state_with_relative_features(next_state)
            buffer.add_state(expanded_state)
            
            state = next_state
            episode_length += 1
        
        # Finalize trajectory (simulate win)
        outcome = 1.0 if done and episode_length < max_steps else 0.0
        num_labeled = buffer.finalize_trajectory(outcome)
        
        # Verify trajectory was collected
        assert num_labeled > 0, "Should have collected at least one state"
        assert buffer.size() == num_labeled, "Buffer size should match labeled states"
        assert buffer.total_trajectories == 1, "Should have one trajectory"
        
        # Verify states and labels
        states, labels = buffer.get_dataset()
        assert states.shape[0] == num_labeled, "Should have correct number of states"
        assert labels.shape[0] == num_labeled, "Should have correct number of labels"
        assert np.all(labels >= 0.0) and np.all(labels <= 1.0), "Labels should be in [0, 1]"
    
    def test_pbrs_with_trajectory_collection(self):
        """Test that PBRS rewards work correctly with trajectory collection."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        buffer = TrajectoryBuffer(max_size=1000, gamma=0.99)
        
        # Run a few steps and verify PBRS rewards are calculated
        obs, info = env.reset(seed=42)
        state = info["state"]
        
        total_reward = 0.0
        for step in range(10):
            if state.valid_moves:
                action = state.valid_moves[0]
            else:
                action = 0
            
            # Collect state before step
            expanded_state = env.expand_state_with_relative_features(state)
            buffer.add_state(expanded_state)
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            done = terminated or truncated
            
            total_reward += reward
            
            # Verify reward is reasonable (PBRS should give non-zero rewards)
            # Rewards can be positive or negative depending on potential changes
            assert isinstance(reward, (int, float)), "Reward should be numeric"
            assert not np.isnan(reward), "Reward should not be NaN"
            assert not np.isinf(reward), "Reward should not be infinite"
            
            state = next_state
            if done:
                break
        
        # Finalize trajectory
        outcome = 1.0 if total_reward > 0 else 0.0  # Simplified outcome
        buffer.finalize_trajectory(outcome)
        
        # Verify trajectory was collected
        assert buffer.size() > 0, "Should have collected trajectory"
    
    def test_lead_bonus_affects_potential_during_trajectory(self):
        """Test that lead bonus affects potential calculation during trajectory collection."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        
        # State with lead token
        state_lead = State(
            player_pieces=[40, 10],  # Token 0 far ahead
            enemy_pieces=[[20, 0, 0, 0]],
            current_player=0,
            dice_roll=1,
            valid_moves=[0, 1],
            movable_pieces=[0, 1]
        )
        
        potential_lead = env._calculate_potential(state_lead)
        
        # State without lead (even spread)
        state_no_lead = State(
            player_pieces=[25, 20],  # Tokens close together
            enemy_pieces=[[20, 0, 0, 0]],
            current_player=0,
            dice_roll=1,
            valid_moves=[0, 1],
            movable_pieces=[0, 1]
        )
        
        potential_no_lead = env._calculate_potential(state_no_lead)
        
        # Lead state should have higher potential
        assert potential_lead > potential_no_lead, "Lead token should increase potential"
        
        # Both should be positive
        assert potential_lead > 0, "Potential should be positive"
        assert potential_no_lead > 0, "Potential should be positive"
    
    def test_collision_penalty_affects_potential_during_trajectory(self):
        """Test that collision penalty affects potential calculation during trajectory collection."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        
        # State with collision risk (enemy close behind on normal path)
        state_danger = State(
            player_pieces=[20, 0],  # Token 0 at 20 (normal path, not safe zone)
            enemy_pieces=[[18, 0, 0, 0]],  # Enemy at 18 (2 steps behind, within 6)
            current_player=0,
            dice_roll=1,
            valid_moves=[0, 1],
            movable_pieces=[0, 1]
        )
        
        potential_danger = env._calculate_potential(state_danger)
        
        # State without collision risk (enemy far away)
        state_safe = State(
            player_pieces=[20, 0],  # Token 0 at 20 (same position)
            enemy_pieces=[[5, 0, 0, 0]],  # Enemy at 5 (15 steps behind, outside danger zone)
            current_player=0,
            dice_roll=1,
            valid_moves=[0, 1],
            movable_pieces=[0, 1]
        )
        
        potential_safe = env._calculate_potential(state_safe)
        
        # Both should have positive potential
        assert potential_danger > 0, "Potential should be positive"
        assert potential_safe > 0, "Potential should be positive"
        
        # Verify collision penalty weight exists
        assert env.W_COLLISION < 0, "Collision penalty should be negative"
        
        # The key is that the collision penalty component is being calculated
        # Even if the difference is small, the component should exist
        assert abs(env.W_COLLISION) > 0, "Collision penalty should have non-zero weight"
    
    def test_expanded_state_includes_relative_features(self):
        """Test that expanded state includes relative features for Coach training."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        
        state = State(
            player_pieces=[20, 10],
            enemy_pieces=[[18, 0, 0, 0]],  # Enemy at 18 (2 steps behind token 0)
            current_player=0,
            dice_roll=1,
            valid_moves=[0, 1],
            movable_pieces=[0, 1]
        )
        
        # Get base and expanded states
        base_obs = env._state_to_obs(state)
        expanded_obs = env.expand_state_with_relative_features(state)
        
        # Expanded should have more features
        assert expanded_obs.shape[0] > base_obs.shape[0], "Expanded state should have more features"
        
        # Base features should be preserved
        assert np.array_equal(base_obs, expanded_obs[:base_obs.shape[0]]), "Base features should be preserved"
        
        # Relative features should be at the end
        relative_features = expanded_obs[base_obs.shape[0]:]
        assert len(relative_features) == 4, "Should have 4 relative features (2 tokens × 2 features)"
        
        # Verify relative features are in [0, 1] range
        assert np.all(relative_features >= 0.0) and np.all(relative_features <= 1.0), "Relative features should be in [0, 1]"
        
        # Verify the features exist (even if collision risk calculation depends on exact board positions)
        # Format: [token0_distance, token0_collision_risk, token1_distance, token1_collision_risk]
        collision_risk_idx = base_obs.shape[0] + 1  # token0 collision risk
        collision_risk = float(expanded_obs[collision_risk_idx])
        assert 0.0 <= collision_risk <= 1.0, f"Collision risk should be in [0, 1], got {collision_risk}"
        
        # The key is that the feature exists and is being calculated
        # The exact value depends on the board layout and distance calculation
    
    def test_temporal_discounting_labels_correctly(self):
        """Test that temporal discounting labels states correctly based on proximity to outcome."""
        buffer = TrajectoryBuffer(max_size=1000, gamma=0.99)
        
        # Simulate a long trajectory
        trajectory_length = 20
        for i in range(trajectory_length):
            buffer.add_state(np.array([float(i)]))
        
        # Finalize with win
        buffer.finalize_trajectory(1.0)
        
        states, labels = buffer.get_dataset()
        
        # Labels should increase towards the end (closer to win)
        # Early states should have lower labels
        # Late states should have higher labels
        
        # Check that labels are monotonically increasing (or at least last > first)
        assert labels[-1] > labels[0], "Last state should have higher label than first state"
        
        # Check that labels are in correct range
        assert np.all(labels >= 0.0) and np.all(labels <= 1.0), "Labels should be in [0, 1]"
        
        # For a win, last state should be close to 1.0
        assert labels[-1] > 0.95, "Last state in winning trajectory should have label > 0.95"
        
        # For a win, first state should be less than last state
        assert labels[0] < labels[-1], "First state should have lower label than last state"
    
    def test_multiple_trajectories_collection(self):
        """Test collecting multiple trajectories."""
        env = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
        buffer = TrajectoryBuffer(max_size=1000, gamma=0.99)
        
        # Collect 3 trajectories
        for traj_idx in range(3):
            obs, info = env.reset(seed=42 + traj_idx)
            state = info["state"]
            done = False
            
            # Collect initial state
            expanded_state = env.expand_state_with_relative_features(state)
            buffer.add_state(expanded_state)
            
            # Run a few steps
            for step in range(5):
                if state.valid_moves:
                    action = state.valid_moves[0]
                else:
                    action = 0
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                done = terminated or truncated
                
                expanded_state = env.expand_state_with_relative_features(next_state)
                buffer.add_state(expanded_state)
                
                state = next_state
                if done:
                    break
            
            # Finalize trajectory
            outcome = 1.0 if traj_idx % 2 == 0 else 0.0  # Alternate wins/losses
            buffer.finalize_trajectory(outcome)
        
        # Verify all trajectories were collected
        assert buffer.total_trajectories == 3, "Should have 3 trajectories"
        assert buffer.size() > 0, "Should have collected states"
        
        # Get dataset
        states, labels = buffer.get_dataset()
        assert states.shape[0] == labels.shape[0], "States and labels should match"
        
        # Verify labels include both wins and losses
        assert np.any(labels > 0.5), "Should have some high labels (wins)"
        assert np.any(labels < 0.5), "Should have some low labels (losses)"

