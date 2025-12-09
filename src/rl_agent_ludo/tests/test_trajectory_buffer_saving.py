"""
Unit tests for Trajectory Buffer Saving functionality.

Tests:
1. Trajectory buffer data is saved correctly
2. Saved data can be loaded and contains correct information
3. Metadata is preserved
4. Edge cases (empty buffer, no collection, etc.)
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from rl_agent_ludo.agents.trajectoryBuffer import TrajectoryBuffer


class TestTrajectoryBufferSaving:
    """Test trajectory buffer saving and loading functionality."""
    
    def test_save_and_load_trajectory_data(self):
        """Test that trajectory data can be saved and loaded correctly."""
        # Create a temporary directory for testing
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create trajectory buffer and add some data
            buffer = TrajectoryBuffer(max_size=100, gamma=0.99)
            
            # Add a trajectory with 5 states
            for i in range(5):
                state = np.array([float(i), float(i+1), float(i+2)], dtype=np.float32)
                buffer.add_state(state)
            
            # Finalize with win
            buffer.finalize_trajectory(1.0)
            
            # Get dataset
            states, labels = buffer.get_dataset()
            
            # Save to file
            save_path = temp_dir / "test_trajectories.npz"
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
                gamma=np.array([buffer.gamma], dtype=np.float32),
                seed=np.array([42], dtype=np.int64),
                num_episodes=np.array([100], dtype=np.int64),
                trajectory_interval=np.array([1], dtype=np.int64),
            )
            
            # Verify file exists
            assert save_path.exists(), "Trajectory file should be created"
            
            # Load and verify
            loaded_data = np.load(save_path)
            
            loaded_states = loaded_data['states']
            loaded_labels = loaded_data['labels']
            
            # Verify states match
            assert np.array_equal(loaded_states, states), "Loaded states should match saved states"
            assert np.array_equal(loaded_labels, labels), "Loaded labels should match saved labels"
            
            # Verify metadata
            assert int(loaded_data['num_samples'][0]) == len(states)
            assert int(loaded_data['state_dim'][0]) == state_dim
            assert abs(float(loaded_data['gamma'][0]) - 0.99) < 0.001, "Gamma should be approximately 0.99 (float32 precision)"
            assert int(loaded_data['seed'][0]) == 42
            assert int(loaded_data['num_episodes'][0]) == 100
            assert int(loaded_data['trajectory_interval'][0]) == 1
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    def test_save_empty_buffer(self):
        """Test that empty buffer doesn't cause errors."""
        buffer = TrajectoryBuffer(max_size=100, gamma=0.99)
        
        # Try to get dataset from empty buffer
        states, labels = buffer.get_dataset()
        
        assert len(states) == 0, "Empty buffer should return empty states"
        assert len(labels) == 0, "Empty buffer should return empty labels"
        
        # Should not raise error
        assert states.shape == (0,), "Empty states should have shape (0,)"
        assert labels.shape == (0,), "Empty labels should have shape (0,)"
    
    def test_save_multiple_trajectories(self):
        """Test saving buffer with multiple trajectories."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            buffer = TrajectoryBuffer(max_size=1000, gamma=0.99)
            
            # Add 3 trajectories
            for traj_idx in range(3):
                for i in range(4):
                    state = np.array([float(traj_idx * 10 + i), float(i)], dtype=np.float32)
                    buffer.add_state(state)
                # Alternate win/loss
                buffer.finalize_trajectory(1.0 if traj_idx % 2 == 0 else 0.0)
            
            # Get dataset
            states, labels = buffer.get_dataset()
            
            # Verify we have all states
            assert len(states) == 12, "Should have 12 states (3 trajectories × 4 states)"
            assert len(labels) == 12, "Should have 12 labels"
            
            # Save
            save_path = temp_dir / "test_multiple_trajectories.npz"
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
                gamma=np.array([buffer.gamma], dtype=np.float32),
            )
            
            # Load and verify
            loaded_data = np.load(save_path)
            loaded_states = loaded_data['states']
            loaded_labels = loaded_data['labels']
            
            assert len(loaded_states) == 12
            assert len(loaded_labels) == 12
            assert np.array_equal(loaded_states, states)
            assert np.array_equal(loaded_labels, labels)
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_large_buffer(self):
        """Test saving a large buffer (tests compression)."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            buffer = TrajectoryBuffer(max_size=10000, gamma=0.99)
            
            # Add a large trajectory
            for i in range(100):
                state = np.random.rand(36).astype(np.float32)  # 36-dim state
                buffer.add_state(state)
            
            buffer.finalize_trajectory(1.0)
            
            states, labels = buffer.get_dataset()
            
            # Save compressed
            save_path = temp_dir / "test_large_buffer.npz"
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
            )
            
            # Verify file exists and is reasonable size (compressed)
            assert save_path.exists()
            file_size = save_path.stat().st_size
            # Should be compressed, so less than uncompressed size
            # Uncompressed would be ~100 * 36 * 4 bytes (states) + 100 * 4 bytes (labels) = ~14.4 KB
            # Compressed should be smaller
            assert file_size > 0, "File should have content"
            assert file_size < 100000, "Compressed file should be reasonable size"
            
            # Load and verify
            loaded_data = np.load(save_path)
            loaded_states = loaded_data['states']
            loaded_labels = loaded_data['labels']
            
            assert len(loaded_states) == 100
            assert loaded_states.shape[1] == 36
            assert np.allclose(loaded_states, states), "Large states should match"
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_metadata_preservation(self):
        """Test that all metadata is correctly preserved."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            buffer = TrajectoryBuffer(max_size=100, gamma=0.95)
            
            # Add some data
            for i in range(3):
                buffer.add_state(np.array([float(i)], dtype=np.float32))
            buffer.finalize_trajectory(1.0)
            
            states, labels = buffer.get_dataset()
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            # Save with metadata
            save_path = temp_dir / "test_metadata.npz"
            test_seed = 12345
            test_episodes = 50000
            test_interval = 150
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
                gamma=np.array([buffer.gamma], dtype=np.float32),
                seed=np.array([test_seed], dtype=np.int64),
                num_episodes=np.array([test_episodes], dtype=np.int64),
                trajectory_interval=np.array([test_interval], dtype=np.int64),
            )
            
            # Load and verify all metadata
            loaded_data = np.load(save_path)
            
            assert int(loaded_data['num_samples'][0]) == 3
            assert int(loaded_data['state_dim'][0]) == 1
            assert abs(float(loaded_data['gamma'][0]) - 0.95) < 0.001
            assert int(loaded_data['seed'][0]) == test_seed
            assert int(loaded_data['num_episodes'][0]) == test_episodes
            assert int(loaded_data['trajectory_interval'][0]) == test_interval
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_with_temporal_discounting_labels(self):
        """Test that temporally discounted labels are correctly saved."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            buffer = TrajectoryBuffer(max_size=100, gamma=0.9)  # Use 0.9 for easier calculation
            
            # Add 3 states
            for i in range(3):
                buffer.add_state(np.array([float(i)], dtype=np.float32))
            
            # Finalize with win
            buffer.finalize_trajectory(1.0)
            
            states, labels = buffer.get_dataset()
            
            # Verify temporal discounting was applied
            # For t=0, T=2, γ=0.9, outcome=1.0:
            # Label_0 = 1.0 * 0.9^2 + 0.5 * (1 - 0.9^2) = 0.81 + 0.095 = 0.905
            # For t=2, T=2: Label_2 = 1.0 * 0.9^0 + 0.5 * (1 - 0.9^0) = 1.0
            expected_label_0 = 1.0 * (0.9 ** 2) + 0.5 * (1 - (0.9 ** 2))
            expected_label_2 = 1.0
            
            assert abs(labels[0] - expected_label_0) < 0.001, f"Label[0] should be {expected_label_0}, got {labels[0]}"
            assert abs(labels[2] - expected_label_2) < 0.001, f"Label[2] should be {expected_label_2}, got {labels[2]}"
            
            # Save
            save_path = temp_dir / "test_labels.npz"
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
            )
            
            # Load and verify labels are preserved
            loaded_data = np.load(save_path)
            loaded_labels = loaded_data['labels']
            
            assert np.allclose(loaded_labels, labels), "Loaded labels should match saved labels"
            assert abs(loaded_labels[0] - expected_label_0) < 0.001
            assert abs(loaded_labels[2] - expected_label_2) < 0.001
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_save_buffer_at_max_capacity(self):
        """Test saving buffer that has reached max capacity (FIFO behavior)."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            buffer = TrajectoryBuffer(max_size=10, gamma=0.99)
            
            # Add more states than max_size
            for i in range(20):
                buffer.add_state(np.array([float(i)], dtype=np.float32))
            buffer.finalize_trajectory(1.0)
            
            # Buffer should be capped at max_size
            assert buffer.size() == 10, "Buffer should be capped at max_size"
            
            # Get dataset (should only have 10 states)
            states, labels = buffer.get_dataset()
            
            assert len(states) == 10, "Dataset should only contain max_size states"
            assert len(labels) == 10
            
            # Save
            save_path = temp_dir / "test_max_capacity.npz"
            state_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
            
            np.savez_compressed(
                save_path,
                states=states,
                labels=labels,
                num_samples=np.array([len(states)], dtype=np.int64),
                state_dim=np.array([state_dim], dtype=np.int64),
            )
            
            # Load and verify
            loaded_data = np.load(save_path)
            assert len(loaded_data['states']) == 10
            assert len(loaded_data['labels']) == 10
            
        finally:
            shutil.rmtree(temp_dir)

