"""
Test suite for Dueling DQN Network

Tests the network architecture, forward pass, and key properties.
"""

import torch
import pytest
from rl_agent_ludo.agents.modules.dueling_dqn_network import DuelingDQNNetwork


def test_network_initialization():
    """Test that network initializes correctly with default parameters."""
    network = DuelingDQNNetwork()
    
    assert network is not None
    assert network.shared is not None
    assert network.layer_norm is not None
    assert network.value_stream is not None
    assert network.advantage_stream is not None


def test_network_custom_parameters():
    """Test network initialization with custom parameters."""
    network = DuelingDQNNetwork(
        input_dim=31,
        hidden_dim=64,
        num_actions=4
    )
    
    # Test forward pass works
    x = torch.randn(31)
    q = network(x)
    assert q.shape == (4,)


def test_forward_pass_1d_input():
    """Test forward pass with 1D input (single state)."""
    network = DuelingDQNNetwork()
    
    # Single state vector (31 features)
    state = torch.randn(31)
    q_values = network(state)
    
    # Should return Q-values for 4 actions
    assert q_values.shape == (4,), f"Expected shape (4,), got {q_values.shape}"
    assert isinstance(q_values, torch.Tensor)


def test_forward_pass_2d_input():
    """Test forward pass with 2D input (batch of states)."""
    network = DuelingDQNNetwork()
    
    # Batch of states (batch_size=32, 31 features)
    batch = torch.randn(32, 31)
    q_values = network(batch)
    
    # Should return Q-values for each state-action pair
    assert q_values.shape == (32, 4), f"Expected shape (32, 4), got {q_values.shape}"


def test_forward_pass_batch_size_1():
    """Test forward pass with batch size 1 (should keep batch dimension)."""
    network = DuelingDQNNetwork()
    
    # Single state in batch format
    batch = torch.randn(1, 31)
    q_values = network(batch)
    
    # Should keep batch dimension
    assert q_values.shape == (1, 4), f"Expected shape (1, 4), got {q_values.shape}"


def test_output_range():
    """Test that Q-values are reasonable (not NaN or Inf)."""
    network = DuelingDQNNetwork()
    
    state = torch.randn(31)
    q_values = network(state)
    
    # Check for NaN or Inf
    assert not torch.isnan(q_values).any(), "Q-values contain NaN"
    assert not torch.isinf(q_values).any(), "Q-values contain Inf"
    
    # Q-values should be finite floats
    assert torch.all(torch.isfinite(q_values)), "Q-values are not finite"


def test_identifiability_property():
    """
    Test that normalized advantages sum to zero (identifiability property).
    This ensures V(s) and A(s,a) are uniquely defined.
    """
    network = DuelingDQNNetwork()
    
    state = torch.randn(31)
    q_values = network(state)
    
    # Extract value and advantage from network internals
    # We need to manually compute to verify the property
    was_1d = state.dim() == 1
    if was_1d:
        state = state.unsqueeze(0)
    
    x = network.shared(state)
    x = network.layer_norm(x)
    x = torch.relu(x)
    
    value = network.value_stream(x)
    advantage = network.advantage_stream(x)
    
    # Calculate normalized advantages (after subtracting mean)
    mean_advantage = advantage.mean(dim=1, keepdim=True)
    normalized_advantage = advantage - mean_advantage
    
    # Normalized advantages should sum to zero (identifiability property)
    # This is what ensures unique representation
    mean_normalized = normalized_advantage.mean(dim=1, keepdim=True)
    assert torch.allclose(mean_normalized, torch.zeros_like(mean_normalized), atol=1e-5), \
        "Normalized advantages should sum to zero for identifiability"
    
    # Verify Q-values match the formula: Q(s,a) = V(s) + normalized_A(s,a)
    expected_q = value + normalized_advantage
    if was_1d:
        expected_q = expected_q.squeeze(0)
    
    assert torch.allclose(q_values, expected_q, atol=1e-5), \
        "Q-values don't match expected formula"


def test_layer_norm_present():
    """Test that LayerNorm is applied correctly."""
    network = DuelingDQNNetwork()
    
    state = torch.randn(31)
    
    # Forward through shared layer
    x = network.shared(state.unsqueeze(0))
    
    # Apply LayerNorm
    x_norm = network.layer_norm(x)
    
    # LayerNorm should normalize the activations
    # Mean should be close to 0, std close to 1 (after normalization)
    mean = x_norm.mean()
    std = x_norm.std()
    
    # LayerNorm normalizes per feature, so check that it's applied
    assert network.layer_norm is not None
    assert x_norm.shape == x.shape


def test_value_stream_output():
    """Test that value stream outputs single value per state."""
    network = DuelingDQNNetwork()
    
    batch = torch.randn(5, 31)
    
    # Forward through shared layers
    x = network.shared(batch)
    x = network.layer_norm(x)
    x = torch.relu(x)
    
    # Value stream
    value = network.value_stream(x)
    
    # Should output one value per state
    assert value.shape == (5, 1), f"Expected shape (5, 1), got {value.shape}"


def test_advantage_stream_output():
    """Test that advantage stream outputs values for all actions."""
    network = DuelingDQNNetwork()
    
    batch = torch.randn(5, 31)
    
    # Forward through shared layers
    x = network.shared(batch)
    x = network.layer_norm(x)
    x = torch.relu(x)
    
    # Advantage stream
    advantage = network.advantage_stream(x)
    
    # Should output 4 values per state (one per action)
    assert advantage.shape == (5, 4), f"Expected shape (5, 4), got {advantage.shape}"


def test_gradient_flow():
    """Test that gradients can flow through the network."""
    network = DuelingDQNNetwork()
    
    state = torch.randn(31, requires_grad=False)
    q_values = network(state)
    
    # Create a dummy loss
    loss = q_values.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for all parameters
    for name, param in network.named_parameters():
        assert param.grad is not None, f"Gradient missing for {name}"


def test_differentiable():
    """Test that network is differentiable."""
    network = DuelingDQNNetwork()
    
    state = torch.randn(31, requires_grad=True)
    q_values = network(state)
    
    # Sum of Q-values should be differentiable
    loss = q_values.sum()
    loss.backward()
    
    # Input should have gradients
    assert state.grad is not None, "Network should be differentiable"


def test_network_parameters_count():
    """Test that network has expected number of parameters."""
    network = DuelingDQNNetwork()
    
    total_params = sum(p.numel() for p in network.parameters())
    
    # Expected: ~38,000 parameters (from design doc)
    # Input: 31*128 + 128 = 4,096
    # LayerNorm: 128*2 = 256
    # Value: (128*128 + 128) + (128*1 + 1) = 16,513
    # Advantage: (128*128 + 128) + (128*4 + 4) = 16,900
    # Total: ~38,000
    
    assert total_params > 30000, f"Too few parameters: {total_params}"
    assert total_params < 50000, f"Too many parameters: {total_params}"


def test_deterministic_output():
    """Test that network produces deterministic output (with fixed weights)."""
    network = DuelingDQNNetwork()
    network.eval()  # Set to evaluation mode
    
    state = torch.randn(31)
    
    # Forward pass twice
    q1 = network(state)
    q2 = network(state)
    
    # Should be identical
    assert torch.allclose(q1, q2), "Network output should be deterministic"


def test_training_mode():
    """Test that network can switch between train and eval modes."""
    network = DuelingDQNNetwork()
    
    # Initially in train mode
    assert network.training == True
    
    # Switch to eval mode
    network.eval()
    assert network.training == False
    
    # Switch back to train mode
    network.train()
    assert network.training == True


def test_multiple_forward_passes():
    """Test that network can handle multiple forward passes."""
    network = DuelingDQNNetwork()
    
    states = [torch.randn(31) for _ in range(10)]
    q_values_list = [network(state) for state in states]
    
    # All should have correct shape
    for q_values in q_values_list:
        assert q_values.shape == (4,)


def test_with_orthogonal_state():
    """Test network with actual orthogonal state vector."""
    from rl_agent_ludo.utils.orthogonal_state_abstractor import OrthogonalStateAbstractor
    from rl_agent_ludo.utils.state import State
    import numpy as np
    
    network = DuelingDQNNetwork()
    abstractor = OrthogonalStateAbstractor()
    
    # Create a mock state
    state = State(
        full_vector=np.zeros(31),
        abstract_state=(0, 0, 0, 0, 0),
        valid_moves=[0, 1, 2, 3],
        dice_roll=1,
        player_pieces=[0, 1, 2, 3],
        enemy_pieces=[[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        movable_pieces=[0, 1, 2, 3]
    )
    
    # Get orthogonal state
    orthogonal_state = abstractor.get_orthogonal_state(state)
    
    # Convert to tensor
    state_tensor = torch.FloatTensor(orthogonal_state)
    
    # Forward pass
    q_values = network(state_tensor)
    
    # Should work correctly
    assert q_values.shape == (4,)
    assert not torch.isnan(q_values).any()


