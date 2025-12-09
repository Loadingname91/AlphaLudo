"""
Win Probability Network (WPN) - The "Coach" Component.

A lightweight neural network that learns to predict win probability from game states.
This network is trained on Monte Carlo data (full game outcomes) and provides
the Potential function Φ(s) for Potential-Based Reward Shaping (PBRS).

Architecture:
- Input: Unified feature vector (28 floats for 2 tokens, 46 floats for 4 tokens)
- Output: Single probability value [0, 1] representing P(Win | State)
- Smaller than the DQN (fewer parameters) for faster training

Training Data:
- States visited during games
- Labeled with final game outcome (1.0 for win, 0.0 for loss)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np


class WinProbabilityNetwork(nn.Module):
    """
    Lightweight neural network for predicting win probability.
    
    Architecture:
    - Input: Unified feature vector (same as DQN input)
    - Hidden layers: Smaller than DQN (e.g., [128, 64] vs [256, 256, 128])
    - Output: Single sigmoid output [0, 1] representing P(Win | State)
    
    Optimized for:
    - Fast training (fewer parameters)
    - Stable predictions (sigmoid output)
    - CPU efficiency
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], device: Optional[str] = None):
        """
        Initialize Win Probability Network.
        
        Args:
            input_dim: Dimension of input feature vector (28 for 2 tokens, 46 for 4 tokens)
            hidden_dims: List of hidden layer dimensions (default: [128, 64] - smaller than DQN)
            device: Device to run on ('cpu' or 'cuda')
        """
        super(WinProbabilityNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer: Single sigmoid output for win probability
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Ensures output is in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict win probability.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
            
        Returns:
            Win probability tensor of shape (batch_size, 1) or (1,)
            Values are in [0, 1] range
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
        
        return self.network(x)
    
    def predict(self, state_vector: np.ndarray) -> float:
        """
        Predict win probability for a single state.
        
        Args:
            state_vector: NumPy array of shape (input_dim,)
            
        Returns:
            Win probability as float in [0, 1]
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).to(self.device)
            prob = self.forward(state_tensor)
            return prob.item()
    
    def train_on_batch(
        self,
        states: List[np.ndarray],
        outcomes: List[float],
        optimizer: optim.Optimizer,
        batch_size: int = 64,
        epochs: int = 1
    ) -> float:
        """
        Train the network on a batch of (state, outcome) pairs.
        
        Args:
            states: List of state vectors (each is np.ndarray of shape (input_dim,))
            outcomes: List of outcomes (1.0 for win, 0.0 for loss)
            optimizer: PyTorch optimizer (e.g., Adam)
            batch_size: Batch size for training
            epochs: Number of epochs to train
            
        Returns:
            Average loss over the training batch
        """
        if len(states) == 0:
            return 0.0
        
        self.train()
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        outcomes_tensor = torch.FloatTensor(np.array(outcomes)).unsqueeze(1).to(self.device)
        
        total_loss = 0.0
        num_batches = 0
        
        # Train for specified epochs
        for epoch in range(epochs):
            # Shuffle data
            indices = torch.randperm(len(states_tensor))
            states_shuffled = states_tensor[indices]
            outcomes_shuffled = outcomes_tensor[indices]
            
            # Mini-batch training
            for i in range(0, len(states_tensor), batch_size):
                batch_states = states_shuffled[i:i + batch_size]
                batch_outcomes = outcomes_shuffled[i:i + batch_size]
                
                # Forward pass
                predictions = self.forward(batch_states)
                
                # Binary cross-entropy loss
                loss = nn.functional.binary_cross_entropy(predictions, batch_outcomes)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save(self, filepath: str) -> None:
        """Save network state to file."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1],  # Exclude output layer
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load network state from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        # Note: input_dim and hidden_dims should match, but we don't recreate the network here

