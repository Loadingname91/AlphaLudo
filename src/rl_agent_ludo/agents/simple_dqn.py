"""
Simple DQN agent for curriculum learning.

A clean, minimal DQN implementation focused on stability and simplicity.
No bells and whistles - just core DQN with experience replay and target network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, Optional
import random


class DQNNetwork(nn.Module):
    """Simple feedforward Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)


class SimpleDQNAgent:
    """
    Simple DQN agent with experience replay and target network.

    Features:
    - Experience replay buffer
    - Target network (updated periodically)
    - Epsilon-greedy exploration
    - No fancy tricks (just core DQN)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_buffer_size: int = 50000,
        batch_size: int = 128,
        target_update_frequency: int = 1000,
        hidden_dims: list = [128, 128],
        device: str = 'cpu',
        seed: int = 42,
    ):
        """Initialize DQN agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.device = device

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Q-network and target network
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        self.total_loss = 0.0
        self.loss_count = 0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            greedy: If True, use greedy policy (epsilon=0)

        Returns:
            action: Selected action
        """
        # Epsilon-greedy exploration
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        # Greedy action from Q-network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if enough samples available).

        Returns:
            loss: Training loss, or None if not enough samples
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values (using target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss (MSE)
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update statistics
        self.step_count += 1
        self.total_loss += loss.item()
        self.loss_count += 1

        # Update target network periodically
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """Decay epsilon (call at end of episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_avg_loss(self) -> float:
        """Get average loss since last call."""
        if self.loss_count == 0:
            return 0.0
        avg_loss = self.total_loss / self.loss_count
        self.total_loss = 0.0
        self.loss_count = 0
        return avg_loss

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
