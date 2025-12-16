"""
T-REX Agent: DQN trained with learned reward function.

This agent uses the reward network learned from trajectory preferences

Key difference from SimpleDQNAgent:
- Uses learned_reward = learned_reward_scale * learned_reward instead of env_reward 
based on either hybrid or pure T-REX approach.
- Otherwise identical DQN training with experience replay and target network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Tuple, Optional
from pathlib import Path
import random


class DQNNetwork(nn.Module):
    """Simple feedforward Q-network (same as SimpleDQNAgent)."""

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


class RewardNetwork(nn.Module):
    """
    Learned reward network (must match training architecture).

    This is a copy of the architecture from reward_network.py.
    """

    def __init__(self, state_dim: int = 16, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Scalar reward output
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns scalar reward for given state."""
        return self.network(state)


class TREXAgent:
    """
    T-REX Agent: DQN that learns from preference-based reward function.

    Key Innovation:
    - Replaces sparse env rewards (win=+1, lose=0) with dense learned rewards
    - Learned rewards provide meaningful feedback at every step
    - Otherwise uses standard DQN algorithm
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        reward_network_path: str,
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
        use_hybrid_rewards: bool = True,
        learned_reward_scale: float = 10.0,
        learned_reward_weight: float = 0.3,
    ):
        """
        Initialize T-REX agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            reward_network_path: Path to trained reward network checkpoint
            use_hybrid_rewards: If True, combine env + learned rewards (recommended!)
            learned_reward_scale: Scale factor for learned rewards (default: 10.0)
            learned_reward_weight: Weight for learned rewards in hybrid (default: 0.3)
            ... (other args same as SimpleDQNAgent)
        """
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

        # Hybrid reward settings
        self.use_hybrid_rewards = use_hybrid_rewards
        self.learned_reward_scale = learned_reward_scale
        self.learned_reward_weight = learned_reward_weight

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Load learned reward network
        print(f"\n{'='*70}")
        print("Loading learned reward network...")
        print(f"{'='*70}")

        self.reward_net = RewardNetwork(state_dim=state_dim, hidden_dim=128).to(device)

        # Load trained weights
        checkpoint = torch.load(reward_network_path, map_location=device)

        # Handle both reward_learner checkpoint and reward_network checkpoint
        if 'state_dict' in checkpoint:
            # Full learner checkpoint
            self.reward_net.load_state_dict(checkpoint['state_dict'])
        else:
            # Direct network checkpoint
            self.reward_net.load_state_dict(checkpoint)

        self.reward_net.eval()  # Set to eval mode (no dropout during inference)

        print(f"✅ Loaded reward network from {reward_network_path}")
        print(f"{'='*70}\n")

        # Q-network and target network (standard DQN)
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

        # T-REX specific statistics
        self.total_learned_reward = 0.0
        self.learned_reward_count = 0

    def get_learned_reward(self, state: np.ndarray) -> float:
        """
        Get learned reward for a state using the reward network.

        Args:
            state: Current state

        Returns:
            learned_reward: Scalar reward predicted by reward network
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            learned_reward = self.reward_net(state_tensor).item()

        # Track statistics
        self.total_learned_reward += learned_reward
        self.learned_reward_count += 1

        return learned_reward

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
        env_reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay buffer with hybrid reward.

        HYBRID APPROACH (use_hybrid_rewards=True):
        - Combines environment reward (sparse but accurate) with learned reward (dense but noisy)
        - total_reward = env_reward + weight * scale * learned_reward
        - Best of both worlds: clear win/loss signal + dense feedback

        PURE T-REX (use_hybrid_rewards=False):
        - Uses only learned reward (original T-REX approach)
        - Can fail if learned rewards are poorly calibrated

        Args:
            state: Current state
            action: Action taken
            env_reward: Environment reward (win/loss/capture signals)
            next_state: Next state
            done: Episode done flag
        """
        # Get learned reward from reward network
        learned_reward = self.get_learned_reward(state)

        if self.use_hybrid_rewards:

            # HYBRID REWARDS: Combine env + learned rewards since some of the rewards are too minor to be  useful.
            # Scale learned reward to reasonable range (0.3 → 3.0 with scale=10)
            scaled_learned = learned_reward * self.learned_reward_scale

            # Weighted combination
            # - env_reward provides clear win/loss signal
            # - learned_reward provides dense step-by-step feedback
            total_reward = env_reward + self.learned_reward_weight * scaled_learned
        else:
            # PURE T-REX: Use only learned reward (original implementation)
            total_reward = learned_reward

        # Store transition
        self.replay_buffer.append((state, action, total_reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (standard DQN, no changes).

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
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)  # These are LEARNED rewards!
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

    def get_avg_learned_reward(self) -> float:
        """Get average learned reward since last call."""
        if self.learned_reward_count == 0:
            return 0.0
        avg_reward = self.total_learned_reward / self.learned_reward_count
        self.total_learned_reward = 0.0
        self.learned_reward_count = 0
        return avg_reward

    def save(self, filepath: str):
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

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
