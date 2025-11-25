"""
Prioritized Experience Replay Buffer

Samples transitions with probability proportional to TD-error.
Uses importance sampling to correct for bias.
"""

import numpy as np
import random
from typing import List, Tuple, Optional
from .sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer with importance sampling.
    """
    
    def __init__(
        self,
        capacity: int = 80000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-6
    ):
        """
        Initialize PER buffer.
        
        Args:
            capacity: Maximum number of transitions
            alpha: Prioritization exponent (0=uniform, 1=full priority)
            beta: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            epsilon: Small constant to ensure non-zero priority
        """
        self.capacity = capacity
        self.alpha = float(alpha)  # Ensure float type
        self.beta = float(beta)  # Ensure float type
        self.beta_end = float(beta_end)  # Ensure float type
        self.epsilon = float(epsilon)  # Ensure float type (fixes string conversion issue)
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0  # Initial priority for new transitions
        self.step = 0
    
    def add(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.
        
        Args:
            state: Current state (31-dim vector)
            action: Action taken
            reward: Reward received
            next_state: Next state (31-dim vector)
            done: Whether episode terminated
        """
        # Use max priority for new transitions
        priority = self.max_priority
        transition = (state, action, reward, next_state, done)
        self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch of transitions.
        
        Returns:
            batch: List of transitions
            indices: Indices of sampled transitions
            weights: Importance sampling weights
        """
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total / batch_size
        
        # Calculate current beta (linearly annealed)
        beta = self.beta + (self.beta_end - self.beta) * (self.step / 100000)
        beta = min(beta, self.beta_end)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)
        
        # Calculate importance sampling weights
        probabilities = np.array(priorities) / self.tree.total
        weights = np.power(self.capacity * probabilities, -beta)
        weights = weights / weights.max()  # Normalize
        
        self.step += 1
        return batch, np.array(indices), weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of transitions to update
            td_errors: TD-errors for those transitions
        """
        # Ensure epsilon is float (handle YAML string conversion)
        epsilon = float(self.epsilon)
        alpha = float(self.alpha)
        
        priorities = np.abs(td_errors) + epsilon
        priorities = np.power(priorities, alpha)
        
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    @property
    def size(self) -> int:
        """Current number of transitions in buffer."""
        return self.tree.n_entries

