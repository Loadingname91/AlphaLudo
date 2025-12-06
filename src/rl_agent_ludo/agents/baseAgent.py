"""
Base Agent Interface.

All agents should inherit from this class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Optional
from ..utils.state import State


class Agent(ABC):
    """
    Base class for all RL agents.
    
    All agents must implement:
    - act(state: State) -> int: Select action given current state
    - push_to_replay_buffer(...): Store experience (if using replay)
    - learn_from_replay(...): Learn from stored experiences (if off-policy)
    - learn_from_rollout(...): Learn from episode rollout (if on-policy)
    """
    
    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        """Return True if agent uses on-policy learning (e.g., policy gradient)."""
        pass
    
    @property
    @abstractmethod
    def needs_replay_learning(self) -> bool:
        """Return True if agent uses experience replay (e.g., DQN, Q-Learning)."""
        pass
    
    @abstractmethod
    def act(self, state: State) -> int:
        """
        Select action given current state.
        
        Args:
            state: Current State object
            
        Returns:
            Action index (into state.valid_moves or state.movable_pieces)
        """
        pass
    
    def push_to_replay_buffer(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        **kwargs
    ) -> None:
        """
        Store experience in replay buffer (if applicable).
        
        Default: No-op (for agents that don't use replay).
        """
        pass
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """
        Learn from replay buffer (if applicable).
        
        Default: No-op (for agents that don't use replay).
        """
        pass
    
    def learn_from_rollout(self, rollout_buffer: list, *args, **kwargs) -> None:
        """
        Learn from episode rollout (if applicable).
        
        Default: No-op (for agents that don't use on-policy learning).
        """
        pass
    
    def on_episode_end(self) -> None:
        """
        Called at the end of each episode.
        
        Use for epsilon decay, episode statistics, etc.
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save agent state to file.
        
        Default: No-op (implement in subclasses that need checkpointing).
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load agent state from file.
        
        Default: No-op (implement in subclasses that need checkpointing).
        """
        pass

