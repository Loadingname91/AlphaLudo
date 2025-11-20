"""
Pillar 3: Agent Interface (Abstract Base Class)

Defines the interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict
from ..utils.state import State


class Agent(ABC):
    """
    Abstract base class for all RL agents.
    
    All agents must implement the act() method and learning methods appropriate to their type.
    """
    
    @property
    @abstractmethod
    def is_on_policy(self) -> bool:
        """
        Whether this agent uses on-policy learning.
        
        Returns:
            True for on-policy (PPO, MCTS), False for off-policy (Q-Learning, DQN)
        """
        pass
    
    @property
    @abstractmethod
    def needs_replay_learning(self) -> bool:
        """
        Whether this agent requires replay buffer for learning.
        
        Returns:
            True if agent needs experience replay (Q-Learning, DQN), False otherwise
        """
        pass

    # ---- Optional debugging hooks -------------------------------------------------

    @property
    def supports_score_debug(self) -> bool:
        """
        Whether this agent exposes detailed scoring debug information.

        Default: False. Override in agents that can provide score breakdowns.
        """
        return False

    def get_last_score_debug(self) -> Dict[str, Any] | None:
        """
        Get debug information for the last action selection, if available.

        Default implementation returns None. Agents that support score debugging
        can override this to return a JSON-serializable dictionary with
        per-move score components and context.
        """
        return None
    
    @abstractmethod
    def act(self, state: State) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current State object
        
        Returns:
            Action index (must be in state.valid_moves)
        """
        pass
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """
        Learn from experience replay buffer.
        
        Default implementation does nothing (for agents that don't need replay).
        Should be overridden by off-policy agents (Q-Learning, DQN).
        """
        pass
    
    def learn_from_rollout(self, rollout_buffer: List[Dict], *args, **kwargs) -> None:
        """
        Learn from a rollout buffer (on-policy learning).
        
        Default implementation does nothing (for agents that don't need rollouts).
        Should be overridden by on-policy agents (PPO).
        
        Args:
            rollout_buffer: List of experience dictionaries from a rollout
        """
        pass
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float, 
                             next_state: State, done: bool, **kwargs) -> None:
        """
        Add experience to replay buffer.
        
        Default implementation does nothing (for agents without replay buffer).
        Should be overridden by agents that use experience replay (DQN).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        pass
    
    def save(self, filepath: str) -> None:
        """
        Save agent model/parameters to file.
        
        Default implementation does nothing.
        Should be overridden by agents that need to save state.
        
        Args:
            filepath: Path to save file
        """
        pass
    
    def load(self, filepath: str) -> None:
        """
        Load agent model/parameters from file.
        
        Default implementation does nothing.
        Should be overridden by agents that need to load state.
        
        Args:
            filepath: Path to load file from
        """
        pass
