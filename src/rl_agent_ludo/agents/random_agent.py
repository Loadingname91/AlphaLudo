"""
RandomAgent: Baseline agent for Phase 0

Selects random valid actions to establish baseline win rate (~25%).
"""

import random
from typing import Optional
from .base_agent import Agent
from ..utils.state import State


class RandomAgent(Agent):
    """
    Random agent that selects random valid actions.
    
    Used as baseline in Phase 0 to validate Trainer, LudoEnv, and MetricsTracker.
    Expected win rate: ~25% (1/4 players).
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random agent.
        
        Args:
            seed: Random seed for reproducibility (optional)
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
    
    @property
    def is_on_policy(self) -> bool:
        """Random agent doesn't learn, so this is not applicable."""
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        """Random agent doesn't learn."""
        return False
    
    def act(self, state: State) -> int:
        """
        Select a random valid action.
        
        Args:
            state: Current State object
        
        Returns:
            Random action index from state.valid_moves
        """
        if not state.valid_moves:
            # Should not happen, but handle gracefully
            return 0
        
        return random.choice(state.valid_moves)
    
    def learn_from_replay(self, *args, **kwargs) -> None:
        """Random agent doesn't learn."""
        pass
    
    def learn_from_rollout(self, rollout_buffer: list, *args, **kwargs) -> None:
        """Random agent doesn't learn."""
        pass
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float,
                             next_state: State, done: bool, **kwargs) -> None:
        """Random agent doesn't use replay buffer."""
        pass
