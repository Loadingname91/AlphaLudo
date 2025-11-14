"""
Pillar 2: RewardShaper (Strategy Pattern)

Implements Strategy design pattern for reward logic, decoupling environment from reward heuristics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class RewardShaper(ABC):
    """
    Abstract base class for reward shaping strategies.
    
    Implements Strategy pattern to allow switching reward heuristics without modifying environment code.
    """
    
    def __init__(self, schema: str):
        """
        Initialize reward shaper.
        
        Args:
            schema: String identifier for the reward schema (e.g., 'sparse', 'dense', 'decoupled-ila')
        """
        self.schema = schema
    
    @abstractmethod
    def get_reward(self, game_events: Dict) -> Tuple[float, Dict]:
        """
        Compute reward from game events.
        
        Args:
            game_events: Dictionary containing game event information such as:
                - 'player_won': bool
                - 'opponent_won': bool
                - 'piece_moved': bool
                - 'piece_captured': bool
                - 'piece_entered_home': bool
                - 'dice_roll': int
                - etc.
        
        Returns:
            Tuple of (net_reward, ila_components_dict):
                - net_reward: Total reward value (float)
                - ila_components: Dictionary of Individual Learning Algorithm components (empty for simple strategies)
        """
        pass


class SparseReward(RewardShaper):
    """
    Sparse reward strategy: reward only on win/loss.
    
    Returns +100 for win, -100 for loss, 0 otherwise.
    """
    
    def __init__(self):
        super().__init__(schema='sparse')
        self.win_reward = 100.0
        self.loss_reward = -100.0
    
    def get_reward(self, game_events: Dict) -> Tuple[float, Dict]:
        """
        Compute sparse reward.
        
        Args:
            game_events: Dictionary containing game events
        
        Returns:
            Tuple of (reward, empty_dict)
        """
        if game_events.get('player_won', False):
            return self.win_reward, {}
        elif game_events.get('opponent_won', False):
            return self.loss_reward, {}
        else:
            return 0.0, {}


# Factory function for creating reward shapers
def create_reward_shaper(schema: str) -> RewardShaper:
    """
    Factory function to create reward shapers based on schema name.
    
    Args:
        schema: Name of the reward schema ('sparse', 'dense', 'decoupled-ila')
    
    Returns:
        RewardShaper instance
    
    Raises:
        ValueError: If schema is not recognized
    """
    strategies = {
        'sparse': SparseReward,
        # 'dense': DenseReward,  # To be implemented in Phase 2
        # 'decoupled-ila': ILAReward,  # To be implemented in Phase 2
    }
    
    if schema not in strategies:
        raise ValueError(f"Unknown reward schema: {schema}. Available: {list(strategies.keys())}")
    
    return strategies[schema]()
