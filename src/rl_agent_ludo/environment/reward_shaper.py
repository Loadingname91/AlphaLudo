"""
Reward shaping strategies.

Implements strategy pattern to allow switching reward heuristics without modifying environment code.
Supports sparse, dense, and context-aware reward schemas.
"""

from abc import ABC, abstractmethod
from ast import Lambda
from typing import Dict, Tuple

from ludopy.player import GOAL_INDEX, STAR_INDEXS, Player

from ..utils.state import State

from ..utils.board_analyser import (
    HOME_INDEX,START_INDEX,GOAL_INDEX,
    HOME_STRETCH_INDEXS,GLOB_INDEXS,STAR_INDEXS
)

# --- Base Reward Constants (As defined in Theory) ---
REWARD_GOAL = 100.0
REWARD_KILL = 50.0
REWARD_SAFETY = 15.0
REWARD_BOOST = 10.0
REWARD_NEUTRAL = 1.0
REWARD_DEATH = -20.0
REWARD_SUICIDE = -10.0  # Applied by Agent potential, but good to track

# Board Constants (matching codebase)
SAFE_GLOBES = {START_INDEX} | set(GLOB_INDEXS)  # {1, 9, 22, 35, 48}
HOME_CORRIDOR_START = 52  # First position in home stretch

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
def create_reward_shaper(schema: str,player_id:int=0) -> RewardShaper:
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
        'context-aware': lambda: ContextualRewardShaper(player_id=player_id),
        # 'dense': DenseReward,  # To be implemented in Phase 2
        # 'decoupled-ila': ILAReward,  # To be implemented in Phase 2
    }
    
    if schema not in strategies:
        raise ValueError(f"Unknown reward schema: {schema}. Available: {list(strategies.keys())}")
    
    return strategies[schema]()



class ContextualRewardShaper(RewardShaper):
    """
    Context Aware Reward Shaper that detects events by comparing board states
    Calculates rewards by comparing Previous State vs Current State
    Provides the 'base rewards' that agent will later scale based on context provided by the state abstractor
    """
    def __init__(self, player_id: int = 0):
        """
        Initialize context aware reward shaper 
        Args:
            player_id : ID of the player whose rewards we are computing 
        """
        super().__init__(schema='context-aware')
        self.player_id = player_id 


    def _infer_moved_piece(self,prev_state:State,next_state:State)->int:
        """
        Try to infer which piece moved by comparing position

        Returns :
            Piece Index that moved or -1 if cannot determine
        """
        for i in range(4):
            if prev_state.player_pieces[i]!=next_state.player_pieces[i]:
                return i
        return -1


    def _count_enemies_on_board(self,state:State)->int:
        """
        Counts total enemy pieces not in home(0) or goal (57)
        """
        count = 0 
        for enemy_list in state.enemy_pieces:
            for pos in enemy_list:
                if pos!=HOME_INDEX and pos !=GOAL_INDEX and pos<GOAL_INDEX:
                    count +=1
        return count

    def _is_safe(self,pos:int)->bool:
        """
        Checks if position is Globe, Start, Home Corridor, or Home.
        
        Args:
            pos: Piece position
        
        Returns:
            True if position is safe
        """
        if pos == HOME_INDEX:
            return True  # Home is safe
        if pos in SAFE_GLOBES:
            return True  # Start or Globe
        if pos >= HOME_CORRIDOR_START and pos < GOAL_INDEX:
            return True  # Home corridor
        if pos == GOAL_INDEX:
            return True  # Goal is safe
        return False

    def compute_passive_reward(self, prev_state: State, next_state: State) -> float:
        """
        Compute passive rewards (e.g. getting killed) by comparing two states.
        Useful for detecting events that happened during opponent turns.
        
        Args:
            prev_state: State after agent's last move
            next_state: State before agent's current move
            
        Returns:
            Accumulated passive reward
        """
        passive_reward = 0.0
        
        # Detect pieces sent home (Death)
        for i in range(4):
            prev_pos = prev_state.player_pieces[i]
            curr_pos = next_state.player_pieces[i]
            
            # If piece was on field and is now home -> Killed
            # Home=0, Goal=57. 
            # Check if it was NOT home and NOT goal, and is NOW home
            if prev_pos != HOME_INDEX and prev_pos != GOAL_INDEX and curr_pos == HOME_INDEX:
                passive_reward += REWARD_DEATH
                
        return passive_reward

    def compute_reward(self,prev_state:State,next_state:State,action_piece_index:int)->float:
        """
        Determines what event occured and returns the base reward

        Args:
            prev_state: State before the action
            next_state: State after the action
            action_piece_index: Index of the piece that moved (0-3)
        
        Returns:
            Base reward value
        """
        reward = 0.0

        # validate action_piece_index
        if action_piece_index <0 or action_piece_index >=4:
            # try to infer which piece was moved by comparing positions
            action_piece_index = self._infer_moved_piece(prev_state, next_state)
            if action_piece_index <0:
                return 0.0
        
        # 1. get the specific piece that was moved
        try:
            prev_pos = prev_state.player_pieces[action_piece_index]
            curr_pos = next_state.player_pieces[action_piece_index]
        except IndexError:
            return 0.0

        # --- Event detection logic ( priority order )

        # A . Goal Event (highest priority)
        if prev_pos < GOAL_INDEX and curr_pos == GOAL_INDEX:
            return REWARD_GOAL

        # B. Kill Event (Teleport)
        # check if any enemy piece got sent back to home
        prev_enemies_on_board = self._count_enemies_on_board(prev_state)
        curr_enemies_on_board = self._count_enemies_on_board(next_state)

        if curr_enemies_on_board < prev_enemies_on_board:
            # An enemy disappeared from the field -> Kill
            return REWARD_KILL
        
        # C. Death Penalty (Moved from Field to Home - e.g. Suicide or strange rule?)
        # PRIORITY FIX: Check this BEFORE Safety
        if prev_pos != HOME_INDEX and curr_pos == HOME_INDEX:
            return REWARD_DEATH
        
        # D. Safety Event 
        # if we moved from unsafe to safe
        was_safe = self._is_safe(prev_pos)
        is_safe = self._is_safe(curr_pos)

        if not was_safe and is_safe:
            return REWARD_SAFETY

        # E. Boost Event ( Star Jump)
        # Calculate raw dice movement vs actual movement 
        dice = prev_state.dice_roll
        expected_pos = prev_pos+dice

        # Handle star jumps : if we land on a star , we jump forward
        if prev_pos!=HOME_INDEX : # cant boost from home
            # check if we landed on a star and jumped further 
            if prev_pos + dice in STAR_INDEXS:
                    # star jump should give extra movement
                    if curr_pos > expected_pos or curr_pos == GOAL_INDEX:
                        return REWARD_BOOST
            # Or if we moved further than expected (general boost detection)
            elif curr_pos > expected_pos and curr_pos !=GOAL_INDEX:
                return REWARD_BOOST

        # F. Neutral Move
        # if we moved and nothing special happened
        if curr_pos != prev_pos:
            return REWARD_NEUTRAL

        return 0.0

    def get_reward(self,game_events:Dict) -> Tuple[float,Dict]:
        """
        Compute reward from game events
        
        Args:
            game_events: Dictionary containing game events and state information 
        
        Returns:
            Tuple of (reward,empty dict for ILA components)
        """

        # Extract state information from game_events
        prev_state = game_events.get('prev_state')
        next_state = game_events.get('next_state')
        action_piece_idx = game_events.get('action_piece_index',-1)

        # if states are not provided , return 0 (fall back for sparse reward)
        if prev_state is None or next_state is None:
            # fallback to checking win/loss
            if game_events.get('player_won',False):
                return REWARD_GOAL,{}
            elif game_events.get('opponent_won',False):
                return REWARD_DEATH, {}
            return 0.0 ,{}
        
        # Compute reward by comparing states
        reward = self.compute_reward(prev_state,next_state,action_piece_idx)
        return reward,{}
