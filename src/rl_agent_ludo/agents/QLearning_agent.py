"""
Tabular Q-Learning agent for Ludo.

Uses context-aware state abstraction and dynamic reward scaling based on game context
(trailing, neutral, leading) and move potentials (kill, boost, safety, etc.).
"""

import pickle
from collections import defaultdict
from typing import Optional, Dict
import numpy as np
import random

from .base_agent import Agent
from rl_agent_ludo.utils.state_abstractor import (
    CONTEXT_LEADING, CONTEXT_TRAILING,
    POT_GOAL, POT_KILL, POT_BOOST, POT_SAFETY, POT_RISK,
    LudoStateAbstractor,
)
from rl_agent_ludo.utils.state import State


class QLearningAgent(Agent):
    """Tabular Q-Learning with context-aware reward scaling."""
    def __init__(
        self,
        seed: Optional[int] = None,
        player_id: int=0,
        learning_rate: float=0.1,
        discount_factor: float=0.9,
        epsilon: float=0.1,
        epsilon_decay: float=0.9995,
        min_epsilon: float=0.01,
        reward_scaling: Optional[Dict] = None,  # Add this - makes config compatible
        debug_scores: bool = False,
        **kwargs  # Accept any extra params to prevent errors
    ):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            
        self.player_id = player_id
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Debugging
        self.debug_scores = debug_scores
        self._last_score_debug = None

        # Q-table: maps state tuple -> array of 4 Q-values (one per piece)
        self.q_table = defaultdict(lambda: np.zeros(4))

        # State abstraction converts raw board state to tactical tuple
        self.state_abstractor = LudoStateAbstractor(player_id)

        # Store reward scaling multipliers (with defaults)
        self._setup_reward_scaling(reward_scaling)

        # Track state/action for Q-update
        self.last_state_tuple = None
        self.last_action = None  # Action index into valid_moves
        self.last_piece_idx = None  # Actual piece index (0-3) that was moved

    
    @property
    def is_on_policy(self) -> bool:
        return False # Q learning is off-policy
    
    @property
    def needs_replay_learning(self) -> bool:
        return True  # Changed from False - trainer calls push_to_replay_buffer for learning
    
    # ---- Optional score debugging hooks ----------------------------------------

    @property
    def supports_score_debug(self) -> bool:
        """Indicate that this agent can provide score breakdowns when enabled."""
        return self.debug_scores

    def get_last_score_debug(self) -> Optional[dict]:
        """Return debug information for the last decision."""
        return self._last_score_debug
    
    def act(self, state: State) -> int:
        """Select action using epsilon-greedy policy."""
        state_tuple = self.state_abstractor.get_abstract_state(state)
        valid_moves = state.valid_moves # Indices [0,1,2,3] usually 
        
        # Debug container
        decision_type = "UNKNOWN"
        selected_q = 0.0
        
        # Exploration: random action
        if random.random() < self.epsilon:
            action = random.choice(valid_moves)
            decision_type = "EXPLORATION"
        else:
            # Exploitation: greedy action
            q_values = self.q_table[state_tuple]

            best_action = valid_moves[0]
            best_q = -float('inf')

            # Find valid move with highest Q-value for corresponding piece
            for act_idx in valid_moves:
                # Map action index to piece index
                if state.movable_pieces:
                    piece_idx = state.movable_pieces[act_idx]
                else:
                    piece_idx = act_idx

                q_val = q_values[piece_idx]

                if q_val > best_q:
                    best_q = q_val
                    best_action = act_idx

            action = best_action
            decision_type = "GREEDY"
            if best_q == -float('inf'):
                decision_type = "FORCED_RANDOM"

        # Store state/action for Q-update
        self.last_state_tuple = state_tuple
        self.last_action = action

        # Store actual piece index that will be moved
        if state.movable_pieces and len(state.movable_pieces) > action:
            self.last_piece_idx = state.movable_pieces[action]
        else:
            self.last_piece_idx = action

        # Capture debug info if enabled
        if self.debug_scores:
            q_values = self.q_table[state_tuple]
            
            # Identify which piece this action moves
            if state.movable_pieces:
                piece_idx = state.movable_pieces[action]
            else:
                piece_idx = action
                
            selected_q = q_values[piece_idx]
            
            # Build components dict (Q-values for all pieces)
            components = {}
            for i in range(4):
                components[f'q_piece_{i}'] = float(q_values[i])

            self._last_score_debug = {
                'piece_idx': int(piece_idx),
                'dice_roll': int(state.dice_roll),
                'total_score': float(selected_q),
                'components': components,
                'decision_type': decision_type,
                'epsilon': self.epsilon,
                'context': state_tuple[4]  # 0=Trailing, 1=Neutral, 2=Leading
            }

        return action 
    
    def push_to_replay_buffer(self, state: State, action: int, reward: float,
                             next_state: State, done: bool, **kwargs) -> None:
        """
        Online Q-learning update (Bellman equation).
        
        Applies context-aware reward scaling before updating Q-values.
        """
        if self.last_state_tuple is None:
            return

        # Get context from state we acted in (tuple: P1,P2,P3,P4,Context)
        context = self.last_state_tuple[4]

        # Get piece index that was actually moved
        if self.last_piece_idx is None:
            piece_idx = self.last_action
        else:
            piece_idx = self.last_piece_idx

        # Get potential of the piece we moved
        action_potential = self.last_state_tuple[piece_idx]

        # Apply context-based reward scaling
        scaled_reward = self._scale_reward(reward, action_potential, context)

        # Calculate target Q-value (standard Q-learning: max over next state)
        next_state_tuple = self.state_abstractor.get_abstract_state(next_state)
        next_q_values = self.q_table[next_state_tuple]
        max_next_q = np.max(next_q_values)

        # Bellman update: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
        current_q = self.q_table[self.last_state_tuple][piece_idx]
        new_q = current_q + self.learning_rate * (scaled_reward + (self.gamma * max_next_q) - current_q)

        self.q_table[self.last_state_tuple][piece_idx] = new_q

    def _setup_reward_scaling(self, reward_scaling: Optional[Dict] = None):
        """Set up reward scaling multipliers from config or use defaults."""
        if reward_scaling is None:
            reward_scaling = {}
        
        # Default multipliers (current hardcoded values)
        default_trailing = {
            'kill': 1.5,
            'boost': 1.2,
            'safety': 0.8,
            'goal': 0.5
        }
        default_leading = {
            'kill': 0.8,
            'safety': 2.0,
            'risk': 1.5
        }
        
        # Merge config with defaults
        trailing = reward_scaling.get('trailing', {})
        leading = reward_scaling.get('leading', {})
        
        self.trailing_scaling = {**default_trailing, **trailing}
        self.leading_scaling = {**default_leading, **leading}
        self.neutral_scaling = reward_scaling.get('neutral', {'multiplier': 1.0})

    def _scale_reward(self, base_reward: float, potential: int, context: int) -> float:
        """Apply context-based multipliers to base reward."""
        multiplier = 1.0
        
        # Map potential constants to string keys
        potential_map = {
            POT_KILL: 'kill',
            POT_BOOST: 'boost',
            POT_SAFETY: 'safety',
            POT_GOAL: 'goal',
            POT_RISK: 'risk',
        }
        potential_key = potential_map.get(potential)
        
        if context == CONTEXT_TRAILING and potential_key:
            multiplier = self.trailing_scaling.get(potential_key, 1.0)
        elif context == CONTEXT_LEADING and potential_key:
            multiplier = self.leading_scaling.get(potential_key, 1.0)
        else:
            multiplier = 1.0  # Default for neutral or unknown
        
        return base_reward * multiplier

    def on_episode_end(self) -> None:
        """Decay epsilon once per episode (not per step)."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save Q-table and epsilon to file."""
        data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to dict for pickling
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load Q-table and epsilon from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.epsilon = data['epsilon']
        self.q_table = defaultdict(lambda: np.zeros(4))
        self.q_table.update(data['q_table'])