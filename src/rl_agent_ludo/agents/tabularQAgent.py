# Board Constants

import pickle
import random

from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import defaultdict
from rl_agent_ludo.utils.state import State
from rl_agent_ludo.agents.baseAgent import Agent

HOME_INDEX = 0 
GOAL_INDEX = 57
# Use the same stars/globes as RuleBasedAgent/LudoEnv for consistency
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]
GLOBE_INDEXES = [1, 9, 22, 35, 48]
HOME_CORRIDOR = list(range(52,57))

# Potential classifications (7-class system for refined state representation)
POT_HOME         = 0  # Token is in base
POT_GOAL         = 1  # Token has finished
POT_SAFE         = 2  # On a Globe or Star (safe position)
POT_RISK_LOW     = 3  # Risk in Q1/Q2 (Early game)
POT_RISK_HIGH    = 4  # Risk in Q3/Q4 (Late game - catastrophic to lose)
POT_KILL         = 5  # Can capture enemy
POT_DANGER       = 6  # Static Danger Zone (Only used in 'hybrid_penalty' mode)
POT_Q1 = 7  # Pos 1-13: The Start
POT_Q2 = 8  # Pos 14-26: First Corner
POT_Q3 = 9  # Pos 27-39: Second Corner
POT_Q4 = 10  # Pos 40-52: The Home Stretch

# Legacy constants for backward compatibility
POT_NULL     = POT_HOME
POT_NEUTRAL  = POT_Q1  # Default mapping
POT_NEUTRAL_EARLY = POT_Q1  # Backward compatibility
POT_NEUTRAL_LATE  = POT_Q4  # Backward compatibility
POT_RISK     = POT_RISK_LOW  # Default to low risk for backward compatibility
POT_BOOST    = POT_SAFE  # Star jumps are safe
POT_SAFETY   = POT_SAFE

# Context labels
CTX_TRAILING = 0
CTX_NEUTRAL  = 1
CTX_LEADING  = 2


class TabularQAgent(Agent):
    """
    Tabular Q-learning agent for Ludo.

    State abstraction modes:
    1. 'compact' (RECOMMENDED for tabular Q-learning): (Context, P1, P2, P3, P4, Dice) - 6-tuple
        - Context: trailing/neutral/leading (0-2)
        - P1-P4: refined potentials (0-9): 10-class quarter-based system with risk split
          0=HOME, 1=GOAL, 2=SAFE, 3=RISK_LOW, 4=RISK_HIGH, 5=KILL, 6=Q1, 7=Q2, 8=Q3, 9=Q4
        - Dice: current dice roll (0-5, represents 1-6)
        - State space: 196,830 states (3 × 10^4 × 6)
        - Solves "Blurry Vision" (quarters) and "Risk Blind Spot" (risk split)
        - Solves "Blind Surgeon" problem by explicitly including dice roll
    2. 'potential': (P1, P2, P3, P4, Context, T1, T2, T3, T4) - 9-tuple
        - P1-P4: potential classifications (0-6)
        - Context: trailing/neutral/leading (0-2)
        - T1-T4: current threat flags (0-1)
    3. 'zone_based': (HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4) - 12-tuple
        - HOME/PATH/SAFE/GOAL: count of tokens in each zone (0-4)
        - EV1-EV4: enemy vulnerable to agent's token (0-1)
        - TV1-TV4: agent's token under attack (0-1)
    4. 'combined'/'enhanced': (HOME,PATH,SAFE,GOAL, EV1,EV2,EV3,EV4, TV1,TV2,TV3,TV4, PROGRESS, CONTEXT) - 14-tuple
        - Zone counts (4) + Vulnerability flags (4) + Threat flags (4)
        - PROGRESS: Token advancement score (0-10)
        - CONTEXT: Game equity context (0-2)
        - Improves upon GitHub repo by adding progress tracking and strategic context

    Q-Table : dict[state, dict[action, float]] (Q value for each action in each state)
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,  # Changed from 0.95 to 0.99 for proper long-term learning
        epsilon: float = 0.1,
        epsilon_decay: float = 0.9995,
        min_epsilon: float = 0.01,
        state_abstraction: str = 'compact',  # 'potential', 'zone_based', 'combined', 'enhanced', or 'compact' (recommended for tabular)
        seed: Optional[int] = None,
    )-> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.state_abstraction = state_abstraction

        # Q-table: state_tuple -> np.array(4) (Q per piece index)
        # State tuple size depends on abstraction mode
        # All use generic Dict[Tuple, np.ndarray] for flexibility
        self.q_table: Dict[Tuple, np.ndarray] = \
            defaultdict(lambda: np.zeros(4, dtype=np.float32))

        # last transition info for online Q-updates
        self._last_state_tuple: Optional[Tuple] = None
        self._last_piece_idx: Optional[int] = None

    
    @property
    def is_on_policy(self) -> bool:
        # Q learning is off policy
        return False
    
    @property
    def needs_replay_learning(self) -> bool:
        return False 
    
    def act(self, state: State) -> int:
        """
        Epsilon Greedy action over piece indices, restricted to valid moves
        """
        if not state.valid_moves:
            return 0

        state_tuple = self._build_state_tuple(state)
        q_values = self.q_table[state_tuple]

        # map valid action indices -> piece indices
        valid_actions = list(state.valid_moves)
        piece_indices = [
            self._action_to_piece_idx(state, action) for action in valid_actions
        ]

        # exploration 
        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
            piece_idx = self._action_to_piece_idx(state, action)
        else:
            # Exploitation: choose action with highest Q over valid piece indices
            best_action = valid_actions[0]
            best_piece_idx = piece_indices[0]
            best_q = q_values[best_piece_idx]

            for action, piece_idx in zip(valid_actions, piece_indices):
                if q_values[piece_idx] > best_q:
                    best_q = q_values[piece_idx]
                    best_action = action
                    best_piece_idx = piece_idx

            action = best_action
            piece_idx = best_piece_idx

        # store last piece/state for Q update
        self._last_state_tuple = state_tuple
        self._last_piece_idx = piece_idx

        return action 

    def push_to_replay_buffer(
        self,
        state:State,
        action:int,
        reward:float,
        next_state:State,
        done:bool,
        **kwargs,
    )-> None:
        """
         Online Tabular Q-learning update

         Assume this is called after each act has been called with state and env has returned reward

        Args:
            state: State object before action
            action: Action index (0-3)
            reward: Reward from environment
            next_state: State object after action
            done: Whether the game is done
            **kwargs: Additional arguments (not used)
        """
        if self._last_state_tuple is None or self._last_piece_idx is None:
            # initialize last state tuple so that next step can update from it 
            self._last_state_tuple = self._build_state_tuple(state)
            return 
        
        s_tuple = self._last_state_tuple
        a_piece = self._last_piece_idx

        # build next state tuple 

        s_next_tuple = self._build_state_tuple(next_state)

        # Reward is now provided directly by environment with rich reward structure
        # No need for additional scaling
        r_scaled = reward

        # Q-learning update
        q_values = self.q_table[s_tuple]
        q_sa = q_values[a_piece]

        # max Q value for next state
        if next_state.valid_moves and not done:
            next_state_indicies = [
                self._action_to_piece_idx(next_state, action) for action in next_state.valid_moves
            ]
            q_next = self.q_table[s_next_tuple]
            max_q_next = max(q_next[piece_idx] for piece_idx in next_state_indicies)
        else:
            max_q_next = 0 
        
        target = r_scaled + (0.0 if done else self.gamma * max_q_next)
        q_values[a_piece] = q_sa + self.alpha * (target - q_sa)

        # prepare for next step 
        self._last_state_tuple = s_next_tuple
        if done:
            # end of the epsiode 
            self._last_state_tuple = None
            self._last_piece_idx = None
            self.on_episode_end()
        
    def learn_from_replay(self,*args,**kwargs)-> None:
        """
        No replay learning needed for tabular Q-learning
        """
        return None 
    
    def learn_from_rollout(self,*args,**kwargs)-> None:
        """
        No Rollout learning needed for tabular Q-learning
        """
        return None
    
    def on_episode_end(self) -> None:
        # run episodic epsilon decay
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
    
    def save(self,filepath:str) -> None:
        """
        Save the Q-table to a file
        """
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        with open(filepath, 'wb') as f:
            pickle.dump(
                {
                    "q_table": q_table_dict,
                    "epsilon": self.epsilon,
                    "state_abstraction": self.state_abstraction,
                },
                f,
            )
    
    def load(self,filepath:str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        # Re-wrap dict as defaultdict
        q_raw = data.get("q_table", {})
        self.q_table = defaultdict(
            lambda: np.zeros(4, dtype=np.float32),
            {k: np.asarray(v, dtype=np.float32) for k, v in q_raw.items()},
        )
        self.epsilon = data.get("epsilon", self.epsilon)
        # Load state_abstraction if saved (for compatibility)
        if "state_abstraction" in data:
            self.state_abstraction = data["state_abstraction"]

    
    def _build_state_tuple(self,state:State) -> Tuple:
        """
        Build state tuple from state based on abstraction mode.

        Returns:
            - 'potential': (P1, P2, P3, P4, Context, T1, T2, T3, T4) - 9-tuple
            - 'zone_based': (HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4) - 12-tuple
            - 'combined'/'enhanced': (HOME,PATH,SAFE,GOAL, EV1,EV2,EV3,EV4, TV1,TV2,TV3,TV4, PROGRESS, CONTEXT) - 14-tuple
            - 'compact': (Context, P1, P2, P3, P4, T1, T2, T3, T4) - 9-tuple (optimized for tabular Q-learning)
                State space: ~12,288 states (manageable for tabular methods)
        """
        if self.state_abstraction == 'compact':
            return self._build_compact_state(state)
        elif self.state_abstraction == 'zone_based':
            return self._build_zone_based_state(state)
        elif self.state_abstraction in ['combined', 'enhanced']:
            return self._build_enhanced_state(state)
        else:  # 'potential'
            return self._build_potential_state(state)
    
    def _build_potential_state(self, state: State) -> Tuple:
        """
        Build (P1, P2, P3, P4, Context, T1, T2, T3, T4) tuple from state.
        
        Returns 9-tuple with potential classifications, context, and current threat flags.
        """
        potentials = []
        threat_flags = []
        
        for piece_idx in range(4):
            potentials.append(self._classify_potential(state, piece_idx))
            
            # Calculate current threat flag (T flag)
            pos = state.player_pieces[piece_idx]
            is_threatened = self._is_token_under_threat(pos, state.enemy_pieces)
            threat_flags.append(1 if is_threatened else 0)
        
        context = self._compute_context(state)
        
        return (
            potentials[0], potentials[1], potentials[2], potentials[3],
            context,
            threat_flags[0], threat_flags[1], threat_flags[2], threat_flags[3]
        )
    
    def _build_zone_based_state(self, state: State) -> Tuple:
        """
        Build zone-based state tuple inspired by referenced repository.
        
        Returns: (HOME, PATH, SAFE, GOAL, EV1, EV2, EV3, EV4, TV1, TV2, TV3, TV4)
        - HOME: count of tokens in base (0-4)
        - PATH: count of tokens on main path (0-4)
        - SAFE: count of tokens in safe zones (0-4)
        - GOAL: count of tokens that reached goal (0-4)
        - EV1-EV4: enemy vulnerable to agent's token (0-1)
        - TV1-TV4: agent's token under attack (0-1)
        """
        player_pieces = state.player_pieces
        
        # Count tokens in each zone
        home_count = sum(1 for pos in player_pieces if pos == HOME_INDEX)
        goal_count = sum(1 for pos in player_pieces if pos == GOAL_INDEX)
        # Safe zones: globes, home corridor, start (but not HOME or GOAL)
        safe_count = sum(1 for pos in player_pieces 
                        if pos not in [HOME_INDEX, GOAL_INDEX] 
                        and self._is_safe_position(pos))
        # Path: tokens on main board (positions 1-51) that are not safe
        path_count = sum(1 for pos in player_pieces 
                        if 1 <= pos <= 51 
                        and not self._is_safe_position(pos))
        
        # Clamp counts to valid range (0-4)
        home_count = min(home_count, 4)
        path_count = min(max(path_count, 0), 4)
        safe_count = min(safe_count, 4)
        goal_count = min(goal_count, 4)
        
        # Check vulnerabilities and threats for each token
        ev_flags = []  # Enemy vulnerable to our tokens
        tv_flags = []  # Our tokens under threat
        
        for piece_idx in range(4):
            pos = player_pieces[piece_idx]
            if pos in [HOME_INDEX, GOAL_INDEX]:
                ev_flags.append(0)
                tv_flags.append(0)
                continue
            
            # Check if this token can capture an enemy (EV)
            can_capture = False
            if self._is_piece_movable(state, piece_idx):
                next_pos = self._simulate_move(pos, state.dice_roll)
                can_capture = self._can_capture(next_pos, state.enemy_pieces)
            ev_flags.append(1 if can_capture else 0)
            
            # Check if this token is under threat (TV)
            is_threatened = self._is_token_under_threat(pos, state.enemy_pieces)
            tv_flags.append(1 if is_threatened else 0)
        
        return (
            home_count, path_count, safe_count, goal_count,
            ev_flags[0], ev_flags[1], ev_flags[2], ev_flags[3],
            tv_flags[0], tv_flags[1], tv_flags[2], tv_flags[3]
        )
    
    def _build_enhanced_state(self, state: State) -> Tuple:
        """
        Enhanced state representation that improves upon GitHub repo approach.

        GitHub repo (8 values): HOME,PATH,SAFE,GOAL,EV1,EV2,TV1,TV2
        Our enhanced version (14 values): Adds progress + strategic context

        Returns: (HOME,PATH,SAFE,GOAL, EV1,EV2,EV3,EV4, TV1,TV2,TV3,TV4, PROGRESS, CONTEXT) - 14-tuple

        What GitHub neglected:
        - PROGRESS: Token advancement score (0-10, discretized)
        - CONTEXT: Game equity context (0-2)
        - Enemy information (already included via EV/TV)
        - Risk levels (binary TV is sufficient, PROGRESS indicates danger)

        Improvements over GitHub:
        - Works with 4 tokens (they only did 2)
        - Adds progress tracking for better long-term planning
        - Adds strategic context for opponent modeling
        """
        player_pieces = state.player_pieces

        # 1. Zone distribution (same as GitHub)
        home_count = sum(1 for pos in player_pieces if pos == HOME_INDEX)
        goal_count = sum(1 for pos in player_pieces if pos == GOAL_INDEX)
        safe_count = sum(1 for pos in player_pieces
                        if pos not in [HOME_INDEX, GOAL_INDEX]
                        and self._is_safe_position(pos))
        path_count = sum(1 for pos in player_pieces
                        if 1 <= pos <= 51
                        and not self._is_safe_position(pos))

        # Clamp counts (0-4 for compatibility)
        home_count = min(home_count, 4)
        path_count = min(max(path_count, 0), 4)
        safe_count = min(safe_count, 4)
        goal_count = min(goal_count, 4)

        # 2. Enemy vulnerability flags (same as GitHub, but for 4 tokens)
        ev_flags = []  # Can our tokens attack enemies?
        for piece_idx in range(4):
            pos = player_pieces[piece_idx]
            if pos in [HOME_INDEX, GOAL_INDEX]:
                ev_flags.append(0)
                continue

            can_capture = False
            if self._is_piece_movable(state, piece_idx):
                next_pos = self._simulate_move(pos, state.dice_roll)
                can_capture = self._can_capture(next_pos, state.enemy_pieces)
            ev_flags.append(1 if can_capture else 0)

        # 3. Threat flags (same as GitHub, but for 4 tokens)
        tv_flags = []  # Are our tokens under threat?
        for piece_idx in range(4):
            pos = player_pieces[piece_idx]
            is_threatened = self._is_token_under_threat(pos, state.enemy_pieces)
            tv_flags.append(1 if is_threatened else 0)

        # 4. Progress score (NEW: What GitHub neglected)
        # Measures how far tokens have advanced (0-100, discretized to 0-10)
        progress_score = self._get_weighted_equity_score(state.player_pieces)
        progress_discrete = min(max(int(progress_score / 10), 0), 10)  # 0-10 scale

        # 5. Game context (NEW: Strategic situation)
        context = self._compute_context(state)

        return (
            # Zone counts (4) - same as GitHub
            home_count, path_count, safe_count, goal_count,
            # Vulnerability flags (4) - same as GitHub but extended
            ev_flags[0], ev_flags[1], ev_flags[2], ev_flags[3],
            # Threat flags (4) - same as GitHub but extended
            tv_flags[0], tv_flags[1], tv_flags[2], tv_flags[3],
            # Progress (1) - NEW: What GitHub missed
            progress_discrete,
            # Context (1) - NEW: Strategic awareness
            context
        )

    def _build_compact_state(self, state: State) -> Tuple:
        """
        Compact state representation optimized for Tabular Q-Learning.
        
        Uses refined 7-class potential system that solves "Neutral Blindness" by
        splitting NEUTRAL into EARLY and LATE based on progress.
        
        CRITICAL FIX: Includes dice_roll explicitly to solve "Blind Surgeon" problem.
        Without dice, the agent cannot distinguish (HOME, dice=1) from (HOME, dice=6),
        causing it to average Q-values incorrectly and miss tactical opportunities.
        
        State Definition: (Context, Potential_1, Potential_2, Potential_3, Potential_4, Dice)
        
        - Context (3 values): Trailing(0), Neutral(1), Leading(2)
        - Potentials (9 values × 4 pieces): Quarter-based 9-class system
          0: HOME - Token is in base
          1: GOAL - Token has finished
          2: RISK - Under threat
          3: KILL - Can capture enemy
          4: SAFE - On a Globe or Star
          5: Q1 - Pos 1-13: The Start
          6: Q2 - Pos 14-26: First Corner
          7: Q3 - Pos 27-39: Second Corner
          8: Q4 - Pos 40-52: The Home Stretch
        - Dice (6 values): Current dice roll (1-6)
        
        Total State Space: 3 × 9^4 × 6 = 3 × 6,561 × 6 = 118,098 states
        
        This solves the "Blind Surgeon" problem:
        - Agent can distinguish (HOME, dice=6) from (HOME, dice=1)
        - Agent learns distinct Q-values for "lucky 6" vs "unlucky 1"
        - Still manageable for tabular Q-learning (43k states)
        """
        # 1. Game context (strategic situation)
        context = self._compute_context(state)
        
        # 2. Refined potentials (7-class system with progress awareness)
        potentials = []
        for piece_idx in range(4):
            potential = self._classify_potential(state, piece_idx)
            # Use the 7-class system directly (no compression needed)
            potentials.append(potential)
        
        # 3. Dice roll (CRITICAL: explicitly included to solve "Blind Surgeon" problem)
        dice_roll = state.dice_roll
        # Normalize dice to 0-5 for state space (1-6 → 0-5)
        dice_normalized = dice_roll - 1
        
        return (
            context,
            potentials[0], potentials[1], 
            potentials[2], potentials[3],
            dice_normalized  # Dice: 0-5 (represents 1-6)
        )
    
    def _build_combined_state(self, state: State) -> Tuple:
        """
        Legacy combined state (13 values) - kept for backward compatibility.
        Use _build_enhanced_state for better performance.
        """
        return self._build_enhanced_state(state)
    
    def _action_to_piece_idx(self, state: State, action_idx: int) -> int:
        """
        Map action to piece index using ABSOLUTE indexing.
        Action 0 = Token 0, Action 1 = Token 1, etc.
        This ensures consistent action-to-token mapping for Q-learning.
        """
        # With absolute indexing, action_idx IS the piece index
        return action_idx
    
    # ------------------------------------------------------------------ #
    # Potential classification
    # ------------------------------------------------------------------ #

    def _classify_potential(self, state: State, piece_idx: int) -> int:
        """
        Classify potential for one piece using 10-class quarter-based system with risk split.
        
        Returns one of:
        - POT_HOME (0): Token is in base
        - POT_GOAL (1): Token has finished
        - POT_SAFE (2): On a Globe or Star (safe position)
        - POT_RISK_LOW (3): Risk in Q1/Q2 (Early game)
        - POT_RISK_HIGH (4): Risk in Q3/Q4 (Late game - catastrophic to lose)
        - POT_KILL (5): Can capture enemy
        - POT_Q1 (6): Pos 1-13 - The Start
        - POT_Q2 (7): Pos 14-26 - First Corner
        - POT_Q3 (8): Pos 27-39 - Second Corner
        - POT_Q4 (9): Pos 40-52 - The Home Stretch
        """
        player_pieces = state.player_pieces
        current_pos = player_pieces[piece_idx]
        dice = state.dice_roll

        # HOME: Token is in base
        if current_pos == HOME_INDEX:
            if not self._is_piece_movable(state, piece_idx):
                return POT_HOME
            # If movable, check what happens when it exits
            next_pos = self._simulate_move(current_pos, dice)
        else:
            next_pos = self._simulate_move(current_pos, dice)

        # GOAL: Token has finished (highest priority)
        if next_pos == GOAL_INDEX or self._is_at_goal(current_pos, dice):
            return POT_GOAL

        # SAFE: On a Globe or Star (safe position) - check before KILL/RISK
        # Check both current position (if already safe) and next position (if moving to safe)
        # Safe positions cannot capture or be captured
        if self._is_safe_position(next_pos) or (current_pos != HOME_INDEX and self._is_safe_position(current_pos)):
            return POT_SAFE

        # KILL: Can capture enemy (tactical priority, but only if not safe)
        if self._can_capture(next_pos, state.enemy_pieces):
            return POT_KILL

        # RISK: Under threat (Granular - Split by position)
        # This fixes the "Risk Blind Spot" - agent now knows losing a token in Q4 is catastrophic
        risk = self._calculate_probability_of_risk(next_pos, state.enemy_pieces)
        if risk > 0.0:
            # If threatened in the second half of board (Q3/Q4), it's HIGH risk
            if next_pos >= 27:
                return POT_RISK_HIGH  # Late game - catastrophic to lose
            else:
                return POT_RISK_LOW  # Early game - annoying but not catastrophic

        # NEUTRAL: Split into Quarters for higher resolution (fixes "Blurry Vision")
        # This provides better granularity than Early/Late split
        if 1 <= next_pos <= 13:
            return POT_Q1  # The Start
        elif 14 <= next_pos <= 26:
            return POT_Q2  # First Corner
        elif 27 <= next_pos <= 39:
            return POT_Q3  # Second Corner
        elif 40 <= next_pos <= 52:
            return POT_Q4  # The Home Stretch
        else:
            # Positions 53-56 are in home corridor, treat as Q4 (almost home)
            return POT_Q4
    
    def _calculate_position_progress(self, pos: int) -> float:
        """
        Calculate progress percentage for a position (0-100%).
        
        Args:
            pos: Position on board
            
        Returns:
            Progress percentage (0.0 = home, 100.0 = goal)
        """
        if pos == HOME_INDEX:
            return 0.0
        if pos == GOAL_INDEX:
            return 100.0
        if pos in HOME_CORRIDOR:
            # In home corridor: progress from 90% to 100%
            corridor_start = HOME_CORRIDOR[0]
            corridor_length = len(HOME_CORRIDOR)
            progress_in_corridor = (pos - corridor_start) / corridor_length
            return 90.0 + (progress_in_corridor * 10.0)
        if 1 <= pos <= 51:
            # On main board: progress from ~2% to ~90%
            # Position 1 is start, position 51 is near corridor
            progress_on_board = (pos - 1) / 51.0
            return 2.0 + (progress_on_board * 88.0)
        return 0.0  # Unknown position

    def _is_piece_movable(self, state: State, piece_idx: int) -> bool:
        """
        Check if this piece is actually one of the movable ones for this state.
        """
        if not state.valid_moves:
            return False

        # If movable_pieces exists, only those pieces can move
        if state.movable_pieces:
            return piece_idx in state.movable_pieces

        # Fallback: allow if its index appears as a valid action
        return piece_idx in state.valid_moves

    # ------------------------------------------------------------------ #
    # Board logic (simplified copy of RuleBasedAgent helpers)
    # ------------------------------------------------------------------ #

    def _simulate_move(self, current_pos: int, dice_roll: int) -> int:
        # Home handling
        if current_pos == HOME_INDEX:
            if dice_roll == 6:
                return 6
            return HOME_INDEX

        # Already finished
        if current_pos == GOAL_INDEX:
            return GOAL_INDEX

        next_pos = current_pos + dice_roll

        # Star jump on main board
        if 1 <= current_pos <= 51 and next_pos in STAR_INDEXES:
            star_idx = STAR_INDEXES.index(next_pos)
            if star_idx < len(STAR_INDEXES) - 1:
                next_pos = STAR_INDEXES[star_idx + 1]
            else:
                next_pos = GOAL_INDEX

        # Goal bounce from corridor
        if current_pos in HOME_CORRIDOR and next_pos > GOAL_INDEX:
            overshoot = next_pos - GOAL_INDEX
            next_pos = GOAL_INDEX - overshoot
            if next_pos < HOME_CORRIDOR[0]:
                next_pos = HOME_CORRIDOR[0]
            elif next_pos > GOAL_INDEX:
                next_pos = GOAL_INDEX

        return next_pos

    def _is_at_goal(self, current_pos: int, dice_roll: int) -> bool:
        if current_pos == HOME_INDEX:
            return False
        if current_pos in HOME_CORRIDOR:
            return current_pos + dice_roll >= GOAL_INDEX
        if 1 <= current_pos <= 51:
            return current_pos + dice_roll >= GOAL_INDEX
        return False

    def _can_capture(self, next_pos: int, enemy_pieces: List[List[int]]) -> bool:
        if self._is_safe_position(next_pos):
            return False
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == next_pos and e_pos not in [HOME_INDEX, GOAL_INDEX]:
                    return True
        return False

    def _is_safe_position(self, pos: int) -> bool:
        return (
            pos == HOME_INDEX
            or pos == GOAL_INDEX
            or pos in HOME_CORRIDOR
            or pos in GLOBE_INDEXES
            or pos == 1  # start globe
        )

    def _is_star_jump(self, current_pos: int, next_pos: int, dice_roll: int) -> bool:
        if current_pos == HOME_INDEX or not (1 <= current_pos <= 51):
            return False
        original_landing = current_pos + dice_roll
        return original_landing in STAR_INDEXES and original_landing != next_pos

    def _get_circular_distance(self, pos1: int, pos2: int) -> int:
        if pos1 in [HOME_INDEX, GOAL_INDEX] or pos2 in [HOME_INDEX, GOAL_INDEX]:
            return 100
        if not (1 <= pos1 <= 51) or not (1 <= pos2 <= 51):
            return 100
        diff = abs(pos1 - pos2)
        return min(diff, 51 - diff)
    
    def _is_token_under_threat(self, token_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """
        Check if a token is under threat from enemy pieces.
        A token is threatened if an enemy can reach it within 6 moves (one dice roll).
        
        Args:
            token_pos: Position of the agent's token
            enemy_pieces: List of enemy piece positions
            
        Returns:
            True if token is under threat, False otherwise
        """
        if self._is_safe_position(token_pos):
            return False
        
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                # Check if enemy is within 6 positions (can reach in one turn)
                dist = self._get_circular_distance(token_pos, e_pos)
                if 1 <= dist <= 6:
                    return True
        return False

    def _calculate_probability_of_risk(
        self,
        next_pos: int,
        enemy_pieces: List[List[int]],
    ) -> float:
        if self._is_safe_position(next_pos):
            return 0.0
        risk_score = 0.0
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos in [HOME_INDEX, GOAL_INDEX]:
                    continue
                dist = self._get_circular_distance(next_pos, e_pos)
                if 1 <= dist <= 6:
                    risk_score += (7 - dist) * 800.0
        return risk_score

    # ------------------------------------------------------------------ #
    # Context & reward scaling
    # ------------------------------------------------------------------ #

    def _get_weighted_equity_score(self, pieces: List[int]) -> float:
        score = 0.0
        for pos in pieces:
            if pos == GOAL_INDEX:
                score += 100
            elif pos in HOME_CORRIDOR:
                score += 50 + pos
            elif pos in GLOBE_INDEXES or pos == 1:
                score += 10 + pos
            elif pos == HOME_INDEX:
                score += 0
            else:
                score += pos
        return score

    def _compute_context(self, state: State) -> int:
        my_score = self._get_weighted_equity_score(state.player_pieces)
        if state.enemy_pieces:
            opp_scores = [
                self._get_weighted_equity_score(enemy)
                for enemy in state.enemy_pieces
            ]
            max_opp = max(opp_scores)
        else:
            max_opp = 0.0
        gap = my_score - max_opp
        if gap < -20:
            return CTX_TRAILING
        elif gap > 20:
            return CTX_LEADING
        return CTX_NEUTRAL

    def _scale_reward(self, base_reward: float, potential: int, context: int) -> float:
        """
        Reward scaling is now handled by the environment's rich reward structure.
        This function is kept for compatibility but just returns the base reward.
        """
        # Environment now provides rich rewards directly, so we just use them as-is
        return base_reward