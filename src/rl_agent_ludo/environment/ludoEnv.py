"""
Ludo Gymnasium Environment.

Supports multiple abstraction modes for A/B testing:

1. 'hybrid': Baseline (Quarters + Risk Split + Clean Rewards)
2. 'hybrid_penalty': Experimental (Adds Static Danger Zones + Penalty Rewards)
"""
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces

from rl_agent_ludo.utils.state import State
from rl_agent_ludo.ludo.game import Game as LudopyGame

# --- Constants for Hybrid Abstraction (Quarter-Based Granularity + Risk Split) ---
CLASS_HOME = 0
CLASS_GOAL = 1
CLASS_SAFE = 2
CLASS_RISK_LOW = 3   # Risk in Q1/Q2 (Early game)
CLASS_RISK_HIGH = 4  # Risk in Q3/Q4 (Late game - catastrophic to lose)
CLASS_KILL = 5
CLASS_DANGER = 6     # NEW: Static Danger (Only used in 'hybrid_penalty')
CLASS_Q1 = 7  # Pos 1-13: The Start
CLASS_Q2 = 8  # Pos 14-26: First Corner
CLASS_Q3 = 9  # Pos 27-39: Second Corner
CLASS_Q4 = 10  # Pos 40-52: The Home Stretch

CTX_TRAILING = 0
CTX_NEUTRAL = 1
CTX_LEADING = 2

HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]
HOME_CORRIDOR = list(range(52, 57))

# Static Danger Zones (Tiles 1-6 steps after enemy spawns)
# Enemy Spawns: 14 (P2), 27 (P3), 40 (P4)
DANGER_INDEXES = [
    14, 15, 16, 17, 18, 19,  # After P2 spawn
    27, 28, 29, 30, 31, 32,  # After P3 spawn
    40, 41, 42, 43, 44, 45   # After P4 spawn
]


class LudoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 4,
        tokens_per_player: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        observation_mode: str = 'hybrid', # 'hybrid' or 'hybrid_penalty'
    ) -> None:
        super().__init__()

        assert num_players in (2, 3, 4), "num_players must be 2, 3, or 4"
        assert tokens_per_player in (2, 4) , "tokens_per_player must be 2 or 4"
        assert 0 <= player_id < num_players, "player_id must be < num_players"

        self.player_id = player_id 
        self.num_players = num_players
        self.tokens_per_player = tokens_per_player
        self.render_mode = render_mode
        self.observation_mode = observation_mode
        self._seed = seed

        self.action_space = spaces.Discrete(4)

        # Observation Space Definition
        if self.observation_mode == 'raw':
            # 4 player pieces + 12 enemy pieces + dice + current_player
            low = np.array([-1] * (4 + 12) + [1] + [0], dtype=np.int32)
            high = np.array([57] * (4 + 12) + [6] + [3], dtype=np.int32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        elif self.observation_mode == 'hybrid':
            # Baseline: Max Class is CLASS_Q4 (10)
            low = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)
            high = np.array([2, 10, 10, 10, 10, 6], dtype=np.int32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
            
        elif self.observation_mode == 'hybrid_penalty':
            # Experimental: Same size (classes go up to 10), logic differs
            low = np.array([0, 0, 0, 0, 0, 1], dtype=np.int32)
            high = np.array([2, 10, 10, 10, 10, 6], dtype=np.int32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

        self.game: Optional[LudopyGame] = None
        self.current_player: int = 0 
        self.current_dice: int = 1
        self.current_move_pieces: List[int] = []
        self._last_raw_obs: Optional[Tuple] = None
        self._prev_state: Optional[State] = None

    def _set_seed(self, seed: Optional[int]) -> None:
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)

    def _ghost_players_for_num_players(self) -> List[int]:
        if self.num_players == 4:
            return []
        if self.num_players == 3:
            return [3]
        if self.num_players == 2:
            return [2, 3]
        return []
    
    def _apply_token_limit(self, pieces: List[int]) -> List[int]:
        if self.tokens_per_player == 4:
            return pieces 
        tokens_on_board = list(pieces)
        for i in range(2, len(tokens_on_board)):
            tokens_on_board[i] = 0
        return tokens_on_board

    def _build_state_from_obs(self, obs: Tuple) -> State:
        dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = obs

        player_pieces = self._apply_token_limit(list(player_pieces))
        enemy_pieces = [self._apply_token_limit(list(ep)) for ep in enemy_pieces]
        move_pieces = [p for p in move_pieces if p < self.tokens_per_player]

        # FIXED: Use ABSOLUTE indexing - valid_moves are the token indices that can move
        # Action 0 = Token 0, Action 1 = Token 1, etc.
        # This ensures consistent action-to-token mapping for Q-learning
        if len(move_pieces) > 0:
            valid_moves = list(move_pieces)  # These are the token indices (0, 1, 2, 3) that can move
            movable_pieces = list(move_pieces)  # Same for backward compatibility
        else:
            valid_moves = []
            movable_pieces = []

        return State(
            player_pieces=player_pieces,
            enemy_pieces=enemy_pieces,
            current_player=self.current_player,
            dice_roll=int(dice),
            valid_moves=valid_moves,
            movable_pieces=movable_pieces,
        )

    def _state_to_obs(self, state: State) -> np.ndarray:
        """Switches between raw and hybrid state generation"""
        if self.observation_mode == 'hybrid':
            hybrid_tuple = self._build_hybrid_state(state)
            return np.asarray(hybrid_tuple, dtype=np.int32)
        else:
            vec = (
                list(state.player_pieces)
                + [p for enemy in state.enemy_pieces for p in enemy]
                + [state.dice_roll, state.current_player]
            )
            return np.asarray(vec, dtype=np.int32)

    # ---------- Hybrid State Logic ----------

    def _build_hybrid_state(self, state: State) -> Tuple[int, ...]:
        """
        Builds the 'Hybrid 7-Class' tuple.
        Returns: (Context, T1_Class, ..., T4_Class, Dice)
        """
        token_classes = []
        for i in range(4):
            if i >= self.tokens_per_player:
                token_classes.append(CLASS_HOME)
                continue
            token_classes.append(self._classify_token_hybrid(state, i))
        
        context = self._compute_context(state)
        # CRITICAL FIX: Add dice_roll to state to resolve partial observability
        return (context,) + tuple(token_classes) + (state.dice_roll,)

    def _classify_token_hybrid(self, state: State, piece_idx: int) -> int:
        pos = state.player_pieces[piece_idx]
        
        # 1. Static Safe States
        if pos == HOME_INDEX:
            return CLASS_HOME
        if pos == GOAL_INDEX:
            return CLASS_GOAL
        if pos in GLOBE_INDEXES or pos in STAR_INDEXES:
            return CLASS_SAFE
        
        # 2. Dynamic Kill (High Priority)
        if self._is_piece_movable(state, piece_idx):
            next_pos = self._simulate_move(pos, state.dice_roll)
            if self._can_capture(next_pos, state.enemy_pieces):
                return CLASS_KILL
        
        # 3. Dynamic Risk (Under Threat)
        if self._is_token_under_threat(pos, state.enemy_pieces):
            if pos >= 27:
                return CLASS_RISK_HIGH
            else:
                return CLASS_RISK_LOW
        
        # 4. Static Danger (ONLY IN EXPERIMENTAL MODE)
        if self.observation_mode == 'hybrid_penalty':
            if pos in DANGER_INDEXES:
                return CLASS_DANGER
        
        # 5. Neutral Progress
        if 1 <= pos <= 13:
            return CLASS_Q1
        elif 14 <= pos <= 26:
            return CLASS_Q2
        elif 27 <= pos <= 39:
            return CLASS_Q3
        else:
            return CLASS_Q4

    def _compute_context(self, state: State) -> int:
        """Simple leading/trailing indicator"""
        my_score = self._get_weighted_equity_score(state.player_pieces)
        if state.enemy_pieces:
            max_opp = max(self._get_weighted_equity_score(enemy) for enemy in state.enemy_pieces)
        else:
            max_opp = 0.0
        
        gap = my_score - max_opp
        if gap < -20:
            return CTX_TRAILING
        elif gap > 20:
            return CTX_LEADING
        return CTX_NEUTRAL

    def _get_weighted_equity_score(self, pieces: List[int]) -> float:
        score = 0.0
        for pos in pieces:
            if pos == GOAL_INDEX: score += 100
            elif pos in HOME_CORRIDOR: score += 50 + pos
            elif pos == HOME_INDEX: score += 0
            else: score += pos
        return score

    def _is_piece_movable(self, state: State, piece_idx: int) -> bool:
        if not state.valid_moves: return False
        if state.movable_pieces: return piece_idx in state.movable_pieces
        return piece_idx in state.valid_moves

    def _simulate_move(self, current_pos: int, dice_roll: int) -> int:
        # Simplified simulation for state classification
        if current_pos == HOME_INDEX:
            if dice_roll == 6: return 1 # Standard ludo rule
            return HOME_INDEX
        if current_pos == GOAL_INDEX: return GOAL_INDEX
        
        next_pos = current_pos + dice_roll
        # Star jump logic
        if next_pos in STAR_INDEXES:
            idx = STAR_INDEXES.index(next_pos)
            if idx < len(STAR_INDEXES) - 1:
                next_pos = STAR_INDEXES[idx + 1]
            else:
                next_pos = GOAL_INDEX
                
        if next_pos > GOAL_INDEX: return GOAL_INDEX
        return next_pos

    def _can_capture(self, next_pos: int, enemy_pieces: List[List[int]]) -> bool:
        # Cannot capture on safe globes/stars
        if next_pos in GLOBE_INDEXES or next_pos in STAR_INDEXES: return False
        if next_pos > 51: return False # Safe corridor
        
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == next_pos:
                    return True
        return False
        
    def _is_token_under_threat(self, token_pos: int, enemy_pieces: List[List[int]]) -> bool:
        if token_pos in GLOBE_INDEXES or token_pos in STAR_INDEXES or token_pos == HOME_INDEX or token_pos == GOAL_INDEX or token_pos > 51:
            return False
            
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos == HOME_INDEX or e_pos == GOAL_INDEX: continue
                # Simple distance check (approximate)
                dist = (token_pos - e_pos) % 52
                # If enemy is 1-6 steps behind us
                if 1 <= (-dist % 52) <= 6:
                    return True
        return False
    
    def _compute_reward_and_done(self, prev_state, current_state, action):
        """
        Pure RL Reward Structure (No +30 Heuristics).
        Gamma=0.99 handles the long term planning.
        """
        if self.game is None: return 0.0, False
        winners = self.game.get_winners_of_game()
        done = len(winners) > 0
        
        reward = 0.0
        
        # Win/Loss (Terminal)
        if done:
            if self.player_id in winners: return 50.0, True
            else: return -50.0, True
            
        if prev_state is None or prev_state.current_player != self.player_id:
            return 0.0, False

        # Reconstruct move details
        if action >= len(prev_state.player_pieces): return 0.0, False
        prev_pos = prev_state.player_pieces[action]
        curr_pos = current_state.player_pieces[action]
        
        # Did we move?
        if prev_pos != curr_pos:
            # Check Capture (+7)
            captured = False
            for enemy in prev_state.enemy_pieces:
                for e_pos in enemy:
                    if e_pos == curr_pos and e_pos not in [HOME_INDEX, GOAL_INDEX, 1] and e_pos not in GLOBE_INDEXES:
                        reward += 7.0
                        captured = True
            
            # Check Goal (+20)
            if curr_pos == GOAL_INDEX:
                reward += 20.0
            
            # Check Safe (+2 small incentive)
            elif curr_pos in GLOBE_INDEXES and prev_pos not in GLOBE_INDEXES:
                reward += 2.0
                
            # Exit Home (+10 encouragement)
            if prev_pos == HOME_INDEX and curr_pos != HOME_INDEX:
                reward += 10.0 
            
            # --- FEATURE PENALTY (ONLY IN EXPERIMENTAL MODE) ---
            if self.observation_mode == 'hybrid_penalty':
                # Entering a threat
                if self._is_token_under_threat(curr_pos, current_state.enemy_pieces):
                    if not self._is_token_under_threat(prev_pos, prev_state.enemy_pieces):
                        reward -= 2.0  # Enter Risk
                    else:
                        reward -= 1.0  # Stay Risk
                
                # Entering Static Danger
                elif curr_pos in DANGER_INDEXES and prev_pos not in DANGER_INDEXES:
                    reward -= 0.5

        # Check if we were captured (Passive penalty -5)
        for i in range(len(prev_state.player_pieces)):
            p_prev = prev_state.player_pieces[i]
            p_curr = current_state.player_pieces[i]
            if p_prev not in [HOME_INDEX, GOAL_INDEX] and p_curr == HOME_INDEX:
                reward -= 5.0

        return reward, False

    # ---------- Gym API ---------- 
    
    def reset(self, seed:Optional[int] = None, options:Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._set_seed(seed if seed is not None else self._seed)
        ghost_players = self._ghost_players_for_num_players()
        self.game = LudopyGame(ghost_players=ghost_players)
        self.game.reset()

        raw_obs , self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs
        dice,_,_,_,_,_ = raw_obs
        self.current_dice = int(dice)

        state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_obs(state)
        info = {"raw_obs": raw_obs, "state": state}
        self._prev_state = state
        return obs_vec, info 
    
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.game is None: raise RuntimeError("Reset first")
        prev_state = self._prev_state

        if self._last_raw_obs is None: raise RuntimeError("No observation available")
        _, move_pieces, _, _, _, _ = self._last_raw_obs
        
        if isinstance(move_pieces, np.ndarray): move_pieces = move_pieces.tolist()
        valid_pieces = [p for p in move_pieces if p < self.tokens_per_player]
        
        if not valid_pieces:
            # No valid pieces - use first move_piece if available, otherwise -1
            piece_to_move = move_pieces[0] if move_pieces else -1
        else:
            # Ensure action is in both valid_pieces AND move_pieces (game checks move_pieces)
            if action in move_pieces and action in valid_pieces:
                piece_to_move = action
            else:
                # Fallback: use first valid piece that's also in move_pieces
                piece_to_move = valid_pieces[0] if valid_pieces else (move_pieces[0] if move_pieces else -1)
        
        # Final safety check: ensure piece_to_move is in move_pieces
        if piece_to_move != -1 and piece_to_move not in move_pieces:
            piece_to_move = move_pieces[0] if move_pieces else -1
        
        _ = self.game.answer_observation(piece_to_move)

        raw_obs, self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs
        dice,_,_,_,_,_ = raw_obs
        self.current_dice = int(dice)

        next_state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_obs(next_state)

        # Rewards: Pure RL
        reward, terminated = self._compute_reward_and_done(prev_state, next_state, action)
        
        self._prev_state = next_state
        return obs_vec, reward, terminated, False, {"raw_obs": raw_obs, "state": next_state}

    def render(self, mode="human"):
        if self.game:
            return self.game.render_environment()
        return None
    
    def close(self):
        pass