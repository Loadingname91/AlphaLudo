"""
Ludo Gymnasium Environment.

Implements a flexible Ludo environment that can run:
- 2 players, 2 tokens each
- 2 players, 4 tokens each
- 3 players ,2 tokens each
- 3 players, 4 tokens each
- 4 players, 4 tokens each

"""
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import random

import gymnasium as gym
from gymnasium import spaces

from rl_agent_ludo.utils.state import State
from rl_agent_ludo.ludo.game import Game as LudopyGame


class LudoEnv(gym.Env):
    """
        Gymnasium compatible Ludo environment wrapping ludopy.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        player_id: int = 0,
        num_players: int = 4,
        tokens_per_player: int = 4,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args : 
            :param player_id: index of the learning agent (0-3 in ludopy numbering)
            :param num_players: 2, 3, or 4 active players
            :param tokens_per_player: 2 or 4 tokens per player (we restrict available pieces)
            :param render_mode: reserved for future rendering
            :param seed: random seed
        """
        super().__init__()

        assert num_players in (2, 3, 4), "num_players must be 2, 3, or 4"
        assert tokens_per_player in (2,4) , "tokens_per_player must be 2 or 4"
        assert 0 <= player_id < num_players, "player_id must be < num_players"

        self.player_id = player_id 
        self.num_players = num_players
        self.tokens_per_player = tokens_per_player
        self.render_mode = render_mode
        self._seed = seed

        # 4 piece actions space ; mask to tokens per player internally depending on the game configuration if its 2p2 or 4p4
        self.action_space = spaces.Discrete(4)

        # Nmeric observation spce metadata ( actual objs is state object)
        # 4 player pieces + up to 3*4 enemy pieces + dice + current_player
        low = np.array([-1] * (4 + 12) + [1] + [0], dtype=np.int32)
        high = np.array([57] * (4 +12) + [6] + [3], dtype=np.int32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.game: Optional[LudopyGame] = None
        self.current_player: int = 0 
        self.current_dice: int = 1
        self.current_move_pieces: List[int] = []
        self._last_raw_obs: Optional[Tuple] = None
        self._prev_state: Optional[State] = None  # Track previous state for reward calculation

    
    # ---------- helpers ---------- 

    def _set_seed(self, seed: Optional[int]) -> None:
        """
        Seed numpy and random if a seed is provided.
        """
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)

    def _ghost_players_for_num_players(self) -> List[int]:
        """
        Map num players (2/3/4) to ludopy ghost_players list.
        player indices are 0..3 in ludopy.
        """
        if self.num_players == 4:
            return []
        if self.num_players == 3:
            return [3]
        if self.num_players == 2:
            return [2,3]
        return []
    
    def _apply_token_limit(self, pieces: List[int]) -> List[int]:
        """
        Enforce tokens per player by freezing extra tokens at home (-1)
        """
        if self.tokens_per_player == 4:
            return pieces 
        # if token per player is 2, then we keep first 2 and mask others home
        tokens_on_board = list(pieces)
        for i in range(2,len(tokens_on_board)):
            tokens_on_board[i] = -1
        return tokens_on_board

    def _build_state_from_obs(self, obs:Tuple) -> State:
        """
        Convert ludopy observation into our state dataclass

        ludopy obs:
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner)
        """

        dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = obs

        # apply token limit to current player's pieces
        player_pieces = self._apply_token_limit(list(player_pieces))

        # enemy_pieces is list of arrays, one per enemy
        enemy_pieces = [self._apply_token_limit(list(ep)) for ep in enemy_pieces]

        # Filter valid moves so we never move beyond tokens_per_player
        # move_pieces contains piece indices 0..3 that ludopy says can move.
        move_pieces = [p for p in move_pieces if p < self.tokens_per_player]

        if len(move_pieces) > 0:
            valid_moves = list(range(len(move_pieces)))
            movable_pieces = list(move_pieces)
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

    def _state_to_vec(self, state: State) -> np.ndarray:
        """
        Convert State object to numpy array matching observation_space.
        
        Returns array of shape (18,): [4 player pieces, 12 enemy pieces, dice, current_player]
        """
        # Flatten: 4 player pieces + 3 enemies × 4 pieces + dice + current_player
        vec = (
            list(state.player_pieces)  # 4 elements
            + [p for enemy in state.enemy_pieces for p in enemy]  # 12 elements (3×4)
            + [state.dice_roll, state.current_player]  # 2 elements
        )
        return np.asarray(vec, dtype=np.int32)

        

    def _is_safe_position(self, pos: int) -> bool:
        """
        Check if a position is safe (globe, home corridor, home, or goal).
        """
        HOME_INDEX = 0
        GOAL_INDEX = 57
        GLOBE_INDEXES = [1, 9, 22, 35, 48]
        HOME_CORRIDOR = list(range(52, 57))
        
        return (
            pos == HOME_INDEX
            or pos == GOAL_INDEX
            or pos in HOME_CORRIDOR
            or pos in GLOBE_INDEXES
        )
    
    def _is_token_under_threat(self, token_pos: int, enemy_pieces: List[List[int]]) -> bool:
        """
        Check if a token is under threat from enemy pieces.
        A token is threatened if an enemy can reach it within 6 moves (one dice roll).
        """
        if self._is_safe_position(token_pos):
            return False
        
        for enemy in enemy_pieces:
            for e_pos in enemy:
                if e_pos in [0, 57] or self._is_safe_position(e_pos):
                    continue
                # Check if enemy is within 6 positions (can reach in one turn)
                # Simple circular distance check
                if 1 <= token_pos <= 51 and 1 <= e_pos <= 51:
                    diff = abs(token_pos - e_pos)
                    dist = min(diff, 51 - diff)
                    if 1 <= dist <= 6:
                        return True
        return False
    
    def _get_furthest_token(self, state: State) -> Optional[int]:
        """
        Get the index of the token that is furthest along the path.
        Returns None if no tokens are on the path.
        """
        HOME_INDEX = 0
        GOAL_INDEX = 57
        HOME_CORRIDOR = list(range(52, 57))
        
        furthest_idx = None
        furthest_pos = -1
        
        for i, pos in enumerate(state.player_pieces):
            if pos == HOME_INDEX or pos == GOAL_INDEX:
                continue
            # For tokens in home corridor, use position directly
            if pos in HOME_CORRIDOR:
                if pos > furthest_pos:
                    furthest_pos = pos
                    furthest_idx = i
            # For tokens on main path
            elif 1 <= pos <= 51:
                if pos > furthest_pos:
                    furthest_pos = pos
                    furthest_idx = i
        
        return furthest_idx
    
    def _compute_rich_reward(
        self, 
        prev_state: Optional[State], 
        current_state: State,
        action: int,
        done: bool
    ) -> float:
        """
        Compute rich reward structure matching reference repository.
        
        Rewards:
        - +5: Successful defense (token under threat, moved to safety)
        - -5: Failed defense (token under threat, was captured)
        - +7: Capture enemy token
        - -7: Being captured by enemy
        - +18: Exit home (move token from home to board)
        - +5: Reach safe zone
        - +20: Reach goal
        - +30: Favorable move (move furthest token when indecisive)
        - -30: Unfavorable move (move closer token when furthest available)
        - +50: Win game
        - -50: Lose game (or -1 to keep magnitude reasonable)
        """
        if prev_state is None:
            # First step, no reward
            return 0.0
        
        reward = 0.0
        HOME_INDEX = 0
        GOAL_INDEX = 57
        HOME_CORRIDOR = list(range(52, 57))
        GLOBE_INDEXES = [1, 9, 22, 35, 48]
        
        # Only compute rewards when it's our turn (player_id == 0)
        # We need to check if the action was taken by our agent
        if prev_state.current_player != self.player_id:
            # Not our turn, but check if any of our tokens were captured during enemy turns
            # (tokens that were on board are now at home)
            reward = 0.0
            for i in range(len(prev_state.player_pieces)):
                prev_pos = prev_state.player_pieces[i]
                curr_pos = current_state.player_pieces[i]
                # Token was on board and is now at home - it was captured
                if prev_pos not in [HOME_INDEX, GOAL_INDEX] and curr_pos == HOME_INDEX:
                    reward -= 7
            return reward
        
        # Get the piece that was moved
        if not prev_state.valid_moves or action >= len(prev_state.valid_moves):
            return 0.0
        
        if prev_state.movable_pieces:
            piece_idx = prev_state.movable_pieces[action]
        else:
            piece_idx = action
        
        if piece_idx >= len(prev_state.player_pieces):
            return 0.0
        
        prev_pos = prev_state.player_pieces[piece_idx]
        curr_pos = current_state.player_pieces[piece_idx]
        
        # 1. Check for capture (offense) - our token lands on enemy position
        captured_enemy = False
        for enemy in prev_state.enemy_pieces:
            for e_pos in enemy:
                if e_pos == curr_pos and e_pos not in [HOME_INDEX, GOAL_INDEX] and not self._is_safe_position(e_pos):
                    captured_enemy = True
                    reward += 7
                    break
            if captured_enemy:
                break
        
        # 2. Check for being captured (defense failure)
        # Our token was captured if it was on the board and is now back at home
        # (and we didn't just move it from home - that would be exiting home, not capture)
        was_captured = False
        if prev_pos not in [HOME_INDEX, GOAL_INDEX] and curr_pos == HOME_INDEX:
            # Token was on board and is now at home - it was captured
            was_captured = True
            reward -= 7
        
        # 3. Check for defense (token was under threat, moved to safety)
        was_under_threat = self._is_token_under_threat(prev_pos, prev_state.enemy_pieces)
        is_now_safe = self._is_safe_position(curr_pos)
        
        if was_under_threat and is_now_safe and not was_captured:
            reward += 5  # Successful defense
        elif was_under_threat and was_captured:
            reward -= 5  # Failed defense
        
        # 4. Check for exiting home
        if prev_pos == HOME_INDEX and curr_pos != HOME_INDEX:
            reward += 18
        
        # 5. Check for reaching safe zone
        if not self._is_safe_position(prev_pos) and is_now_safe and curr_pos != GOAL_INDEX:
            reward += 5
        
        # 6. Check for reaching goal
        if prev_pos != GOAL_INDEX and curr_pos == GOAL_INDEX:
            reward += 20
        
        # 7. Check for favorable/unfavorable moves (strategic decision)
        # This happens when we have multiple valid moves and need to choose
        if len(prev_state.valid_moves) > 1:
            furthest_idx = self._get_furthest_token(prev_state)
            if furthest_idx is not None:
                if piece_idx == furthest_idx:
                    reward += 30  # Favorable: moved furthest token
                else:
                    # Check if furthest token was available but we moved a different one
                    if furthest_idx in (prev_state.movable_pieces or []):
                        reward -= 30  # Unfavorable: should have moved furthest
        
        # 8. Win/loss rewards
        if done:
            winners = self.game.get_winners_of_game() if self.game else []
            if self.player_id in winners:
                reward += 50
            else:
                reward -= 1  # Keep loss penalty reasonable (or use -50 if preferred)
        
        return reward
    
    def _compute_reward_and_done(self, prev_state: Optional[State] = None, current_state: Optional[State] = None, action: int = 0) -> Tuple[float, bool]:
        """
        Compute rich reward structure matching reference repository.
        Falls back to sparse reward if prev_state is None.
        """
        if self.game is None:
            return 0.0, False 
        
        winners = self.game.get_winners_of_game()
        done = len(winners) > 0
        
        if prev_state is not None and current_state is not None:
            # Use rich reward structure
            reward = self._compute_rich_reward(prev_state, current_state, action, done)
            return reward, done
        else:
            # Fallback to sparse reward
            if not done:
                return 0.0, False
            if self.player_id in winners:
                return 1.0, True
            else:
                return -1.0, True
        
    # ---------- Gymnasium API ---------- 
    
    def reset(self, seed:Optional[int] = None, options:Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return initial (observation, info).
        
        Returns:
            observation: numpy array matching observation_space (18 elements)
            info: dict containing full State object and raw_obs
        """
        # Prefer explicit seed, fall back to env-level seed
        self._set_seed(seed if seed is not None else self._seed)

        # Underlying Game currently only takes ghost_players; it always has 4 players and 4 tokens.
        # We therefore instantiate with ghost players based on num_players, and enforce tokens_per_player
        # via _apply_token_limit and action masking.
        ghost_players = self._ghost_players_for_num_players()
        self.game = LudopyGame(ghost_players=ghost_players)
        self.game.reset()

        raw_obs , self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs

        dice,move_pieces,player_pieces,enemy_pieces,player_is_winner,there_is_winner = raw_obs
        self.current_dice = int(dice)
        self.current_move_pieces = list(move_pieces) if len(move_pieces) >0 else []

        state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_vec(state)
        info: Dict[str, Any] = {
            "raw_obs": raw_obs,
            "state": state,  # Full State object for agents that need it
        }
        
        # Initialize previous state tracking with initial state
        self._prev_state = state

        return obs_vec, info 
    
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply one action (index into current valid moves) in the environment.
        
        Returns:
            observation: numpy array matching observation_space (18 elements)
            reward: float reward
            terminated: bool indicating if episode ended
            truncated: bool indicating if episode was truncated
            info: dict containing full State object and raw_obs
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        # Store previous state BEFORE action (from last observation)
        prev_state = self._prev_state if self._prev_state else None
        
        # Map action index -> actual piece index via current_move_pieces
        valid_moves = self.current_move_pieces

        if len(valid_moves) == 0:
            # No valid moves; pass 0
            piece_to_move = 0
        else:
            # Clamp action index; valid_moves may have been filtered for token limit
            if action < 0 or action >= len(valid_moves):
                piece_to_move = valid_moves[0]
            else:
                piece_to_move = valid_moves[action]

        # Execute in ludopy
        _ = self.game.answer_observation(piece_to_move)

        # Next observation
        raw_obs, self.current_player = self.game.get_observation()
        self._last_raw_obs = raw_obs

        dice, move_pieces, player_pieces, enemy_pieces, player_is_winner, there_is_winner = raw_obs
        self.current_dice = int(dice)
        self.current_move_pieces = list(move_pieces) if len(move_pieces) > 0 else []

        next_state = self._build_state_from_obs(raw_obs)
        obs_vec = self._state_to_vec(next_state)

        # Compute reward using rich reward structure
        # Only give reward if it was our agent's turn
        if prev_state and prev_state.current_player == self.player_id:
            reward, terminated = self._compute_reward_and_done(prev_state, next_state, action)
        else:
            # Not our turn or first step - use sparse reward
            reward, terminated = self._compute_reward_and_done()
        truncated = False  # no time-limit yet

        # Update previous state for next step (store current state)
        self._prev_state = next_state

        info: Dict[str, Any] = {
            "raw_obs": raw_obs,
            "state": next_state,  # Full State object for agents that need it
        }

        return obs_vec, reward, terminated, truncated, info
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            - "human": Display the board using OpenCV (requires opencv-python)
            - "rgb_array": Return the board image as numpy array (RGB)
            
        Returns:
            Board image as numpy array (RGB) if mode="rgb_array", None if mode="human"
        """
        if self.game is None:
            raise RuntimeError("Environment not reset. Call reset() before render().")
        
        board_img = self.game.render_environment()
        
        if mode == "human":
            try:
                import cv2
                # Convert RGB to BGR for OpenCV display
                board_img_bgr = cv2.cvtColor(board_img, cv2.COLOR_RGB2BGR)
                
                # Get image dimensions
                img_height, img_width = board_img_bgr.shape[:2]
                
                # Set maximum display size for small screens (conservative defaults)
                # These values work well for screens as small as 1280x720 or 1366x768
                max_display_width = 1024   # Max width for small screens
                max_display_height = 768  # Max height for small screens
                
                # Calculate scale factor to fit within max dimensions while maintaining aspect ratio
                scale_w = max_display_width / img_width
                scale_h = max_display_height / img_height
                scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale
                
                # Resize if needed to fit small screens
                if scale < 1.0:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    board_img_bgr = cv2.resize(
                        board_img_bgr, 
                        (new_width, new_height), 
                        interpolation=cv2.INTER_AREA
                    )
                
                window_name = f"Ludo - Player {self.player_id}"
                cv2.imshow(window_name, board_img_bgr)
                cv2.waitKey(1)  # Non-blocking wait (1ms) - allows window to update
                return None
            except ImportError:
                print("Warning: opencv-python not installed. Cannot display board. Install with: pip install opencv-python")
                return board_img
        elif mode == "rgb_array":
            return board_img
        else:
            raise ValueError(f"Unknown render mode: {mode}. Supported modes: 'human', 'rgb_array'")
    
    def close(self) -> None:
        """
        Cleanup any open resources, including OpenCV windows.
        """
        # Close OpenCV windows if they exist
        try:
            import cv2
            cv2.destroyAllWindows()
        except ImportError:
            pass  # opencv-python not installed, nothing to close
        
        if self.game is not None:
            self.game.reset()
        self.game = None
        self.current_player = 0
        self.current_dice = 1
        self.current_move_pieces = []
        self._last_raw_obs = None
        self._prev_state = None
        
