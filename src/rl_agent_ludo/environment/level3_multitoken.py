"""
Level 3: Multi-Token Coordination

Game Rules:
- 2 players, 2 tokens each
- Full track (60 positions: 0=home, 1-59=track, 60=goal)
- NO six-to-exit rule (can always leave home)
- YES capturing (land on opponent → send them home)
- YES safe zones (positions 10, 20, 30, 40, 50)
- Win: BOTH tokens reach goal

State: ~14D vector (aggregate representation)
  My tokens:  [avg_pos, leading_pos, trailing_pos, num_home, num_goal, num_vulnerable, num_safe]
  Opp tokens: [avg_pos, leading_pos, trailing_pos, num_home, num_goal, num_vulnerable, num_safe]

Actions: 0=move token0, 1=move token1, 2=pass
  - Action masking used (can only move valid tokens)

Reward:
  - Progress reward (+1 per position moved)
  - Capture bonus (+30)
  - Captured penalty (-30)
  - Safe zone bonus (+5)
  - Token completion (+50 for getting one token home)
  - Win bonus (+100 for both tokens home)
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Level3MultiTokenLudo(gym.Env):
    """
    Level 3 Ludo with 2 tokens per player.

    Key challenge: Token coordination - which token to move when?
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Track configuration
    TRACK_LENGTH = 60
    GOAL_POSITION = 60
    HOME_POSITION = 0

    # Safe zones
    SAFE_ZONES = [10, 20, 30, 40, 50]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        # Action space: 0=move token0, 1=move token1, 2=pass
        self.action_space = spaces.Discrete(3)

        # Observation space: 14D (7 for my tokens + 7 for opponent tokens)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # Game state: 2 tokens per player
        self.my_tokens = [self.HOME_POSITION, self.HOME_POSITION]
        self.opp_tokens = [self.HOME_POSITION, self.HOME_POSITION]

        self.current_player = 0
        self.current_dice = 0
        self.done = False
        self.winner = None

        # Statistics
        self.total_steps = 0
        self.max_steps = 1000  # Longer since 2 tokens
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0
        self.num_my_tokens_finished = 0

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        self.np_random = np.random.default_rng(seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        # Reset game state
        self.my_tokens = [self.HOME_POSITION, self.HOME_POSITION]
        self.opp_tokens = [self.HOME_POSITION, self.HOME_POSITION]
        self.current_player = 0
        self.done = False
        self.winner = None
        self.total_steps = 0
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0
        self.num_my_tokens_finished = 0

        # Roll dice for player 0
        self.current_dice = self._roll_dice()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step of the game."""
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        self.total_steps += 1

        # Only process action for player 0 (our agent)
        if self.current_player == 0:
            reward = self._execute_player_action(action)
        else:
            # Opponent's turn (random policy)
            reward = 0.0
            self._execute_opponent_turn()

        # Check win conditions
        if self._all_tokens_at_goal(self.my_tokens):
            self.done = True
            self.winner = 0
            reward += 100.0  # Win bonus
        elif self._all_tokens_at_goal(self.opp_tokens):
            self.done = True
            self.winner = 1
            reward -= 100.0  # Loss penalty

        # Check max steps
        truncated = self.total_steps >= self.max_steps
        if truncated:
            self.done = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, self.done, truncated, info

    def _execute_player_action(self, action: int) -> float:
        """Execute player 0's action and return reward."""
        reward = 0.0

        if action in [0, 1]:  # Move token 0 or token 1
            token_idx = action

            # Check if this token can move
            if self._can_move_token(self.my_tokens, token_idx, self.current_dice):
                old_pos = self.my_tokens[token_idx]
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.my_tokens[token_idx] = new_pos

                # Progress reward
                progress = new_pos - old_pos
                reward += progress * 1.0

                # Token completion bonus
                if new_pos == self.GOAL_POSITION and old_pos < self.GOAL_POSITION:
                    reward += 50.0
                    self.num_my_tokens_finished += 1

                # Check captures (only if not at goal)
                if new_pos < self.GOAL_POSITION and new_pos > 0:
                    if not self._is_safe_zone(new_pos):
                        # Check if we captured any opponent token
                        for opp_idx in range(len(self.opp_tokens)):
                            if self.opp_tokens[opp_idx] == new_pos and self.opp_tokens[opp_idx] < self.GOAL_POSITION:
                                # Capture!
                                self.opp_tokens[opp_idx] = self.HOME_POSITION
                                reward += 30.0
                                self.num_captures_by_me += 1

                # Safe zone bonus
                if self._is_safe_zone(new_pos) and not self._is_safe_zone(old_pos):
                    reward += 5.0
            else:
                # Invalid action (tried to move token that can't move)
                reward -= 1.0
        else:  # Pass (action == 2)
            # Check if we could have moved any token
            if self._can_move_token(self.my_tokens, 0, self.current_dice) or \
               self._can_move_token(self.my_tokens, 1, self.current_dice):
                # Passed when we could move (bad action)
                reward -= 1.0

        # Switch to opponent
        self.current_player = 1
        self.current_dice = self._roll_dice()

        return reward

    def _execute_opponent_turn(self):
        """Execute opponent's turn (random policy)."""
        # Random policy: Pick a random movable token
        movable_tokens = []
        for idx in range(len(self.opp_tokens)):
            if self._can_move_token(self.opp_tokens, idx, self.current_dice):
                movable_tokens.append(idx)

        if len(movable_tokens) > 0:
            # 80% move a random valid token, 20% pass
            if self.np_random.random() < 0.8:
                token_idx = self.np_random.choice(movable_tokens)
                old_pos = self.opp_tokens[token_idx]
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.opp_tokens[token_idx] = new_pos

                # Check if opponent captured us
                if new_pos < self.GOAL_POSITION and new_pos > 0:
                    if not self._is_safe_zone(new_pos):
                        for my_idx in range(len(self.my_tokens)):
                            if self.my_tokens[my_idx] == new_pos and self.my_tokens[my_idx] < self.GOAL_POSITION:
                                # We got captured!
                                self.my_tokens[my_idx] = self.HOME_POSITION
                                self.num_captures_of_me += 1
                                # If this token was at goal, decrement counter
                                if self.my_tokens[my_idx] == self.GOAL_POSITION:
                                    self.num_my_tokens_finished = max(0, self.num_my_tokens_finished - 1)

        # Switch back to player 0
        self.current_player = 0
        self.current_dice = self._roll_dice()

    def _can_move_token(self, tokens: List[int], token_idx: int, dice: int) -> bool:
        """Check if a specific token can move."""
        if token_idx < 0 or token_idx >= len(tokens):
            return False
        current_pos = tokens[token_idx]
        # Can't move if already at goal
        return current_pos < self.GOAL_POSITION

    def _all_tokens_at_goal(self, tokens: List[int]) -> bool:
        """Check if all tokens are at goal."""
        return all(pos >= self.GOAL_POSITION for pos in tokens)

    def _is_safe_zone(self, position: int) -> bool:
        """Check if position is a safe zone."""
        return position in self.SAFE_ZONES

    def _is_vulnerable(self, tokens: List[int], token_idx: int) -> bool:
        """Check if a token is vulnerable to capture."""
        pos = tokens[token_idx]
        if pos == self.HOME_POSITION or pos >= self.GOAL_POSITION:
            return False
        if self._is_safe_zone(pos):
            return False
        return True

    def _roll_dice(self) -> int:
        """Roll a 6-sided dice."""
        return self.np_random.integers(1, 7)

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (14D vector).

        My tokens (7D):
          - avg_position_norm
          - leading_position_norm (furthest ahead)
          - trailing_position_norm (furthest behind)
          - num_at_home (0, 1, or 2) normalized to [0, 1]
          - num_at_goal (0, 1, or 2) normalized to [0, 1]
          - num_vulnerable (0, 1, or 2) normalized to [0, 1]
          - num_in_safe (0, 1, or 2) normalized to [0, 1]

        Opponent tokens (7D): same structure
        """
        # My tokens statistics
        my_positions = [min(p, self.GOAL_POSITION) for p in self.my_tokens]
        my_avg_pos = np.mean(my_positions) / self.GOAL_POSITION
        my_leading_pos = max(my_positions) / self.GOAL_POSITION
        my_trailing_pos = min(my_positions) / self.GOAL_POSITION
        my_num_home = sum(1 for p in self.my_tokens if p == self.HOME_POSITION) / 2.0
        my_num_goal = sum(1 for p in self.my_tokens if p >= self.GOAL_POSITION) / 2.0
        my_num_vuln = sum(1 for i in range(len(self.my_tokens)) if self._is_vulnerable(self.my_tokens, i)) / 2.0
        my_num_safe = sum(1 for p in self.my_tokens if self._is_safe_zone(p) and p > 0 and p < self.GOAL_POSITION) / 2.0

        # Opponent tokens statistics
        opp_positions = [min(p, self.GOAL_POSITION) for p in self.opp_tokens]
        opp_avg_pos = np.mean(opp_positions) / self.GOAL_POSITION
        opp_leading_pos = max(opp_positions) / self.GOAL_POSITION
        opp_trailing_pos = min(opp_positions) / self.GOAL_POSITION
        opp_num_home = sum(1 for p in self.opp_tokens if p == self.HOME_POSITION) / 2.0
        opp_num_goal = sum(1 for p in self.opp_tokens if p >= self.GOAL_POSITION) / 2.0
        opp_num_vuln = sum(1 for i in range(len(self.opp_tokens)) if self._is_vulnerable(self.opp_tokens, i)) / 2.0
        opp_num_safe = sum(1 for p in self.opp_tokens if self._is_safe_zone(p) and p > 0 and p < self.GOAL_POSITION) / 2.0

        obs = np.array([
            # My tokens (7D)
            my_avg_pos,
            my_leading_pos,
            my_trailing_pos,
            my_num_home,
            my_num_goal,
            my_num_vuln,
            my_num_safe,
            # Opponent tokens (7D)
            opp_avg_pos,
            opp_leading_pos,
            opp_trailing_pos,
            opp_num_home,
            opp_num_goal,
            opp_num_vuln,
            opp_num_safe,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        # Action mask: which actions are valid?
        action_mask = np.zeros(3, dtype=bool)
        action_mask[0] = self._can_move_token(self.my_tokens, 0, self.current_dice)
        action_mask[1] = self._can_move_token(self.my_tokens, 1, self.current_dice)
        action_mask[2] = True  # Can always pass

        return {
            'my_tokens': self.my_tokens.copy(),
            'opp_tokens': self.opp_tokens.copy(),
            'current_player': self.current_player,
            'current_dice': self.current_dice,
            'done': self.done,
            'winner': self.winner,
            'total_steps': self.total_steps,
            'num_captures_by_me': self.num_captures_by_me,
            'num_captures_of_me': self.num_captures_of_me,
            'num_my_tokens_finished': self.num_my_tokens_finished,
            'action_mask': action_mask,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print("\n" + "="*60)
            print(f"Step: {self.total_steps}")
            print(f"Current Player: {self.current_player}")
            print(f"Dice: {self.current_dice}")
            print(f"My tokens: {self.my_tokens} ({self.num_my_tokens_finished}/2 finished)")
            print(f"Opp tokens: {self.opp_tokens}")
            print(f"Captures: By me={self.num_captures_by_me}, Of me={self.num_captures_of_me}")

            # Visual track
            track = ['-'] * (self.GOAL_POSITION + 1)

            # Mark safe zones
            for safe_pos in self.SAFE_ZONES:
                track[safe_pos] = 'S'

            # Mark tokens (use letters for multiple tokens at same position)
            for pos in self.my_tokens:
                p = min(pos, self.GOAL_POSITION)
                if track[p] in ['-', 'S']:
                    track[p] = '0'
                else:
                    track[p] = 'M'  # Multiple tokens here

            for pos in self.opp_tokens:
                p = min(pos, self.GOAL_POSITION)
                if track[p] in ['-', 'S']:
                    track[p] = '1'
                elif track[p] in ['0', 'M']:
                    track[p] = 'X'  # Collision
                else:
                    track[p] = 'O'  # Multiple opponent tokens

            # Print track in chunks
            for i in range(0, self.GOAL_POSITION + 1, 20):
                chunk = track[i:i+20]
                print(f"[{i:2d}-{min(i+19, self.GOAL_POSITION):2d}] " + "".join(chunk))

            if self.done:
                if self.winner is not None:
                    print(f"\n🏆 Player {self.winner} WINS!")
                else:
                    print("\n⏱️  Game truncated")
            print("="*60)

    def close(self):
        """Clean up resources."""
        pass


def make_level3_env(seed: Optional[int] = None, render_mode: Optional[str] = None):
    """Create a Level 3 multi-token Ludo environment."""
    return Level3MultiTokenLudo(render_mode=render_mode, seed=seed)
