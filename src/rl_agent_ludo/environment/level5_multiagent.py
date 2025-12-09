"""
Level 5: Multi-Agent Chaos (4 players, 2 tokens each)

Final level before the ultimate challenge:
- 4 players competing simultaneously
- 2 tokens per player (must finish both to win)
- Full rules: six-to-exit, capturing, safe zones
- 16D state space (aggregate opponent features)
- 3 actions: move token0, move token1, pass

Target: 52%+ win rate (much harder with 3 opponents!)
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, List, Optional


class Level5MultiAgentLudo(gym.Env):
    """Level 5: 4-player Ludo with 2 tokens each."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Game constants
    TRACK_LENGTH = 60
    GOAL_POSITION = 60
    HOME_POSITION = 0
    SAFE_ZONES = [10, 20, 30, 40, 50]
    NUM_PLAYERS = 4
    TOKENS_PER_PLAYER = 2

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        super().__init__()

        self.render_mode = render_mode
        self.np_random = np.random.default_rng(seed)

        # State: 16D vector
        # - My tokens (9D): avg_pos, leading, trailing, num_home, num_finished,
        #                   vulnerable_0, vulnerable_1, can_exit_0, can_exit_1
        # - All opponents (6D): avg_pos, leading, trailing, num_home, num_finished, num_vulnerable
        # - Dice (1D): normalized current dice
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(16,), dtype=np.float32
        )

        # Actions: 0 = move token 0, 1 = move token 1, 2 = pass
        self.action_space = gym.spaces.Discrete(3)

        # Game state
        self.my_tokens = [self.HOME_POSITION] * self.TOKENS_PER_PLAYER
        self.opponent_tokens = [
            [self.HOME_POSITION] * self.TOKENS_PER_PLAYER
            for _ in range(self.NUM_PLAYERS - 1)
        ]  # 3 opponents with 2 tokens each
        self.current_dice = 0
        self.turn_count = 0
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Reset all tokens to home
        self.my_tokens = [self.HOME_POSITION] * self.TOKENS_PER_PLAYER
        self.opponent_tokens = [
            [self.HOME_POSITION] * self.TOKENS_PER_PLAYER
            for _ in range(self.NUM_PLAYERS - 1)
        ]
        self.current_dice = self._roll_dice()
        self.turn_count = 0
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        reward = 0.0
        terminated = False

        # Execute player's action
        if action < 2:  # Move token 0 or 1
            token_idx = action
            if self._can_move_token(self.my_tokens, token_idx, self.current_dice):
                old_pos = self.my_tokens[token_idx]
                new_pos = self._calculate_new_position(old_pos, self.current_dice)
                self.my_tokens[token_idx] = new_pos

                # Reward shaping
                if new_pos == self.GOAL_POSITION:
                    reward += 100.0  # Token reached goal
                else:
                    reward += 1.0  # Progress reward

                # Check for captures
                capture_reward = self._check_captures(new_pos, player=0)
                reward += capture_reward
                self.num_captures_by_me += (capture_reward > 0)

        # Check if player won (both tokens at goal)
        if self._check_win(self.my_tokens):
            reward += 500.0  # Win bonus
            terminated = True
            winner = 0
        else:
            # Execute opponent turns
            winner = self._execute_all_opponent_turns()
            if winner is not None:
                reward -= 500.0  # Opponent won
                terminated = True

        # Roll dice for next turn
        self.current_dice = self._roll_dice()
        self.turn_count += 1

        # Truncate if game too long
        truncated = self.turn_count >= 300

        observation = self._get_observation()
        info = self._get_info()
        if terminated or truncated:
            info['winner'] = winner if winner is not None else -1

        return observation, reward, terminated, truncated, info

    def _execute_all_opponent_turns(self) -> Optional[int]:
        """Execute turns for all 3 opponents. Returns winner player number if any won."""
        for opp_idx in range(self.NUM_PLAYERS - 1):
            # Roll dice for this opponent
            dice = self._roll_dice()

            # Random policy: choose random valid action (80% move, 20% pass)
            valid_moves = []
            for token_idx in range(self.TOKENS_PER_PLAYER):
                if self._can_move_token(self.opponent_tokens[opp_idx], token_idx, dice):
                    valid_moves.append(token_idx)

            if valid_moves and self.np_random.random() < 0.8:
                token_idx = self.np_random.choice(valid_moves)
                old_pos = self.opponent_tokens[opp_idx][token_idx]
                new_pos = self._calculate_new_position(old_pos, dice)
                self.opponent_tokens[opp_idx][token_idx] = new_pos

                # Check if opponent captured our token
                for my_token_idx in range(self.TOKENS_PER_PLAYER):
                    if (self.my_tokens[my_token_idx] == new_pos and
                        new_pos not in self.SAFE_ZONES and
                        new_pos > self.HOME_POSITION and
                        new_pos < self.GOAL_POSITION):
                        self.my_tokens[my_token_idx] = self.HOME_POSITION
                        self.num_captures_of_me += 1

                # Check if opponent captured another opponent's token
                for other_opp_idx in range(self.NUM_PLAYERS - 1):
                    if other_opp_idx == opp_idx:
                        continue
                    for token_idx2 in range(self.TOKENS_PER_PLAYER):
                        if (self.opponent_tokens[other_opp_idx][token_idx2] == new_pos and
                            new_pos not in self.SAFE_ZONES and
                            new_pos > self.HOME_POSITION and
                            new_pos < self.GOAL_POSITION):
                            self.opponent_tokens[other_opp_idx][token_idx2] = self.HOME_POSITION

            # Check if this opponent won
            if self._check_win(self.opponent_tokens[opp_idx]):
                return opp_idx + 1  # Player 1, 2, or 3

        return None

    def _check_captures(self, new_pos: int, player: int) -> float:
        """Check if landing on new_pos captures any opponent tokens. Returns reward."""
        if new_pos in self.SAFE_ZONES or new_pos <= self.HOME_POSITION or new_pos >= self.GOAL_POSITION:
            return 0.0

        capture_reward = 0.0

        # Check all opponent tokens
        for opp_idx in range(self.NUM_PLAYERS - 1):
            for token_idx in range(self.TOKENS_PER_PLAYER):
                if self.opponent_tokens[opp_idx][token_idx] == new_pos:
                    # Capture!
                    self.opponent_tokens[opp_idx][token_idx] = self.HOME_POSITION
                    capture_reward += 30.0

        return capture_reward

    def _can_move_token(self, tokens: List[int], token_idx: int, dice: int) -> bool:
        """Check if token can move with current dice (six-to-exit rule)."""
        if token_idx < 0 or token_idx >= len(tokens):
            return False

        current_pos = tokens[token_idx]

        # Can't move if already at goal
        if current_pos >= self.GOAL_POSITION:
            return False

        # Six-to-exit rule: need 6 to leave home
        if current_pos == self.HOME_POSITION:
            return dice == 6

        return True

    def _calculate_new_position(self, old_pos: int, dice: int) -> int:
        """Calculate new position after moving."""
        if old_pos == self.HOME_POSITION:
            # Leaving home (must have rolled 6)
            return 1
        else:
            return min(old_pos + dice, self.GOAL_POSITION)

    def _check_win(self, tokens: List[int]) -> bool:
        """Check if all tokens reached goal."""
        return all(pos >= self.GOAL_POSITION for pos in tokens)

    def _roll_dice(self) -> int:
        """Roll a 6-sided die."""
        return int(self.np_random.integers(1, 7))

    def _is_safe_zone(self, position: int) -> bool:
        """Check if position is a safe zone."""
        return position in self.SAFE_ZONES

    def _is_vulnerable(self, tokens: List[int], token_idx: int) -> bool:
        """Check if token is vulnerable to capture."""
        pos = tokens[token_idx]
        if pos == self.HOME_POSITION or pos >= self.GOAL_POSITION:
            return False
        if self._is_safe_zone(pos):
            return False
        return True

    def _get_observation(self) -> np.ndarray:
        """
        Return 16D observation vector:
        - My tokens (9D): avg_pos, leading, trailing, num_home, num_finished,
                          vulnerable_0, vulnerable_1, can_exit_0, can_exit_1
        - All opponents (6D): avg_pos, leading, trailing, num_home, num_finished, num_vulnerable
        - Dice (1D): normalized current dice
        """
        # My tokens features
        my_positions = [p for p in self.my_tokens if p < self.GOAL_POSITION]
        if len(my_positions) > 0:
            my_avg_pos = np.mean(my_positions) / self.GOAL_POSITION
            my_leading = max(my_positions) / self.GOAL_POSITION
            my_trailing = min(my_positions) / self.GOAL_POSITION
        else:
            my_avg_pos = 1.0
            my_leading = 1.0
            my_trailing = 1.0

        my_num_home = sum(1 for p in self.my_tokens if p == self.HOME_POSITION) / self.TOKENS_PER_PLAYER
        my_num_finished = sum(1 for p in self.my_tokens if p >= self.GOAL_POSITION) / self.TOKENS_PER_PLAYER
        my_vulnerable_0 = 1.0 if self._is_vulnerable(self.my_tokens, 0) else 0.0
        my_vulnerable_1 = 1.0 if self._is_vulnerable(self.my_tokens, 1) else 0.0
        my_can_exit_0 = 1.0 if (self.my_tokens[0] == self.HOME_POSITION and self.current_dice == 6) else 0.0
        my_can_exit_1 = 1.0 if (self.my_tokens[1] == self.HOME_POSITION and self.current_dice == 6) else 0.0

        # All opponents aggregated (6 tokens total from 3 players)
        all_opp_tokens = []
        for opp_tokens in self.opponent_tokens:
            all_opp_tokens.extend(opp_tokens)

        opp_positions = [p for p in all_opp_tokens if p < self.GOAL_POSITION]
        if len(opp_positions) > 0:
            opp_avg_pos = np.mean(opp_positions) / self.GOAL_POSITION
            opp_leading = max(opp_positions) / self.GOAL_POSITION
            opp_trailing = min(opp_positions) / self.GOAL_POSITION
        else:
            opp_avg_pos = 1.0
            opp_leading = 1.0
            opp_trailing = 1.0

        total_opp_tokens = len(all_opp_tokens)
        opp_num_home = sum(1 for p in all_opp_tokens if p == self.HOME_POSITION) / total_opp_tokens
        opp_num_finished = sum(1 for p in all_opp_tokens if p >= self.GOAL_POSITION) / total_opp_tokens

        # Count vulnerable opponent tokens
        opp_num_vulnerable = 0
        for opp_idx in range(self.NUM_PLAYERS - 1):
            for token_idx in range(self.TOKENS_PER_PLAYER):
                if self._is_vulnerable(self.opponent_tokens[opp_idx], token_idx):
                    opp_num_vulnerable += 1
        opp_num_vulnerable /= total_opp_tokens

        # Dice
        dice_norm = self.current_dice / 6.0

        obs = np.array([
            # My tokens (9D)
            my_avg_pos, my_leading, my_trailing, my_num_home, my_num_finished,
            my_vulnerable_0, my_vulnerable_1, my_can_exit_0, my_can_exit_1,
            # All opponents (6D)
            opp_avg_pos, opp_leading, opp_trailing, opp_num_home, opp_num_finished, opp_num_vulnerable,
            # Dice (1D)
            dice_norm,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict:
        """Return info dict with action mask and game state."""
        # Action mask: which actions are valid?
        action_mask = np.zeros(3, dtype=bool)
        action_mask[0] = self._can_move_token(self.my_tokens, 0, self.current_dice)
        action_mask[1] = self._can_move_token(self.my_tokens, 1, self.current_dice)
        action_mask[2] = True  # Can always pass

        # Ensure at least one action is valid
        if not np.any(action_mask[:2]):
            action_mask[2] = True

        return {
            'action_mask': action_mask,
            'my_tokens': self.my_tokens.copy(),
            'opponent_tokens': [tokens.copy() for tokens in self.opponent_tokens],
            'current_dice': self.current_dice,
            'turn_count': self.turn_count,
            'num_my_tokens_finished': sum(1 for p in self.my_tokens if p >= self.GOAL_POSITION),
            'num_captures_by_me': self.num_captures_by_me,
            'num_captures_of_me': self.num_captures_of_me,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print(f"\n{'='*60}")
            print(f"Turn {self.turn_count}, Dice: {self.current_dice}")
            print(f"Player 0 (me): {self.my_tokens}")
            for i, tokens in enumerate(self.opponent_tokens):
                print(f"Player {i+1}: {tokens}")
            print(f"Captures: by me={self.num_captures_by_me}, of me={self.num_captures_of_me}")
            print(f"{'='*60}")

    def close(self):
        """Clean up resources."""
        pass
