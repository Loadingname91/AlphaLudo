"""
Level 2: Interaction (Add Capturing)

Game Rules:
- 2 players, 1 token each
- Full track (60 positions: 0=home, 1-59=track, 60=goal)
- NO six-to-exit rule (can always leave home)
- YES capturing (land on opponent → send them home)
- YES safe zones (positions where you can't be captured)
- Win: First token to reach goal

State: 8D vector
  [my_pos_norm, opp_pos_norm, am_i_vulnerable, is_opp_vulnerable,
   am_i_in_safe, is_opp_in_safe, distance_to_opp_norm, my_progress]

Actions: 0=move token, 1=pass (if can't move)

Reward:
  - Dense distance-based shaping (+1 per position)
  - Capture bonus (+30)
  - Captured penalty (-30)
  - Safe zone bonus (+5)
  - Terminal win/loss (+100/-100)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Level2InteractionLudo(gym.Env):
    """
    Level 2 Ludo with capturing mechanics.

    Adds strategic depth: offensive (capture opponent) and defensive (avoid capture).
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Track configuration
    TRACK_LENGTH = 60
    GOAL_POSITION = 60
    HOME_POSITION = 0

    # Safe zones (can't be captured here)
    # Every 10 positions for simplicity
    SAFE_ZONES = [10, 20, 30, 40, 50]

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        # Action space: 0=move, 1=pass
        self.action_space = spaces.Discrete(2)

        # Observation space: 8D
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Game state
        self.player_positions = [self.HOME_POSITION, self.HOME_POSITION]
        self.current_player = 0
        self.current_dice = 0
        self.done = False
        self.winner = None

        # Episode statistics
        self.total_steps = 0
        self.max_steps = 500
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0

        # Random seed
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
        self.player_positions = [self.HOME_POSITION, self.HOME_POSITION]
        self.current_player = 0
        self.done = False
        self.winner = None
        self.total_steps = 0
        self.num_captures_by_me = 0
        self.num_captures_of_me = 0

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
        if self.player_positions[0] >= self.GOAL_POSITION:
            self.done = True
            self.winner = 0
            reward += 100.0
        elif self.player_positions[1] >= self.GOAL_POSITION:
            self.done = True
            self.winner = 1
            reward -= 100.0

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
        old_pos = self.player_positions[0]

        if action == 0:  # Move
            can_move = self._can_move(0, self.current_dice)
            if can_move:
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.player_positions[0] = new_pos

                # Progress reward
                progress = new_pos - old_pos
                reward += progress * 1.0

                # Check if we captured opponent
                if new_pos == self.player_positions[1] and new_pos > 0 and new_pos < self.GOAL_POSITION:
                    if not self._is_safe_zone(new_pos):
                        # Capture!
                        self.player_positions[1] = self.HOME_POSITION
                        reward += 30.0  # Capture bonus
                        self.num_captures_by_me += 1

                # Safe zone bonus (entered safe zone)
                if self._is_safe_zone(new_pos) and not self._is_safe_zone(old_pos):
                    reward += 5.0
            else:
                reward -= 1.0  # Invalid action
        else:  # Pass
            can_move = self._can_move(0, self.current_dice)
            if can_move:
                reward -= 1.0  # Passed when could move

        # Switch to opponent
        self.current_player = 1
        self.current_dice = self._roll_dice()

        return reward

    def _execute_opponent_turn(self):
        """Execute opponent's turn (random policy)."""
        can_move = self._can_move(1, self.current_dice)
        old_pos = self.player_positions[1]

        if can_move:
            # Random policy: 80% move, 20% pass
            if self.np_random.random() < 0.8:
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.player_positions[1] = new_pos

                # Check if opponent captured us
                if new_pos == self.player_positions[0] and new_pos > 0 and new_pos < self.GOAL_POSITION:
                    if not self._is_safe_zone(new_pos):
                        # We got captured!
                        self.player_positions[0] = self.HOME_POSITION
                        self.num_captures_of_me += 1

        # Switch back to player 0
        self.current_player = 0
        self.current_dice = self._roll_dice()

    def _can_move(self, player: int, dice: int) -> bool:
        """Check if player can move."""
        current_pos = self.player_positions[player]
        return current_pos < self.GOAL_POSITION

    def _is_safe_zone(self, position: int) -> bool:
        """Check if position is a safe zone."""
        return position in self.SAFE_ZONES

    def _is_vulnerable(self, player: int) -> bool:
        """
        Check if player is vulnerable to capture.

        Vulnerable if:
        - Not at home
        - Not at goal
        - Not in safe zone
        - Opponent can reach this position with current dice
        """
        my_pos = self.player_positions[player]
        opp_player = 1 - player

        # Not vulnerable if at home, goal, or safe zone
        if my_pos == self.HOME_POSITION or my_pos >= self.GOAL_POSITION:
            return False
        if self._is_safe_zone(my_pos):
            return False

        # Check if opponent can land on us
        # (This is approximate - we don't know future dice rolls)
        # For simplicity, just check if we're not in a safe zone and not home/goal
        return True

    def _roll_dice(self) -> int:
        """Roll a 6-sided dice."""
        return self.np_random.integers(1, 7)

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (8D vector).

        Returns:
            [my_pos_norm, opp_pos_norm, am_i_vulnerable, is_opp_vulnerable,
             am_i_in_safe, is_opp_in_safe, distance_to_opp_norm, my_progress]
        """
        my_pos = self.player_positions[0]
        opp_pos = self.player_positions[1]

        # Normalize positions
        my_pos_norm = my_pos / self.GOAL_POSITION
        opp_pos_norm = opp_pos / self.GOAL_POSITION

        # Vulnerability flags
        am_i_vulnerable = float(self._is_vulnerable(0))
        is_opp_vulnerable = float(self._is_vulnerable(1))

        # Safe zone flags
        am_i_in_safe = float(self._is_safe_zone(my_pos))
        is_opp_in_safe = float(self._is_safe_zone(opp_pos))

        # Distance to opponent (for capture opportunities)
        if my_pos < opp_pos:
            distance_to_opp = opp_pos - my_pos
        else:
            distance_to_opp = self.GOAL_POSITION  # Behind us, set to max
        distance_to_opp_norm = min(distance_to_opp, 60) / 60.0

        # Overall progress
        my_progress = my_pos / self.GOAL_POSITION

        obs = np.array([
            my_pos_norm,
            opp_pos_norm,
            am_i_vulnerable,
            is_opp_vulnerable,
            am_i_in_safe,
            is_opp_in_safe,
            distance_to_opp_norm,
            my_progress,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            'player_positions': self.player_positions.copy(),
            'current_player': self.current_player,
            'current_dice': self.current_dice,
            'done': self.done,
            'winner': self.winner,
            'total_steps': self.total_steps,
            'num_captures_by_me': self.num_captures_by_me,
            'num_captures_of_me': self.num_captures_of_me,
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            print("\n" + "="*60)
            print(f"Step: {self.total_steps}")
            print(f"Current Player: {self.current_player}")
            print(f"Dice: {self.current_dice}")
            print(f"Player 0 (Agent): Position {self.player_positions[0]}/{self.GOAL_POSITION}")
            print(f"Player 1 (Opponent): Position {self.player_positions[1]}/{self.GOAL_POSITION}")
            print(f"Captures: By me={self.num_captures_by_me}, Of me={self.num_captures_of_me}")

            # Visual track with safe zones
            track = ['-'] * (self.GOAL_POSITION + 1)

            # Mark safe zones
            for safe_pos in self.SAFE_ZONES:
                track[safe_pos] = 'S'

            # Mark player positions
            p0_pos = min(self.player_positions[0], self.GOAL_POSITION)
            p1_pos = min(self.player_positions[1], self.GOAL_POSITION)

            if p0_pos == p1_pos:
                track[p0_pos] = 'X'
            else:
                track[p0_pos] = '0'
                track[p1_pos] = '1'

            # Print track in chunks of 20
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


def make_level2_env(seed: Optional[int] = None, render_mode: Optional[str] = None):
    """Create a Level 2 Ludo environment."""
    return Level2InteractionLudo(render_mode=render_mode, seed=seed)
