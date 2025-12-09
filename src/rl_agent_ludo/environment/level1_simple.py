"""
Level 1: Simplest Ludo Environment

Game Rules:
- 2 players, 1 token each
- Linear track (60 positions: 0=home, 1-59=track, 60=goal)
- NO six-to-exit rule (can always leave home)
- NO capturing
- NO safe zones
- Win: First token to reach goal

State: 4D vector [my_pos_norm, opp_pos_norm, can_i_move, can_opp_move]
Actions: 0=move token, 1=pass (if can't move)
Reward: Dense distance-based shaping + terminal win/loss
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Level1SimpleLudo(gym.Env):
    """
    Simplest possible Ludo environment for curriculum learning.

    This is Level 1 of 6 in the progressive difficulty curriculum.
    Goal: Learn basic "move forward to win" strategy.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    # Track configuration
    TRACK_LENGTH = 60  # 0=home, 1-59=track, 60=goal
    GOAL_POSITION = 60
    HOME_POSITION = 0

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        # Action space: 0=move, 1=pass
        self.action_space = spaces.Discrete(2)

        # Observation space: 4D [my_pos_norm, opp_pos_norm, can_i_move, can_opp_move]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Game state
        self.player_positions = [self.HOME_POSITION, self.HOME_POSITION]  # [player0, player1]
        self.current_player = 0  # 0 or 1
        self.current_dice = 0
        self.done = False
        self.winner = None

        # For reward shaping
        self.prev_my_position = self.HOME_POSITION

        # Episode statistics
        self.total_steps = 0
        self.max_steps = 500  # Safety cap

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

        # Roll dice for player 0
        self.current_dice = self._roll_dice()

        # Store for reward shaping
        self.prev_my_position = self.player_positions[0]

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the game.

        Args:
            action: 0=move, 1=pass

        Returns:
            observation, reward, terminated, truncated, info
        """
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
            reward += 100.0  # Win bonus
        elif self.player_positions[1] >= self.GOAL_POSITION:
            self.done = True
            self.winner = 1
            reward -= 100.0  # Loss penalty

        # Check max steps (truncation)
        truncated = self.total_steps >= self.max_steps
        if truncated:
            self.done = True

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, self.done, truncated, info

    def _execute_player_action(self, action: int) -> float:
        """
        Execute player 0's action and return reward.

        Returns:
            reward: Dense distance-based shaping reward
        """
        reward = 0.0

        if action == 0:  # Move
            can_move = self._can_move(0, self.current_dice)
            if can_move:
                # Move the token
                old_pos = self.player_positions[0]
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.player_positions[0] = new_pos

                # Dense distance-based shaping reward
                # Progress reward: +1 for each position moved
                progress = (new_pos - old_pos)
                reward += progress * 1.0

                # Update prev position for next step
                self.prev_my_position = new_pos
            else:
                # Tried to move but couldn't (invalid action)
                reward -= 1.0  # Small penalty
        else:  # Pass
            # Passing is valid if we can't move
            can_move = self._can_move(0, self.current_dice)
            if can_move:
                # Passed when we could move (bad action)
                reward -= 1.0

        # Switch to opponent's turn
        self.current_player = 1
        self.current_dice = self._roll_dice()

        return reward

    def _execute_opponent_turn(self):
        """Execute opponent's turn (random policy)."""
        can_move = self._can_move(1, self.current_dice)

        if can_move:
            # Random policy: 80% move, 20% pass (like RandomAgent)
            if self.np_random.random() < 0.8:
                # Move opponent token
                old_pos = self.player_positions[1]
                new_pos = min(old_pos + self.current_dice, self.GOAL_POSITION)
                self.player_positions[1] = new_pos
            # else: pass (don't move)

        # Switch back to player 0's turn
        self.current_player = 0
        self.current_dice = self._roll_dice()

    def _can_move(self, player: int, dice: int) -> bool:
        """
        Check if player can move with given dice roll.

        In Level 1:
        - Can always leave home (no six-to-exit rule)
        - Can move if not already at goal
        """
        current_pos = self.player_positions[player]
        return current_pos < self.GOAL_POSITION

    def _roll_dice(self) -> int:
        """Roll a 6-sided dice."""
        return self.np_random.integers(1, 7)  # 1-6 inclusive

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (4D vector).

        Returns:
            [my_pos_norm, opp_pos_norm, can_i_move, can_opp_move]
        """
        my_pos = self.player_positions[0]
        opp_pos = self.player_positions[1]

        # Normalize positions to [0, 1]
        my_pos_norm = my_pos / self.GOAL_POSITION
        opp_pos_norm = opp_pos / self.GOAL_POSITION

        # Can move flags (only meaningful for current player)
        if self.current_player == 0:
            can_i_move = float(self._can_move(0, self.current_dice))
            can_opp_move = 0.0  # Unknown (not opponent's turn)
        else:
            can_i_move = 0.0  # Not our turn
            can_opp_move = float(self._can_move(1, self.current_dice))

        obs = np.array(
            [my_pos_norm, opp_pos_norm, can_i_move, can_opp_move],
            dtype=np.float32
        )

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging/logging."""
        return {
            'player_positions': self.player_positions.copy(),
            'current_player': self.current_player,
            'current_dice': self.current_dice,
            'done': self.done,
            'winner': self.winner,
            'total_steps': self.total_steps,
        }

    def render(self):
        """Render the environment (simple text display)."""
        if self.render_mode == "human":
            print("\n" + "="*60)
            print(f"Step: {self.total_steps}")
            print(f"Current Player: {self.current_player}")
            print(f"Dice: {self.current_dice}")
            print(f"Player 0 (Agent): Position {self.player_positions[0]}/{self.GOAL_POSITION}")
            print(f"Player 1 (Opponent): Position {self.player_positions[1]}/{self.GOAL_POSITION}")

            # Visual track
            track = ['-'] * (self.GOAL_POSITION + 1)
            p0_pos = min(self.player_positions[0], self.GOAL_POSITION)
            p1_pos = min(self.player_positions[1], self.GOAL_POSITION)

            if p0_pos == p1_pos:
                track[p0_pos] = 'X'  # Both at same position
            else:
                track[p0_pos] = '0'
                track[p1_pos] = '1'

            print("Track: [" + "".join(track) + "]")

            if self.done:
                if self.winner is not None:
                    print(f"\n🏆 Player {self.winner} WINS!")
                else:
                    print("\n⏱️  Game truncated (max steps)")
            print("="*60)

    def close(self):
        """Clean up resources."""
        pass


# Convenience function for creating the environment
def make_level1_env(seed: Optional[int] = None, render_mode: Optional[str] = None):
    """Create a Level 1 simple Ludo environment."""
    return Level1SimpleLudo(render_mode=render_mode, seed=seed)
