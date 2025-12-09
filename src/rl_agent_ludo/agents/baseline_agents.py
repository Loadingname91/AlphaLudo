"""
Baseline agents for benchmarking.

These simple agents serve as baselines to measure learning progress:
- RandomAgent: Selects random valid actions
- GreedyAgent: Always moves forward when possible
"""

import numpy as np
from typing import Any, Dict


class RandomAgent:
    """Agent that selects random valid actions."""

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """
        Select random action.

        Works for Level 1, 2, and 3:
        - Level 1 (4D, 2 actions): observation[2] is can_move flag
        - Level 2 (8D, 2 actions): no can_move flag, just random
        - Level 3 (14D, 3 actions): use action_mask from info if available
        """
        # Check if we have action_mask in info (Level 3+)
        if info is not None and 'action_mask' in info:
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                # Pick random valid action, but prefer moving over passing
                move_actions = [a for a in valid_actions if a < 2]  # 0 and 1 are move actions
                if len(move_actions) > 0 and self.np_random.random() < 0.8:
                    return int(self.np_random.choice(move_actions))
                else:
                    return int(self.np_random.choice(valid_actions))
            else:
                return 2  # Pass if no valid actions (shouldn't happen)

        # Check observation dimension for Level 1 vs 2
        if len(observation) == 4:
            # Level 1: use can_move flag
            can_move = observation[2] > 0.5
            if can_move:
                return 0 if self.np_random.random() < 0.8 else 1
            else:
                return 1
        else:
            # Level 2: no can_move flag, just be random
            # 80% move, 20% pass
            return 0 if self.np_random.random() < 0.8 else 1

    def reset(self):
        """Reset agent state (if any)."""
        pass


class GreedyAgent:
    """Agent that always moves forward when possible."""

    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """
        Select greedy action (always move if possible).

        Works for Level 1, 2, and 3:
        - Level 1 (4D, 2 actions): observation[2] is can_move flag
        - Level 2 (8D, 2 actions): always try to move
        - Level 3 (14D, 3 actions): move the most advanced token if possible
        """
        # Check if we have action_mask in info (Level 3+)
        if info is not None and 'action_mask' in info:
            action_mask = info['action_mask']
            # Prefer moving token over passing
            if action_mask[0]:  # Can move token 0
                return 0
            elif action_mask[1]:  # Can move token 1
                return 1
            else:
                return 2  # Pass

        # Check observation dimension for Level 1 vs 2
        if len(observation) == 4:
            # Level 1: use can_move flag
            can_move = observation[2] > 0.5
            return 0 if can_move else 1
        else:
            # Level 2: always try to move (greedy)
            return 0

    def reset(self):
        """Reset agent state (if any)."""
        pass


class AlwaysMoveAgent:
    """
    Agent that always tries to move (even when invalid).
    Useful for testing environment's handling of invalid actions.
    """

    def act(self, observation: np.ndarray, info: Dict[str, Any] = None) -> int:
        """Always return action 0 (move)."""
        return 0

    def reset(self):
        """Reset agent state (if any)."""
        pass
