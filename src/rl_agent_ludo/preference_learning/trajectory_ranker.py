"""
Trajectory Ranker for T-REX.

Creates preference pairs from collected trajectories by ranking them
based on game outcomes, captures, and episode length.
"""

import pickle
import random
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class TrajectoryRanker:
    """
    Ranks trajectories and creates preference pairs for T-REX.

    Ranking criteria (in order):
    1. Win > Loss
    2. Among wins: More captures > Fewer captures
    3. Among wins: Shorter episode > Longer episode
    4. Among losses: Survived longer > Died early
    """

    def __init__(self):
        self.preference_pairs = []

    def rank_pair(self, traj_i: Dict, traj_j: Dict) -> int:
        """
        Compare two trajectories.

        Args:
            traj_i: First trajectory
            traj_j: Second trajectory

        Returns:
            1 if traj_i > traj_j (i is better)
            -1 if traj_j > traj_i (j is better)
            0 if equal (skip this pair)
        """
        # Rule 1: Win > Loss
        if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'loss':
            return 1
        if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'win':
            return -1

        # Both won or both lost - need secondary criteria
        if traj_i['outcome'] == 'win' and traj_j['outcome'] == 'win':
            # Rule 2: Among wins, more captures > fewer
            if traj_i['num_captures'] > traj_j['num_captures']:
                return 1
            elif traj_i['num_captures'] < traj_j['num_captures']:
                return -1
            else:
                # Rule 3: Among equal captures, shorter > longer (more efficient)
                if traj_i['episode_length'] < traj_j['episode_length']:
                    return 1
                elif traj_i['episode_length'] > traj_j['episode_length']:
                    return -1

        if traj_i['outcome'] == 'loss' and traj_j['outcome'] == 'loss':
            # Rule 4: Among losses, survived longer > died early
            if traj_i['episode_length'] > traj_j['episode_length']:
                return 1
            elif traj_i['episode_length'] < traj_j['episode_length']:
                return -1

        return 0  # Equal - will be skipped

    def create_preference_pairs(self, trajectories: List[Dict],
                               max_pairs: int = 10000,
                               seed: int = 42) -> List[Tuple]:
        """
        Create preference pairs from trajectories.

        Args:
            trajectories: List of trajectory dictionaries
            max_pairs: Maximum number of pairs to create
            seed: Random seed for reproducibility

        Returns:
            preference_pairs: List of (better_traj, worse_traj) tuples
        """
        print(f"\n{'='*70}")
        print(f"Creating preference pairs from {len(trajectories)} trajectories...")
        print(f"{'='*70}")

        if len(trajectories) < 2:
            raise ValueError(f"Need at least 2 trajectories, got {len(trajectories)}")

        random.seed(seed)
        preference_pairs = []
        attempted = 0
        max_attempts = max_pairs * 3  # Try 3x to get enough valid pairs

        # Sample pairs randomly
        while len(preference_pairs) < max_pairs and attempted < max_attempts:
            i, j = random.sample(range(len(trajectories)), 2)
            traj_i = trajectories[i]
            traj_j = trajectories[j]

            ranking = self.rank_pair(traj_i, traj_j)

            if ranking == 1:
                # i > j
                preference_pairs.append((traj_i, traj_j))
            elif ranking == -1:
                # j > i
                preference_pairs.append((traj_j, traj_i))
            # If ranking == 0, skip this pair

            attempted += 1

        print(f"Created {len(preference_pairs)} preference pairs")
        print(f"   (Attempted {attempted} pairings, {len(preference_pairs)/attempted*100:.1f}% valid)")

        # Statistics
        self._print_statistics(preference_pairs)

        self.preference_pairs = preference_pairs
        return preference_pairs

    def _print_statistics(self, pairs: List[Tuple]):
        """Print statistics about preference pairs."""
        print(f"\nPreference Pair Statistics:")

        better_wins = sum(1 for better, worse in pairs if better['outcome'] == 'win')
        worse_losses = sum(1 for better, worse in pairs if worse['outcome'] == 'loss')
        win_vs_loss = sum(1 for b, w in pairs if b['outcome'] == 'win' and w['outcome'] == 'loss')

        print(f"  Total pairs: {len(pairs)}")
        print(f"  Better trajectories that won: {better_wins} ({better_wins/len(pairs)*100:.1f}%)")
        print(f"  Worse trajectories that lost: {worse_losses} ({worse_losses/len(pairs)*100:.1f}%)")
        print(f"  Win vs Loss pairs: {win_vs_loss} ({win_vs_loss/len(pairs)*100:.1f}%)")

        # Average metrics
        avg_better_captures = np.mean([b['num_captures'] for b, w in pairs])
        avg_worse_captures = np.mean([w['num_captures'] for b, w in pairs])
        avg_better_length = np.mean([b['episode_length'] for b, w in pairs])
        avg_worse_length = np.mean([w['episode_length'] for b, w in pairs])

        print(f"\n  Better trajectories:")
        print(f"    Avg captures: {avg_better_captures:.2f}")
        print(f"    Avg length: {avg_better_length:.1f}")
        print(f"  Worse trajectories:")
        print(f"    Avg captures: {avg_worse_captures:.2f}")
        print(f"    Avg length: {avg_worse_length:.1f}")

    def save_pairs(self, filepath: str):
        """Save preference pairs to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.preference_pairs, f)

        print(f"\nSaved {len(self.preference_pairs)} pairs to {filepath}")

    def load_pairs(self, filepath: str):
        """Load preference pairs from disk."""
        with open(filepath, 'rb') as f:
            self.preference_pairs = pickle.load(f)

        print(f"Loaded {len(self.preference_pairs)} pairs from {filepath}")
        return self.preference_pairs

    def split_train_val(self, train_ratio: float = 0.8,
                       seed: int = 42) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Split preference pairs into train and validation sets.

        Args:
            train_ratio: Ratio of training data
            seed: Random seed

        Returns:
            train_pairs, val_pairs
        """
        random.seed(seed)
        pairs = self.preference_pairs.copy()
        random.shuffle(pairs)

        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        print(f"\nTrain/Val Split:")
        print(f"  Train: {len(train_pairs)} pairs ({train_ratio*100:.0f}%)")
        print(f"  Val: {len(val_pairs)} pairs ({(1-train_ratio)*100:.0f}%)")

        return train_pairs, val_pairs
