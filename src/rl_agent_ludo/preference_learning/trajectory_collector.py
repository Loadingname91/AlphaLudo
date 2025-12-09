"""
Trajectory Collector for T-REX.

Runs existing agents and records full game trajectories with metadata
for preference-based reward learning.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch


class TrajectoryCollector:
    """
    Collects and saves game trajectories from agents.

    Each trajectory includes:
    - Full state sequence
    - Action sequence
    - Game outcome (win/loss)
    - Metadata (captures, episode length, agent type)
    """

    def __init__(self, save_dir: str = "checkpoints/level6/trajectories"):
        """
        Initialize trajectory collector.

        Args:
            save_dir: Directory to save collected trajectories
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories = []

        print(f"📁 Trajectory save directory: {self.save_dir}")

    def collect_trajectory(self, env, agent, episode_id: int,
                          seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Run one episode and record full trajectory.

        Args:
            env: Gymnasium environment
            agent: Agent to collect trajectory from
            episode_id: Unique episode identifier
            seed: Random seed for reproducibility

        Returns:
            trajectory: Dictionary with trajectory data
        """
        states = []
        actions = []
        env_rewards = []

        # Reset environment
        if seed is not None:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()

        done = False
        step = 0
        num_captures = 0
        got_captured = 0

        # Run episode
        while not done and step < 1000:
            # Record state
            states.append(state.copy())

            # Agent acts
            # Try greedy parameter (for DQN agents), fallback to no parameter (for baseline agents)
            try:
                action = agent.act(state, greedy=False)
            except TypeError:
                action = agent.act(state)
            actions.append(action)

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            env_rewards.append(reward)

            # Track captures (if info available)
            if 'captures_made' in info:
                num_captures = info['captures_made']
            if 'times_captured' in info:
                got_captured = info['times_captured']

            state = next_state
            done = terminated or truncated
            step += 1

        # Determine outcome
        winner = info.get('winner', -1)
        outcome = 'win' if winner == 0 else 'loss'

        # Create trajectory dictionary
        trajectory = {
            'episode_id': episode_id,
            'states': states,
            'actions': actions,
            'env_rewards': env_rewards,  # Will be ignored by T-REX
            'outcome': outcome,
            'winner': winner,
            'num_captures': num_captures,
            'got_captured': got_captured,
            'episode_length': step,
            'agent_type': getattr(agent, 'name', 'unknown'),
            'final_reward': sum(env_rewards),
        }

        return trajectory

    def collect_batch(self, env, agent, num_episodes: int,
                     batch_name: str = "default",
                     seed_start: int = 42,
                     verbose: bool = True) -> List[Dict]:
        """
        Collect multiple trajectories and save to disk.

        Args:
            env: Gymnasium environment
            agent: Agent to collect from
            num_episodes: Number of episodes to collect
            batch_name: Name for this batch (for saving)
            seed_start: Starting seed for reproducibility
            verbose: Print progress

        Returns:
            batch_trajectories: List of trajectory dictionaries
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Collecting {num_episodes} trajectories from agent: {getattr(agent, 'name', 'unknown')}")
            print(f"{'='*70}")

        batch_trajectories = []
        win_count = 0

        for i in range(num_episodes):
            # Collect trajectory with unique seed
            traj = self.collect_trajectory(
                env, agent,
                episode_id=i,
                seed=seed_start + i
            )
            batch_trajectories.append(traj)

            # Track statistics
            if traj['outcome'] == 'win':
                win_count += 1

            # Progress logging
            if verbose and (i + 1) % 100 == 0:
                current_win_rate = win_count / (i + 1) * 100
                print(f"  Progress: {i+1}/{num_episodes} | "
                      f"Win Rate: {current_win_rate:.1f}% | "
                      f"Avg Length: {np.mean([t['episode_length'] for t in batch_trajectories]):.1f}")

        # Final statistics
        if verbose:
            self._print_batch_statistics(batch_trajectories)

        # Save batch
        save_path = self.save_dir / f"{batch_name}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(batch_trajectories, f)

        if verbose:
            print(f"💾 Saved {len(batch_trajectories)} trajectories to {save_path}\n")

        self.trajectories.extend(batch_trajectories)
        return batch_trajectories

    def _print_batch_statistics(self, trajectories: List[Dict]):
        """Print statistics about collected trajectories."""
        wins = sum(1 for t in trajectories if t['outcome'] == 'win')
        losses = len(trajectories) - wins

        avg_length = np.mean([t['episode_length'] for t in trajectories])
        avg_captures = np.mean([t['num_captures'] for t in trajectories])
        avg_captured = np.mean([t['got_captured'] for t in trajectories])
        avg_reward = np.mean([t['final_reward'] for t in trajectories])

        print(f"\n📊 Batch Statistics:")
        print(f"  Total Episodes: {len(trajectories)}")
        print(f"  Wins: {wins} ({wins/len(trajectories)*100:.1f}%)")
        print(f"  Losses: {losses} ({losses/len(trajectories)*100:.1f}%)")
        print(f"  Avg Episode Length: {avg_length:.1f} steps")
        print(f"  Avg Captures Made: {avg_captures:.2f}")
        print(f"  Avg Times Captured: {avg_captured:.2f}")
        print(f"  Avg Final Reward: {avg_reward:.2f}")

    def load_trajectories(self, batch_name: str) -> List[Dict]:
        """
        Load saved trajectories from disk.

        Args:
            batch_name: Name of batch to load

        Returns:
            trajectories: List of trajectory dictionaries
        """
        load_path = self.save_dir / f"{batch_name}.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {load_path}")

        with open(load_path, 'rb') as f:
            trajectories = pickle.load(f)

        print(f"✅ Loaded {len(trajectories)} trajectories from {load_path}")
        return trajectories

    def get_all_saved_batches(self) -> List[str]:
        """Get list of all saved batch names."""
        batch_files = list(self.save_dir.glob("*.pkl"))
        batch_names = [f.stem for f in batch_files]
        return batch_names

    def load_all_trajectories(self) -> List[Dict]:
        """Load all saved trajectory batches."""
        all_trajectories = []
        batch_names = self.get_all_saved_batches()

        print(f"📂 Found {len(batch_names)} saved batches")

        for batch_name in batch_names:
            trajs = self.load_trajectories(batch_name)
            all_trajectories.extend(trajs)

        print(f"✅ Total trajectories loaded: {len(all_trajectories)}")
        return all_trajectories

    def summarize_all(self):
        """Print summary of all saved trajectories."""
        all_trajs = self.load_all_trajectories()

        print(f"\n{'='*70}")
        print("SUMMARY OF ALL COLLECTED TRAJECTORIES")
        print(f"{'='*70}")

        # Group by agent type
        by_agent = {}
        for traj in all_trajs:
            agent_type = traj['agent_type']
            if agent_type not in by_agent:
                by_agent[agent_type] = []
            by_agent[agent_type].append(traj)

        # Print per-agent statistics
        for agent_type, trajs in by_agent.items():
            wins = sum(1 for t in trajs if t['outcome'] == 'win')
            print(f"\n{agent_type}:")
            print(f"  Count: {len(trajs)}")
            print(f"  Win Rate: {wins/len(trajs)*100:.1f}%")
            print(f"  Avg Length: {np.mean([t['episode_length'] for t in trajs]):.1f}")
