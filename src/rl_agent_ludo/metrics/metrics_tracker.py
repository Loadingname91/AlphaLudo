"""
Pillar 4: MetricsTracker (Raw Data Collection)

Lightweight metrics collection class with no analysis dependencies.
Collects raw data and saves to JSON/CSV files for offline analysis.
"""

import json
import csv
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict


class MetricsTracker:
    """
    Collects raw training metrics during training loop.
    
    Lightweight implementation with no analysis dependencies (no pandas, matplotlib).
    Exports raw data to JSON/CSV for offline analysis via analysis.py.
    """
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment (used in output filenames)
            output_dir: Directory to save metrics files (default: "results")
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Episode-level metrics
        self.episodes: List[Dict[str, Any]] = []
        
        # Step-level metrics (optional, can be large)
        self.steps: List[Dict[str, Any]] = []
        
        # Current episode tracking
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards: List[float] = []
        self.episode_steps: List[Dict[str, Any]] = []
        
        # Running statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.wins = 0
        self.losses = 0
        
        # Optional score debugging lines (human-readable, space-efficient)
        self.score_debug_lines: List[str] = []
    
    def log_metrics(
        self,
        state,
        action: int,
        reward: float,
        info: Dict[str, Any],
        score_debug: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log metrics for a single step.
        
        Args:
            state: Current State object
            action: Action taken
            reward: Reward received
            info: Additional information dictionary
        """
        # Create step record
        step_data = {
            'episode': self.current_episode,
            'step': self.current_step,
            'state': str(state.abstract_state),  # Convert tuple to string for JSON serialization
            'action': int(action),
            'reward': float(reward),
            'dice_roll': int(state.dice_roll),
            'valid_moves_count': len(state.valid_moves),
        }
        
        # Add any additional info
        step_data.update({k: v for k, v in info.items() if self._is_serializable(v)})

        # Optionally record detailed score debug information from agent
        # We keep this in a separate, human-readable log instead of bloating JSON.
        if score_debug is not None:
            line = self._format_score_debug_line(
                episode=self.current_episode,
                step=self.current_step,
                state=state,
                action=action,
                reward=reward,
                score_debug=score_debug,
            )
            self.score_debug_lines.append(line)
        
        # Store step data
        self.steps.append(step_data)
        self.episode_steps.append(step_data)
        self.episode_rewards.append(reward)
        
        self.current_step += 1
        self.total_steps += 1
    
    def log_episode(self, episode_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log metrics for completed episode.
        
        Args:
            episode_data: Optional dictionary with additional episode-level metrics
        """
        # Calculate episode statistics
        total_reward = sum(self.episode_rewards)
        
        # Use episode_length from episode_data if provided (more accurate), 
        # otherwise fall back to len(self.episode_steps)
        if episode_data and 'episode_length' in episode_data:
            episode_length = episode_data['episode_length']
        else:
            episode_length = len(self.episode_steps)
        
        # Create episode record
        episode_record = {
            'episode': self.current_episode,
            'total_reward': float(total_reward),
            'episode_length': int(episode_length),
            'avg_reward': float(total_reward / episode_length) if episode_length > 0 else 0.0,
            'total_steps': self.total_steps,
        }
        
        # Add any additional episode data (this will override episode_length if provided)
        if episode_data:
            episode_record.update({k: v for k, v in episode_data.items() if self._is_serializable(v)})
            
            # Track wins/losses
            if episode_data.get('won') is True:
                self.wins += 1
                episode_record['won'] = True
                episode_record['lost'] = False
            elif episode_data.get('won') is False:
                self.losses += 1
                episode_record['won'] = False
                episode_record['lost'] = True
            else:
                episode_record['won'] = None
                episode_record['lost'] = None
        
        # Calculate win rate
        if self.total_episodes > 0:
            episode_record['cumulative_win_rate'] = float(self.wins / self.total_episodes)
        else:
            episode_record['cumulative_win_rate'] = 0.0
        
        # Store episode record
        self.episodes.append(episode_record)
        
        # Reset episode tracking
        self.current_episode += 1
        self.total_episodes += 1
        self.current_step = 0
        self.episode_rewards = []
        self.episode_steps = []
    
    def save_metrics(self) -> Dict[str, str]:
        """
        Save all collected metrics to disk.
        
        Returns:
            Dictionary with filepaths of saved files:
                - 'episodes_json': Path to episodes JSON file
                - 'episodes_csv': Path to episodes CSV file
                - 'steps_json': Path to steps JSON file (if steps were collected)
        """
        saved_files = {}
        
        # Save episodes to JSON
        episodes_json_path = os.path.join(
            self.output_dir, 
            f"{self.experiment_name}_episodes.json"
        )
        with open(episodes_json_path, 'w') as f:
            json.dump(self.episodes, f, indent=2)
        saved_files['episodes_json'] = episodes_json_path
        
        # Save episodes to CSV
        episodes_csv_path = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_episodes.csv"
        )
        if self.episodes:
            fieldnames = self.episodes[0].keys()
            with open(episodes_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.episodes)
            saved_files['episodes_csv'] = episodes_csv_path
        
        # NOTE: We intentionally skip saving per-step metrics to JSON to keep
        # disk usage manageable. Detailed step/score debugging is handled via
        # compact .log files instead.
        
        # Save score debug log (if any)
        if self.score_debug_lines:
            score_log_path = os.path.join(
                self.output_dir,
                f"{self.experiment_name}_score_debug.log"
            )
            with open(score_log_path, 'w') as f:
                f.writelines(self.score_debug_lines)
            saved_files['score_debug_log'] = score_log_path
        
        return saved_files
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected metrics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.episodes:
            return {
                'total_episodes': 0,
                'total_steps': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
            }
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': float(self.wins / self.total_episodes) if self.total_episodes > 0 else 0.0,
            'avg_episode_length': float(sum(ep['episode_length'] for ep in self.episodes) / len(self.episodes)),
            'avg_episode_reward': float(sum(ep['total_reward'] for ep in self.episodes) / len(self.episodes)),
        }
    
    def _is_serializable(self, value: Any) -> bool:
        """
        Check if value is JSON serializable.
        
        Args:
            value: Value to check
        
        Returns:
            True if serializable, False otherwise
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
    
    def reset(self) -> None:
        """Reset all metrics (for new experiment)."""
        self.episodes = []
        self.steps = []
        self.current_episode = 0
        self.current_step = 0
        self.episode_rewards = []
        self.episode_steps = []
        self.total_episodes = 0
        self.total_steps = 0
        self.wins = 0
        self.losses = 0
        self.score_debug_lines = []

    def save_partial_metrics(self, suffix: str) -> Dict[str, str]:
        """
        Save a snapshot of current metrics to disk without resetting internal state.

        Useful for checkpoint-style logging (e.g., every N episodes) so that large
        runs can be inspected incrementally.

        Args:
            suffix: String suffix to distinguish this snapshot

        Returns:
            Dictionary with filepaths of saved files for this snapshot.
        """
        saved_files: Dict[str, str] = {}

        # Episodes snapshot
        episodes_json_path = os.path.join(
            self.output_dir,
            f"{self.experiment_name}_episodes_{suffix}.json"
        )
        with open(episodes_json_path, 'w') as f:
            json.dump(self.episodes, f, indent=2)
        saved_files['episodes_json'] = episodes_json_path

        # Score debug snapshot (if any)
        if self.score_debug_lines:
            score_log_path = os.path.join(
                self.output_dir,
                f"{self.experiment_name}_score_debug_{suffix}.log"
            )
            with open(score_log_path, 'w') as f:
                f.writelines(self.score_debug_lines)
            saved_files['score_debug_log'] = score_log_path

        return saved_files

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _format_score_debug_line(
        self,
        episode: int,
        step: int,
        state,
        action: int,
        reward: float,
        score_debug: Dict[str, Any],
    ) -> str:
        """
        Convert score_debug information into a compact, human-readable log line.

        This keeps disk usage low while still exposing the key scoring signal.
        """
        components = score_debug.get('components', {}) or {}
        # Only log non-zero components to keep lines short
        non_zero_components = {
            name: value for name, value in components.items()
            if isinstance(value, (int, float)) and value != 0
        }

        parts = [
            f"ep={episode}",
            f"step={step}",
            f"dice={score_debug.get('dice_roll')}",
            f"piece={score_debug.get('piece_idx')}",
            f"action={int(action)}",
            f"reward={float(reward)}",
            f"total={score_debug.get('total_score')}",
        ]

        if non_zero_components:
            comp_str = ", ".join(f"{k}={v}" for k, v in non_zero_components.items())
            parts.append(f"components=[{comp_str}]")

        return " ".join(str(p) for p in parts if p is not None) + "\n"
