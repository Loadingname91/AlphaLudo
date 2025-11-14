"""
Pillar 5: Trainer (Orchestrator)

Main application orchestrator managing training loop, environment-agent interaction, and logging.
"""

import os
import random
import numpy as np
import logging
import time
from typing import Dict, Any, Optional
from tqdm import tqdm
from ..environment.ludo_env import LudoEnv
from ..agents.base_agent import Agent
from ..metrics.metrics_tracker import MetricsTracker
from ..utils.state import State


class Trainer:
    """
    Main training orchestrator.
    
    Manages training loop, environment-agent interaction, and logging.
    Handles both on-policy and off-policy training modes.
    """
    
    def __init__(
        self,
        env: LudoEnv,
        agent: Agent,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        experiment_logger: Optional[Any] = None
    ):
        """
        Initialize trainer.
        
        Args:
            env: LudoEnv instance
            agent: Agent instance
            config: Training configuration dictionary
            logger: Python logger for console output (optional)
            experiment_logger: Experiment logger (TensorBoard/WandB, optional)
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.logger = logger or self._create_logger()
        self.experiment_logger = experiment_logger
        
        # Training parameters from config
        self.num_episodes = config['training'].get('num_episodes', 1000)
        self.max_steps_per_episode = config['training'].get('max_steps_per_episode', 1000)
        self.seed = config['training'].get('seed', None)
        self.log_interval = config['training'].get('log_interval', 100)
        self.save_interval = config['training'].get('save_interval', None)
        
        # Metrics tracker
        experiment_name = config['experiment'].get('name', 'default_experiment')
        output_dir = config['experiment'].get('output_dir', 'results')
        self.metrics_tracker = MetricsTracker(experiment_name, output_dir)
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
    
    def run(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Selects appropriate training loop based on agent type (on-policy vs off-policy).
        
        Returns:
            Dictionary with training results and summary
        """
        self.logger.info(f"Starting training with {self.num_episodes} episodes")
        self.logger.info(f"Agent type: {type(self.agent).__name__}")
        self.logger.info(f"Is on-policy: {self.agent.is_on_policy}")
        self.logger.info(f"Needs replay learning: {self.agent.needs_replay_learning}")
        
        # Set seeds for reproducibility
        if self.seed is not None:
            self._set_seeds(self.seed)
            self.logger.info(f"Random seed set to: {self.seed}")
        
        # Select training loop based on agent type
        if self.agent.is_on_policy:
            self.logger.info("Using on-policy training loop")
            results = self._run_on_policy_loop()
        else:
            self.logger.info("Using off-policy training loop")
            results = self._run_off_policy_loop()
        
        # Save final metrics
        saved_files = self.metrics_tracker.save_metrics()
        self.logger.info(f"Metrics saved to: {saved_files}")
        
        # Get summary
        summary = self.metrics_tracker.get_summary()
        self.logger.info(f"Training complete. Summary: {summary}")
        
        # Close environment rendering if enabled
        if hasattr(self.env, 'close'):
            self.env.close()
        
        results['summary'] = summary
        results['saved_files'] = saved_files
        
        return results
    
    def _run_on_policy_loop(self) -> Dict[str, Any]:
        """
        On-policy training loop (for PPO, MCTS).
        
        Collects rollouts and learns from them.
        
        Returns:
            Dictionary with training results
        """
        # Create progress bar
        pbar = tqdm(
            total=self.num_episodes,
            desc="Training",
            unit="ep",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            # Set episode number BEFORE reset so seed can use it
            self.metrics_tracker.current_episode = episode
            self.env.set_episode(episode)
            # Reset environment
            state = self.env.reset()
            episode_rewards = []
            episode_info = {}
            
            # Collect rollout
            rollout_buffer = []
            done = False
            step = 0
            
            while not done and step < self.max_steps_per_episode:
                # Check whose turn it is
                is_learning_agent_turn = (self.env.current_player == self.env.player_id)
                
                if is_learning_agent_turn:
                    # Learning agent's turn - get action from agent
                    action = self.agent.act(state)
                    
                    # Environment step
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Store experience
                    experience = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'info': info
                    }
                    rollout_buffer.append(experience)
                    
                    # Log metrics
                    self.metrics_tracker.log_metrics(state, action, reward, info)
                    episode_rewards.append(reward)
                    
                    # Update state
                    state = next_state
                    step += 1
                    self.step_count += 1
                    
                    # Log to experiment logger if available
                    if self.experiment_logger:
                        self._log_to_experiment_logger(step, reward, info)
                else:
                    # Opponent's turn - environment handles it automatically
                    # Use dummy action (will be ignored by environment)
                    next_state, reward, done, info = self.env.step(0)
                    
                    # Update state to track game progress
                    state = next_state
                    step += 1
                    self.step_count += 1
            
            # Learn from rollout (if agent implements it)
            if hasattr(self.agent, 'learn_from_rollout'):
                self.agent.learn_from_rollout(rollout_buffer)
            
            # Log episode
            episode_info['total_reward'] = sum(episode_rewards)
            episode_info['episode_length'] = step
            
            # Add learning agent player ID (from last info dict)
            if 'learning_agent_player_id' in info:
                episode_info['learning_agent_player_id'] = info.get('learning_agent_player_id')
            
            # Add winner information if game ended
            if done:
                if 'won' in info:
                    episode_info['won'] = info.get('won')
                if 'winner_player_id' in info:
                    episode_info['winner_player_id'] = info.get('winner_player_id')
                if 'winners' in info:
                    episode_info['winners'] = info.get('winners')
            
            self.metrics_tracker.log_episode(episode_info)
            self.episode_count += 1
            
            # Update progress bar with statistics
            summary = self.metrics_tracker.get_summary()
            
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            episodes_completed = episode + 1
            episodes_remaining = self.num_episodes - episodes_completed
            
            if episodes_completed > 0:
                avg_time_per_episode = elapsed_time / episodes_completed
                estimated_remaining = avg_time_per_episode * episodes_remaining
            else:
                estimated_remaining = 0
            
            # Format time strings
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining)
            
            # Update progress bar
            pbar.set_postfix({
                'Win Rate': f"{summary['win_rate']:.1%}",
                'Avg Reward': f"{summary['avg_episode_reward']:.1f}",
                'Elapsed': elapsed_str,
                'ETA': remaining_str
            })
            pbar.update(1)
            
            # Periodic detailed logging
            if (episode + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{self.num_episodes} | "
                    f"Win Rate: {summary['win_rate']:.3f} | "
                    f"Avg Reward: {summary['avg_episode_reward']:.2f} | "
                    f"Elapsed: {elapsed_str} | "
                    f"ETA: {remaining_str}"
                )
        
        pbar.close()
        
        return {
            'episodes': self.episode_count,
            'steps': self.step_count,
            'mode': 'on_policy'
        }
    
    def _run_off_policy_loop(self) -> Dict[str, Any]:
        """
        Off-policy training loop (for Q-Learning, DQN, Random).
        
        Collects experiences and learns from replay buffer.
        
        Returns:
            Dictionary with training results
        """
        # Create progress bar
        pbar = tqdm(
            total=self.num_episodes,
            desc="Training",
            unit="ep",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            # Set episode number BEFORE reset so seed can use it
            self.metrics_tracker.current_episode = episode
            self.env.set_episode(episode)
            # Reset environment
            state = self.env.reset()
            episode_rewards = []
            episode_info = {}
            
            done = False
            step = 0
            
            while not done and step < self.max_steps_per_episode:
                # Check whose turn it is
                is_learning_agent_turn = (self.env.current_player == self.env.player_id)
                
                if is_learning_agent_turn:
                    # Learning agent's turn - get action from agent
                    action = self.agent.act(state)
                    
                    # Environment step
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Push to replay buffer (if agent needs it)
                    if self.agent.needs_replay_learning:
                        self.agent.push_to_replay_buffer(state, action, reward, next_state, done, **info)
                    
                    # Learn from replay (if agent needs it)
                    if self.agent.needs_replay_learning:
                        self.agent.learn_from_replay()
                    
                    # Log metrics
                    self.metrics_tracker.log_metrics(state, action, reward, info)
                    episode_rewards.append(reward)
                    
                    # Update state
                    state = next_state
                    step += 1
                    self.step_count += 1
                    
                    # Log to experiment logger if available
                    if self.experiment_logger:
                        self._log_to_experiment_logger(step, reward, info)
                else:
                    # Opponent's turn - environment handles it automatically
                    # Use dummy action (will be ignored by environment)
                    next_state, reward, done, info = self.env.step(0)
                    
                    # Update state to track game progress
                    state = next_state
                    step += 1
                    self.step_count += 1
            
            # Log episode
            episode_info['total_reward'] = sum(episode_rewards)
            episode_info['episode_length'] = step
            
            # Add learning agent player ID (from last info dict)
            if 'learning_agent_player_id' in info:
                episode_info['learning_agent_player_id'] = info.get('learning_agent_player_id')
            
            # Add winner information if game ended
            if done:
                if 'won' in info:
                    episode_info['won'] = info.get('won')
                if 'winner_player_id' in info:
                    episode_info['winner_player_id'] = info.get('winner_player_id')
                if 'winners' in info:
                    episode_info['winners'] = info.get('winners')
            
            self.metrics_tracker.log_episode(episode_info)
            self.episode_count += 1
            
            # Update progress bar with statistics
            summary = self.metrics_tracker.get_summary()
            
            # Calculate time estimates
            elapsed_time = time.time() - start_time
            episodes_completed = episode + 1
            episodes_remaining = self.num_episodes - episodes_completed
            
            if episodes_completed > 0:
                avg_time_per_episode = elapsed_time / episodes_completed
                estimated_remaining = avg_time_per_episode * episodes_remaining
            else:
                estimated_remaining = 0
            
            # Format time strings
            elapsed_str = self._format_time(elapsed_time)
            remaining_str = self._format_time(estimated_remaining)
            
            # Update progress bar
            pbar.set_postfix({
                'Win Rate': f"{summary['win_rate']:.1%}",
                'Avg Reward': f"{summary['avg_episode_reward']:.1f}",
                'Elapsed': elapsed_str,
                'ETA': remaining_str
            })
            pbar.update(1)
            
            # Periodic detailed logging
            if (episode + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Episode {episode + 1}/{self.num_episodes} | "
                    f"Win Rate: {summary['win_rate']:.3f} | "
                    f"Avg Reward: {summary['avg_episode_reward']:.2f} | "
                    f"Elapsed: {elapsed_str} | "
                    f"ETA: {remaining_str}"
                )
            
            # Save checkpoint (if configured)
            if self.save_interval and (episode + 1) % self.save_interval == 0:
                self._save_checkpoint(episode + 1)
        
        pbar.close()
        
        return {
            'episodes': self.episode_count,
            'steps': self.step_count,
            'mode': 'off_policy'
        }
    
    def _set_seeds(self, seed: int) -> None:
        """
        Set random seeds for reproducibility.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Set PyTorch seed if available (for future neural network agents)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass  # PyTorch not installed, skip
    
    def _log_to_experiment_logger(self, step: int, reward: float, info: Dict[str, Any]) -> None:
        """
        Log metrics to experiment logger (TensorBoard/WandB).
        
        Args:
            step: Current step number
            reward: Reward value
            info: Additional information dictionary
        """
        if self.experiment_logger is None:
            return
        
        # Try TensorBoard-style logging
        if hasattr(self.experiment_logger, 'add_scalar'):
            self.experiment_logger.add_scalar('train/reward', reward, step)
            if 'won' in info:
                self.experiment_logger.add_scalar('train/won', 1.0 if info['won'] else 0.0, step)
        
        # Try WandB-style logging
        elif hasattr(self.experiment_logger, 'log'):
            self.experiment_logger.log({'reward': reward, 'step': step})
    
    def _save_checkpoint(self, episode: int) -> None:
        """
        Save training checkpoint.
        
        Args:
            episode: Current episode number
        """
        # Save agent model if it supports saving
        if hasattr(self.agent, 'save'):
            checkpoint_dir = self.config['training'].get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_episode_{episode}.pth")
            self.agent.save(checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds to human-readable string.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted time string (e.g., "1h 23m 45s" or "2m 30s" or "45s")
        """
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"
    
    def _create_logger(self) -> logging.Logger:
        """Create default logger."""
        logger = logging.getLogger('RLAgentLudo')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
