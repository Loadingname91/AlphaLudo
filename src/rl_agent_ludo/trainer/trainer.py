"""
Training orchestrator.

Manages the training loop, coordinates environment-agent interactions, and handles logging.
Supports both on-policy and off-policy learning algorithms.
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
from ..environment.reward_shaper import ContextualRewardShaper
from ..utils.state_abstractor import CONTEXT_TRAILING, CONTEXT_NEUTRAL, CONTEXT_LEADING


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
        experiment_logger: Optional[Any] = None,
        use_context_aware_rewards:bool = False, # for checking in context aware rewards
        resume_from_checkpoint: Optional[str] = None,
        resume_run_path: Optional[str] = None  # Path to existing run directory to resume logging
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
        
        # ---------------------------------------------------------------------
        # Metrics tracker (per-run, structured output directory)
        #
        # Directory structure:
        #   {output_dir}/
        #       {agent_type}/
        #           reward_{reward_schema}/
        #               seed_{seed}/
        #                   {experiment_name}_{timestamp}/
        #
        # This makes it easy to inspect runs per reward schema and per seed.
        # ---------------------------------------------------------------------
        experiment_name = config['experiment'].get('name', 'default_experiment')
        base_output_dir = config['experiment'].get('output_dir', 'results')
        
        # Extract agent type to create a subdirectory
        agent_type = config['agent'].get('type', 'unknown_agent')
        
        # Ensure base_output_dir includes agent type if not already present
        if agent_type not in base_output_dir:
            base_output_dir = os.path.join(base_output_dir, agent_type)
        
        # Determine reward schema (from env) and seed (from training/experiment/env)
        reward_schema = getattr(env, "reward_schema", None) or config.get('environment', {}).get('reward_schema', 'unknown')
        seed_value = (
            self.seed
            or getattr(env, "seed", None)
            or config.get('experiment', {}).get('seed', None)
        )
        seed_str = str(seed_value) if seed_value is not None else "unknown"
        
        # Enrich directory structure with reward schema and seed
        base_output_dir = os.path.join(
            base_output_dir,
            f"reward_{reward_schema}",
            f"seed_{seed_str}",
        )
            
        if resume_run_path:
            # Use existing run directory
            run_output_dir = resume_run_path
            self.logger.info(f"Resuming logging in existing directory: {run_output_dir}")
            resume_metrics = True
        else:
            # Create new timestamped directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            run_output_dir = os.path.join(base_output_dir, f"{experiment_name}_{timestamp}")
            resume_metrics = False
            
        self.metrics_tracker = MetricsTracker(experiment_name, run_output_dir, resume=resume_metrics)
        self.run_output_dir = run_output_dir  # optional: for logging/introspection
        
        # Training state
        self.episode_count = 0
        self.step_count = 0
        
        # Resume from checkpoint
        self.start_episode = 0
        if resume_from_checkpoint:
            self.start_episode = self._extract_episode_from_checkpoint(resume_from_checkpoint)
            self.episode_count = self.start_episode
            self.logger.info(f"Resuming training from episode {self.start_episode}")
        
        # Context-aware reward shaper
        self.use_context_aware_rewards = use_context_aware_rewards
        if self.use_context_aware_rewards:
            self.reward_shaper = ContextualRewardShaper(player_id=env.player_id)
            # Inject into env to ensure step() calculates passive/active rewards correctly
            self.env.reward_shaper = self.reward_shaper
            # Track context frequencies for metrics
            self.context_counts = {CONTEXT_TRAILING: 0, CONTEXT_NEUTRAL: 0, CONTEXT_LEADING: 0}
        else:
            self.reward_shaper = None
    
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
            results = self.run_on_policy_loop()
        else:
            self.logger.info("Using off-policy training loop")
            results = self.run_off_policy_loop()
        
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
    
    def run_on_policy_loop(self) -> Dict[str, Any]:
        """
        On-policy training loop (for PPO, MCTS).
        
        Collects rollouts and learns from them.
        Updated to use ContextualRewardShaper when enabled.
        
        Returns:
            Dictionary with training results
        """
        # Create progress bar
        pbar = tqdm(
            total=self.num_episodes,
            initial=self.start_episode,
            desc="Training",
            unit="ep",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        if self.start_episode > 0:
            pbar.update(0)  # Initialize progress bar at start_episode
        
        start_time = time.time()
        
        for episode in range(self.start_episode, self.num_episodes):
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
                    
                    # Store previous state BEFORE step for reward calculation
                    prev_state = state

                    # Optional score debugging information from agent
                    score_debug = None
                    if self.config['training'].get('enable_score_debug', False) and \
                            getattr(self.agent, 'supports_score_debug', False):
                        score_debug = self.agent.get_last_score_debug()

                    # Environment step
                    next_state, env_reward, done, info = self.env.step(action)
                    
                    # Calculate context-aware reward if enabled
                    # Handled by LudoEnv if injected
                    reward = env_reward
                    
                    # Track context frequency
                    if self.use_context_aware_rewards and hasattr(self.agent, 'last_state_tuple') and self.agent.last_state_tuple:
                        context = self.agent.last_state_tuple[4]
                        self.context_counts[context] = self.context_counts.get(context, 0) + 1
                    
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
                    self.metrics_tracker.log_metrics(state, action, reward, info, score_debug=score_debug)
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
                    next_state, reward, done, info = self.env.step(0)
                    state = next_state
                    step += 1
                    self.step_count += 1
            
            # Learn from rollout (if agent implements it)
            if hasattr(self.agent, 'learn_from_rollout'):
                self.agent.learn_from_rollout(rollout_buffer)
            
            # Log episode (with context info if enabled)
            episode_info['total_reward'] = sum(episode_rewards)
            episode_info['episode_length'] = step
            
            if self.use_context_aware_rewards:
                episode_info['context_trailing'] = self.context_counts.get(CONTEXT_TRAILING, 0)
                episode_info['context_neutral'] = self.context_counts.get(CONTEXT_NEUTRAL, 0)
                episode_info['context_leading'] = self.context_counts.get(CONTEXT_LEADING, 0)
            
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
            
            # Notify agent that episode ended (for epsilon decay, etc.)
            if hasattr(self.agent, 'on_episode_end'):
                self.agent.on_episode_end()
            
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
            'mode': 'on_policy',
            'context_counts': self.context_counts if self.use_context_aware_rewards else None
        }
    
    def run_off_policy_loop(self) -> Dict[str, Any]:
        """
        Off-policy training loop (for Q-Learning, DQN, Random).
        
        Collects experiences and learns from replay buffer.
        Uses N-step pending experience to ensure next_state is always an Agent Turn state.
        
        Returns:
            Dictionary with training results
        """
        # Create progress bar
        pbar = tqdm(
            total=self.num_episodes,
            initial=self.start_episode,
            desc="Training",
            unit="ep",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        if self.start_episode > 0:
            pbar.update(0)  # Initialize progress bar at start_episode
        
        start_time = time.time()
        
        for episode in range(self.start_episode, self.num_episodes):
            # Set episode number BEFORE reset so seed can use it
            self.metrics_tracker.current_episode = episode
            self.env.set_episode(episode)
            # Reset environment
            state = self.env.reset()
            episode_rewards = []
            episode_info = {}
            
            done = False
            step = 0
            
            # Buffer for storing experience until next agent turn
            pending_experience = None
            
            while not done and step < self.max_steps_per_episode:
                # Check whose turn it is
                is_learning_agent_turn = (self.env.current_player == self.env.player_id)
                
                if is_learning_agent_turn:
                    # 1. Process pending experience from PREVIOUS agent turn
                    if pending_experience is not None:
                        prev_s = pending_experience['state']
                        prev_a = pending_experience['action']
                        prev_r = pending_experience['reward']
                        prev_info = pending_experience['info']
                        
                        # CURRENT state is the next_state for the previous action
                        # (We skipped opponent turns to get here)
                        current_s_as_next = state 
                        
                        # Push to buffer
                        if self.agent.needs_replay_learning:
                            self.agent.push_to_replay_buffer(
                                prev_s, prev_a, prev_r, current_s_as_next, False, **prev_info
                            )
                            self.agent.learn_from_replay()
                        
                        pending_experience = None

                    # 2. Learning agent's turn - get action from agent
                    action = self.agent.act(state)
                    
                    # Store previous state BEFORE step for reward calculation
                    prev_state = state

                    # Optional score debugging information from agent
                    score_debug = None
                    if self.config['training'].get('enable_score_debug', False) and \
                            getattr(self.agent, 'supports_score_debug', False):
                        score_debug = self.agent.get_last_score_debug()

                    current_s_for_prev = state # The state at start of turn
                    
                    # Environment step (returns reward which includes passive/wait penalty)
                    next_state, total_reward, done, info = self.env.step(action)
                    
                    # 3. Decompose Reward (Active vs Passive)
                    # Extract rewards
                    passive_reward = info.get('passive_reward', 0.0)
                    active_reward = total_reward - passive_reward # Roughly, if additive
                    
                    # Handle Pending Experience (Previous Action)
                    if pending_experience is not None:
                        # Add passive penalty (death) to previous action's reward
                        pending_experience['reward'] += passive_reward
                        
                        # Push previous transition
                        if self.agent.needs_replay_learning:
                            self.agent.push_to_replay_buffer(
                                pending_experience['state'], 
                                pending_experience['action'], 
                                pending_experience['reward'], 
                                current_s_for_prev, # The state we arrived at (start of this turn)
                                False, # Previous action did not end episode (we are here)
                                **pending_experience['info']
                            )
                            self.agent.learn_from_replay()
                    
                    # Create New Pending Experience
                    pending_experience = {
                        'state': current_s_for_prev, # State we acted in
                        'action': action,
                        'reward': active_reward, # Reward for the action itself
                        'info': info
                    }
                    
                    # Track context frequency (if agent tracks state tuples)
                    if self.use_context_aware_rewards and hasattr(self.agent, 'last_state_tuple') and self.agent.last_state_tuple:
                        context = self.agent.last_state_tuple[4]
                        self.context_counts[context] = self.context_counts.get(context, 0) + 1
                    
                    # Log metrics (including optional score breakdown)
                    self.metrics_tracker.log_metrics(state, action, total_reward, info, score_debug=score_debug)
                    episode_rewards.append(total_reward)
                    
                    # Update state
                    state = next_state
                    step += 1
                    self.step_count += 1
                    
                    # Log to experiment logger
                    if self.experiment_logger:
                        self._log_to_experiment_logger(step, total_reward, info)
                        
                else:
                    # Opponent's turn
                    # Use dummy action
                    next_state, reward, done, info = self.env.step(0)
                    
                    # We do NOT accumulate opponent reward into pending_experience here
                    # because LudoEnv calculates passive_reward at start of next agent turn.
                    
                    # Update state
                    state = next_state
                    step += 1
                    self.step_count += 1
            
            # End of Episode
            # Push the final pending experience
            if pending_experience is not None:
                final_reward = pending_experience['reward']
                
                self.agent.push_to_replay_buffer(
                    pending_experience['state'],
                    pending_experience['action'],
                    final_reward,
                    state, # Terminal state
                    True, # Done
                    **pending_experience['info']
                )
                self.agent.learn_from_replay()

            # Log episode
            episode_info['total_reward'] = sum(episode_rewards)
            episode_info['episode_length'] = step
            
            # Add context frequency metrics if using context-aware rewards
            if self.use_context_aware_rewards:
                episode_info['context_trailing'] = self.context_counts.get(CONTEXT_TRAILING, 0)
                episode_info['context_neutral'] = self.context_counts.get(CONTEXT_NEUTRAL, 0)
                episode_info['context_leading'] = self.context_counts.get(CONTEXT_LEADING, 0)
            
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
            
            # Notify agent that episode ended (for epsilon decay, etc.)
            if hasattr(self.agent, 'on_episode_end'):
                self.agent.on_episode_end()
            
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

                # Optionally save a partial snapshot of metrics (including score_debug)
                if self.config['training'].get('enable_score_debug', False):
                    suffix = f"ep_{episode + 1}"
                    self.metrics_tracker.save_partial_metrics(suffix)
        
        pbar.close()
        
        return {
            'episodes': self.episode_count,
            'steps': self.step_count,
            'mode': 'off_policy',
            'context_counts': self.context_counts if self.use_context_aware_rewards else None
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
    
    def _extract_episode_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Extract episode number from checkpoint filename.
        
        Args:
            checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/agent_episode_8000.pth')
            
        Returns:
            Episode number extracted from filename, or 0 if not found
        """
        import re
        filename = os.path.basename(checkpoint_path)
        # Match pattern like "agent_episode_8000.pth"
        match = re.search(r'episode_(\d+)', filename)
        if match:
            return int(match.group(1))
        else:
            self.logger.warning(f"Could not extract episode number from checkpoint filename: {filename}")
            return 0
