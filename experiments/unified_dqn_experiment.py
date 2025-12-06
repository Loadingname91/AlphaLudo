"""
Unified DQN Experiment Script

Trains a Unified DQN agent using the "Egocentric Physics" approach with:
- Unified Feature Vector (46 floats for 4 tokens, 28 floats for 2 tokens)
- Action Masking (Logit Masking) for invalid moves
- Delta-Progress + Event Impulses + ILA Penalty reward structure

CPU Performance Optimizations:
- Large batch size (256) for better CPU core utilization
- Train less frequently (every 16 steps) to reduce overhead
- Multiple gradient steps per training call (8) for better CPU efficiency
- Optimized tensor operations using torch.from_numpy
- PyTorch threading configured to use all available CPU cores
"""

import gymnasium as gym
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)
from rl_agent_ludo.agents.unifiedDQNAgent import (
    create_unified_dqn_agent_2tokens,
    create_unified_dqn_agent_4tokens,
)


def run_experiment(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 2024,
    verbose: bool = True,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,  # Not used when epsilon_schedule is set
    min_epsilon: float = 0.01,
    epsilon_schedule: str = 'exponential',  # 'linear', 'exponential', or 'adaptive'
    epsilon_decay_fraction: float = 0.9,  # Fraction of episodes to decay over (0.9 = 90%)
    batch_size: int = 256,  # Large batch size for better CPU utilization
    replay_buffer_size: int = 150000,
    target_update_frequency: int = 1000,
    train_frequency: int = 32,  # Train every 32 steps (reduced frequency for speed)
    gradient_steps: int = 1,  # Single gradient step per training call (optimized for speed)
    hidden_dims: list = [128, 128, 64],  # Smaller network for faster training
    device: str = None,  # 'cuda', 'cpu', or None for auto-detect
) -> dict:
    """
    Run a Unified DQN experiment.
    
    This trains the Unified DQN agent using the unified feature vector
    with action masking and the new reward structure.
    """
    # Create environment based on tokens_per_player
    if tokens_per_player == 2:
        env = UnifiedLudoEnv2Tokens(
            player_id=0,
            num_players=num_players,
            seed=seed,
        )
    elif tokens_per_player == 4:
        env = UnifiedLudoEnv4Tokens(
            player_id=0,
            num_players=num_players,
            seed=seed,
        )
    else:
        raise ValueError(f"tokens_per_player must be 2 or 4, got {tokens_per_player}")

    # Auto-detect device if not specified
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print(f"Using device: {device}")
        if device == 'cuda':
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create agent based on tokens_per_player
    # Use epsilon_decay=1.0 to disable automatic decay (we'll use manual scheduling like dqn_selfplay.py)
    if tokens_per_player == 2:
        agent = create_unified_dqn_agent_2tokens(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=1.0,  # Disable automatic decay, use manual scheduling
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            gradient_steps=gradient_steps,
            hidden_dims=hidden_dims,
            device=device,
            seed=seed,
        )
    else:  # tokens_per_player == 4
        agent = create_unified_dqn_agent_4tokens(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=epsilon,
            epsilon_decay=1.0,  # Disable automatic decay, use manual scheduling
            min_epsilon=min_epsilon,
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_update_frequency=target_update_frequency,
            train_frequency=train_frequency,
            gradient_steps=gradient_steps,
            hidden_dims=hidden_dims,
            device=device,
            seed=seed,
        )

    wins = 0
    losses = 0
    episode_lengths = []
    rewards = []
    
    # Debug stats for periodic logging (similar to dqn_selfplay.py)
    debug_stats = {
        'win_by_episode': [],
        'reward_by_episode': [],
        'exploration_rate': [],
        'replay_buffer_size': [],
    }
    
    # Log every N episodes for debugging (more frequent for better monitoring)
    log_interval = max(1, num_episodes // 50)  # Log ~50 times during training

    print(
        f"Running Unified DQN experiment with {num_episodes} episodes "
        f"({num_players} players, {tokens_per_player} tokens, "
        f"unified state abstraction)..."
    )

    tqdm_bar = tqdm(total=num_episodes, desc="Running episodes", disable=not verbose)

    for episode in range(num_episodes):
        # Manual epsilon scheduling (optimized for long training runs)
        decay_duration = int(num_episodes * epsilon_decay_fraction)
        progress = min(1.0, episode / max(decay_duration, 1))
        
        if epsilon_schedule == 'exponential':
            # Exponential decay: ε_t = ε_min + (ε_0 - ε_min) * (decay_rate)^t
            # For long runs: decay_rate calculated to reach min_epsilon at decay_duration
            # Using: decay_rate = (min_epsilon / epsilon)^(1/decay_duration)
            # This gives smooth exponential decay
            if decay_duration > 0:
                decay_rate = (min_epsilon / epsilon) ** (1.0 / decay_duration)
                current_epsilon = min_epsilon + (epsilon - min_epsilon) * (decay_rate ** episode)
            else:
                current_epsilon = epsilon
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
            
        elif epsilon_schedule == 'adaptive':
            # Adaptive decay: Slower decay if performance is improving
            # Use exponential but adjust based on recent win rate
            if decay_duration > 0:
                base_decay_rate = (min_epsilon / epsilon) ** (1.0 / decay_duration)
                # Adjust decay rate based on recent performance (if available)
                if len(debug_stats['win_by_episode']) > 100:
                    recent_window = min(100, len(debug_stats['win_by_episode']))
                    recent_win_rate = sum(debug_stats['win_by_episode'][-recent_window:]) / recent_window
                    # If winning more, decay faster; if struggling, decay slower
                    performance_factor = 0.8 + (recent_win_rate * 0.4)  # Range: 0.8-1.2
                    adjusted_decay_rate = base_decay_rate ** performance_factor
                else:
                    adjusted_decay_rate = base_decay_rate
                current_epsilon = min_epsilon + (epsilon - min_epsilon) * (adjusted_decay_rate ** episode)
            else:
                current_epsilon = epsilon
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
            
        else:  # 'linear' (default, matches dqn_selfplay.py)
            # Linear decay: Simple and predictable
            current_epsilon = epsilon - (epsilon - min_epsilon) * progress
            current_epsilon = max(min(current_epsilon, epsilon), min_epsilon)
        
        agent.epsilon = current_epsilon
        
        obs, info = env.reset(seed=seed + episode)
        state = info["state"]
        action_mask = info["action_mask"]
        done = False
        episode_length = 0
        episode_reward = 0.0

        # Safety cap on steps per episode
        max_steps = 10000

        while not done and episode_length < max_steps:
            prev_state = state
            prev_obs = obs.copy()
            prev_action_mask = action_mask.copy()
            
            # Select action with masking
            action = agent.act(state, obs=obs, action_mask=action_mask)

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            next_action_mask = info["action_mask"]
            done = terminated or truncated

            # Store experience and train
            agent.push_to_replay_buffer(
                prev_state,
                action,
                reward,
                next_state,
                done,
                obs=prev_obs,
                next_obs=obs,
                action_mask=prev_action_mask,
                next_action_mask=next_action_mask,
            )

            state = next_state
            action_mask = next_action_mask

            episode_reward += reward
            episode_length += 1

        episode_lengths.append(episode_length)
        rewards.append(episode_reward)

        # Check actual game outcome (similar to dqn_selfplay.py)
        episode_won = False
        if done:
            # Unwrap environment if it's wrapped (e.g., TimeLimit)
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            # Access game object to check winners
            if hasattr(unwrapped_env, 'game') and unwrapped_env.game:
                winners = unwrapped_env.game.get_winners_of_game()
                episode_won = unwrapped_env.player_id in winners if hasattr(unwrapped_env, 'player_id') else False
            # Fallback: check terminal reward if game object not available
            elif reward == 100.0:  # Win reward
                episode_won = True
        
        # Track wins and losses
        if episode_won:
            wins += 1
        elif done:  # Game ended but agent didn't win
            losses += 1
        
        # Store debug stats
        debug_stats['win_by_episode'].append(1 if episode_won else 0)
        debug_stats['reward_by_episode'].append(episode_reward)

        tqdm_bar.update(1)
        tqdm_bar.set_postfix(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            epsilon=agent.epsilon,
            buffer_size=len(agent.replay_buffer),
        )

        # Debug logging every N episodes (similar to dqn_selfplay.py)
        if (episode + 1) % log_interval == 0 or episode == 0:
            replay_buffer_size = len(agent.replay_buffer)
            debug_stats['exploration_rate'].append(agent.epsilon)
            debug_stats['replay_buffer_size'].append(replay_buffer_size)
            
            # Calculate recent win rate (last 10% of episodes so far)
            recent_window = max(10, (episode + 1) // 10)
            recent_wins = sum(debug_stats['win_by_episode'][-recent_window:])
            recent_win_rate = recent_wins / recent_window if recent_window > 0 else 0.0
            
            # Calculate recent average reward
            recent_avg_reward = np.mean(debug_stats['reward_by_episode'][-recent_window:]) if recent_window > 0 else 0.0
            
            # Overall win rate so far
            overall_win_rate = wins / (episode + 1) if episode > 0 else 0.0
            
            # Use tqdm.write() to avoid interfering with progress bar
            tqdm.write(
                f"  Episode {episode+1}/{num_episodes}: "
                f"Replay buffer={replay_buffer_size}, "
                f"ε={agent.epsilon:.3f}, "
                f"Overall win rate={overall_win_rate:.2%}, "
                f"Recent win rate={recent_win_rate:.2%}, "
                f"Recent avg reward={recent_avg_reward:.1f}"
            )

    env.close()
    
    # Final debug summary (similar to dqn_selfplay.py)
    if verbose:
        print(f"\n  Debug Summary:")
        print(f"    Final replay buffer size: {len(agent.replay_buffer)}")
        print(f"    Final epsilon: {agent.epsilon:.4f}")
        if debug_stats['replay_buffer_size']:
            print(f"    Replay buffer trend: {debug_stats['replay_buffer_size'][0]} → {debug_stats['replay_buffer_size'][-1]}")

    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "draws": num_episodes - wins - losses,
        "win_rate": wins / num_episodes if num_episodes > 0 else 0.0,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "debug_stats": debug_stats,  # Include debug stats for analysis
        "config": {
            "num_players": num_players,
            "tokens_per_player": tokens_per_player,
            "seed": seed,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "min_epsilon": min_epsilon,
            "epsilon_schedule": epsilon_schedule,
            "epsilon_decay_fraction": epsilon_decay_fraction,
            "batch_size": batch_size,
            "replay_buffer_size": replay_buffer_size,
            "target_update_frequency": target_update_frequency,
            "train_frequency": train_frequency,
            "gradient_steps": gradient_steps,
            "hidden_dims": hidden_dims,
            "device": device,
        },
    }

    return stats


def save_results(results: dict, agent_name: str, base_seed: int):
    """Save experiment results to a JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{agent_name}_seed{base_seed}_{timestamp}"
    
    filepath_json = results_dir / f"{filename_base}.json"
    data_to_save = {
        "agent_name": agent_name,
        "timestamp": timestamp,
        "base_seed": base_seed,
        "experiments": results,
        "summary": {},
    }

    for config_name, stats in results.items():
        data_to_save["summary"][config_name] = {
            "win_rate": stats["win_rate"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "draws": stats["draws"],
            "avg_episode_length": stats["avg_episode_length"],
            "avg_reward": stats["avg_reward"],
        }

    with open(filepath_json, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {filepath_json}")
    print(f"{'=' * 60}")

    return filepath_json


def run_quick_test(
    num_episodes: int = 1000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 42,
) -> dict:
    """
    Quick test function for running a single Unified DQN experiment configuration.
    """
    print("=" * 60)
    print(f"Unified DQN Quick Test: {num_episodes} episodes ({tokens_per_player} tokens)")
    print("=" * 60)
    
    results = {}
    config_name = f"unified_dqn_{num_players}p{tokens_per_player}t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed,
    )
    
    print("\n" + "=" * 60)
    print("QUICK TEST RESULTS")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(f"  Avg Episode Length: {stats['avg_episode_length']:.1f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f}")
    
    save_results(results, "UnifiedDQNAgent_QuickTest", seed)
    return results


def main(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 42,
    device: str = None,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    epsilon_schedule: str = 'exponential',  # 'linear', 'exponential', or 'adaptive'
    epsilon_decay_fraction: float = 0.9,  # Fraction of episodes to decay over
    batch_size: int = None,
    replay_buffer_size: int = 100000,
    target_update_frequency: int = 1000,
    train_frequency: int = None,
    gradient_steps: int = None,
    hidden_dims: list = None,
):
    """Run Unified DQN experiments with specified configuration."""
    agent_name = "UnifiedDQNAgent"
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GPU-optimized vs CPU-optimized hyperparameters (if not specified)
    if batch_size is None:
        batch_size = 256  # Default to 256 for faster training (works well for both CPU and GPU)
    
    if train_frequency is None:
        if device == 'cuda':
            train_frequency = 4
        else:
            train_frequency = 32  # Train less frequently for CPU (optimized for speed)
    
    if gradient_steps is None:
        gradient_steps = 1  # Single gradient step for speed (works well with large batch_size)
    
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]  # Smaller network for faster training

    results = {}

    # Run experiment with specified configuration
    print("=" * 60)
    print(f"Unified DQN Experiment: {tokens_per_player} tokens, {num_players} players (Device: {device})")
    print("=" * 60)
    config_name = f"unified_dqn_{num_players}p{tokens_per_player}t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        min_epsilon=min_epsilon,
        epsilon_schedule=epsilon_schedule,
        epsilon_decay_fraction=epsilon_decay_fraction,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=hidden_dims,
        device=device,
    )

    # Save results
    save_results(results, agent_name, seed)

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Draws: {stats['draws']}")
        print(f"  Avg Episode Length: {stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f} ± {stats['std_reward']:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Unified DQN experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test (1000 episodes)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tokens", type=int, default=4, choices=[2, 4], help="Tokens per player (2 or 4)")
    parser.add_argument("--players", type=int, default=4, choices=[2, 4], help="Number of players (2 or 4)")
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'], 
                       help="Device to use ('cuda' for GPU, 'cpu' for CPU, None for auto-detect)")
    
    # Hyperparameters (similar to dqn_selfplay.py)
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Epsilon decay rate (not used with --epsilon_schedule)")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--epsilon_schedule", type=str, default='exponential', choices=['linear', 'exponential', 'adaptive'],
                       help="Epsilon decay schedule: 'linear' (simple), 'exponential' (recommended for long runs), 'adaptive' (performance-based)")
    parser.add_argument("--epsilon_decay_fraction", type=float, default=0.9,
                       help="Fraction of episodes to decay epsilon over (0.9 = 90%%, default: 0.9 for long runs)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (default: 256 for faster training)")
    parser.add_argument("--replay_buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--target_update_frequency", type=int, default=100, help="Steps between target network updates")
    parser.add_argument("--train_frequency", type=int, default=None, help="Steps between training calls (None for auto)")
    parser.add_argument("--gradient_steps", type=int, default=None, help="Gradient steps per training call (None for auto)")
    parser.add_argument("--hidden_dims", type=str, default="128,128,64", 
                       help="Hidden layer dimensions (comma-separated, e.g., '256,256,128')")
    
    args = parser.parse_args()
    
    # Parse hidden_dims
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',')]
    
    if args.quick:
        results = {}
        config_name = f"unified_dqn_quick_test_{args.tokens}t"
        results[config_name] = run_experiment(
            num_episodes=1000,
            num_players=args.players,
            tokens_per_player=args.tokens,
            seed=args.seed,
            device=args.device,
            learning_rate=args.alpha,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            epsilon_schedule=args.epsilon_schedule,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            target_update_frequency=args.target_update_frequency,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
            hidden_dims=hidden_dims,
        )
        save_results(results, "UnifiedDQNAgent_QuickTest", args.seed)
    else:
        main(
            num_episodes=args.episodes,
            num_players=args.players,
            tokens_per_player=args.tokens,
            seed=args.seed,
            device=args.device,
            learning_rate=args.alpha,
            discount_factor=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            epsilon_schedule=args.epsilon_schedule,
            epsilon_decay_fraction=args.epsilon_decay_fraction,
            batch_size=args.batch_size,
            replay_buffer_size=args.replay_buffer_size,
            target_update_frequency=args.target_update_frequency,
            train_frequency=args.train_frequency,
            gradient_steps=args.gradient_steps,
            hidden_dims=hidden_dims,
        )

