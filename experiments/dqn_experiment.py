"""
DQN Experiment Script

Trains a DQN agent using combined state abstraction (13-tuple).
Based on the reference implementation from: https://github.com/raffaele-aurucci/Ludo_Game_AI

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

import rl_agent_ludo.environment.ludoEnv as ludoEnv
from rl_agent_ludo.agents.dqnAgent import DQNAgent


def run_experiment(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 2024,
    verbose: bool = True,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    batch_size: int = 256,  # Large batch size for better CPU utilization (8 cores)
    replay_buffer_size: int = 10000,
    target_update_frequency: int = 100,
    train_frequency: int = 16,  # Train every 16 steps (less frequent = faster)
    gradient_steps: int = 8,  # More gradient steps per training call (better CPU utilization)
    hidden_dims: list = [128, 128, 64],
    device: str = None,  # 'cuda', 'cpu', or None for auto-detect
) -> dict:
    """
    Run a DQN experiment.

    This trains the DQN agent using combined state abstraction (13-tuple).
    """
    env = gym.make(
        "Ludo-v0",
        player_id=0,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
    )

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
    
    agent = DQNAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
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

    print(
        f"Running DQN experiment with {num_episodes} episodes "
        f"({num_players} players, {tokens_per_player} tokens, "
        f"combined state abstraction)..."
    )

    tqdm_bar = tqdm(total=num_episodes, desc="Running episodes", disable=not verbose)

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        state = info["state"]
        done = False
        episode_length = 0
        episode_reward = 0.0

        # Safety cap on steps per episode
        max_steps = 5000

        while not done and episode_length < max_steps:
            prev_state = state
            action = agent.act(state)

            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            done = terminated or truncated

            # Store experience and train
            agent.push_to_replay_buffer(
                prev_state,
                action,
                reward,
                next_state,
                done,
            )

            state = next_state

            episode_reward += reward
            episode_length += 1

        episode_lengths.append(episode_length)
        rewards.append(episode_reward)

        # Track wins and losses
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1

        tqdm_bar.update(1)
        tqdm_bar.set_postfix(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            epsilon=agent.epsilon,
            buffer_size=len(agent.replay_buffer),
        )

        if (episode + 1) % 100 == 0 and verbose:
            print(f"\nEpisode {episode + 1} completed")
            print(f"Win rate: {wins / (episode + 1):.3f}")
            print(f"Loss rate: {losses / (episode + 1):.3f}")
            print(f"Average episode length: {np.mean(episode_lengths[-100:]):.2f}")
            print(f"Average episode reward: {np.mean(rewards[-100:]):.3f}")
            print(f"Current epsilon: {agent.epsilon:.4f}")
            print(f"Replay buffer size: {len(agent.replay_buffer)}")
            print(f"Training steps: {agent.training_steps}")

    env.close()

    stats = {
        "num_episodes": num_episodes,
        "wins": wins,
        "losses": losses,
        "draws": num_episodes - wins - losses,
        "win_rate": wins / num_episodes,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "config": {
            "num_players": num_players,
            "tokens_per_player": tokens_per_player,
            "seed": seed,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "min_epsilon": min_epsilon,
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
    seed: int = 42,
) -> dict:
    """
    Quick test function for running a single DQN experiment configuration.
    """
    print("=" * 60)
    print(f"DQN Quick Test: {num_episodes} episodes")
    print("=" * 60)
    
    results = {}
    config_name = f"dqn_combined_{num_players}p4t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=4,
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
    
    save_results(results, "DQNAgent_Combined_QuickTest", seed)
    return results


def main(device=None):
    """Run DQN experiments with different configurations."""
    agent_name = "DQNAgent"
    base_seed = 42
    
    # Auto-detect device if not specified
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # GPU-optimized vs CPU-optimized hyperparameters
    if device == 'cuda':
        batch_size = 512
        train_frequency = 4
        gradient_steps = 1
    else:
        batch_size = 256
        train_frequency = 16
        gradient_steps = 8

    results = {}

    # Experiment 1: Default hyperparameters (matching reference implementation)
    print("=" * 60)
    print(f"DQN Experiment 1: Default hyperparameters (Device: {device})")
    print("=" * 60)
    config_name = "dqn_default_10k"
    results[config_name] = run_experiment(
        num_episodes=10000,
        num_players=4,
        tokens_per_player=4,
        seed=base_seed,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        batch_size=batch_size,
        replay_buffer_size=10000,
        target_update_frequency=100,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=[128, 128, 64],
        device=device,
    )

    # Save results
    save_results(results, agent_name, base_seed)

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
    
    parser = argparse.ArgumentParser(description="Run DQN experiments")
    parser.add_argument("--quick", action="store_true", help="Run quick test (1000 episodes)")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, choices=['cuda', 'cpu'], 
                       help="Device to use ('cuda' for GPU, 'cpu' for CPU, None for auto-detect)")
    
    args = parser.parse_args()
    
    if args.quick:
        results = {}
        config_name = "dqn_quick_test"
        results[config_name] = run_experiment(
            num_episodes=1000,
            num_players=4,
            tokens_per_player=4,
            seed=args.seed,
            device=args.device,
        )
        save_results(results, "DQNAgent_QuickTest", args.seed)
    else:
        main(device=args.device)

