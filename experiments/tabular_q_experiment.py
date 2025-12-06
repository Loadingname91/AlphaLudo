# experiments/tabular_q_experiment.py

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
from rl_agent_ludo.agents.tabularQAgent import TabularQAgent


def run_experiment(
    num_episodes: int = 5000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 2024,
    verbose: bool = True,
    state_abstraction: str = 'potential',  # 'potential', 'zone_based', or 'combined'
) -> dict:
    """
    Run a Tabular Q-Learning experiment.

    This both trains and evaluates the agent online over num_episodes.
    """

    env = gym.make(
        "Ludo-v0",
        player_id=0,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
    )

    agent = TabularQAgent(seed=seed, state_abstraction=state_abstraction)

    wins = 0
    losses = 0
    episode_lengths = []
    rewards = []

    print(
        f"Running Tabular Q-Learning experiment with {num_episodes} episodes "
        f"({num_players} players, {tokens_per_player} tokens, "
        f"state_abstraction={state_abstraction})..."
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

            # Online Q-learning update
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

        # Track wins and losses (same convention as rule-based experiment)
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1

        tqdm_bar.update(1)
        tqdm_bar.set_postfix(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            epsilon=getattr(agent, "epsilon", None),
        )

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1} completed")
            print(f"Win rate: {wins / (episode + 1):.3f}")
            print(f"Loss rate: {losses / (episode + 1):.3f}")
            print(f"Average episode length: {np.mean(episode_lengths):.2f}")
            print(f"Average episode reward: {np.mean(rewards):.3f}")
            print(f"Total wins: {wins}")
            print(f"Total losses: {losses}")
            print(f"Total episodes: {episode + 1}")
            print(f"Total steps: {int(np.sum(episode_lengths))}")
            print(f"Total rewards: {np.sum(rewards):.2f}")
            print(f"Current epsilon: {agent.epsilon:.4f}")

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
            "state_abstraction": state_abstraction,
        },
    }

    return stats


def save_results(results: dict, agent_name: str, base_seed: int, episode_metrics: list = None):
    """
    Save experiment results to a JSON file and detailed metrics to a CSV log.
    """
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{agent_name}_seed{base_seed}_{timestamp}"
    
    # 1. Save JSON summary
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

    # 2. Save Episode Metrics Log (CSV format for easy plotting)
    if episode_metrics:
        filepath_log = results_dir / f"{filename_base}.csv"
        with open(filepath_log, "w") as f:
            f.write("config,episode,win_rate,loss_rate,avg_length,avg_reward,epsilon\n")
            for metric in episode_metrics:
                f.write(
                    f"{metric['config']},{metric['episode']},"
                    f"{metric['win_rate']:.4f},{metric['loss_rate']:.4f},"
                    f"{metric['avg_length']:.2f},{metric['avg_reward']:.4f},"
                    f"{metric['epsilon']:.6f}\n"
                )
        print(f"Detailed log saved to: {filepath_log}")

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {filepath_json}")
    print(f"{'=' * 60}")

    return filepath_json


def run_quick_test(
    state_abstraction: str = 'zone_based',
    num_episodes: int = 1000,
    num_players: int = 4,
    seed: int = 42,
) -> dict:
    """
    Quick test function for running a single experiment configuration.
    
    Args:
        state_abstraction: 'potential', 'zone_based', or 'combined'
        num_episodes: Number of episodes to run
        num_players: Number of players in the game
        seed: Random seed
    
    Returns:
        Dictionary with experiment results
    """
    print("=" * 60)
    print(f"Quick Test: {state_abstraction} abstraction, {num_episodes} episodes")
    print("=" * 60)
    
    results = {}
    config_name = f"{state_abstraction}_{num_players}p4t_{num_episodes//1000}k"
    results[config_name] = run_experiment(
        num_episodes=num_episodes,
        num_players=num_players,
        tokens_per_player=4,
        seed=seed,
        state_abstraction=state_abstraction,
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
    
    save_results(results, f"TabularQAgent_{state_abstraction.capitalize()}_QuickTest", seed)
    return results


def main():
    """Run Tabular Q-Learning experiments with different configurations."""
    agent_name = "TabularQAgent"
    base_seed = 42

    results = {}

    # Experiment 1: 4 players, 4 tokens, potential-based
    print("=" * 60)
    print("TabularQ Experiment 1: 4 players, 4 tokens per player (potential-based)")
    print("=" * 60)
    results["4p4t_potential"] = run_experiment(
        num_episodes=10000,  # 10k episodes for hyperparameter check
        num_players=4,
        tokens_per_player=4,
        seed=base_seed,
        state_abstraction='potential',
    )

    # Experiment 2: 4 players, 4 tokens, zone-based
    print("\n" + "=" * 60)
    print("TabularQ Experiment 2: 4 players, 4 tokens per player (zone-based)")
    print("=" * 60)
    results["4p4t_zone"] = run_experiment(
        num_episodes=10000,  # 10k episodes for hyperparameter check
        num_players=4,
        tokens_per_player=4,
        seed=base_seed,
        state_abstraction='zone_based',
    )

    # Experiment 3: 2 players, 4 tokens, potential-based
    print("\n" + "=" * 60)
    print("TabularQ Experiment 3: 2 players, 4 tokens per player (potential-based)")
    print("=" * 60)
    results["2p4t_potential"] = run_experiment(
        num_episodes=10000,  # 10k episodes for hyperparameter check
        num_players=2,
        tokens_per_player=4,
        seed=base_seed,
        state_abstraction='potential',
    )

    # Experiment 4: 2 players, 4 tokens, zone-based
    print("\n" + "=" * 60)
    print("TabularQ Experiment 4: 2 players, 4 tokens per player (zone-based)")
    print("=" * 60)
    results["2p4t_zone"] = run_experiment(
        num_episodes=10000,  # 10k episodes for hyperparameter check
        num_players=2,
        tokens_per_player=4,
        seed=base_seed,
        state_abstraction='zone_based',
    )

    print("\n" + "=" * 60)
    print("TABULAR Q EXPERIMENT SUMMARY")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(
            f"  Avg Episode Length: "
            f"{stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}"
        )
        print(
            f"  Avg Reward: "
            f"{stats['avg_reward']:.3f} ± {stats['std_reward']:.3f}"
        )

    save_results(results, agent_name, base_seed)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Tabular Q-Learning experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "quick", "zone", "potential"],
        default="full",
        help="Experiment mode: 'full' (all experiments), 'quick' (single quick test), "
             "'zone' (zone-based only), 'potential' (potential-based only)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of episodes (for quick/zone/potential modes)"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=4,
        help="Number of players (for quick/zone/potential modes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        run_quick_test(
            state_abstraction='zone_based',
            num_episodes=args.episodes,
            num_players=args.players,
            seed=args.seed,
        )
    elif args.mode == "zone":
        results = {}
        results["zone_test"] = run_experiment(
            num_episodes=args.episodes,
            num_players=args.players,
            tokens_per_player=4,
            seed=args.seed,
            state_abstraction='zone_based',
        )
        save_results(results, "TabularQAgent_ZoneBased", args.seed)
    elif args.mode == "potential":
        results = {}
        results["potential_test"] = run_experiment(
            num_episodes=args.episodes,
            num_players=args.players,
            tokens_per_player=4,
            seed=args.seed,
            state_abstraction='potential',
        )
        save_results(results, "TabularQAgent_Potential", args.seed)
    else:  # full
        main()