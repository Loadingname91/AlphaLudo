import gymnasium as gym
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# add src to path 
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import rl_agent_ludo.environment.ludoEnv as ludoEnv
from rl_agent_ludo.agents.ruleBasedAgent import RuleBasedAgent

def run_experiment(
    num_episodes: int = 10000,
    num_players: int = 4,
    tokens_per_player: int = 4,
    seed: int = 2024,
    verbose: bool = True,
) -> dict:
    """
    Run a rule-based experiment.
    
    Args:
        num_episodes: Number of episodes to run
        num_players: Number of players
        tokens_per_player: Number of tokens per player
        seed: Random seed
        verbose: Whether to print verbose output
    
    Returns:
        dict: Dictionary containing the results of the experiment
    """
    env = gym.make("Ludo-v0", player_id=0, num_players=num_players, tokens_per_player=tokens_per_player)

    agent = RuleBasedAgent(seed=seed)

    wins = 0
    losses = 0 
    episode_lengths = []
    rewards = []

    print(f"Running rule-based experiment with {num_episodes} episodes...")

    tqdm_bar = tqdm(total=num_episodes, desc="Running episodes", disable=not verbose)

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        state = info['state']
        done = False
        episode_length = 0
        episode_reward = 0

        while not done:
            action = agent.act(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = info['state']
            done = terminated or truncated

            episode_reward += reward 
            episode_length += 1

            # safety limit 
            if episode_length > 5000:
                break 
        
        episode_lengths.append(episode_length)
        rewards.append(episode_reward)
        
        # Track wins and losses
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        
        tqdm_bar.update(1)
        tqdm_bar.set_postfix(episode=episode, reward=episode_reward, length=episode_length)

        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1} completed")
            print(f"Win rate: {wins/(episode+1)}")
            print(f"Loss rate: {losses/(episode+1)}")
            print(f"Average episode length: {np.mean(episode_lengths)}")
            print(f"Average episode reward: {np.mean(rewards)}")
            print(f"Total wins: {wins}")
            print(f"Total losses: {losses}")
            print(f"Total episodes: {episode+1}")
            print(f"Total steps: {np.sum(episode_lengths)}")
            print(f"Total rewards: {np.sum(rewards)}")
            print(f"Average reward: {np.mean(rewards)}")
            print(f"Average episode length: {np.mean(episode_lengths)}")
        
    env.close()

    stats = {
        'num_episodes': num_episodes,
        'wins': wins,
        'losses': losses,
        'draws': num_episodes - wins - losses,
        'win_rate': wins / num_episodes,
        'avg_episode_length': float(np.mean(episode_lengths)),
        'std_episode_length': float(np.std(episode_lengths)),
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'config': {
            'num_players': num_players,
            'tokens_per_player': tokens_per_player,
            'seed': seed,
        }
    }
    
    return stats

def save_results(results: dict, agent_name: str, base_seed: int):
    """
    Save experiment results to a JSON file in the results directory.
    
    Args:
        results: Dictionary containing experiment results
        agent_name: Name of the agent used
        base_seed: Base seed used for experiments
    """
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp, agent name, and seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent_name}_seed{base_seed}_{timestamp}.json"
    filepath = results_dir / filename
    
    # Prepare data to save
    data_to_save = {
        'agent_name': agent_name,
        'timestamp': timestamp,
        'base_seed': base_seed,
        'experiments': results,
        'summary': {}
    }
    
    # Calculate summary statistics
    for config_name, stats in results.items():
        data_to_save['summary'][config_name] = {
            'win_rate': stats['win_rate'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'draws': stats['draws'],
            'avg_episode_length': stats['avg_episode_length'],
            'avg_reward': stats['avg_reward'],
        }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {filepath}")
    print(f"{'=' * 60}")
    
    return filepath


def main():
    """Run experiments with different configurations."""
    agent_name = "RuleBasedAgent"
    base_seed = 42
    
    results = {}
    
    # Experiment 1: 4 players, 4 tokens
    print("=" * 60)
    print("Experiment 1: 4 players, 4 tokens per player")
    print("=" * 60)
    results['4p4t'] = run_experiment(
        num_episodes=500,
        num_players=4,
        tokens_per_player=4,
        seed=base_seed
    )
    
    # Experiment 2: 2 players, 4 tokens
    print("\n" + "=" * 60)
    print("Experiment 2: 2 players, 4 tokens per player")
    print("=" * 60)
    results['2p4t'] = run_experiment(
        num_episodes=500,
        num_players=2,
        tokens_per_player=4,
        seed=base_seed
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    for config, stats in results.items():
        print(f"\n{config}:")
        print(f"  Win Rate: {stats['win_rate']:.2%}")
        print(f"  Wins: {stats['wins']}/{stats['num_episodes']}")
        print(f"  Avg Episode Length: {stats['avg_episode_length']:.1f} ± {stats['std_episode_length']:.1f}")
        print(f"  Avg Reward: {stats['avg_reward']:.3f} ± {stats['std_reward']:.3f}")
    
    # Save results to file
    save_results(results, agent_name, base_seed)
    
    return results


if __name__ == "__main__":
    results = main()