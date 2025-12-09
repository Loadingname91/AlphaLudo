"""
Comprehensive evaluation of all trained models (Levels 1-5).
Tests each model and saves detailed results for visualization.
"""

import sys
from pathlib import Path
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level1_simple import Level1SimpleLudo
from rl_agent_ludo.environment.level2_interaction import Level2InteractionLudo
from rl_agent_ludo.environment.level3_multitoken import Level3MultiTokenLudo
from rl_agent_ludo.environment.level4_stochastic import Level4StochasticLudo
from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent


def evaluate_model(env, agent, num_episodes=500, level_name="Level"):
    """Evaluate a trained agent."""
    print(f"\n{'='*80}")
    print(f"Evaluating {level_name}")
    print(f"{'='*80}")

    wins = 0
    episode_rewards = []
    episode_lengths = []
    captures_by_agent = []
    captures_of_agent = []

    for ep in tqdm(range(num_episodes), desc=f"{level_name} Evaluation"):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 500:
            action = agent.act(obs, greedy=True)  # Greedy evaluation
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated
            steps += 1

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        winner = info.get('winner', -1)
        if winner == 0:
            wins += 1

        # Track captures if available
        if 'captures_by_agent' in info:
            captures_by_agent.append(info['captures_by_agent'])
        if 'captures_of_agent' in info:
            captures_of_agent.append(info['captures_of_agent'])

    # Calculate statistics
    win_rate = wins / num_episodes
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    results = {
        'level': level_name,
        'num_episodes': num_episodes,
        'win_rate': float(win_rate),
        'wins': wins,
        'avg_reward': float(avg_reward),
        'std_reward': float(std_reward),
        'avg_length': float(avg_length),
        'std_length': float(std_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
    }

    if captures_by_agent:
        results['avg_captures_by_agent'] = float(np.mean(captures_by_agent))
        results['avg_captures_of_agent'] = float(np.mean(captures_of_agent))
        results['captures_by_agent'] = [int(c) for c in captures_by_agent]
        results['captures_of_agent'] = [int(c) for c in captures_of_agent]

    # Print summary
    print(f"\n{'='*60}")
    print(f"{level_name} Results:")
    print(f"{'='*60}")
    print(f"  Win Rate: {win_rate:.1%} ({wins}/{num_episodes})")
    print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Avg Length: {avg_length:.1f} ± {std_length:.1f} steps")
    if captures_by_agent:
        print(f"  Captures by agent: {np.mean(captures_by_agent):.2f}/episode")
        print(f"  Captures of agent: {np.mean(captures_of_agent):.2f}/episode")
    print(f"{'='*60}\n")

    return results


def main():
    """Evaluate all models and save results."""

    device = torch.device("cpu")
    results_dir = Path("results/evaluations")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'num_eval_episodes': 500,
        'levels': {}
    }

    # Level 1: Basic Movement
    print("\n" + "="*80)
    print("LEVEL 1: Basic Movement")
    print("="*80)
    env1 = Level1SimpleLudo(seed=42)
    agent1 = SimpleDQNAgent(
        state_dim=4,
        action_dim=2,
        hidden_dims=[128, 128],
        device=str(device)
    )
    checkpoint_path = "checkpoints/level1/best_model_ep002000_wr0.930_20251208_161834.pth"
    agent1.load(checkpoint_path)
    results1 = evaluate_model(env1, agent1, num_episodes=500, level_name="Level 1")
    all_results['levels']['level1'] = results1

    # Level 2: Opponent Interaction
    print("\n" + "="*80)
    print("LEVEL 2: Opponent Interaction")
    print("="*80)
    env2 = Level2InteractionLudo(seed=42)
    agent2 = SimpleDQNAgent(
        state_dim=8,
        action_dim=2,
        hidden_dims=[128, 128],
        device=str(device)
    )
    checkpoint_path = "checkpoints/level2/best_model_ep002500_wr0.830_20251208_164448.pth"
    agent2.load(checkpoint_path)
    results2 = evaluate_model(env2, agent2, num_episodes=500, level_name="Level 2")
    all_results['levels']['level2'] = results2

    # Level 3: Multi-Token Strategy
    print("\n" + "="*80)
    print("LEVEL 3: Multi-Token Strategy")
    print("="*80)
    env3 = Level3MultiTokenLudo(seed=42)
    agent3 = SimpleDQNAgent(
        state_dim=14,
        action_dim=3,
        hidden_dims=[128, 128],
        device=str(device)
    )
    checkpoint_path = "checkpoints/level3/best_model_ep004000_wr0.790_20251208_175158.pth"
    agent3.load(checkpoint_path)
    results3 = evaluate_model(env3, agent3, num_episodes=500, level_name="Level 3")
    all_results['levels']['level3'] = results3

    # Level 4: Stochastic Dynamics
    print("\n" + "="*80)
    print("LEVEL 4: Stochastic Dynamics")
    print("="*80)
    env4 = Level4StochasticLudo(seed=42)
    agent4 = SimpleDQNAgent(
        state_dim=16,
        action_dim=3,
        hidden_dims=[128, 128],
        device=str(device)
    )
    checkpoint_path = "checkpoints/level4/best_model_ep001500_wr0.625_20251208_181658.pth"
    agent4.load(checkpoint_path)
    results4 = evaluate_model(env4, agent4, num_episodes=500, level_name="Level 4")
    all_results['levels']['level4'] = results4

    # Level 5: Multi-Agent Chaos
    print("\n" + "="*80)
    print("LEVEL 5: Multi-Agent Chaos")
    print("="*80)
    env5 = Level5MultiAgentLudo(seed=42)
    agent5 = SimpleDQNAgent(
        state_dim=16,
        action_dim=3,
        hidden_dims=[128, 128],
        device=str(device)
    )
    checkpoint_path = "checkpoints/level5/best_model_ep014000_wr0.610_20251208_195012.pth"
    agent5.load(checkpoint_path)
    results5 = evaluate_model(env5, agent5, num_episodes=500, level_name="Level 5")
    all_results['levels']['level5'] = results5

    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"all_models_evaluation_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Level':<15} {'Win Rate':<15} {'Avg Reward':<20} {'Avg Length':<15}")
    print("-" * 65)

    for level_key in ['level1', 'level2', 'level3', 'level4', 'level5']:
        data = all_results['levels'][level_key]
        print(f"{data['level']:<15} {data['win_rate']:.1%} ({data['wins']}/500)   "
              f"{data['avg_reward']:>6.1f} ± {data['std_reward']:<6.1f}   "
              f"{data['avg_length']:>5.1f} ± {data['std_length']:<5.1f}")

    print("\n" + "="*80)
    print(f"Results saved to: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()
