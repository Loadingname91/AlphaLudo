"""
Demo script to visualize trained agents playing at each level.
Shows live rendering of the game with the trained model.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import time
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level1_simple import Level1SimpleLudo
from rl_agent_ludo.environment.level2_interaction import Level2InteractionLudo
from rl_agent_ludo.environment.level3_multitoken import Level3MultiTokenLudo
from rl_agent_ludo.environment.level4_stochastic import Level4StochasticLudo
from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent


def demo_level(level, env, agent, num_episodes=5):
    """Demo a trained agent playing."""
    print(f"\n{'='*80}")
    print(f"DEMO: {level}")
    print(f"{'='*80}\n")
    print("Watch the agent play in the terminal!")
    print(f"Running {num_episodes} episodes...\n")

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < 500:
            action = agent.act(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            done = terminated or truncated
            steps += 1

            # Small delay so you can see what's happening
            time.sleep(0.1)

        winner = info.get('winner', -1)
        result = "WON!" if winner == 0 else "LOST"
        print(f"\nEpisode {ep+1} finished: {result}")
        print(f"  Steps: {steps}, Total Reward: {episode_reward:.1f}")
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description='Demo trained agents with rendering')
    parser.add_argument('--level', type=int, choices=[1, 2, 3, 4, 5],
                       help='Which level to demo (1-5), or omit to demo all')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')

    args = parser.parse_args()
    device = torch.device("cpu")

    levels_config = {
        1: {
            'name': 'Level 1: Basic Movement',
            'env': Level1SimpleLudo(render_mode="human", seed=42),
            'state_dim': 4,
            'action_dim': 2,
            'checkpoint': 'checkpoints/level1/best_model_ep002000_wr0.930_20251208_161834.pth'
        },
        2: {
            'name': 'Level 2: Opponent Interaction',
            'env': Level2InteractionLudo(render_mode="human", seed=42),
            'state_dim': 8,
            'action_dim': 2,
            'checkpoint': 'checkpoints/level2/best_model_ep002500_wr0.830_20251208_164448.pth'
        },
        3: {
            'name': 'Level 3: Multi-Token Strategy',
            'env': Level3MultiTokenLudo(render_mode="human", seed=42),
            'state_dim': 14,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level3/best_model_ep004000_wr0.790_20251208_175158.pth'
        },
        4: {
            'name': 'Level 4: Stochastic Dynamics',
            'env': Level4StochasticLudo(render_mode="human", seed=42),
            'state_dim': 16,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level4/best_model_ep001500_wr0.625_20251208_181658.pth'
        },
        5: {
            'name': 'Level 5: Multi-Agent Chaos',
            'env': Level5MultiAgentLudo(render_mode="human", seed=42),
            'state_dim': 16,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level5/best_model_ep014000_wr0.610_20251208_195012.pth'
        }
    }

    levels_to_run = [args.level] if args.level else [1, 2, 3, 4, 5]

    for level_num in levels_to_run:
        config = levels_config[level_num]

        # Load agent
        agent = SimpleDQNAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dims=[128, 128],
            device='cpu'
        )
        agent.load(config['checkpoint'])

        # Demo
        demo_level(config['name'], config['env'], agent, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
