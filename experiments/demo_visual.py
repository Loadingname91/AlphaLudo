"""
Demo script with GRAPHICAL visualization using CV2 windows.
Shows a visual board with pieces moving in real-time!
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

from rl_agent_ludo.environment.standard_board_visualizer import StandardLudoBoardVisualizer


def extract_positions_level1(env):
    """Extract positions for Level 1."""
    return [[env.player_positions[0]], [env.player_positions[1]]]


def extract_positions_level2(env):
    """Extract positions for Level 2."""
    return [[env.player_positions[0]], [env.player_positions[1]]]


def extract_positions_level3(env):
    """Extract positions for Level 3 (2 tokens each)."""
    return [
        env.my_tokens.copy(),
        env.opp_tokens.copy()
    ]


def extract_positions_level4(env):
    """Extract positions for Level 4 (2 tokens each)."""
    return [
        env.my_tokens.copy(),
        env.opp_tokens.copy()
    ]


def extract_positions_level5(env):
    """Extract positions for Level 5 (4 players, 2 tokens each)."""
    # Player 0 (agent) - my_tokens
    positions = [env.my_tokens.copy()]

    # Players 1-3 (opponents) - opponent_tokens
    for opp_tokens in env.opponent_tokens:
        positions.append(opp_tokens.copy())

    return positions


def demo_level_visual(level_num, level_name, env, agent, extract_fn, num_episodes=3):
    """Demo a trained agent with visual rendering."""
    print(f"\n{'='*80}")
    print(f"DEMO: {level_name}")
    print(f"{'='*80}\n")
    print("🎮 Watch the CV2 window for visual gameplay!")
    print(f"Running {num_episodes} episodes...\n")
    print("Press 'q' in the CV2 window to quit early")
    print("Or close the window to stop\n")

    # Determine number of players and tokens
    if level_num <= 2:
        num_players = 2
        tokens_per_player = 1
    elif level_num <= 4:
        num_players = 2
        tokens_per_player = 2
    else:  # Level 5
        num_players = 4
        tokens_per_player = 2

    # Create visualizer (use standard board layout)
    viz = StandardLudoBoardVisualizer(level_num, num_players, tokens_per_player)

    try:
        for ep in range(num_episodes):
            print(f"\n--- Episode {ep+1}/{num_episodes} ---")
            obs, info = env.reset(seed=42 + ep)
            done = False
            episode_reward = 0
            steps = 0

            while not done and steps < 500:
                # Extract current state for visualization
                positions = extract_fn(env)

                state_info = {
                    'player_positions': positions,
                    'current_player': getattr(env, 'current_player', 0),
                    'dice': getattr(env, 'current_dice', None),
                    'winner': getattr(env, 'winner', None) if getattr(env, 'done', False) else None,
                    'step': steps
                }

                # Render
                viz.render(state_info)

                # Check for quit
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    print("\nQuitting early...")
                    viz.close()
                    return

                # Agent acts
                action = agent.act(obs, greedy=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                done = terminated or truncated
                steps += 1

                # Small delay
                time.sleep(0.05)

            # Show final state
            positions = extract_fn(env)
            state_info = {
                'player_positions': positions,
                'current_player': getattr(env, 'current_player', 0),
                'dice': getattr(env, 'current_dice', None),
                'winner': getattr(env, 'winner', None),
                'step': steps
            }
            viz.render(state_info)

            winner = info.get('winner', -1)
            result = "WON! 🎉" if winner == 0 else "LOST 😢"
            print(f"Episode {ep+1} finished: {result}")
            print(f"  Steps: {steps}, Total Reward: {episode_reward:.1f}")

            # Pause before next episode
            print("Next episode in 2 seconds...")
            time.sleep(2)

    finally:
        viz.close()


def main():
    parser = argparse.ArgumentParser(description='Demo with graphical visualization')
    parser.add_argument('--level', type=int, choices=[1, 2, 3, 4, 5], required=True,
                       help='Which level to demo (1-5)')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')

    args = parser.parse_args()
    device = torch.device("cpu")

    levels_config = {
        1: {
            'name': 'Level 1: Basic Movement',
            'env': Level1SimpleLudo(seed=42),
            'extract_fn': extract_positions_level1,
            'state_dim': 4,
            'action_dim': 2,
            'checkpoint': 'checkpoints/level1/best_model_ep002000_wr0.930_20251208_161834.pth'
        },
        2: {
            'name': 'Level 2: Opponent Interaction',
            'env': Level2InteractionLudo(seed=42),
            'extract_fn': extract_positions_level2,
            'state_dim': 8,
            'action_dim': 2,
            'checkpoint': 'checkpoints/level2/best_model_ep002500_wr0.830_20251208_164448.pth'
        },
        3: {
            'name': 'Level 3: Multi-Token Strategy',
            'env': Level3MultiTokenLudo(seed=42),
            'extract_fn': extract_positions_level3,
            'state_dim': 14,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level3/best_model_ep004000_wr0.790_20251208_175158.pth'
        },
        4: {
            'name': 'Level 4: Stochastic Dynamics',
            'env': Level4StochasticLudo(seed=42),
            'extract_fn': extract_positions_level4,
            'state_dim': 16,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level4/best_model_ep001500_wr0.625_20251208_181658.pth'
        },
        5: {
            'name': 'Level 5: Multi-Agent Chaos',
            'env': Level5MultiAgentLudo(seed=42),
            'extract_fn': extract_positions_level5,
            'state_dim': 16,
            'action_dim': 3,
            'checkpoint': 'checkpoints/level5/best_model_ep014000_wr0.610_20251208_195012.pth'
        }
    }

    config = levels_config[args.level]

    # Load agent
    agent = SimpleDQNAgent(
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        hidden_dims=[128, 128],
        device='cpu'
    )
    agent.load(config['checkpoint'])

    # Demo with visualization
    demo_level_visual(
        args.level,
        config['name'],
        config['env'],
        agent,
        config['extract_fn'],
        num_episodes=args.episodes
    )


if __name__ == "__main__":
    import cv2
    main()
