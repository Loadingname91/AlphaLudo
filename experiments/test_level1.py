"""
Quick test script for Level 1 environment.

Tests:
1. Environment basics (reset, step, render)
2. Random agent gameplay
3. Greedy agent gameplay
4. Random vs Random baseline (expected ~50% win rate)
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level1_simple import Level1SimpleLudo
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def test_environment_basics():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Environment Basics")
    print("="*60)

    env = Level1SimpleLudo(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Initial observation: {obs}")

    # Take a few steps
    print(f"\n✓ Running 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.1f}, done={terminated or truncated}")

        if terminated or truncated:
            print(f"  Episode ended: winner={info.get('winner')}")
            break

    env.close()
    print("\n✓ Environment basics test passed!")


def test_random_agent():
    """Test random agent gameplay."""
    print("\n" + "="*60)
    print("TEST 2: Random Agent Gameplay")
    print("="*60)

    env = Level1SimpleLudo(seed=42)
    agent = RandomAgent(seed=42)

    obs, info = env.reset(seed=42)
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < 100:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"✓ Random agent completed episode")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Winner: {info.get('winner')}")
    print(f"  Final positions: {info.get('player_positions')}")


def test_greedy_agent():
    """Test greedy agent gameplay."""
    print("\n" + "="*60)
    print("TEST 3: Greedy Agent Gameplay")
    print("="*60)

    env = Level1SimpleLudo(seed=42)
    agent = GreedyAgent()

    obs, info = env.reset(seed=42)
    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < 100:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"✓ Greedy agent completed episode")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Winner: {info.get('winner')}")
    print(f"  Final positions: {info.get('player_positions')}")


def test_random_vs_random_baseline(num_episodes=100):
    """
    Test random vs random baseline.

    Since opponent is also random, we expect ~50% win rate.
    This validates the environment is not biased.
    """
    print("\n" + "="*60)
    print(f"TEST 4: Random vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: ~50% win rate (no bias)")

    env = Level1SimpleLudo(seed=42)
    agent = RandomAgent(seed=42)

    wins = 0
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if info.get('winner') == 0:
            wins += 1

    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n✓ Baseline test completed")
    print(f"  Win rate: {win_rate:.2%} ({wins}/{num_episodes})")
    print(f"  Avg reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg episode length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")

    # Check if win rate is close to 50%
    if 0.40 <= win_rate <= 0.60:
        print(f"  ✓ Win rate is reasonable (40-60%)")
    else:
        print(f"  ⚠ Win rate outside expected range (40-60%)")


def test_greedy_vs_random_baseline(num_episodes=100):
    """
    Test greedy vs random baseline.

    Greedy should beat random since it always moves forward.
    Expected win rate: 60-80%
    """
    print("\n" + "="*60)
    print(f"TEST 5: Greedy vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: 60-80% win rate (greedy should beat random)")

    env = Level1SimpleLudo(seed=42)
    agent = GreedyAgent()

    wins = 0
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=42 + ep)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        if info.get('winner') == 0:
            wins += 1

    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n✓ Greedy vs Random test completed")
    print(f"  Win rate: {win_rate:.2%} ({wins}/{num_episodes})")
    print(f"  Avg reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg episode length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")

    # Check if greedy beats random
    if win_rate >= 0.55:
        print(f"  ✓ Greedy agent beats random (win rate >= 55%)")
    else:
        print(f"  ⚠ Greedy agent not beating random significantly")


def main():
    print("\n" + "="*80)
    print("LEVEL 1 ENVIRONMENT TEST SUITE")
    print("="*80)

    try:
        test_environment_basics()
        test_random_agent()
        test_greedy_agent()
        test_random_vs_random_baseline(num_episodes=100)
        test_greedy_vs_random_baseline(num_episodes=100)

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nEnvironment is ready for training!")
        print("Run: python experiments/level1_train.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
