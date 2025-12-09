"""
Quick test script for Level 2 environment.

Tests:
1. Environment basics (reset, step, capturing)
2. Safe zones work correctly
3. Capture mechanics work
4. Random vs Random baseline
5. Greedy vs Random baseline
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level2_interaction import Level2InteractionLudo
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def test_environment_basics():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Environment Basics")
    print("="*60)

    env = Level2InteractionLudo(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Safe zones: {env.SAFE_ZONES}")
    print(f"  Initial observation: {obs}")

    # Take a few steps
    print(f"\n✓ Running 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.1f}, "
              f"captures_by_me={info['num_captures_by_me']}, "
              f"captures_of_me={info['num_captures_of_me']}")

        if terminated or truncated:
            break

    env.close()
    print("\n✓ Environment basics test passed!")


def test_capture_mechanics():
    """Test that capturing works correctly."""
    print("\n" + "="*60)
    print("TEST 2: Capture Mechanics")
    print("="*60)

    env = Level2InteractionLudo(seed=42)

    # Run multiple episodes looking for captures
    total_captures_by_agent = 0
    total_captures_of_agent = 0
    episodes_with_captures = 0

    for ep in range(50):
        obs, info = env.reset(seed=42 + ep)
        done = False
        steps = 0

        while not done and steps < 100:
            action = 0  # Always try to move
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        total_captures_by_agent += info['num_captures_by_me']
        total_captures_of_agent += info['num_captures_of_me']

        if info['num_captures_by_me'] > 0 or info['num_captures_of_me'] > 0:
            episodes_with_captures += 1

    print(f"✓ Capture mechanics test completed (50 episodes)")
    print(f"  Total captures by agent: {total_captures_by_agent}")
    print(f"  Total captures of agent: {total_captures_of_agent}")
    print(f"  Episodes with captures: {episodes_with_captures}/50")

    if total_captures_by_agent > 0 or total_captures_of_agent > 0:
        print(f"  ✓ Capturing works!")
    else:
        print(f"  ⚠ No captures observed (might need more episodes)")


def test_safe_zones():
    """Test that safe zones prevent captures."""
    print("\n" + "="*60)
    print("TEST 3: Safe Zone Protection")
    print("="*60)

    env = Level2InteractionLudo(seed=42)
    print(f"Safe zones at positions: {env.SAFE_ZONES}")

    # Check that _is_safe_zone works
    for pos in env.SAFE_ZONES:
        assert env._is_safe_zone(pos), f"Position {pos} should be safe zone"

    print(f"✓ Safe zone detection works")

    # Check vulnerability detection
    env.player_positions = [15, 20]  # Agent at 15, opponent at 20 (safe)
    assert env._is_vulnerable(0), "Agent at 15 should be vulnerable"
    assert not env._is_vulnerable(1), "Opponent at 20 (safe zone) should not be vulnerable"

    print(f"✓ Vulnerability detection works")
    print(f"  Position 15 (track): vulnerable ✓")
    print(f"  Position 20 (safe zone): not vulnerable ✓")


def test_random_vs_random_baseline(num_episodes=100):
    """Test random vs random baseline."""
    print("\n" + "="*60)
    print(f"TEST 4: Random vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: ~50% win rate")

    env = Level2InteractionLudo(seed=42)
    agent = RandomAgent(seed=42)

    wins = 0
    total_rewards = []
    episode_lengths = []
    total_captures_by = 0
    total_captures_of = 0

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
        total_captures_by += info['num_captures_by_me']
        total_captures_of += info['num_captures_of_me']

        if info.get('winner') == 0:
            wins += 1

    win_rate = wins / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)

    print(f"\n✓ Baseline test completed")
    print(f"  Win rate: {win_rate:.2%} ({wins}/{num_episodes})")
    print(f"  Avg reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg episode length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Total captures by agent: {total_captures_by}")
    print(f"  Total captures of agent: {total_captures_of}")

    if 0.40 <= win_rate <= 0.60:
        print(f"  ✓ Win rate is reasonable (40-60%)")
    else:
        print(f"  ⚠ Win rate outside expected range (40-60%)")


def test_greedy_vs_random_baseline(num_episodes=100):
    """Test greedy vs random baseline."""
    print("\n" + "="*60)
    print(f"TEST 5: Greedy vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: 60-80% win rate")

    env = Level2InteractionLudo(seed=42)
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

    if win_rate >= 0.55:
        print(f"  ✓ Greedy agent beats random (win rate >= 55%)")
    else:
        print(f"  ⚠ Greedy agent not beating random significantly")


def main():
    print("\n" + "="*80)
    print("LEVEL 2 ENVIRONMENT TEST SUITE (Capturing + Safe Zones)")
    print("="*80)

    try:
        test_environment_basics()
        test_capture_mechanics()
        test_safe_zones()
        test_random_vs_random_baseline(num_episodes=100)
        test_greedy_vs_random_baseline(num_episodes=100)

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nLevel 2 environment is ready for training!")
        print("Run: python experiments/level2_train.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
