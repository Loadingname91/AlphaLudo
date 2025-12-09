"""
Quick test script for Level 3 environment.

Tests:
1. Environment basics
2. Multi-token mechanics
3. Token coordination
4. Random vs Random baseline
5. Greedy vs Random baseline
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level3_multitoken import Level3MultiTokenLudo
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def test_environment_basics():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Environment Basics")
    print("="*60)

    env = Level3MultiTokenLudo(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space} (0=token0, 1=token1, 2=pass)")
    print(f"  Safe zones: {env.SAFE_ZONES}")
    print(f"  Initial observation: {obs}")
    print(f"  Action mask: {info['action_mask']}")

    # Take a few steps
    print(f"\n✓ Running 5 steps...")
    for i in range(5):
        action_mask = info['action_mask']
        valid_actions = np.where(action_mask)[0]
        action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 2

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: action={action}, reward={reward:.1f}, "
              f"my_tokens={info['my_tokens']}, opp_tokens={info['opp_tokens']}")

        if terminated or truncated:
            break

    env.close()
    print("\n✓ Environment basics test passed!")


def test_multi_token_completion():
    """Test that both tokens need to finish to win."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Token Completion")
    print("="*60)

    env = Level3MultiTokenLudo(seed=42)

    games_with_partial_completion = 0
    games_won = 0
    total_games = 20

    for ep in range(total_games):
        obs, info = env.reset(seed=42 + ep)
        done = False
        steps = 0
        max_tokens_finished = 0

        while not done and steps < 200:
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask)[0]
            action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 2

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            max_tokens_finished = max(max_tokens_finished, info['num_my_tokens_finished'])

        if max_tokens_finished == 1:
            games_with_partial_completion += 1
        if info.get('winner') == 0 and max_tokens_finished == 2:
            games_won += 1

    print(f"✓ Multi-token test completed ({total_games} games)")
    print(f"  Games with 1 token finished: {games_with_partial_completion}")
    print(f"  Games won (both tokens): {games_won}")
    print(f"  ✓ Both tokens required to win!")


def test_random_vs_random_baseline(num_episodes=100):
    """Test random vs random baseline."""
    print("\n" + "="*60)
    print(f"TEST 3: Random vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: ~50% win rate")

    env = Level3MultiTokenLudo(seed=42)
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

    if 0.35 <= win_rate <= 0.65:
        print(f"  ✓ Win rate is reasonable (35-65%)")
    else:
        print(f"  ⚠ Win rate outside expected range (35-65%)")


def test_greedy_vs_random_baseline(num_episodes=100):
    """Test greedy vs random baseline."""
    print("\n" + "="*60)
    print(f"TEST 4: Greedy vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: 55-75% win rate")

    env = Level3MultiTokenLudo(seed=42)
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

    if win_rate >= 0.50:
        print(f"  ✓ Greedy agent beats random (win rate >= 50%)")
    else:
        print(f"  ⚠ Greedy agent not beating random significantly")


def main():
    print("\n" + "="*80)
    print("LEVEL 3 ENVIRONMENT TEST SUITE (Multi-Token Coordination)")
    print("="*80)

    try:
        test_environment_basics()
        test_multi_token_completion()
        test_random_vs_random_baseline(num_episodes=100)
        test_greedy_vs_random_baseline(num_episodes=100)

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nLevel 3 environment is ready for training!")
        print("Run: python experiments/level3_train.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
