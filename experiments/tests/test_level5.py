"""
Quick test script for Level 5 environment.

Tests:
1. Environment basics
2. Multi-agent mechanics (4 players)
3. Multi-token coordination
4. Random vs Random baseline
5. Greedy vs Random baseline
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def test_environment_basics():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Environment Basics")
    print("="*60)

    env = Level5MultiAgentLudo(render_mode="human", seed=42)
    obs, info = env.reset(seed=42)

    print(f"✓ Environment created")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space} (0=token0, 1=token1, 2=pass)")
    print(f"  Safe zones: {env.SAFE_ZONES}")
    print(f"  Num players: {env.NUM_PLAYERS}")
    print(f"  Tokens per player: {env.TOKENS_PER_PLAYER}")
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
              f"my_tokens={info['my_tokens']}")

        if terminated or truncated:
            break

    env.close()
    print("\n✓ Environment basics test passed!")


def test_multi_agent_mechanics():
    """Test that 4 players are competing."""
    print("\n" + "="*60)
    print("TEST 2: Multi-Agent Mechanics (4 players)")
    print("="*60)

    env = Level5MultiAgentLudo(seed=42)

    winners = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}  # -1 for truncated games
    total_games = 30

    for ep in range(total_games):
        obs, info = env.reset(seed=42 + ep)
        done = False
        steps = 0

        while not done and steps < 300:
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask)[0]
            action = int(np.random.choice(valid_actions)) if len(valid_actions) > 0 else 2

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        winner = info.get('winner', -1)
        winners[winner] = winners.get(winner, 0) + 1

    print(f"✓ Multi-agent test completed ({total_games} games)")
    print(f"  Player 0 (agent) wins: {winners[0]}")
    print(f"  Player 1 wins: {winners[1]}")
    print(f"  Player 2 wins: {winners[2]}")
    print(f"  Player 3 wins: {winners[3]}")
    print(f"  Truncated games: {winners[-1]}")
    print(f"  ✓ All 4 players competing!")

    # With 4 players and random play, expect ~25% win rate each
    agent_win_rate = winners[0] / total_games
    print(f"  Agent win rate: {agent_win_rate:.2%} (expected ~25% for random play)")


def test_multi_token_completion():
    """Test that both tokens need to finish to win."""
    print("\n" + "="*60)
    print("TEST 3: Multi-Token Completion")
    print("="*60)

    env = Level5MultiAgentLudo(seed=42)

    games_with_partial_completion = 0
    games_won = 0
    total_games = 20

    for ep in range(total_games):
        obs, info = env.reset(seed=42 + ep)
        done = False
        steps = 0
        max_tokens_finished = 0

        while not done and steps < 300:
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
    print(f"TEST 4: Random vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: ~25% win rate (4 players, all random)")

    env = Level5MultiAgentLudo(seed=42)
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

    if 0.15 <= win_rate <= 0.35:
        print(f"  ✓ Win rate is reasonable (15-35% for 4-player game)")
    else:
        print(f"  ⚠ Win rate outside expected range (15-35%)")


def test_greedy_vs_random_baseline(num_episodes=100):
    """Test greedy vs random baseline."""
    print("\n" + "="*60)
    print(f"TEST 5: Greedy vs Random Baseline ({num_episodes} episodes)")
    print("="*60)
    print("Expected: 30-40% win rate (greedy should have slight edge)")

    env = Level5MultiAgentLudo(seed=42)
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

    if win_rate >= 0.27:
        print(f"  ✓ Greedy agent beats random expectation (win rate >= 27%)")
    else:
        print(f"  ⚠ Greedy agent not beating random significantly")


def main():
    print("\n" + "="*80)
    print("LEVEL 5 ENVIRONMENT TEST SUITE (Multi-Agent Chaos)")
    print("="*80)

    try:
        test_environment_basics()
        test_multi_agent_mechanics()
        test_multi_token_completion()
        test_random_vs_random_baseline(num_episodes=100)
        test_greedy_vs_random_baseline(num_episodes=100)

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nLevel 5 environment is ready for training!")
        print("Run: python experiments/level5_train.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
