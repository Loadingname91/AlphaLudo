#!/usr/bin/env python3
"""
Reward Verification Script for UnifiedLudoEnv.

This script runs the environment and verifies reward values are correct
for different scenarios. It provides detailed output about reward distribution.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.unifiedLudoEnv import (
    UnifiedLudoEnv2Tokens,
    UnifiedLudoEnv4Tokens,
)

# Board constants
HOME_INDEX = 0
GOAL_INDEX = 57
GLOBE_INDEXES = [1, 9, 22, 35, 48]
STAR_INDEXES = [5, 12, 18, 25, 31, 38, 44, 51]


def analyze_rewards(env, num_episodes=20, max_steps_per_episode=500):
    """Analyze reward distribution and verify correctness."""
    print(f"\n{'='*60}")
    print(f"Analyzing rewards for {env.__class__.__name__}")
    print(f"{'='*60}\n")
    
    all_rewards = []
    event_counts = {
        'exit_home': 0,
        'safe_zone': 0,
        'kill': 0,
        'goal': 0,
        'win': 0,
        'death': 0,
        'normal_move': 0,
    }
    
    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_rewards = []
        episode_events = []
        
        for step in range(max_steps_per_episode):
            if info["state"].valid_moves:
                action = info["state"].valid_moves[0]
                prev_state = info["state"]
                prev_pos = prev_state.player_pieces[action]
                
                obs, reward, terminated, truncated, info = env.step(action)
                next_state = info["state"]
                curr_pos = next_state.player_pieces[action]
                
                episode_rewards.append(reward)
                all_rewards.append(reward)
                
                # Classify event
                if prev_pos == HOME_INDEX and curr_pos != HOME_INDEX:
                    event_counts['exit_home'] += 1
                    episode_events.append(('exit_home', reward))
                    if reward < 9.0 or reward > 12.0:
                        print(f"  ⚠️  Episode {episode}, Step {step}: Exit home reward {reward:.2f} (expected ~10.0)")
                
                elif (prev_pos not in GLOBE_INDEXES and curr_pos in GLOBE_INDEXES) or \
                     (prev_pos not in STAR_INDEXES and curr_pos in STAR_INDEXES):
                    event_counts['safe_zone'] += 1
                    episode_events.append(('safe_zone', reward))
                    if reward < 4.0 or reward > 8.0:
                        print(f"  ⚠️  Episode {episode}, Step {step}: Safe zone reward {reward:.2f} (expected ~5.0)")
                
                elif prev_pos not in [HOME_INDEX, GOAL_INDEX] and curr_pos == HOME_INDEX:
                    event_counts['death'] += 1
                    episode_events.append(('death', reward))
                    if reward > -15.0:
                        print(f"  ⚠️  Episode {episode}, Step {step}: Death penalty {reward:.2f} (expected <= -20.0)")
                
                elif curr_pos == GOAL_INDEX and prev_pos != GOAL_INDEX:
                    event_counts['goal'] += 1
                    episode_events.append(('goal', reward))
                    if reward < 29.0 or reward > 35.0:
                        print(f"  ⚠️  Episode {episode}, Step {step}: Goal reward {reward:.2f} (expected ~30.0)")
                
                elif reward >= 14.0 and reward <= 18.0:
                    # Likely a kill (check if we're on an enemy position)
                    for enemy in prev_state.enemy_pieces:
                        if curr_pos in enemy and curr_pos not in [HOME_INDEX, GOAL_INDEX, 1]:
                            if curr_pos not in GLOBE_INDEXES:
                                event_counts['kill'] += 1
                                episode_events.append(('kill', reward))
                                break
                
                elif reward == 100.0:
                    event_counts['win'] += 1
                    episode_events.append(('win', reward))
                
                elif reward == -50.0:
                    episode_events.append(('loss', reward))
                
                else:
                    event_counts['normal_move'] += 1
                
                if terminated or truncated:
                    if reward == 100.0:
                        print(f"  ✅ Episode {episode}: Won! (reward: {reward:.2f})")
                    elif reward == -50.0:
                        print(f"  ❌ Episode {episode}: Lost (reward: {reward:.2f})")
                    break
            else:
                # No valid moves, step anyway
                obs, reward, terminated, truncated, info = env.step(0)
                if terminated or truncated:
                    break
        
        # Print episode summary if interesting events occurred
        if episode_events:
            interesting = [e for e in episode_events if e[0] in ['exit_home', 'safe_zone', 'kill', 'goal', 'win', 'death']]
            if interesting:
                print(f"  Episode {episode} events: {interesting[:3]}...")  # Show first 3
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Reward Statistics:")
    print(f"{'='*60}")
    if all_rewards:
        print(f"  Total rewards collected: {len(all_rewards)}")
        print(f"  Mean reward: {np.mean(all_rewards):.4f}")
        print(f"  Std reward: {np.std(all_rewards):.4f}")
        print(f"  Min reward: {np.min(all_rewards):.2f}")
        print(f"  Max reward: {np.max(all_rewards):.2f}")
        print(f"  Median reward: {np.median(all_rewards):.4f}")
        
        # Check for issues
        nan_count = sum(1 for r in all_rewards if np.isnan(r))
        inf_count = sum(1 for r in all_rewards if np.isinf(r))
        extreme_count = sum(1 for r in all_rewards if abs(r) > 200.0)
        
        print(f"\n  Issues found:")
        print(f"    NaN rewards: {nan_count}")
        print(f"    Inf rewards: {inf_count}")
        print(f"    Extreme rewards (>200): {extreme_count}")
        
        if nan_count > 0:
            print(f"  ❌ ERROR: Found {nan_count} NaN rewards!")
        if inf_count > 0:
            print(f"  ❌ ERROR: Found {inf_count} Inf rewards!")
        if extreme_count > 0:
            print(f"  ⚠️  WARNING: Found {extreme_count} extreme rewards!")
        
        print(f"\n  Event counts:")
        for event, count in event_counts.items():
            print(f"    {event}: {count}")
        
        # Verify reward ranges
        print(f"\n  Reward Range Verification:")
        normal_rewards = [r for r in all_rewards if -100.0 <= r <= 100.0]
        print(f"    Rewards in [-100, 100]: {len(normal_rewards)}/{len(all_rewards)} ({100*len(normal_rewards)/len(all_rewards):.1f}%)")
        
        if len(normal_rewards) == len(all_rewards):
            print(f"  ✅ All rewards are in reasonable range!")
        else:
            print(f"  ⚠️  Some rewards are outside reasonable range")
    else:
        print("  ⚠️  No rewards collected!")
    
    print(f"{'='*60}\n")


def main():
    """Run reward verification for both configurations."""
    print("\n" + "="*60)
    print("Unified Ludo Environment - Reward Verification")
    print("="*60)
    
    # Test 2 tokens
    env_2tokens = UnifiedLudoEnv2Tokens(player_id=0, num_players=2, seed=42)
    analyze_rewards(env_2tokens, num_episodes=20, max_steps_per_episode=500)
    
    # Test 4 tokens
    env_4tokens = UnifiedLudoEnv4Tokens(player_id=0, num_players=4, seed=42)
    analyze_rewards(env_4tokens, num_episodes=20, max_steps_per_episode=500)
    
    print("\n✅ Reward verification complete!\n")


if __name__ == "__main__":
    main()

