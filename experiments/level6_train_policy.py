"""
Level 6 - Phase 4: Train Policy with Learned Reward (T-REX)

This script trains a DQN agent using the learned reward function from Phase 2-3
instead of the sparse environment rewards.

Key Innovation:
- Uses learned_reward = reward_net(state) for every step
- Provides dense, meaningful feedback vs sparse win/lose signal
- Expected improvement: 61% → 63-67% win rate

Usage:
    # Quick test (100 episodes)
    python3 experiments/level6_train_policy.py --quick

    # Full training (15000 episodes, ~2-3 hours)
    python3 experiments/level6_train_policy.py

    # Custom episodes
    python3 experiments/level6_train_policy.py --episodes 10000
"""

import sys
from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.trex_agent import TREXAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def evaluate_agent(agent, env, num_episodes=100, greedy=True, seed=42):
    """Evaluate agent performance."""
    wins = 0
    total_rewards = []
    episode_lengths = []
    total_captures_by = 0
    total_captures_of = 0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            action = agent.act(obs, greedy=greedy)
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

    return {
        'win_rate': wins / num_episodes,
        'wins': wins,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'total_captures_by_agent': total_captures_by,
        'total_captures_of_agent': total_captures_of,
        'avg_captures_by_agent': total_captures_by / num_episodes,
        'avg_captures_of_agent': total_captures_of / num_episodes,
    }


def train_level6_trex(
    reward_network_path='checkpoints/level6/reward_network_best.pth',
    num_episodes=15000,
    eval_frequency=500,
    num_eval_episodes=100,
    learning_rate=5e-5,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    batch_size=128,
    replay_buffer_size=300000,
    target_update_frequency=1000,
    hidden_dims=[128, 128],
    seed=42,
    save_checkpoints=True,
    checkpoint_dir='checkpoints/level6',
    device='cpu',
    use_hybrid_rewards=True,
    learned_reward_scale=10.0,
    learned_reward_weight=0.3,
):
    """Train Level 6 T-REX agent with learned reward function."""
    print("="*80)
    print("LEVEL 6 - PHASE 4: TRAIN POLICY WITH LEARNED REWARD (T-REX)")
    print("="*80)
    if use_hybrid_rewards:
        print(f"Mode: HYBRID REWARDS (env + learned)")
        print(f"   Innovation: Combines sparse env rewards with dense learned rewards")
        print(f"   Formula: reward = env_reward + {learned_reward_weight} × {learned_reward_scale} × learned_reward")
        print(f"   Benefits: Clear win/loss signal + dense step-by-step feedback")
    else:
        print(f"Mode: PURE T-REX (learned rewards only)")
        print(f"Innovation: Using learned reward function instead of sparse env rewards")
    print(f"Target: 63-67% win rate (vs Level 5's 61%)")
    print(f"Training episodes: {num_episodes}")
    print(f"\nReward Network: {reward_network_path}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Discount factor: {discount_factor}")
    print(f"  Epsilon: {epsilon_start} → {epsilon_min} (decay={epsilon_decay})")
    print(f"  Batch size: {batch_size}")
    print(f"  Replay buffer: {replay_buffer_size}")
    print(f"  State dim: 16D")
    print(f"  Device: {device}")
    print("="*80)

    # Create environment
    env = Level5MultiAgentLudo(seed=seed)

    # Create T-REX agent (uses learned reward function!)
    agent = TREXAgent(
        state_dim=16,
        action_dim=3,
        reward_network_path=reward_network_path,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        replay_buffer_size=replay_buffer_size,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        hidden_dims=hidden_dims,
        device=device,
        seed=seed,
        use_hybrid_rewards=use_hybrid_rewards,
        learned_reward_scale=learned_reward_scale,
        learned_reward_weight=learned_reward_weight,
    )

    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = Path(__file__).parent.parent / checkpoint_dir
        checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Training statistics
    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_wins': [],
        'losses': [],
        'learned_rewards': [],  # Track learned rewards
        'eval_history': [],
    }

    # Best model tracking
    best_win_rate = 0.0
    best_model_path = None

    # Progress bar
    pbar = tqdm(total=num_episodes, desc="T-REX Training")

    # Training loop
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            action = agent.act(obs)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience (env_reward is ignored, learned reward is used instead!)
            agent.store_experience(obs, action, env_reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                training_stats['losses'].append(loss)

            obs = next_obs
            episode_reward += env_reward  # Track env reward for logging
            episode_steps += 1

        # Episode finished
        agent.episode_count += 1
        agent.decay_epsilon()

        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_steps)
        training_stats['episode_wins'].append(1 if info.get('winner') == 0 else 0)

        # Track average learned reward per episode
        avg_learned_reward = agent.get_avg_learned_reward()
        training_stats['learned_rewards'].append(avg_learned_reward)

        # Update progress bar
        recent_win_rate = np.mean(training_stats['episode_wins'][-100:]) if len(training_stats['episode_wins']) >= 100 else 0.0
        recent_learned_reward = np.mean(training_stats['learned_rewards'][-100:]) if len(training_stats['learned_rewards']) >= 100 else 0.0
        pbar.update(1)
        pbar.set_postfix({
            'win_rate': f"{recent_win_rate:.2%}",
            'epsilon': f"{agent.epsilon:.3f}",
            'learned_r': f"{recent_learned_reward:.2f}",
            'buffer': len(agent.replay_buffer),
        })

        # Periodic evaluation
        if (episode + 1) % eval_frequency == 0:
            eval_stats = evaluate_agent(
                agent, env, num_episodes=num_eval_episodes, greedy=True, seed=seed + 100000 + episode
            )

            training_stats['eval_history'].append({
                'episode': episode + 1,
                **eval_stats
            })

            tqdm.write(f"\n{'='*70}")
            tqdm.write(f"Evaluation at episode {episode + 1}:")
            tqdm.write(f"  Win rate: {eval_stats['win_rate']:.2%} ({eval_stats['wins']}/{num_eval_episodes})")
            tqdm.write(f"  Avg reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            tqdm.write(f"  Avg length: {eval_stats['avg_length']:.1f} ± {eval_stats['std_length']:.1f}")
            tqdm.write(f"  Captures by agent: {eval_stats['avg_captures_by_agent']:.2f}/episode")
            tqdm.write(f"  Captures of agent: {eval_stats['avg_captures_of_agent']:.2f}/episode")
            tqdm.write(f"  Avg learned reward: {recent_learned_reward:.2f}")
            tqdm.write(f"  Epsilon: {agent.epsilon:.4f}")
            tqdm.write(f"  Replay buffer: {len(agent.replay_buffer)}")

            # Save best model
            if save_checkpoints and eval_stats['win_rate'] > best_win_rate:
                best_win_rate = eval_stats['win_rate']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = checkpoint_path / f"trex_best_ep{episode+1:06d}_wr{best_win_rate:.3f}_{timestamp}.pth"
                agent.save(str(best_model_path))
                tqdm.write(f"  New best T-REX model saved: {best_model_path.name}")

            # Check if target reached
            if eval_stats['win_rate'] >= 0.63:
                tqdm.write(f"\n{'='*70}")
                tqdm.write(f"TARGET REACHED! Win rate: {eval_stats['win_rate']:.2%} >= 63%")
                tqdm.write(f"   T-REX successfully improved over Level 5 baseline!")
                tqdm.write(f"{'='*70}")

            # Compare to Level 5 baseline
            level5_baseline = 0.61
            improvement = (eval_stats['win_rate'] - level5_baseline) / level5_baseline * 100
            if eval_stats['win_rate'] > level5_baseline:
                tqdm.write(f"  Improvement over Level 5: +{improvement:.1f}%")
            else:
                tqdm.write(f"  Still below Level 5: {improvement:.1f}%")

            tqdm.write(f"{'='*70}\n")

    pbar.close()

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    final_eval = evaluate_agent(agent, env, num_episodes=500, greedy=True, seed=seed + 999999)

    print(f"Win rate: {final_eval['win_rate']:.2%} ({final_eval['wins']}/500)")
    print(f"Avg reward: {final_eval['avg_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Avg length: {final_eval['avg_length']:.1f} ± {final_eval['std_length']:.1f}")
    print(f"Avg captures by agent: {final_eval['avg_captures_by_agent']:.2f}")
    print(f"Avg captures of agent: {final_eval['avg_captures_of_agent']:.2f}")

    # Save final model
    if save_checkpoints:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = checkpoint_path / f"trex_final_ep{num_episodes}_wr{final_eval['win_rate']:.3f}_{timestamp}.pth"
        agent.save(str(final_model_path))
        print(f"\nFinal model saved: {final_model_path.name}")

    # Save training statistics
    if save_checkpoints:
        stats_path = checkpoint_path / f"training_stats_{timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        print(f"Training stats saved: {stats_path.name}")

    print("\n" + "="*80)
    print("T-REX TRAINING COMPLETE!")
    print("="*80)
    print(f"Best win rate: {best_win_rate:.2%}")
    print(f"Final win rate: {final_eval['win_rate']:.2%}")

    level5_baseline = 0.61
    if final_eval['win_rate'] > level5_baseline:
        improvement = (final_eval['win_rate'] - level5_baseline) / level5_baseline * 100
        print(f"Improvement over Level 5: +{improvement:.1f}%")
        print(f"\nT-REX successfully learned from preferences!")
    else:
        print(f"Did not exceed Level 5 baseline ({level5_baseline:.1%})")
        print(f"   Consider: More trajectories, longer training, or hyperparameter tuning")

    if best_model_path:
        print(f"\nBest model: {best_model_path.name}")

    print("="*80)

    return agent, training_stats


def main():
    parser = argparse.ArgumentParser(description="Train Level 6 T-REX policy")

    # Mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (100 episodes)')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (default: 15000, quick: 100)')
    parser.add_argument('--eval_freq', type=int, default=None,
                       help='Evaluation frequency (default: 500, quick: 50)')
    parser.add_argument('--num_eval', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')

    # Reward network
    parser.add_argument('--reward_network', type=str,
                       default='checkpoints/level6/reward_network_best.pth',
                       help='Path to trained reward network')

    # DQN hyperparameters (match Level 5 for comparison)
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial epsilon for exploration (default: 1.0)')
    parser.add_argument('--epsilon_min', type=float, default=0.05,
                       help='Minimum epsilon (default: 0.05)')
    parser.add_argument('--epsilon_decay', type=float, default=0.999,
                       help='Epsilon decay rate (default: 0.999)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training (default: 128)')
    parser.add_argument('--buffer_size', type=int, default=300000,
                       help='Replay buffer size (default: 300000)')
    parser.add_argument('--target_update', type=int, default=1000,
                       help='Target network update frequency (default: 1000)')
    parser.add_argument('--hidden_dims', type=str, default='128,128',
                       help='Hidden layer dimensions (default: 128,128)')

    # Hybrid reward settings
    parser.add_argument('--no_hybrid', action='store_true',
                       help='Disable hybrid rewards (use pure T-REX)')
    parser.add_argument('--reward_scale', type=float, default=10.0,
                       help='Scale factor for learned rewards (default: 10.0)')
    parser.add_argument('--reward_weight', type=float, default=0.3,
                       help='Weight for learned rewards in hybrid (default: 0.3)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*80)
    print("LEVEL 6 - PHASE 4: TRAIN POLICY WITH LEARNED REWARD")
    print("="*80)

    # Set parameters
    if args.quick:
        print("\nQUICK MODE")
        num_episodes = args.episodes if args.episodes else 100
        eval_freq = args.eval_freq if args.eval_freq else 50
    else:
        print("\nFULL MODE")
        num_episodes = args.episodes if args.episodes else 15000
        eval_freq = args.eval_freq if args.eval_freq else 500

    # Parse hidden_dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    print(f"\nConfiguration:")
    print(f"  Training episodes: {num_episodes}")
    print(f"  Evaluation frequency: {eval_freq}")
    print(f"  Num eval episodes: {args.num_eval}")
    print(f"  Reward network: {args.reward_network}")
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Discount factor (gamma): {args.gamma}")
    print(f"  Epsilon: {args.epsilon} → {args.epsilon_min} (decay={args.epsilon_decay})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Replay buffer size: {args.buffer_size}")
    print(f"  Target update freq: {args.target_update}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {args.device}")
    print(f"\nReward Settings:")
    print(f"  Hybrid rewards: {not args.no_hybrid}")
    if not args.no_hybrid:
        print(f"  Learned reward scale: {args.reward_scale}")
        print(f"  Learned reward weight: {args.reward_weight}")
    print("="*80 + "\n")

    # Check if reward network exists
    reward_network_path = Path(args.reward_network)
    if not reward_network_path.exists():
        print(f"Error: Reward network not found at {args.reward_network}")
        print(f"   Please run Phase 2-3 first:")
        print(f"   python3 experiments/level6_learn_reward.py")
        return

    # Train
    agent, stats = train_level6_trex(
        reward_network_path=args.reward_network,
        num_episodes=num_episodes,
        eval_frequency=eval_freq,
        num_eval_episodes=args.num_eval,
        learning_rate=args.lr,
        discount_factor=args.gamma,
        epsilon_start=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        replay_buffer_size=args.buffer_size,
        target_update_frequency=args.target_update,
        hidden_dims=hidden_dims,
        seed=args.seed,
        device=args.device,
        use_hybrid_rewards=not args.no_hybrid,
        learned_reward_scale=args.reward_scale,
        learned_reward_weight=args.reward_weight,
    )

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Phase 4 complete - T-REX policy trained")
    print("Phase 5: Evaluate and compare")
    print("   Command: python3 experiments/level6_evaluate.py")
    print("="*80)


if __name__ == "__main__":
    main()
