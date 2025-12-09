"""
Level 1 Training Script

Train a DQN agent on the simplest Ludo environment (2p1t, no capturing).
Goal: Achieve 65%+ win rate against random opponent in 10k episodes.
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

from rl_agent_ludo.environment.level1_simple import Level1SimpleLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent, GreedyAgent


def evaluate_agent(agent, env, num_episodes=100, greedy=True, seed=42):
    """
    Evaluate agent performance.

    Args:
        agent: Agent to evaluate
        env: Environment
        num_episodes: Number of evaluation episodes
        greedy: Use greedy policy (no exploration)
        seed: Random seed

    Returns:
        dict: Evaluation statistics
    """
    wins = 0
    total_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        steps = 0

        while not done:
            if hasattr(agent, 'act'):
                if isinstance(agent, SimpleDQNAgent):
                    action = agent.act(obs, greedy=greedy)
                else:
                    action = agent.act(obs, info)
            else:
                raise ValueError("Agent must have act() method")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)
        episode_lengths.append(steps)

        # Check if agent won
        if info.get('winner') == 0:
            wins += 1

    return {
        'win_rate': wins / num_episodes,
        'wins': wins,
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }


def train_level1(
    num_episodes=10000,
    eval_frequency=500,
    num_eval_episodes=100,
    learning_rate=1e-4,
    discount_factor=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    batch_size=128,
    replay_buffer_size=50000,
    target_update_frequency=1000,
    hidden_dims=[128, 128],
    seed=42,
    save_checkpoints=True,
    checkpoint_dir='checkpoints/level1',
    device='cpu',
):
    """
    Train Level 1 agent.

    Args:
        num_episodes: Number of training episodes
        eval_frequency: Evaluate every N episodes
        num_eval_episodes: Number of episodes for evaluation
        ... (other hyperparameters)

    Returns:
        dict: Training statistics
    """
    print("="*80)
    print("LEVEL 1 TRAINING: Simplest Ludo (2p1t, no capturing)")
    print("="*80)
    print(f"Target: 65%+ win rate in {num_episodes} episodes")
    print(f"Hyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Discount factor: {discount_factor}")
    print(f"  Epsilon: {epsilon_start} → {epsilon_min} (decay={epsilon_decay})")
    print(f"  Batch size: {batch_size}")
    print(f"  Replay buffer: {replay_buffer_size}")
    print(f"  Target update freq: {target_update_frequency}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Device: {device}")
    print("="*80)

    # Create environment
    env = Level1SimpleLudo(seed=seed)

    # Create DQN agent
    agent = SimpleDQNAgent(
        state_dim=4,
        action_dim=2,
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
        'eval_history': [],
    }

    # Best model tracking
    best_win_rate = 0.0
    best_model_path = None

    # Progress bar
    pbar = tqdm(total=num_episodes, desc="Training")

    # Training loop
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_steps = 0

        while not done:
            # Select action
            action = agent.act(obs)

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)

            # Train
            loss = agent.train_step()
            if loss is not None:
                training_stats['losses'].append(loss)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

        # Episode finished
        agent.episode_count += 1
        agent.decay_epsilon()

        # Store statistics
        training_stats['episode_rewards'].append(episode_reward)
        training_stats['episode_lengths'].append(episode_steps)
        training_stats['episode_wins'].append(1 if info.get('winner') == 0 else 0)

        # Update progress bar
        recent_win_rate = np.mean(training_stats['episode_wins'][-100:]) if len(training_stats['episode_wins']) >= 100 else 0.0
        pbar.update(1)
        pbar.set_postfix({
            'win_rate': f"{recent_win_rate:.2%}",
            'epsilon': f"{agent.epsilon:.3f}",
            'reward': f"{episode_reward:.1f}",
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

            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"Evaluation at episode {episode + 1}:")
            tqdm.write(f"  Win rate: {eval_stats['win_rate']:.2%} ({eval_stats['wins']}/{num_eval_episodes})")
            tqdm.write(f"  Avg reward: {eval_stats['avg_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            tqdm.write(f"  Avg length: {eval_stats['avg_length']:.1f} ± {eval_stats['std_length']:.1f}")
            tqdm.write(f"  Epsilon: {agent.epsilon:.4f}")
            tqdm.write(f"  Replay buffer: {len(agent.replay_buffer)}")

            # Save best model
            if save_checkpoints and eval_stats['win_rate'] > best_win_rate:
                best_win_rate = eval_stats['win_rate']
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = checkpoint_path / f"best_model_ep{episode+1:06d}_wr{best_win_rate:.3f}_{timestamp}.pth"
                agent.save(str(best_model_path))
                tqdm.write(f"  🏆 New best model saved: {best_model_path.name}")

            # Check if target reached
            if eval_stats['win_rate'] >= 0.65:
                tqdm.write(f"\n{'='*60}")
                tqdm.write(f"🎉 TARGET REACHED! Win rate: {eval_stats['win_rate']:.2%} >= 65%")
                tqdm.write(f"{'='*60}")

            tqdm.write(f"{'='*60}\n")

    pbar.close()

    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    final_eval = evaluate_agent(
        agent, env, num_episodes=num_eval_episodes*2, greedy=True, seed=seed + 200000
    )

    print(f"Win rate: {final_eval['win_rate']:.2%} ({final_eval['wins']}/{num_eval_episodes*2})")
    print(f"Avg reward: {final_eval['avg_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"Avg length: {final_eval['avg_length']:.1f} ± {final_eval['std_length']:.1f}")

    # Save final model
    if save_checkpoints:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = checkpoint_path / f"final_model_ep{num_episodes}_wr{final_eval['win_rate']:.3f}_{timestamp}.pth"
        agent.save(str(final_model_path))
        print(f"\nFinal model saved: {final_model_path}")

    if best_model_path:
        print(f"Best model: {best_model_path} (win rate: {best_win_rate:.2%})")

    # Save training statistics
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"level1_training_{timestamp}.json"

    results = {
        'config': {
            'num_episodes': num_episodes,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'epsilon_start': epsilon_start,
            'epsilon_min': epsilon_min,
            'epsilon_decay': epsilon_decay,
            'batch_size': batch_size,
            'replay_buffer_size': replay_buffer_size,
            'target_update_frequency': target_update_frequency,
            'hidden_dims': hidden_dims,
            'seed': seed,
        },
        'final_evaluation': final_eval,
        'best_win_rate': best_win_rate,
        'best_model_path': str(best_model_path) if best_model_path else None,
        'eval_history': training_stats['eval_history'],
    }

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {results_path}")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Level 1 agent")
    parser.add_argument('--episodes', type=int, default=10000, help="Number of training episodes")
    parser.add_argument('--eval_freq', type=int, default=500, help="Evaluation frequency")
    parser.add_argument('--num_eval', type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=1.0, help="Initial epsilon")
    parser.add_argument('--epsilon_min', type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help="Epsilon decay rate")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--buffer_size', type=int, default=50000, help="Replay buffer size")
    parser.add_argument('--target_update', type=int, default=1000, help="Target network update frequency")
    parser.add_argument('--hidden_dims', type=str, default='128,128', help="Hidden layer dimensions")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cpu', help="Device (cpu/cuda)")
    parser.add_argument('--no_save', action='store_true', help="Don't save checkpoints")

    args = parser.parse_args()

    # Parse hidden dims
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]

    # Train
    train_level1(
        num_episodes=args.episodes,
        eval_frequency=args.eval_freq,
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
        save_checkpoints=not args.no_save,
        device=args.device,
    )


if __name__ == '__main__':
    main()
