#!/usr/bin/env python3
"""
Self-play training experiment for DQN (Deep Q-Network).

Implements multi-phase training approach similar to Tabular Q-Learning:
1. Phase 1: Train against random opponents
2. Phase 2: Train against best agent from Phase 1 (self-play)

CPU-Optimized for 8-core ARM CPU with MKLDNN support.
"""

import gymnasium as gym
import numpy as np
import sys
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Register environment and import
import rl_agent_ludo.environment  # noqa: F401 - registers Ludo-v0
from rl_agent_ludo.agents.dqnAgent import DQNAgent


def train_against_opponent(
    agent: DQNAgent,
    opponent: Optional[DQNAgent],
    num_episodes: int,
    num_players: int = 2,
    tokens_per_player: int = 4,
    seed: int = 42,
    phase_name: str = "Training",
    epsilon_start: Optional[float] = None,
    epsilon_end: Optional[float] = None,
    observation_mode: str = 'hybrid',
) -> dict:
    """
    Train agent against an opponent (or random if opponent is None).
    
    Args:
        agent: The learning agent
        opponent: Fixed opponent agent (or None for random)
        num_episodes: Number of training episodes
        num_players: Number of players
        tokens_per_player: Tokens per player
        seed: Random seed
        phase_name: Name of training phase for logging
    
    Returns:
        Training statistics
    """
    env = gym.make(
        "Ludo-v0",
        player_id=0,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        observation_mode=observation_mode,
    )
    
    wins = 0
    losses = 0
    episode_lengths = []
    rewards = []
    
    # Debug stats for periodic logging
    debug_stats = {
        'win_by_episode': [],
        'reward_by_episode': [],
        'exploration_rate': [],
        'replay_buffer_size': [],
    }
    
    # Log every N episodes for debugging (more frequent for better monitoring)
    log_interval = max(1, num_episodes // 50)  # Log ~50 times during training (more frequent than tabular)
    
    print(f"\n{phase_name}: {num_episodes} episodes")
    print(f"  Agent: DQNAgent (learning)")
    print(f"  Opponent: {'DQNAgent (frozen)' if opponent else 'Random'}")
    
    for episode in tqdm(range(num_episodes), desc=phase_name):
        # Optional: episode-wise epsilon scheduling (Decay over first 75%)
        if epsilon_start is not None and epsilon_end is not None:
            # Decay duration is 75% of total episodes
            decay_duration = int(num_episodes * 0.75)
            
            # Calculate progress (capped at 1.0 so it stays at min_epsilon for the last 25%)
            progress = min(1.0, episode / max(decay_duration, 1))
            
            current_epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress
            current_epsilon = max(min(current_epsilon, epsilon_start), epsilon_end)
            agent.epsilon = current_epsilon

        obs, info = env.reset(seed=seed + episode)
        state = info["state"]
        done = False
        episode_length = 0
        episode_reward = 0.0
        max_steps = 5000
        
        while not done and episode_length < max_steps:
            prev_state = state
            
            # Agent acts when it's player 0's turn
            if state.current_player == 0:
                action = agent.act(state)
            elif opponent and state.current_player == 1:
                # Opponent acts (no learning)
                action = opponent.act(state)
            else:
                # Other players or no opponent: random action
                action = np.random.choice(state.valid_moves) if state.valid_moves else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = info["state"]
            done = terminated or truncated
            
            # Only update the learning agent (player 0)
            if prev_state.current_player == 0:
                agent.push_to_replay_buffer(
                    prev_state,
                    action,
                    reward,
                    next_state,
                    done,
                )
                episode_reward += reward
            
            state = next_state
            episode_length += 1
        
        # Check actual game outcome (not reward sign)
        episode_won = False
        if done:
            # Unwrap environment if it's wrapped (e.g., TimeLimit)
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            # Access game object to check winners
            if hasattr(unwrapped_env, 'game') and unwrapped_env.game:
                winners = unwrapped_env.game.get_winners_of_game()
                episode_won = unwrapped_env.player_id in winners if hasattr(unwrapped_env, 'player_id') else False
        
        episode_lengths.append(episode_length)
        rewards.append(episode_reward)
        debug_stats['win_by_episode'].append(1 if episode_won else 0)
        debug_stats['reward_by_episode'].append(episode_reward)
        
        # Debug logging every N episodes
        if (episode + 1) % log_interval == 0 or episode == 0:
            replay_buffer_size = len(agent.replay_buffer)
            debug_stats['exploration_rate'].append(agent.epsilon)
            debug_stats['replay_buffer_size'].append(replay_buffer_size)
            
            # Calculate recent win rate (last 10% of episodes so far)
            recent_window = max(10, (episode + 1) // 10)
            recent_wins = sum(debug_stats['win_by_episode'][-recent_window:])
            recent_win_rate = recent_wins / recent_window if recent_window > 0 else 0.0
            
            # Calculate recent average reward
            recent_avg_reward = np.mean(debug_stats['reward_by_episode'][-recent_window:]) if recent_window > 0 else 0.0
            
            tqdm.write(
                f"  Episode {episode+1}/{num_episodes}: "
                f"Replay buffer={replay_buffer_size}, "
                f"ε={agent.epsilon:.3f}, "
                f"Recent win rate={recent_win_rate:.2%}, "
                f"Recent avg reward={recent_avg_reward:.1f}"
            )
        
        if episode_won:
            wins += 1
        elif done:  # Game ended but agent didn't win
            losses += 1
    
    env.close()
    
    # Final debug summary
    print(f"\n  Debug Summary:")
    print(f"    Final replay buffer size: {len(agent.replay_buffer)}")
    print(f"    Final epsilon: {agent.epsilon:.4f}")
    if debug_stats['replay_buffer_size']:
        print(f"    Replay buffer trend: {debug_stats['replay_buffer_size'][0]} → {debug_stats['replay_buffer_size'][-1]}")
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": num_episodes - wins - losses,
        "win_rate": wins / num_episodes,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "debug_stats": debug_stats,
    }


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 1000,
    num_players: int = 2,
    tokens_per_player: int = 4,
    seed: int = 9999,
    opponent: Optional[DQNAgent] = None,
    observation_mode: str = 'hybrid',
) -> dict:
    """
    Evaluate agent performance (no learning).
    
    Args:
        agent: Agent to evaluate
        num_episodes: Number of evaluation episodes
        num_players: Number of players
        tokens_per_player: Tokens per player
        seed: Random seed for evaluation
        opponent: Optional opponent agent
    
    Returns:
        Evaluation statistics
    """
    # Temporarily disable learning
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Greedy policy for evaluation
    
    env = gym.make(
        "Ludo-v0",
        player_id=0,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        observation_mode=observation_mode,
    )
    
    wins = 0
    losses = 0
    episode_lengths = []
    rewards = []
    
    print(f"\nEvaluation: {num_episodes} episodes (greedy policy)")
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=seed + episode)
        state = info["state"]
        done = False
        episode_length = 0
        episode_reward = 0.0
        max_steps = 5000
        
        while not done and episode_length < max_steps:
            if state.current_player == 0:
                action = agent.act(state)
            elif opponent and state.current_player == 1:
                action = opponent.act(state)
            else:
                action = np.random.choice(state.valid_moves) if state.valid_moves else 0
            
            obs, reward, terminated, truncated, info = env.step(action)
            state = info["state"]
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        # Check actual game outcome (not reward sign)
        episode_won = False
        if done:
            # Unwrap environment if it's wrapped (e.g., TimeLimit)
            unwrapped_env = env
            while hasattr(unwrapped_env, 'env'):
                unwrapped_env = unwrapped_env.env
            # Access game object to check winners
            if hasattr(unwrapped_env, 'game') and unwrapped_env.game:
                winners = unwrapped_env.game.get_winners_of_game()
                episode_won = unwrapped_env.player_id in winners if hasattr(unwrapped_env, 'player_id') else False
        
        episode_lengths.append(episode_length)
        rewards.append(episode_reward)
        
        if episode_won:
            wins += 1
        elif done:  # Game ended but agent didn't win
            losses += 1
    
    env.close()
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": num_episodes - wins - losses,
        "win_rate": wins / num_episodes,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
    }


def run_selfplay_experiment(
    state_abstraction: str = 'combined',
    num_players: int = 2,
    tokens_per_player: int = 4,
    phase1_episodes: int = 10000,
    phase2_episodes: int = 10000,
    eval_episodes: int = 1000,
    learning_rate: float = 0.001,
    discount_factor: float = 0.95,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    min_epsilon: float = 0.01,
    batch_size: int = 128,  # Reduced for faster training
    replay_buffer_size: int = 170000,
    target_update_frequency: int = 1000,
    train_frequency: int = 32,  # Train every 32 steps (reduced frequency for speed)
    gradient_steps: int = 1,  # Single gradient step per training (optimized for speed)
    hidden_dims: list = [128, 128, 64],
    seed: int = 42,
    device: str = 'cpu',
    observation_mode: str = 'hybrid',
) -> dict:
    """
    Run multi-phase self-play training experiment for DQN.
    
    Args:
        state_abstraction: 'potential', 'zone_based', or 'combined'
        num_players: Number of players (2 or 4)
        tokens_per_player: Tokens per player
        phase1_episodes: Episodes for Phase 1 (vs random)
        phase2_episodes: Episodes for Phase 2 (self-play)
        eval_episodes: Episodes for evaluation
        learning_rate: Learning rate
        discount_factor: Discount factor (gamma)
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        min_epsilon: Minimum epsilon
        batch_size: Batch size for training
        replay_buffer_size: Size of replay buffer
        target_update_frequency: Steps between target network updates
        train_frequency: Steps between training calls
        gradient_steps: Number of gradient steps per training call
        hidden_dims: Hidden layer dimensions
        seed: Random seed
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Dictionary with all phase results
    """
    print("=" * 70)
    print("MULTI-PHASE SELF-PLAY TRAINING (DQN)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  State Abstraction: {state_abstraction}")
    print(f"  Players: {num_players}")
    print(f"  Tokens per player: {tokens_per_player}")
    print(f"  Learning rate (α): {learning_rate}")
    print(f"  Discount factor (γ): {discount_factor}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Min Epsilon: {min_epsilon}")
    print(f"  Epsilon decay schedule: linear {epsilon:.3f} → {min_epsilon:.3f} per phase")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print("=" * 70)
    
    results = {}
    
    # Phase 1: Train against random
    print("\n" + "=" * 70)
    print("PHASE 1: Training Against Random Opponents")
    print("=" * 70)
    
    agent_phase1 = DQNAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=1.0,  # Use manual epsilon schedule
        min_epsilon=min_epsilon,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=hidden_dims,
        state_abstraction=state_abstraction,
        device=device,
        seed=seed,
    )
    
    results["phase1_training"] = train_against_opponent(
        agent=agent_phase1,
        opponent=None,
        num_episodes=phase1_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed,
        phase_name="Phase 1 Training",
        epsilon_start=epsilon,
        epsilon_end=min_epsilon,
    )
    
    # Evaluate Phase 1 agent
    results["phase1_eval"] = evaluate_agent(
        agent=agent_phase1,
        num_episodes=eval_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed + 100000,
        observation_mode=observation_mode,
    )
    
    print(f"\nPhase 1 Results:")
    print(f"  Training Win Rate: {results['phase1_training']['win_rate']:.2%}")
    print(f"  Evaluation Win Rate: {results['phase1_eval']['win_rate']:.2%}")
    print(f"  Final Epsilon: {agent_phase1.epsilon:.4f}")
    
    # Save Phase 1 agent
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    phase1_path = checkpoint_dir / f"dqn_phase1_{state_abstraction}_{seed}.pth"
    agent_phase1.save(str(phase1_path))
    print(f"  Saved to: {phase1_path}")
    
    # Phase 2: Self-play training
    print("\n" + "=" * 70)
    print("PHASE 2: Self-Play Training")
    print("=" * 70)
    
    # Create new agent for learning, initialized with Phase 1 weights
    agent_phase2 = DQNAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,  # Reset epsilon for Phase 2
        epsilon_decay=1.0,  # Use manual epsilon schedule
        min_epsilon=min_epsilon,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=hidden_dims,
        state_abstraction=state_abstraction,
        device=device,
        seed=seed + 1,
    )
    agent_phase2.load(str(phase1_path))
    agent_phase2.epsilon = epsilon  # Reset epsilon after loading
    
    # Create frozen copy of Phase 1 agent as opponent
    opponent = DQNAgent(
        learning_rate=0.0,  # No learning (not used but required)
        discount_factor=discount_factor,
        epsilon=0.01,  # Nearly greedy
        epsilon_decay=1.0,
        min_epsilon=0.01,
        batch_size=batch_size,
        replay_buffer_size=replay_buffer_size,
        target_update_frequency=target_update_frequency,
        train_frequency=train_frequency,
        gradient_steps=gradient_steps,
        hidden_dims=hidden_dims,
        state_abstraction=state_abstraction,
        device=device,
        seed=seed + 2,
    )
    opponent.load(str(phase1_path))
    opponent.epsilon = 0.01
    
    results["phase2_training"] = train_against_opponent(
        agent=agent_phase2,
        opponent=opponent,
        num_episodes=phase2_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed + 10000,
        phase_name="Phase 2 Self-Play",
        epsilon_start=epsilon,
        epsilon_end=min_epsilon,
        observation_mode=observation_mode,
    )
    
    # Evaluate Phase 2 agent
    results["phase2_eval_vs_random"] = evaluate_agent(
        agent=agent_phase2,
        num_episodes=eval_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed + 200000,
        opponent=None,
        observation_mode=observation_mode,
    )
    
    results["phase2_eval_vs_phase1"] = evaluate_agent(
        agent=agent_phase2,
        num_episodes=eval_episodes,
        num_players=num_players,
        tokens_per_player=tokens_per_player,
        seed=seed + 300000,
        opponent=opponent,
        observation_mode=observation_mode,
    )
    
    print(f"\nPhase 2 Results:")
    print(f"  Training Win Rate (vs Phase 1): {results['phase2_training']['win_rate']:.2%}")
    print(f"  Eval Win Rate (vs Random): {results['phase2_eval_vs_random']['win_rate']:.2%}")
    print(f"  Eval Win Rate (vs Phase 1): {results['phase2_eval_vs_phase1']['win_rate']:.2%}")
    print(f"  Final Epsilon: {agent_phase2.epsilon:.4f}")
    
    # Save Phase 2 agent
    phase2_path = checkpoint_dir / f"dqn_phase2_{state_abstraction}_{seed}.pth"
    agent_phase2.save(str(phase2_path))
    print(f"  Saved to: {phase2_path}")
    
    # Summary
    results["config"] = {
        "state_abstraction": state_abstraction,
        "num_players": num_players,
        "tokens_per_player": tokens_per_player,
        "phase1_episodes": phase1_episodes,
        "phase2_episodes": phase2_episodes,
        "learning_rate": learning_rate,
        "discount_factor": discount_factor,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "min_epsilon": min_epsilon,
        "batch_size": batch_size,
        "replay_buffer_size": replay_buffer_size,
        "target_update_frequency": target_update_frequency,
        "train_frequency": train_frequency,
        "gradient_steps": gradient_steps,
        "hidden_dims": hidden_dims,
        "device": device,
        "seed": seed,
    }
    
    return results


def save_results(results: dict, filename: str):
    """Save experiment results to JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"dqn_selfplay_{filename}_{timestamp}.json"
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {filepath}")
    print(f"{'=' * 70}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-play training for DQN")
    parser.add_argument(
        "--abstraction",
        type=str,
        choices=["compact", "potential", "zone_based", "combined", "enhanced"],
        default="compact",
        help="State abstraction method: 'compact' (6-tuple, RECOMMENDED), 'potential' (9-tuple), 'zone_based' (12-tuple), or 'combined' (13-tuple)"
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--tokens", type=int, default=4, help="Tokens per player")
    parser.add_argument("--phase1", type=int, default=10000, help="Phase 1 episodes")
    parser.add_argument("--phase2", type=int, default=10000, help="Phase 2 episodes")
    parser.add_argument("--eval", type=int, default=1000, help="Evaluation episodes")
    parser.add_argument("--alpha", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum epsilon")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--obs_mode", type=str, default="hybrid", choices=["hybrid", "hybrid_penalty"], help="Observation mode: 'hybrid' (baseline) or 'hybrid_penalty' (experimental)")
    
    args = parser.parse_args()
    
    results = run_selfplay_experiment(
        state_abstraction=args.abstraction,
        num_players=args.players,
        tokens_per_player=args.tokens,
        phase1_episodes=args.phase1,
        phase2_episodes=args.phase2,
        eval_episodes=args.eval,
        learning_rate=args.alpha,
        discount_factor=args.gamma,
        epsilon=args.epsilon,
        min_epsilon=args.min_epsilon,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        observation_mode=args.obs_mode,
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nPhase 1 (vs Random):")
    print(f"  Evaluation Win Rate: {results['phase1_eval']['win_rate']:.2%}")
    print(f"\nPhase 2 (Self-Play):")
    print(f"  Training Win Rate (vs Phase 1): {results['phase2_training']['win_rate']:.2%}")
    print(f"  Eval Win Rate (vs Random): {results['phase2_eval_vs_random']['win_rate']:.2%}")
    print(f"  Eval Win Rate (vs Phase 1): {results['phase2_eval_vs_phase1']['win_rate']:.2%}")
    
    improvement = (
        results['phase2_eval_vs_random']['win_rate'] - 
        results['phase1_eval']['win_rate']
    ) * 100
    print(f"\nImprovement (Phase 2): {improvement:+.2f}%")
    
    save_results(results, f"{args.abstraction}_{args.players}p{args.tokens}t")

