#!/usr/bin/env python3
"""
Self-play training experiment for Tabular Q-Learning.

Implements multi-phase training approach similar to Ludo_Game_AI repository:
1. Phase 1: Train against random opponents
2. Phase 2: Train against best agent from Phase 1 (self-play)

Reference: https://github.com/raffaele-aurucci/Ludo_Game_AI
"""

import gymnasium as gym
import numpy as np
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Register environment and import
import rl_agent_ludo.environment  # noqa: F401 - registers Ludo-v0
from rl_agent_ludo.agents.tabularQAgent import TabularQAgent


def train_against_opponent(
    agent: TabularQAgent,
    opponent: Optional[TabularQAgent],
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
    
    # Debug tracking
    debug_stats = {
        'q_table_size': [],
        'exploration_rate': [],
        'avg_q_values': [],
        'reward_by_episode': [],
        'win_by_episode': [],
        'action_selection': {'explore': 0, 'exploit': 0},
    }
    
    print(f"\n{phase_name}: {num_episodes} episodes")
    print(f"  Agent: TabularQAgent (learning)")
    print(f"  Opponent: {'TabularQAgent (frozen)' if opponent else 'Random'}")
    
    # Log every N episodes for debugging
    log_interval = max(1, num_episodes // 20)  # Log ~20 times during training
    
    for episode in tqdm(range(num_episodes), desc=phase_name):
        # Optional: episode-wise epsilon scheduling (linear decay)
        if epsilon_start is not None and epsilon_end is not None:
            # Linear decay over the entire training horizon for this phase
            progress = episode / max(num_episodes - 1, 1)
            current_epsilon = epsilon_start - (epsilon_start - epsilon_end) * progress
            # Clamp to [epsilon_end, epsilon_start]
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
                # Track exploration vs exploitation
                state_tuple = agent._build_state_tuple(state)
                if random.random() < agent.epsilon:
                    debug_stats['action_selection']['explore'] += 1
                else:
                    debug_stats['action_selection']['exploit'] += 1
                
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
        debug_stats['reward_by_episode'].append(episode_reward)
        debug_stats['win_by_episode'].append(1 if episode_won else 0)
        
        # Debug logging every N episodes
        if (episode + 1) % log_interval == 0 or episode == 0:
            q_table_size = len(agent.q_table)
            avg_q = np.mean([np.mean(q_vals) for q_vals in agent.q_table.values()]) if agent.q_table else 0.0
            
            debug_stats['q_table_size'].append(q_table_size)
            debug_stats['exploration_rate'].append(agent.epsilon)
            debug_stats['avg_q_values'].append(avg_q)
            
            # Calculate recent win rate (last 10% of episodes so far)
            recent_window = max(10, (episode + 1) // 10)
            recent_wins = sum(debug_stats['win_by_episode'][-recent_window:])
            recent_win_rate = recent_wins / recent_window if recent_window > 0 else 0.0
            
            # Calculate recent average reward
            recent_avg_reward = np.mean(debug_stats['reward_by_episode'][-recent_window:]) if recent_window > 0 else 0.0
            
            tqdm.write(
                f"  Episode {episode+1}/{num_episodes}: "
                f"Q-table size={q_table_size}, "
                f"ε={agent.epsilon:.3f}, "
                f"Recent win rate={recent_win_rate:.2%}, "
                f"Recent avg reward={recent_avg_reward:.1f}, "
                f"Avg Q={avg_q:.2f}"
            )
        
        if episode_won:
            wins += 1
        elif done:  # Game ended but agent didn't win
            losses += 1
    
    env.close()
    
    # Final debug summary
    total_actions = debug_stats['action_selection']['explore'] + debug_stats['action_selection']['exploit']
    explore_pct = (debug_stats['action_selection']['explore'] / total_actions * 100) if total_actions > 0 else 0
    exploit_pct = (debug_stats['action_selection']['exploit'] / total_actions * 100) if total_actions > 0 else 0
    
    print(f"\n  Debug Summary:")
    print(f"    Final Q-table size: {len(agent.q_table)}")
    print(f"    Exploration: {explore_pct:.1f}%, Exploitation: {exploit_pct:.1f}%")
    print(f"    Final epsilon: {agent.epsilon:.4f}")
    if debug_stats['avg_q_values']:
        print(f"    Final avg Q-value: {debug_stats['avg_q_values'][-1]:.2f}")
        print(f"    Q-value trend: {debug_stats['avg_q_values'][0]:.2f} → {debug_stats['avg_q_values'][-1]:.2f}")
    
    return {
        "wins": wins,
        "losses": losses,
        "draws": num_episodes - wins - losses,
        "win_rate": wins / num_episodes,
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "debug_stats": {
            "final_q_table_size": len(agent.q_table),
            "exploration_rate": explore_pct,
            "exploitation_rate": exploit_pct,
            "final_epsilon": float(agent.epsilon),
            "final_avg_q": float(debug_stats['avg_q_values'][-1]) if debug_stats['avg_q_values'] else 0.0,
            "q_value_trend": [float(x) for x in debug_stats['avg_q_values']],
            "q_table_growth": [int(x) for x in debug_stats['q_table_size']],
        }
    }


def evaluate_agent(
    agent: TabularQAgent,
    num_episodes: int = 1000,
    num_players: int = 2,
    tokens_per_player: int = 4,
    seed: int = 9999,
    opponent: Optional[TabularQAgent] = None,
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
    state_abstraction: str = 'zone_based',
    num_players: int = 2,
    tokens_per_player: int = 4,
    phase1_episodes: int = 10000,
    phase2_episodes: int = 10000,
    eval_episodes: int = 1000,
    learning_rate: float = 0.3,
    discount_factor: float = 0.99,  # Changed from 0.5 to 0.99 for proper long-term learning
    epsilon: float = 0.1,
    epsilon_decay: float = 1.0,
    min_epsilon: float = 0.05,
    seed: int = 42,
    observation_mode: str = 'hybrid',
) -> dict:
    """
    Run multi-phase self-play training experiment.
    
    Args:
        state_abstraction: 'potential', 'zone_based', or 'combined'
        num_players: Number of players (2 or 4)
        tokens_per_player: Tokens per player
        phase1_episodes: Episodes for Phase 1 (vs random)
        phase2_episodes: Episodes for Phase 2 (self-play)
        eval_episodes: Episodes for evaluation
        learning_rate: Alpha parameter
        discount_factor: Gamma parameter
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        min_epsilon: Minimum epsilon for linear decay
        seed: Random seed
    
    Returns:
        Dictionary with all phase results
    """
    print("=" * 70)
    print("MULTI-PHASE SELF-PLAY TRAINING")
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
    print("=" * 70)
    
    results = {}
    
    # Phase 1: Train against random
    print("\n" + "=" * 70)
    print("PHASE 1: Training Against Random Opponents")
    print("=" * 70)
    
    # Use manual (linear) epsilon schedule in this script, so fix decay to 1.0
    agent_phase1 = TabularQAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=1.0,
        min_epsilon=min_epsilon,
        seed=seed,
        state_abstraction=state_abstraction,
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
    phase1_path = checkpoint_dir / f"phase1_{state_abstraction}_{seed}.pkl"
    agent_phase1.save(str(phase1_path))
    print(f"  Saved to: {phase1_path}")
    
    # Phase 2: Self-play training
    print("\n" + "=" * 70)
    print("PHASE 2: Self-Play Training")
    print("=" * 70)
    
    # Create new agent for learning, initialized with Phase 1 Q-table
    agent_phase2 = TabularQAgent(
        learning_rate=0.02, # decreased from 0.3 to 0.02 for better long-term learning for phase 2
        discount_factor=discount_factor,
        epsilon=epsilon,  # Reset epsilon for Phase 2
        epsilon_decay=1.0,  # Use manual epsilon schedule
        min_epsilon=min_epsilon,
        seed=seed + 1,
        state_abstraction=state_abstraction,
    )
    agent_phase2.load(str(phase1_path))
    agent_phase2.epsilon = epsilon  # Reset epsilon after loading
    
    # Create frozen copy of Phase 1 agent as opponent
    opponent = TabularQAgent(
        learning_rate=0.0,  # No learning
        discount_factor=discount_factor,
        epsilon=0.01,  # Nearly greedy
        epsilon_decay=1.0,  # No decay
        seed=seed + 2,
        state_abstraction=state_abstraction,
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
    phase2_path = checkpoint_dir / f"phase2_{state_abstraction}_{seed}.pkl"
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
        "seed": seed,
    }
    
    return results


def save_results(results: dict, filename: str):
    """Save experiment results to JSON file."""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"selfplay_{filename}_{timestamp}.json"
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print(f"Results saved to: {filepath}")
    print(f"{'=' * 70}")
    
    return filepath


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-play training for Tabular Q-Learning")
    parser.add_argument(
        "--abstraction",
        type=str,
        choices=["potential", "zone_based", "combined", "enhanced", "compact"],
        default="zone_based",
        help="State abstraction method: 'potential' (9-tuple), 'zone_based' (12-tuple), or 'combined' (17-tuple)"
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--tokens", type=int, default=4, help="Tokens per player")
    parser.add_argument("--phase1", type=int, default=10000, help="Phase 1 episodes")
    parser.add_argument("--phase2", type=int, default=10000, help="Phase 2 episodes")
    parser.add_argument("--eval", type=int, default=1000, help="Evaluation episodes")
    parser.add_argument("--alpha", type=float, default=0.3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (recommended: 0.99 for long-term learning)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Initial epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon for linear decay")
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

