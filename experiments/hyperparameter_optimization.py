#!/usr/bin/env python3
"""
Bayesian Hyperparameter Optimization for Tabular Q-Learning Self-Play

Uses Optuna (Tree-structured Parzen Estimator) for efficient hyperparameter search.
More efficient than grid search - explores promising regions first.

Usage:
    python experiments/hyperparameter_optimization.py \
        --abstraction combined \
        --players 2 --tokens 4 \
        --trials 50 \
        --quick  # Use fewer episodes per trial for faster search
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
except ImportError:
    print("ERROR: optuna not installed. Install with: pip install optuna")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Import after path setup
from experiments.tabular_q_selfplay import run_selfplay_experiment


def objective(
    trial: optuna.Trial,
    state_abstraction: str,
    num_players: int,
    tokens_per_player: int,
    phase1_episodes: int,
    phase2_episodes: int,
    eval_episodes: int,
    seed: int,
) -> float:
    """
    Objective function for Optuna optimization.
    
    Returns the win rate to maximize (Phase 2 eval vs Random).
    """
    # Suggest hyperparameters
    alpha = trial.suggest_float("alpha", 0.05, 0.2, log=False)
    gamma = trial.suggest_float("gamma", 0.7, 0.95, log=False)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.2, log=False)
    min_epsilon = trial.suggest_float("min_epsilon", 0.01, 0.1, log=False)
    
    # Ensure min_epsilon <= epsilon
    if min_epsilon > epsilon:
        min_epsilon = epsilon * 0.5
    
    # Run experiment
    try:
        results = run_selfplay_experiment(
            state_abstraction=state_abstraction,
            num_players=num_players,
            tokens_per_player=tokens_per_player,
            phase1_episodes=phase1_episodes,
            phase2_episodes=phase2_episodes,
            eval_episodes=eval_episodes,
            learning_rate=alpha,
            discount_factor=gamma,
            epsilon=epsilon,
            min_epsilon=min_epsilon,
            seed=seed + trial.number,  # Different seed per trial
        )
        
        # Objective: Phase 2 eval win rate vs Random
        objective_value = results['phase2_eval_vs_random']['win_rate']
        
        # Report intermediate value (Phase 1 win rate)
        trial.set_user_attr("phase1_win_rate", results['phase1_eval']['win_rate'])
        trial.set_user_attr("phase2_vs_phase1", results['phase2_eval_vs_phase1']['win_rate'])
        trial.set_user_attr("improvement", 
            (results['phase2_eval_vs_random']['win_rate'] - 
             results['phase1_eval']['win_rate']) * 100)
        
        return objective_value
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0  # Return worst case if trial fails


def run_optimization(
    state_abstraction: str = "combined",
    num_players: int = 2,
    tokens_per_player: int = 4,
    phase1_episodes: int = 20000,  # Reduced for faster search
    phase2_episodes: int = 10000,   # Reduced for faster search
    eval_episodes: int = 2000,
    n_trials: int = 10,
    seed: int = 42,
    quick: bool = False,
    study_name: str = None,
) -> Dict[str, Any]:
    """
    Run Bayesian hyperparameter optimization.
    
    Args:
        quick: If True, use even fewer episodes (10k/5k) for faster search
    """
    if quick:
        phase1_episodes = 10000
        phase2_episodes = 5000
        eval_episodes = 1000
        print("⚡ Quick mode: Using reduced episodes for faster search")
    
    print("=" * 70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  State Abstraction: {state_abstraction}")
    print(f"  Players: {num_players}, Tokens: {tokens_per_player}")
    print(f"  Phase 1 Episodes: {phase1_episodes:,}")
    print(f"  Phase 2 Episodes: {phase2_episodes:,}")
    print(f"  Eval Episodes: {eval_episodes:,}")
    print(f"  Number of Trials: {n_trials}")
    print(f"  Optimization Method: Tree-structured Parzen Estimator (TPE)")
    print(f"\nSearch Space:")
    print(f"  α (learning rate): [0.05, 0.2]")
    print(f"  γ (discount): [0.7, 0.95]")
    print(f"  ε (epsilon): [0.05, 0.2]")
    print(f"  min_ε (min epsilon): [0.01, 0.1]")
    print("=" * 70)
    
    # Create study
    if study_name is None:
        study_name = f"ludo_{state_abstraction}_{num_players}p{tokens_per_player}t"
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize win rate
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    # Run optimization
    print(f"\n🚀 Starting optimization ({n_trials} trials)...")
    print("   This may take a while. Each trial runs full training.\n")
    
    study.optimize(
        lambda trial: objective(
            trial,
            state_abstraction,
            num_players,
            tokens_per_player,
            phase1_episodes,
            phase2_episodes,
            eval_episodes,
            seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    
    best_trial = study.best_trial
    print(f"\n✅ Best Trial: #{best_trial.number}")
    print(f"   Win Rate: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print(f"\n📊 Best Hyperparameters:")
    for param, value in best_trial.params.items():
        print(f"   {param}: {value:.4f}")
    
    print(f"\n📈 Trial Statistics:")
    print(f"   Phase 1 Win Rate: {best_trial.user_attrs.get('phase1_win_rate', 0):.2%}")
    print(f"   Phase 2 vs Phase 1: {best_trial.user_attrs.get('phase2_vs_phase1', 0):.2%}")
    print(f"   Improvement: {best_trial.user_attrs.get('improvement', 0):+.2f}%")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study
    study_file = results_dir / f"optuna_study_{study_name}_{timestamp}.pkl"
    import pickle
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"\n💾 Study saved to: {study_file}")
    
    # Save best params as JSON
    best_params = {
        "best_trial": best_trial.number,
        "best_value": best_trial.value,
        "hyperparameters": best_trial.params,
        "attributes": best_trial.user_attrs,
        "config": {
            "state_abstraction": state_abstraction,
            "num_players": num_players,
            "tokens_per_player": tokens_per_player,
            "phase1_episodes": phase1_episodes,
            "phase2_episodes": phase2_episodes,
            "eval_episodes": eval_episodes,
            "n_trials": n_trials,
            "seed": seed,
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            for t in study.trials
        ],
    }
    
    json_file = results_dir / f"optuna_best_{study_name}_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"💾 Best params saved to: {json_file}")
    
    # Print command to run with best params
    print("\n" + "=" * 70)
    print("RECOMMENDED COMMAND (with best hyperparameters):")
    print("=" * 70)
    print(f"\npython experiments/tabular_q_selfplay.py \\")
    print(f"  --abstraction {state_abstraction} \\")
    print(f"  --players {num_players} --tokens {tokens_per_player} \\")
    print(f"  --phase1 50000 --phase2 25000 \\")
    print(f"  --eval 5000 \\")
    print(f"  --alpha {best_trial.params['alpha']:.4f} \\")
    print(f"  --gamma {best_trial.params['gamma']:.4f} \\")
    print(f"  --epsilon {best_trial.params['epsilon']:.4f} \\")
    print(f"  --min_epsilon {best_trial.params['min_epsilon']:.4f}")
    print("=" * 70)
    
    return best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for Tabular Q-Learning"
    )
    parser.add_argument(
        "--abstraction",
        type=str,
        choices=["potential", "zone_based", "combined"],
        default="combined",
        help="State abstraction method",
    )
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--tokens", type=int, default=4, help="Tokens per player")
    parser.add_argument(
        "--phase1",
        type=int,
        default=20000,
        help="Phase 1 episodes per trial (reduced for faster search)",
    )
    parser.add_argument(
        "--phase2",
        type=int,
        default=10000,
        help="Phase 2 episodes per trial (reduced for faster search)",
    )
    parser.add_argument("--eval", type=int, default=2000, help="Evaluation episodes")
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer episodes (10k/5k) for faster search",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    run_optimization(
        state_abstraction=args.abstraction,
        num_players=args.players,
        tokens_per_player=args.tokens,
        phase1_episodes=args.phase1,
        phase2_episodes=args.phase2,
        eval_episodes=args.eval,
        n_trials=args.trials,
        seed=args.seed,
        quick=args.quick,
    )

