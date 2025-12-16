"""
Level 6 - Phase 2-3: Learn Reward Function from Trajectories.

This script:
1. Loads collected trajectories
2. Creates preference pairs (better > worse rankings)
3. Trains reward network using Bradley-Terry model
4. Saves learned reward function

Usage:
    python3 experiments/level6_learn_reward.py [--quick]

    --quick: Quick test mode (fewer pairs, fewer epochs)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
from rl_agent_ludo.preference_learning.trajectory_collector import TrajectoryCollector
from rl_agent_ludo.preference_learning.trajectory_ranker import TrajectoryRanker
from rl_agent_ludo.preference_learning.reward_network import RewardLearner


def main():
    parser = argparse.ArgumentParser(description="Learn reward function from trajectories")
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (1k pairs, 20 epochs)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default: 100, quick: 20)')
    parser.add_argument('--pairs', type=int, default=None,
                       help='Number of preference pairs (default: 10000, quick: 1000)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    print("="*80)
    print("LEVEL 6 - PHASE 2-3: LEARN REWARD FUNCTION FROM TRAJECTORIES")
    print("="*80)

    # Set parameters
    if args.quick:
        print("\nQUICK MODE")
        num_pairs = args.pairs if args.pairs else 1000
        num_epochs = args.epochs if args.epochs else 20
    else:
        print("\nFULL MODE")
        num_pairs = args.pairs if args.pairs else 10000
        num_epochs = args.epochs if args.epochs else 100

    print(f"Preference pairs: {num_pairs}")
    print(f"Training epochs: {num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Seed: {args.seed}")

    # Load Trajectories
    print("\n" + "="*80)
    print("[1/3] LOADING TRAJECTORIES")
    print("="*80)

    collector = TrajectoryCollector(save_dir="checkpoints/level6/trajectories")

    # Load all available batches
    all_trajectories = []
    batch_files = list(Path("checkpoints/level6/trajectories").glob("*.pkl"))

    if not batch_files:
        print("No trajectory files found in checkpoints/level6/trajectories/")
        print("   Please run level6_collect_trajectories.py first")
        return

    for batch_file in batch_files:
        batch_name = batch_file.stem
        trajs = collector.load_trajectories(batch_name)
        all_trajectories.extend(trajs)

    print(f"\nTotal trajectories loaded: {len(all_trajectories)}")

    # Print statistics
    wins = sum(1 for t in all_trajectories if t['outcome'] == 'win')
    losses = len(all_trajectories) - wins

    print(f"\nTrajectory Statistics:")
    print(f"  Wins: {wins} ({wins/len(all_trajectories)*100:.1f}%)")
    print(f"  Losses: {losses} ({losses/len(all_trajectories)*100:.1f}%)")

    # Group by agent type
    by_agent = {}
    for traj in all_trajectories:
        agent_type = traj['agent_type']
        if agent_type not in by_agent:
            by_agent[agent_type] = []
        by_agent[agent_type].append(traj)

    for agent_type, trajs in by_agent.items():
        agent_wins = sum(1 for t in trajs if t['outcome'] == 'win')
        print(f"  {agent_type}: {len(trajs)} trajectories, "
              f"{agent_wins/len(trajs)*100:.1f}% win rate")

    # Create Preference Pairs
    print("\n" + "="*80)
    print("[2/3] CREATING PREFERENCE PAIRS")
    print("="*80)

    ranker = TrajectoryRanker()
    preference_pairs = ranker.create_preference_pairs(
        all_trajectories,
        max_pairs=num_pairs,
        seed=args.seed
    )

    if len(preference_pairs) < 100:
        print(f"\nWarning: Only {len(preference_pairs)} pairs created.")
        print("   This may not be enough for good learning.")
        print("   Consider collecting more diverse trajectories.")

    # Save pairs
    ranker.save_pairs("checkpoints/level6/preference_pairs.pkl")

    # Split train/val
    train_pairs, val_pairs = ranker.split_train_val(train_ratio=0.8, seed=args.seed)

    # Train Reward Network
    print("\n" + "="*80)
    print("[3/3] TRAINING REWARD NETWORK")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Get state dimension from first trajectory
    state_dim = len(all_trajectories[0]['states'][0])
    print(f"State dimension: {state_dim}")

    # Create learner
    learner = RewardLearner(
        state_dim=state_dim,
        hidden_dim=128,
        learning_rate=args.lr,
        device=device
    )

    # Train
    learner.train(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        num_epochs=num_epochs,
        batch_size=args.batch_size,
        patience=10,
        verbose=True
    )

    # Save final model
    learner.save("checkpoints/level6/reward_network_final.pth")

    # Summary
    print("\n" + "="*80)
    print("REWARD LEARNING COMPLETE!")
    print("="*80)
    print(f"Trajectories used: {len(all_trajectories)}")
    print(f"Preference pairs created: {len(preference_pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    print(f"Training epochs: {len(learner.train_losses)}")
    print(f"Final validation loss: {learner.val_losses[-1]:.4f}")
    print(f"Final validation accuracy: {learner.val_accuracies[-1]:.3f}")

    print(f"\nSaved Files:")
    print(f"  - checkpoints/level6/preference_pairs.pkl")
    print(f"  - checkpoints/level6/reward_network_best.pth")
    print(f"  - checkpoints/level6/reward_network_final.pth")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Phase 2-3 complete - Reward function learned")
    print("Phase 4: Train policy with learned reward")
    print("   Command: python3 experiments/level6_train_policy.py")
    print("\nThe reward network is ready to use for training!")
    print("="*80)


if __name__ == "__main__":
    main()
