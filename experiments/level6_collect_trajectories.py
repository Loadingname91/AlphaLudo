"""
Level 6 - Phase 1: Collect Trajectories for T-REX.

This script collects game trajectories from existing trained agents
to create training data for preference-based reward learning.

Collects from:
1. Level 5 agent (best performance)
2. Random agent (negative examples)
3. Level 3 agent (medium skill - optional)

Usage:
    python experiments/level6_collect_trajectories.py [--quick]

    --quick: Collect smaller batch for testing (50 per agent)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import torch
from rl_agent_ludo.environment.level5_multiagent import Level5MultiAgentLudo
from rl_agent_ludo.agents.simple_dqn import SimpleDQNAgent
from rl_agent_ludo.agents.baseline_agents import RandomAgent
from rl_agent_ludo.preference_learning.trajectory_collector import TrajectoryCollector


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories for T-REX")
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (50 trajectories per agent)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    print("="*80)
    print("LEVEL 6 - PHASE 1: TRAJECTORY COLLECTION FOR T-REX")
    print("="*80)

    # Set collection size
    if args.quick:
        print("\nQUICK MODE: Collecting 50 trajectories per agent")
        num_level5 = 50
        num_random = 30
        num_level3 = 20
    else:
        print("\nFULL MODE: Collecting 1000+ trajectories")
        num_level5 = 500
        num_random = 300
        num_level3 = 200

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Environment (4-player full game)
    print("\nInitializing Level 5 environment (4 players, 2 tokens each)...")
    env = Level5MultiAgentLudo()
    print(f"   State dim: {env.observation_space.shape[0]}")
    print(f"   Action dim: {env.action_space.n}")

    # Collector
    collector = TrajectoryCollector(save_dir="checkpoints/level6/trajectories")

    # Collect from Level 5 Agent
    print("\n" + "="*80)
    print("[1/3] COLLECTING FROM LEVEL 5 TRAINED AGENT")
    print("="*80)

    level5_checkpoint = Path("checkpoints/level5/best_model.pth")

    if not level5_checkpoint.exists():
        print(f"Level 5 checkpoint not found: {level5_checkpoint}")
        print("   Please train Level 5 first or provide correct path")
        print("Skipping Level 5 collection...")
        level5_collected = False
    else:
        print(f"Loading Level 5 agent from {level5_checkpoint}...")

        try:
            level5_agent = SimpleDQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                device=device
            )
            level5_agent.load(str(level5_checkpoint))
            level5_agent.name = "level5_trained"
            level5_agent.epsilon = 0.0  # Greedy evaluation
            print("Level 5 agent loaded successfully")

            print(f"\nCollecting {num_level5} trajectories from Level 5 agent...")
            collector.collect_batch(
                env, level5_agent,
                num_episodes=num_level5,
                batch_name="level5_demos",
                seed_start=args.seed,
                verbose=True
            )
            level5_collected = True

        except Exception as e:
            print(f"Error loading Level 5 agent: {e}")
            print("Skipping Level 5 collection...")
            level5_collected = False

    # Collect from Random Agent
    print("\n" + "="*80)
    print("[2/3] COLLECTING FROM RANDOM AGENT (Negative Examples)")
    print("="*80)

    print("Initializing random agent...")
    random_agent = RandomAgent(seed=args.seed + 5000)
    random_agent.name = "random"

    print(f"\nCollecting {num_random} trajectories from Random agent...")
    collector.collect_batch(
        env, random_agent,
        num_episodes=num_random,
        batch_name="random_demos",
        seed_start=args.seed + 1000,
        verbose=True
    )

    # Collect from Level 3 Agent (Optional)
    print("\n" + "="*80)
    print("[3/3] COLLECTING FROM LEVEL 3 AGENT (Medium Skill - Optional)")
    print("="*80)

    level3_checkpoint = Path("checkpoints/level3/best_model.pth")

    if not level3_checkpoint.exists():
        print(f"Level 3 checkpoint not found: {level3_checkpoint}")
        print("   This is optional - you can proceed without it")
        print("Skipping Level 3 collection...")
        level3_collected = False
    else:
        try:
            # Note: Level 3 uses 2-player environment
            from rl_agent_ludo.environment.level3_multitoken import Level3MultiTokenLudo

            print("Initializing Level 3 environment (2 players, 2 tokens each)...")
            env3 = Level3MultiTokenLudo()

            print(f"Loading Level 3 agent from {level3_checkpoint}...")
            level3_agent = SimpleDQNAgent(
                state_dim=env3.observation_space.shape[0],
                action_dim=env3.action_space.n,
                device=device
            )
            level3_agent.load(str(level3_checkpoint))
            level3_agent.name = "level3_trained"
            level3_agent.epsilon = 0.0
            print("Level 3 agent loaded successfully")

            # Create separate collector for Level 3 (different environment)
            collector_l3 = TrajectoryCollector(save_dir="checkpoints/level6/trajectories")

            print(f"\nCollecting {num_level3} trajectories from Level 3 agent...")
            collector_l3.collect_batch(
                env3, level3_agent,
                num_episodes=num_level3,
                batch_name="level3_demos",
                seed_start=args.seed + 2000,
                verbose=True
            )
            level3_collected = True

        except Exception as e:
            print(f"Error with Level 3: {e}")
            print("Skipping Level 3 collection...")
            level3_collected = False

    # Summary
    print("\n" + "="*80)
    print("TRAJECTORY COLLECTION COMPLETE!")
    print("="*80)

    total_trajectories = 0

    if level5_collected:
        print(f"Level 5 demos: {num_level5} trajectories")
        total_trajectories += num_level5

    print(f"Random demos: {num_random} trajectories")
    total_trajectories += num_random

    if level3_collected:
        print(f"Level 3 demos: {num_level3} trajectories")
        total_trajectories += num_level3

    print(f"\nTotal collected: {total_trajectories} trajectories")
    print(f"Saved to: checkpoints/level6/trajectories/")

    # Verify saved files
    print("\nSaved files:")
    saved_batches = collector.get_all_saved_batches()
    for batch in saved_batches:
        print(f"   - {batch}.pkl")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Phase 1 complete - Trajectories collected")
    print("Phase 2-3: Run level6_learn_reward.py to train reward network")
    print(f"   Command: python experiments/level6_learn_reward.py")
    print("\nTip: You can run this script with --quick flag for faster testing")
    print("="*80)


if __name__ == "__main__":
    main()
