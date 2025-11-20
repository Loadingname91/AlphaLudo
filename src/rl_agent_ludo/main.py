"""
Main entry point for RL Agent Ludo training.

Usage:
    python -m rl_agent_ludo.main --config configs/default_config.yaml
"""

import argparse
import sys
import os
from pathlib import Path
from utils.visualizer import visualize_board_indices
# Add src to path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from rl_agent_ludo.environment.ludo_env import LudoEnv
from rl_agent_ludo.agents.agent_registry import AgentRegistry
from rl_agent_ludo.trainer.trainer import Trainer
from rl_agent_ludo.utils.config_loader import load_config


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train RL Agent for Ludo')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the board'
    )
    parser.add_argument(
        '--show-render',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Show the render of the board'
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default='assets/board.png',
        help='Path to save the board visualization'
    )
    parser.add_argument(
        '--show-indices',
       type=lambda x: x.lower() == 'true',
       default=True,
        help='Show the legend on the board'
    )
    parser.add_argument(
        '--show-legend',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Show the legend on the board'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Override experiment name from config'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help='Override number of episodes from config'
    )
    args = parser.parse_args()

    if args.visualize:
        visualize_board_indices(
            save_path=args.save_path,
            show_track_indices=True,
            show_tile_coords=True,
            show_legend=True
            )
        sys.exit(0)
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        print(f"Please create a config file or use the default: configs/default_config.yaml")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Override config with command-line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.num_episodes:
        config['training']['num_episodes'] = args.num_episodes
    
    # Ensure experiment_name is set
    if config.get('experiment', {}).get('name') is None:
        config['experiment']['name'] = 'default_experiment'
    
    # Ensure output_dir is set
    if config.get('experiment', {}).get('output_dir') is None:
        config['experiment']['output_dir'] = 'results'
    
    print("=" * 60)
    print("RL Agent Ludo - Training")
    print("=" * 60)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Agent Type: {config['agent']['type']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    print(f"Output Dir: {config['experiment']['output_dir']}")
    print("=" * 60)
    
    # Create environment
    env_config = config.get('environment', {})
    env = LudoEnv(
        reward_schema=env_config.get('reward_schema', 'sparse'),
        player_id=env_config.get('player_id', 0),
        seed=config.get('seed') or config.get('experiment', {}).get('seed'),
        render=env_config.get('render', False)
    )
    
    # Create agent
    agent_config = config['agent']
    if isinstance(agent_config, dict):
        agent = AgentRegistry.create_agent(agent_config)
    else:
        # Fallback: assume agent type is in top-level config
        agent = AgentRegistry.create_agent({'type': config['agent']['type']})
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        config=config
    )
    
    # Run training
    try:
        results = trainer.run()
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Summary: {results.get('summary', {})}")
        print(f"Saved files: {results.get('saved_files', {})}")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save metrics even on interruption
        saved_files = trainer.metrics_tracker.save_metrics()
        print(f"Metrics saved to: {saved_files}")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
