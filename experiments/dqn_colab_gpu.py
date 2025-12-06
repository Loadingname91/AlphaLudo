#!/usr/bin/env python3
"""
DQN Training Script for Google Colab GPU

Copy-paste this into Google Colab cells or run as a script.
Make sure to enable GPU: Runtime → Change runtime type → GPU
"""

# Cell 1: Check GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  No GPU detected! Please enable GPU in Runtime → Change runtime type")

# Cell 2: Setup (choose one method)
# Method 1: Clone from GitHub
# !git clone https://github.com/yourusername/RLagentLudo.git
# OR if your repo is named AlphaLudo:
# !git clone https://github.com/yourusername/AlphaLudo.git

# Method 2: Upload files manually via Colab file browser, then:
# !unzip -q /content/RLagentLudo.zip
# OR if your repo is named AlphaLudo:
# !unzip -q /content/AlphaLudo.zip

# Cell 3: Install dependencies
# !pip install -q gymnasium numpy torch tqdm matplotlib opencv-python

# Cell 4: Import and run training
if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    
    # Auto-detect repository path (works for both RLagentLudo and AlphaLudo)
    possible_paths = [
        '/content/AlphaLudo',
        '/content/RLagentLudo',
        Path.cwd(),  # Current working directory
    ]
    
    repo_path = None
    for path in possible_paths:
        test_path = Path(path)
        if (test_path / "src" / "rl_agent_ludo").exists():
            repo_path = test_path
            break
    
    if repo_path is None:
        # Try to find it in current directory
        current = Path.cwd()
        if (current / "src" / "rl_agent_ludo").exists():
            repo_path = current
        else:
            raise FileNotFoundError(
                f"Could not find repository. Tried: {possible_paths}\n"
                f"Current directory: {current}\n"
                f"Please ensure the repository is in /content/AlphaLudo or /content/RLagentLudo"
            )
    
    print(f"Using repository path: {repo_path}")
    
    # Add paths - IMPORTANT: add src directory so rl_agent_ludo can be imported
    src_path = str(repo_path / "src")
    experiments_path = str(repo_path / "experiments")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if experiments_path not in sys.path:
        sys.path.insert(0, experiments_path)
    
    print(f"Added to sys.path:")
    print(f"  - {src_path}")
    print(f"  - {experiments_path}")
    
    # Verify imports work
    try:
        import gymnasium as gym
        import numpy as np
        from datetime import datetime
        import json
        
        # This import registers the Ludo-v0 environment
        import rl_agent_ludo.envs  # noqa: F401
        print("✓ Successfully imported rl_agent_ludo.envs")
        
        from rl_agent_ludo.agents.dqnAgent import DQNAgent
        print("✓ Successfully imported DQNAgent")
        
        from dqn_experiment import run_experiment, save_results
        print("✓ Successfully imported run_experiment")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"\nDebugging info:")
        print(f"  sys.path: {sys.path[:5]}")
        print(f"  Checking if rl_agent_ludo exists: {(repo_path / 'src' / 'rl_agent_ludo').exists()}")
        print(f"  Checking if envs exists: {(repo_path / 'src' / 'rl_agent_ludo' / 'envs').exists()}")
        raise
    
    # Run training with GPU-optimized settings
    print("=" * 60)
    print("Starting DQN Training on GPU")
    print("=" * 60)
    
    results = run_experiment(
        num_episodes=10000,
        num_players=4,
        tokens_per_player=4,
        seed=42,
        verbose=True,
        learning_rate=0.001,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        batch_size=512,  # Larger batch size for GPU
        replay_buffer_size=10000,
        target_update_frequency=100,
        train_frequency=4,  # Train more frequently on GPU
        gradient_steps=1,  # Single gradient step (GPU handles batches efficiently)
        hidden_dims=[128, 128, 64],
        device='cuda',  # Explicitly use GPU
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Avg Episode Length: {results['avg_episode_length']:.1f}")
    print(f"Avg Reward: {results['avg_reward']:.3f}")
    
    # Save results
    results_dir = repo_path / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"dqn_gpu_colab_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    
    # Download (uncomment to download automatically)
    # from google.colab import files
    # files.download(str(filepath))

