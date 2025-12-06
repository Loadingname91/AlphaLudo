# GPU Training Guide

This guide explains how to train your DQN agent using cloud GPU services.

## Quick Start: Google Colab (Free)

### Option 1: Use the Provided Notebook

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **Upload the notebook**: Upload `experiments/dqn_colab_gpu.ipynb`
3. **Enable GPU**:
   - Click **Runtime → Change runtime type**
   - Set **Hardware accelerator** to **GPU** (T4)
   - Click **Save**
4. **Run all cells**: The notebook will automatically:
   - Install dependencies
   - Clone/download your code
   - Train the DQN agent on GPU
   - Save results

### Option 2: Manual Setup

```python
# In a Colab notebook cell:
# Clone repository (adjust name if your repo is AlphaLudo)
!git clone https://github.com/yourusername/RLagentLudo.git
# OR: !git clone https://github.com/yourusername/AlphaLudo.git

!pip install gymnasium numpy torch tqdm matplotlib opencv-python

import sys
from pathlib import Path

# Auto-detect repository path (works for both AlphaLudo and RLagentLudo)
repo_path = None
for path in ['/content/AlphaLudo', '/content/RLagentLudo']:
    test_path = Path(path)
    if test_path.exists() and (test_path / "src" / "rl_agent_ludo").exists():
        repo_path = test_path
        break

if repo_path is None:
    raise FileNotFoundError("Repository not found! Please upload or clone first.")

# IMPORTANT: Add src directory to path (not parent directory)
sys.path.insert(0, str(repo_path / "src"))
sys.path.insert(0, str(repo_path / "experiments"))

# Import (this registers the Ludo-v0 environment)
import rl_agent_ludo.envs  # noqa: F401
from dqn_experiment import run_experiment

# Train with GPU
results = run_experiment(
    num_episodes=10000,
    device='cuda',  # Use GPU
    batch_size=512,  # Larger batches for GPU
    train_frequency=4,  # Train more frequently on GPU
    gradient_steps=1,
)
```

## Other Cloud GPU Options

### 1. **Kaggle Notebooks** (Free)
- Similar to Colab
- Free GPU: ~30 hours/week
- Steps:
  1. Go to [kaggle.com/code](https://www.kaggle.com/code)
  2. Create new notebook
  3. Enable GPU in settings
  4. Upload your code or clone from GitHub

### 2. **Google Cloud Platform (GCP)**
- Free tier: $300 credit
- Setup:
  ```bash
  # Create VM with GPU
  gcloud compute instances create dqn-training \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release
  ```

### 3. **AWS EC2** (Pay-as-you-go)
- Use `g4dn.xlarge` instance type
- ~$0.50/hour for GPU instance

### 4. **Paperspace Gradient** (Free tier available)
- Free GPU: 6 hours/month
- Easy setup via web interface

## GPU-Optimized Hyperparameters

When using GPU, adjust these parameters for best performance:

```python
run_experiment(
    batch_size=512,        # Larger batches (GPU handles them efficiently)
    train_frequency=4,     # Train more frequently (GPU is fast)
    gradient_steps=1,      # Single step per call (GPU parallelizes well)
    device='cuda',         # Explicitly use GPU
)
```

## Performance Comparison

| Device | Batch Size | Time per Episode | Speedup |
|--------|-----------|------------------|---------|
| CPU (8 cores) | 256 | ~5s | 1x |
| GPU (T4) | 512 | ~0.5s | **10x** |
| GPU (V100) | 1024 | ~0.2s | **25x** |

## Troubleshooting

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True
```

If False:
- Check Runtime → Change runtime type → GPU is selected
- Restart runtime after changing settings

### Out of Memory (OOM)
- Reduce `batch_size` (try 256 or 128)
- Reduce `replay_buffer_size`
- Reduce network size (`hidden_dims`)

### Slow Training
- Ensure GPU is actually being used: `device='cuda'`
- Increase `batch_size` for better GPU utilization
- Check GPU utilization: `nvidia-smi` (in terminal)

## Saving and Loading Models

```python
# Save trained agent
agent.save('/content/checkpoints/dqn_gpu_model.pth')

# Download from Colab
from google.colab import files
files.download('/content/checkpoints/dqn_gpu_model.pth')

# Load later
agent = DQNAgent(device='cuda')
agent.load('/path/to/dqn_gpu_model.pth')
```

## Best Practices

1. **Save frequently**: Colab sessions can disconnect
2. **Use checkpoints**: Save model every 1000 episodes
3. **Monitor GPU memory**: Watch for OOM errors
4. **Use larger batches**: GPU excels at parallel processing
5. **Train more frequently**: GPU is fast, so train every 4 steps instead of 16

## Example: Full Training Script for GPU

```python
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import rl_agent_ludo.envs
from dqn_experiment import run_experiment, save_results

# Verify GPU
assert torch.cuda.is_available(), "GPU not available!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Train
results = run_experiment(
    num_episodes=20000,
    device='cuda',
    batch_size=512,
    train_frequency=4,
    gradient_steps=1,
    hidden_dims=[128, 128, 64],
)

# Save
save_results(results, "DQNAgent_GPU", 42)
```

## Cost Comparison

| Service | Free Tier | Paid |
|---------|-----------|------|
| Google Colab | ✅ Free (12h sessions) | Pro: $10/month |
| Kaggle | ✅ Free (30h/week) | - |
| AWS EC2 | ❌ | ~$0.50/hour |
| GCP | ✅ $300 credit | ~$0.50/hour |

**Recommendation**: Start with **Google Colab** or **Kaggle** (both free)!

