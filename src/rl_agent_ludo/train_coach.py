"""
Coach Training Script (Phase 2: Data Infrastructure)

Trains the Win Probability Network (WPN) using the trajectory dataset 
collected during Phase 1.

Theoretical Basis:
- Input: Egocentric state vectors (expanded with relative features, Markovian).
- Label: Temporally discounted outcome (0.0 to 1.0).
- Loss: Binary Cross-Entropy (BCE) to minimize prediction error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import sys

# Import local modules
# Add src directory to path so we can import rl_agent_ludo
sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_agent_ludo.agents.winProbabilityNetwork import WinProbabilityNetwork


def interpret_val_loss(val_loss: float) -> tuple[str, str]:
    """
    Interpret validation loss in terms of model learning stage.
    
    Returns:
        Tuple of (stage, emoji) describing the learning stage
    """
    if val_loss >= 0.693:
        return "Random", "🎲"
    elif val_loss >= 0.600:
        return "Weak", "📉"
    elif val_loss >= 0.500:
        return "Basic", "📊"
    elif val_loss >= 0.450:
        return "Learning", "📈"
    else:
        return "Expert", "🎯"


def train_coach(
    trajectory_path: str,
    output_path: str,
    tokens: int,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = None,
    hidden_dims: list = None,
    verbose: bool = True
):
    """
    Train the Win Probability Network (Coach) on trajectory data.
    
    Args:
        trajectory_path: Path to .npz trajectory file
        output_path: Where to save the trained model weights
        tokens: Number of tokens (2 or 4) for dimension validation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to use ('cpu' or 'cuda', None for auto-detect)
        hidden_dims: Hidden layer dimensions (default: [128, 64])
        verbose: Whether to print progress
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        tqdm.write(f"🔧 Using device: {device}")

    # 1. Load Data from .npz file
    if verbose:
        tqdm.write(f"📂 Loading trajectory data from: {trajectory_path}")
    
    try:
        data = np.load(trajectory_path)
        
        # Extract states and labels
        states = data['states']
        labels = data['labels']
        
        # Extract metadata
        metadata = {
            'num_samples': int(data['num_samples'][0]) if 'num_samples' in data else len(states),
            'state_dim': int(data['state_dim'][0]) if 'state_dim' in data else states.shape[1] if len(states.shape) > 1 else states.shape[0],
            'gamma': float(data['gamma'][0]) if 'gamma' in data else 0.99,
            'seed': int(data['seed'][0]) if 'seed' in data else None,
            'num_episodes': int(data['num_episodes'][0]) if 'num_episodes' in data else None,
            'trajectory_interval': int(data['trajectory_interval'][0]) if 'trajectory_interval' in data else None,
        }
        
        if verbose:
            tqdm.write(f"   ✅ Loaded {len(states):,} samples")
            tqdm.write(f"   State dimension: {metadata['state_dim']}")
            tqdm.write(f"   Gamma: {metadata['gamma']:.4f}")
            if metadata['seed'] is not None:
                tqdm.write(f"   Seed: {metadata['seed']}")
            if metadata['num_episodes'] is not None:
                tqdm.write(f"   Episodes: {metadata['num_episodes']:,}")
        
    except Exception as e:
        tqdm.write(f"❌ Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Validate State Dimension
    input_dim = states.shape[1] if len(states.shape) > 1 else states.shape[0]
    
    # Expected dimensions: base_obs + relative_features
    # 2 tokens: 28 (base) + 4 (2 tokens * 2 features) = 32
    # 4 tokens: 46 (base) + 8 (4 tokens * 2 features) = 54
    expected_dim = 32 if tokens == 2 else 54
    
    if input_dim != expected_dim:
        tqdm.write(f"⚠️  Warning: State dimension mismatch!")
        tqdm.write(f"   Expected: {expected_dim} (for {tokens} tokens)")
        tqdm.write(f"   Got: {input_dim}")
        tqdm.write(f"   Continuing anyway...")
    
    # 3. Check Data Quality
    if verbose:
        mean_label = labels.mean()
        std_label = labels.std()
        tqdm.write(f"\n📊 Data Statistics:")
        tqdm.write(f"   Label mean: {mean_label:.4f} (0.5 = balanced Win/Loss)")
        tqdm.write(f"   Label std: {std_label:.4f}")
        tqdm.write(f"   Label range: [{labels.min():.4f}, {labels.max():.4f}]")
        
        # Count win-like vs loss-like labels
        win_like = np.sum(labels > 0.5)
        loss_like = np.sum(labels <= 0.5)
        tqdm.write(f"   Labels > 0.5: {win_like:,} ({win_like/len(labels)*100:.1f}%)")
        tqdm.write(f"   Labels <= 0.5: {loss_like:,} ({loss_like/len(labels)*100:.1f}%)")

    # 4. Train/Validation Split (80/20)
    if verbose:
        tqdm.write(f"\n🔄 Splitting data: 80% train, 20% validation...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        states, labels, test_size=0.2, random_state=42, shuffle=True
    )
    
    if verbose:
        tqdm.write(f"   Train samples: {len(X_train):,}")
        tqdm.write(f"   Validation samples: {len(X_val):,}")

    # 5. Create DataLoaders
    # Convert to tensors and add channel dim for BCE: (N,) -> (N, 1)
    train_ds = TensorDataset(
        torch.from_numpy(X_train.astype(np.float32)), 
        torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val.astype(np.float32)), 
        torch.from_numpy(y_val.astype(np.float32)).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 6. Initialize Network
    if hidden_dims is None:
        hidden_dims = [128, 64]  # Smaller than DQN for faster training
    
    if verbose:
        tqdm.write(f"\n🧠 Initializing Win Probability Network...")
        tqdm.write(f"   Input dimension: {input_dim}")
        tqdm.write(f"   Hidden layers: {hidden_dims}")
        tqdm.write(f"   Output: 1 (sigmoid probability)")
    
    model = WinProbabilityNetwork(
        input_dim=input_dim, 
        hidden_dims=hidden_dims,
        device=device
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary Cross Entropy

    # 7. Training Loop
    if verbose:
        tqdm.write(f"\n🚀 Starting Coach Training...")
        tqdm.write(f"   Epochs: {epochs}")
        tqdm.write(f"   Batch size: {batch_size}")
        tqdm.write(f"   Learning rate: {learning_rate}")
        tqdm.write(f"{'='*60}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Track learning milestones
    milestones = {
        'random': False,      # >= 0.693
        'weak': False,        # >= 0.600
        'basic': False,       # >= 0.500
        'learning': False,    # >= 0.450
        'expert': False,      # < 0.450
    }
    
    # Create progress bar for epochs
    epoch_bar = tqdm(range(epochs), desc="Training epochs", disable=not verbose)
    
    for epoch in epoch_bar:
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        # Progress bar for training batches
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                        leave=False, disable=not verbose)
        
        for inputs, targets in train_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress bar
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # Progress bar for validation batches
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                      leave=False, disable=not verbose)
        
        with torch.no_grad():
            for inputs, targets in val_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
                
                # Update progress bar
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Interpret validation loss
        stage, emoji = interpret_val_loss(avg_val_loss)
        
        # Track milestones
        if not milestones['random'] and avg_val_loss < 0.693:
            milestones['random'] = True
            if verbose:
                tqdm.write(f"  🎯 Milestone: Passed Random Guessing (0.693) → Val Loss: {avg_val_loss:.4f}")
        if not milestones['weak'] and avg_val_loss < 0.600:
            milestones['weak'] = True
            if verbose:
                tqdm.write(f"  🎯 Milestone: Entered Weak Learning (0.600) → Val Loss: {avg_val_loss:.4f}")
        if not milestones['basic'] and avg_val_loss < 0.500:
            milestones['basic'] = True
            if verbose:
                tqdm.write(f"  🎯 Milestone: Entered Basic Heuristics (0.500) → Val Loss: {avg_val_loss:.4f}")
        if not milestones['learning'] and avg_val_loss < 0.450:
            milestones['learning'] = True
            if verbose:
                tqdm.write(f"  🎯 Milestone: Entered Deep Learning (0.450) → Val Loss: {avg_val_loss:.4f}")
        if not milestones['expert'] and avg_val_loss < 0.450:
            milestones['expert'] = True
        
        # Update epoch progress bar with interpretation
        epoch_bar.set_postfix({
            'train': f'{avg_train_loss:.4f}',
            'val': f'{avg_val_loss:.4f}',
            'best': f'{best_val_loss:.4f}',
            'stage': f'{emoji} {stage}'
        })
        
        # Save checkpoint if validation improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state_dict
            torch.save(model.state_dict(), str(output_file))
            
            if verbose:
                abs_path = output_file.resolve()
                tqdm.write(f"  ⭐ Epoch {epoch+1:02d}/{epochs}: New best model saved!")
                tqdm.write(f"     Path: {abs_path}")
                tqdm.write(f"     Val Loss: {avg_val_loss:.4f} ({emoji} {stage})")
    
    if verbose:
        final_stage, final_emoji = interpret_val_loss(val_losses[-1] if val_losses else best_val_loss)
        best_stage, best_emoji = interpret_val_loss(best_val_loss)
        
        output_file = Path(output_path)
        abs_output_path = output_file.resolve()
        
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"✅ Training Complete!")
        tqdm.write(f"   Model saved to: {abs_output_path}")
        tqdm.write(f"\n📊 Validation Loss Analysis:")
        tqdm.write(f"   Best Validation Loss: {best_val_loss:.4f} ({best_emoji} {best_stage})")
        tqdm.write(f"   Final Validation Loss: {val_losses[-1]:.4f} ({final_emoji} {final_stage})")
        tqdm.write(f"   Final Train Loss: {train_losses[-1]:.4f}")
        
        tqdm.write(f"\n🎯 Learning Milestones Reached:")
        tqdm.write(f"   {'✅' if milestones['random'] else '❌'} Random Guessing Threshold (< 0.693)")
        tqdm.write(f"   {'✅' if milestones['weak'] else '❌'} Weak Learning Threshold (< 0.600)")
        tqdm.write(f"   {'✅' if milestones['basic'] else '❌'} Basic Heuristics Threshold (< 0.500)")
        tqdm.write(f"   {'✅' if milestones['learning'] else '❌'} Deep Learning Threshold (< 0.450)")
        
        tqdm.write(f"\n📚 Interpretation Guide:")
        tqdm.write(f"   ≥ 0.693 (🎲 Random): Model is guessing randomly")
        tqdm.write(f"   0.600-0.693 (📉 Weak): Model is learning basic patterns")
        tqdm.write(f"   0.500-0.600 (📊 Basic): Model has general heuristics")
        tqdm.write(f"   0.450-0.500 (📈 Learning): Model is gaining insights")
        tqdm.write(f"   < 0.450 (🎯 Expert): Model has deep game understanding")
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Win Probability Network (Coach) on trajectory data")
    parser.add_argument("--data", type=str, required=True, 
                       help="Path to the saved trajectory .npz file")
    parser.add_argument("--output", type=str, default="coach_model.pth", 
                       help="Where to save the trained model weights")
    parser.add_argument("--tokens", type=int, default=2, choices=[2, 4],
                       help="Number of tokens (2 or 4) for dimension validation")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, 
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                       help="Learning rate")
    parser.add_argument("--hidden_dims", type=str, default="128,64",
                       help="Hidden layer dimensions (comma-separated, e.g., '128,64')")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda'],
                       help="Device to use (None for auto-detect)")
    
    args = parser.parse_args()
    
    # Parse hidden_dims
    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(',')]
    
    train_coach(
        trajectory_path=args.data,
        output_path=args.output,
        tokens=args.tokens,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        hidden_dims=hidden_dims,
        verbose=True
    )

