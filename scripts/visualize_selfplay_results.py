#!/usr/bin/env python3
"""
Visualize self-play training results and compare with baseline.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results():
    """Load the self-play results."""
    results_dir = Path(__file__).parent.parent / "results"
    
    # Find the most recent self-play results
    selfplay_files = list(results_dir.glob("selfplay_*.json"))
    if not selfplay_files:
        print("No self-play results found!")
        return None
    
    latest_file = max(selfplay_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def plot_win_rates(results):
    """Plot win rate progression across training phases."""
    phases = ['Phase 1\n(vs Random)', 'Phase 2\n(vs Random)', 'Phase 2\n(vs Phase 1)']
    win_rates = [
        results['phase1_eval']['win_rate'] * 100,
        results['phase2_eval_vs_random']['win_rate'] * 100,
        results['phase2_eval_vs_phase1']['win_rate'] * 100
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(phases, win_rates, color=['#3498db', '#2ecc71', '#f39c12'], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add reference line at 50%
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Random Baseline (50%)', alpha=0.7)
    
    # Add reference band for reference repo (64-70%)
    ax.axhspan(64, 70, alpha=0.2, color='green', label='Reference Repo Range (64-70%)')
    
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Self-Play Training: Win Rate Progression', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/selfplay_win_rates.png', dpi=300, bbox_inches='tight')
    print("Saved: results/selfplay_win_rates.png")
    plt.close()

def plot_improvement(results):
    """Plot the improvement delta from self-play."""
    baseline = results['phase1_eval']['win_rate'] * 100
    improved = results['phase2_eval_vs_random']['win_rate'] * 100
    delta = improved - baseline
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['Baseline\n(Phase 1)', 'After Self-Play\n(Phase 2)']
    values = [baseline, improved]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add improvement annotation
    ax.annotate(f'Improvement:\n+{delta:.1f}%',
                xy=(0.5, baseline), xytext=(0.5, (baseline + improved) / 2),
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_ylabel('Win Rate vs Random (%)', fontsize=12, fontweight='bold')
    ax.set_title('Self-Play Training Improvement', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random Baseline', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/selfplay_improvement.png', dpi=300, bbox_inches='tight')
    print("Saved: results/selfplay_improvement.png")
    plt.close()

def plot_episode_lengths(results):
    """Plot average episode lengths across phases."""
    phases = ['Phase 1\nTraining', 'Phase 1\nEval', 'Phase 2\nTraining', 'Phase 2\nEval\n(vs Random)']
    lengths = [
        results['phase1_training']['avg_episode_length'],
        results['phase1_eval']['avg_episode_length'],
        results['phase2_training']['avg_episode_length'],
        results['phase2_eval_vs_random']['avg_episode_length']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(phases, lengths, color=['#3498db', '#9b59b6', '#2ecc71', '#f39c12'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Average Episode Length (moves)', fontsize=12, fontweight='bold')
    ax.set_title('Episode Length Progression (Shorter = More Efficient)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(lengths) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/selfplay_episode_lengths.png', dpi=300, bbox_inches='tight')
    print("Saved: results/selfplay_episode_lengths.png")
    plt.close()

def plot_comparison_with_reference(results):
    """Compare our results with the reference repository."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Win rates comparison
    categories = ['Reference\n(Q-Agent)', 'Reference\n(DQL)', 'Our Agent\n(Phase 2)']
    win_rates = [64.58, 70.96, results['phase2_eval_vs_random']['win_rate'] * 100]
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    bars1 = ax1.bar(categories, win_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Win Rate Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 80)
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Training context comparison
    contexts = ['Reference\n(2 tokens)', 'Our Agent\n(4 tokens)']
    complexity = [2, 4]  # tokens per player
    episodes = [30000, 5000]  # training episodes
    
    x = np.arange(len(contexts))
    width = 0.35
    
    bars2a = ax2.bar(x - width/2, complexity, width, label='Tokens/Player', 
                     color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2b = ax2.bar(x + width/2, [e/10000 for e in episodes], width, 
                     label='Training (10k episodes)', 
                     color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Training Configuration Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(contexts)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add annotations
    for i, (c, e) in enumerate(zip(complexity, episodes)):
        ax2.text(i - width/2, c, str(c), ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(i + width/2, e/10000, f'{e//1000}k', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/selfplay_comparison_reference.png', dpi=300, bbox_inches='tight')
    print("Saved: results/selfplay_comparison_reference.png")
    plt.close()

def print_summary(results):
    """Print a text summary of the results."""
    print("\n" + "="*70)
    print("SELF-PLAY TRAINING RESULTS SUMMARY")
    print("="*70)
    
    config = results['config']
    print(f"\nConfiguration:")
    print(f"  State Abstraction: {config['state_abstraction']}")
    print(f"  Game Setup: {config['num_players']} players, {config['tokens_per_player']} tokens/player")
    print(f"  Training: Phase 1={config['phase1_episodes']}, Phase 2={config['phase2_episodes']}")
    print(f"  Hyperparameters: α={config['learning_rate']}, γ={config['discount_factor']}, ε={config['epsilon']}")
    
    print(f"\nPhase 1 Results (Baseline):")
    p1_eval = results['phase1_eval']
    print(f"  Win Rate vs Random: {p1_eval['win_rate']*100:.2f}%")
    print(f"  Record: {p1_eval['wins']}-{p1_eval['losses']}")
    print(f"  Avg Episode Length: {p1_eval['avg_episode_length']:.1f} moves")
    
    print(f"\nPhase 2 Results (After Self-Play):")
    p2_eval = results['phase2_eval_vs_random']
    p2_vs_p1 = results['phase2_eval_vs_phase1']
    print(f"  Win Rate vs Random: {p2_eval['win_rate']*100:.2f}%")
    print(f"  Record: {p2_eval['wins']}-{p2_eval['losses']}")
    print(f"  Win Rate vs Phase 1: {p2_vs_p1['win_rate']*100:.2f}%")
    print(f"  Avg Episode Length: {p2_eval['avg_episode_length']:.1f} moves")
    
    improvement = (p2_eval['win_rate'] - p1_eval['win_rate']) * 100
    print(f"\n{'='*70}")
    print(f"IMPROVEMENT: +{improvement:.2f}% absolute ({p1_eval['win_rate']*100:.1f}% → {p2_eval['win_rate']*100:.1f}%)")
    print(f"{'='*70}")
    
    print(f"\nComparison with Reference Repository:")
    print(f"  Reference Best (DQL): 70.96%")
    print(f"  Reference Q-Agent: 64.58%")
    print(f"  Our Agent: {p2_eval['win_rate']*100:.2f}%")
    print(f"  Gap to Q-Agent: {64.58 - p2_eval['win_rate']*100:.2f}%")
    print(f"  Note: Reference used 2 tokens/player, we use 4 (2x complexity)")
    print(f"  Note: Reference trained 30k episodes/phase, we used 5k (6x less)")
    
    print("\n" + "="*70)

def main():
    """Main function."""
    results = load_results()
    if results is None:
        return
    
    print_summary(results)
    
    print("\nGenerating visualizations...")
    plot_win_rates(results)
    plot_improvement(results)
    plot_episode_lengths(results)
    plot_comparison_with_reference(results)
    
    print("\n✅ All visualizations generated successfully!")
    print("📊 Check the 'results/' directory for PNG files.")

if __name__ == '__main__':
    main()

