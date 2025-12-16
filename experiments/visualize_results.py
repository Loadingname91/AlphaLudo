"""
Visualize evaluation results for all trained models.
Creates comprehensive plots and charts for presentation.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_win_rates(results, output_dir):
    """Plot win rates across all levels."""
    levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    if 'level6' in results['levels']:
        levels.append('level6')

    level_names = [results['levels'][l]['level'] for l in levels]
    win_rates = [results['levels'][l]['win_rate'] * 100 for l in levels]

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(levels))

    bars = ax.bar(x, win_rates, label='Win Rate', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Curriculum Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Curriculum Learning Results: Win Rates Across All Levels',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'win_rates.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'win_rates.png'}")
    plt.close()


def plot_rewards(results, output_dir):
    """Plot average rewards with error bars."""
    levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    if 'level6' in results['levels']:
        levels.append('level6')

    level_names = [results['levels'][l]['level'] for l in levels]
    avg_rewards = [results['levels'][l]['avg_reward'] for l in levels]
    std_rewards = [results['levels'][l]['std_reward'] for l in levels]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(levels))

    bars = ax.bar(x, avg_rewards, color='#9b59b6', alpha=0.7, yerr=std_rewards,
                   capsize=10, error_kw={'linewidth': 2, 'ecolor': '#34495e'})

    ax.set_xlabel('Curriculum Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Episode Reward', fontsize=12, fontweight='bold')
    ax.set_title('Average Reward Performance Across Curriculum Levels',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1, alpha=0.5)

    # Add value labels
    for i, (bar, reward, std) in enumerate(zip(bars, avg_rewards, std_rewards)):
        ax.text(bar.get_x() + bar.get_width()/2., reward + std + 20,
               f'{reward:.0f}±{std:.0f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'rewards.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'rewards.png'}")
    plt.close()


def plot_episode_lengths(results, output_dir):
    """Plot average episode lengths."""
    levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    if 'level6' in results['levels']:
        levels.append('level6')

    level_names = [results['levels'][l]['level'] for l in levels]
    avg_lengths = [results['levels'][l]['avg_length'] for l in levels]
    std_lengths = [results['levels'][l]['std_length'] for l in levels]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(levels))

    bars = ax.bar(x, avg_lengths, color='#e74c3c', alpha=0.7, yerr=std_lengths,
                   capsize=10, error_kw={'linewidth': 2, 'ecolor': '#34495e'})

    ax.set_xlabel('Curriculum Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Episode Length (steps)', fontsize=12, fontweight='bold')
    ax.set_title('Game Efficiency: Average Episode Length Across Levels',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(level_names)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bar, length, std in zip(bars, avg_lengths, std_lengths):
        ax.text(bar.get_x() + bar.get_width()/2., length + std + 2,
               f'{length:.1f}±{std:.1f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'episode_lengths.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'episode_lengths.png'}")
    plt.close()


def plot_comprehensive_dashboard(results, output_dir):
    """Create a comprehensive dashboard with all metrics."""
    levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    if 'level6' in results['levels']:
        levels.append('level6')

    level_names = ['L1', 'L2', 'L3', 'L4', 'L5']
    if 'level6' in results['levels']:
        level_names.append('L6')

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Win Rate Progression
    ax1 = fig.add_subplot(gs[0, :2])
    win_rates = [results['levels'][l]['win_rate'] * 100 for l in levels]

    x = np.arange(len(levels))
    ax1.plot(x, win_rates, marker='o', linewidth=3, markersize=10,
             color='#2ecc71', label='Win Rate')
    ax1.set_xlabel('Level', fontweight='bold')
    ax1.set_ylabel('Win Rate (%)', fontweight='bold')
    ax1.set_title('Win Rate Progression Across Levels', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(level_names)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. Rewards Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    rewards = [results['levels'][l]['avg_reward'] for l in levels]
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    ax2.barh(level_names, rewards, color=colors, alpha=0.7)
    ax2.set_xlabel('Avg Reward', fontweight='bold')
    ax2.set_title('Average Rewards', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='-', linewidth=1)

    # 3. Episode Length Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    lengths = [results['levels'][l]['avg_length'] for l in levels]
    ax3.bar(level_names, lengths, color='#34495e', alpha=0.7)
    ax3.set_ylabel('Steps', fontweight='bold')
    ax3.set_title('Episode Length', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # 4. Performance Summary Table
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    table_data.append(['Level', 'Win Rate', 'Avg Reward', 'Avg Length', 'Wins'])

    for i, level in enumerate(levels):
        data = results['levels'][level]
        wr = data['win_rate'] * 100
        table_data.append([
            data['level'],
            f"{wr:.1f}%",
            f"{data['avg_reward']:.0f}",
            f"{data['avg_length']:.1f}",
            f"{data['wins']}/{data['num_episodes']}"
        ])

    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.2, 0.2, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows with alternating colors
    for i in range(1, 6):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f8f9fa')

    ax4.set_title('Performance Summary', fontweight='bold', pad=20, fontsize=12)

    # Overall title
    num_levels = len(levels)
    fig.suptitle(f'Curriculum Learning Results Dashboard - All {num_levels} Levels',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'dashboard.png'}")
    plt.close()


def create_summary_report(results, output_dir):
    """Create a text summary report."""
    report = []
    report.append("="*80)
    report.append("COMPREHENSIVE EVALUATION RESULTS")
    report.append(f"Evaluation Date: {results['evaluation_date']}")
    report.append(f"Episodes per Level: {results['num_eval_episodes']}")
    report.append("="*80)
    report.append("")

    levels = ['level1', 'level2', 'level3', 'level4', 'level5']
    if 'level6' in results['levels']:
        levels.append('level6')

    for level in levels:
        data = results['levels'][level]
        wr = data['win_rate'] * 100

        report.append(f"{data['level']}")
        report.append("-" * 40)
        report.append(f"  Win Rate: {wr:.1f}%")
        report.append(f"  Wins: {data['wins']}/{data['num_episodes']}")
        report.append(f"  Avg Reward: {data['avg_reward']:.1f} ± {data['std_reward']:.1f}")
        report.append(f"  Avg Length: {data['avg_length']:.1f} ± {data['std_length']:.1f} steps")

        if 'avg_captures_by_agent' in data:
            report.append(f"  Captures by agent: {data['avg_captures_by_agent']:.2f}/episode")
            report.append(f"  Captures of agent: {data['avg_captures_of_agent']:.2f}/episode")

        report.append("")

    report.append("="*80)
    report.append("OVERALL SUMMARY")
    report.append("="*80)

    avg_win_rate = np.mean([results['levels'][l]['win_rate'] * 100 for l in levels])
    report.append(f"Average Win Rate Across All Levels: {avg_win_rate:.1f}%")

    total_wins = sum(results['levels'][l]['wins'] for l in levels)
    total_episodes = sum(results['levels'][l]['num_episodes'] for l in levels)
    report.append(f"Total Wins: {total_wins}/{total_episodes}")
    report.append("")

    report_text = "\n".join(report)

    # Save to file
    with open(output_dir / 'evaluation_summary.txt', 'w') as f:
        f.write(report_text)

    print(f"✓ Saved: {output_dir / 'evaluation_summary.txt'}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='Visualize evaluation results')
    parser.add_argument('--results', type=str,
                       default='results/evaluations/all_models_evaluation_20251209_040011.json',
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str,
                       default='results/visualizations',
                       help='Output directory for plots')

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Generate plots
    plot_win_rates(results, output_dir)
    plot_rewards(results, output_dir)
    plot_episode_lengths(results, output_dir)
    plot_comprehensive_dashboard(results, output_dir)

    # Generate text summary
    create_summary_report(results, output_dir)

    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
