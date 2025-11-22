import csv
import sys
import os

def mean(data):
    if not data:
        return 0.0
    return sum(data) / len(data)

def analyze_results(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    episodes = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                row['won'] = row['won'] == 'True'
                row['total_reward'] = float(row['total_reward'])
                row['episode_length'] = int(row['episode_length'])
                
                # These appear to be cumulative in the CSV
                row['total_steps_cum'] = int(row['total_steps']) if row['total_steps'] else 0
                row['context_trailing_cum'] = int(row['context_trailing']) if row['context_trailing'] else 0
                row['context_neutral_cum'] = int(row['context_neutral']) if row['context_neutral'] else 0
                row['context_leading_cum'] = int(row['context_leading']) if row['context_leading'] else 0
                
                episodes.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    total_episodes = len(episodes)
    print(f"Total Episodes: {total_episodes}")
    print("-" * 30)

    # Overall Stats
    wins = sum(1 for ep in episodes if ep['won'])
    overall_win_rate = (wins / total_episodes) * 100 if total_episodes > 0 else 0
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")
    
    # Bucket Analysis
    window = 1000
    num_buckets = total_episodes // window
    
    print(f"\nAnalysis per {window} episodes:")
    print(f"{'Episode Range':<20} | {'Win Rate':<10} | {'Avg Reward':<12} | {'Avg Length':<10}")
    print("-" * 60)
    
    bucket_win_rates = []

    for i in range(num_buckets):
        start_idx = i * window
        end_idx = (i + 1) * window
        bucket = episodes[start_idx:end_idx]
        
        bucket_wins = sum(1 for ep in bucket if ep['won'])
        win_rate = (bucket_wins / len(bucket)) * 100
        
        avg_reward = mean([ep['total_reward'] for ep in bucket])
        avg_length = mean([ep['episode_length'] for ep in bucket])
        
        bucket_win_rates.append(win_rate)

        range_str = f"{start_idx}-{end_idx}"
        print(f"{range_str:<20} | {win_rate:6.2f}%   | {avg_reward:10.2f}   | {avg_length:8.2f}")

    # Last 1000 episodes detailed
    last_1000 = episodes[-1000:] if total_episodes >= 1000 else episodes
    
    if last_1000:
        last_wins = sum(1 for ep in last_1000 if ep['won'])
        last_win_rate = (last_wins / len(last_1000)) * 100
        last_avg_reward = mean([ep['total_reward'] for ep in last_1000])
        last_avg_length = mean([ep['episode_length'] for ep in last_1000])

        print("\nLast 1000 Episodes Stats:")
        print(f"Win Rate: {last_win_rate:.2f}%")
        print(f"Average Reward: {last_avg_reward:.2f}")
        print(f"Average Length: {last_avg_length:.2f}")
        
        # Context Analysis for Last 1000 using cumulative difference
        start_ep = episodes[-1001] if len(episodes) > 1000 else episodes[0]
        end_ep = episodes[-1]
        
        diff_trailing = end_ep['context_trailing_cum'] - start_ep['context_trailing_cum']
        diff_neutral = end_ep['context_neutral_cum'] - start_ep['context_neutral_cum']
        diff_leading = end_ep['context_leading_cum'] - start_ep['context_leading_cum']
        
        total_context_steps = diff_trailing + diff_neutral + diff_leading
        
        if total_context_steps > 0:
            print("\nContext Distribution (Last 1000 episodes):")
            print(f"Trailing: {diff_trailing / total_context_steps * 100:.2f}%")
            print(f"Neutral:  {diff_neutral / total_context_steps * 100:.2f}%")
            print(f"Leading:  {diff_leading / total_context_steps * 100:.2f}%")

    # Check improvement trend in the last few buckets
    if len(bucket_win_rates) >= 2:
        last_rate = bucket_win_rates[-1]
        prev_rate = bucket_win_rates[-2]
        diff = last_rate - prev_rate
        print(f"\nTrend (Last vs Prev bucket win rate): {'Improving' if diff > 0 else 'Declining'} ({diff:+.2f}%)")

if __name__ == "__main__":
    csv_path = "/home/loadingname/RLagentLudo/results/q_learning_context_aware_20251121_234620/q_learning_context_aware_episodes.csv"
    analyze_results(csv_path)

