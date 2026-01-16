#!/usr/bin/env python3
"""
Plotting utilities for stream-bench online training results.

Generates two types of plots:
1. Test Performance: Shows accuracy progression across test windows
2. Training Progress: Shows pre/post train accuracy, improvement, and playbook growth

Can be run as a standalone script:
    python stream-bench/plot.py --run_dir results/ace_run_20260115_213336_bird_online

Or imported and used programmatically:
    from plot import plot_online_performance, plot_training_progress
    plot_online_performance(save_path, mode)
    plot_training_progress(save_path, mode)
"""
import os
import sys
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for saving plots without display
import matplotlib.pyplot as plt


def plot_online_performance(save_path, mode='online'):
    """
    Generate performance plots for online mode showing how accuracy changes over steps.

    Args:
        save_path: Path where results are saved and where plot will be saved
        mode: Run mode (should be 'online', default: 'online')
    """
    if mode != 'online':
        print(f"Skipping plot generation - only available for online mode (current mode: {mode})")
        return

    # Load test results from the saved JSON file (which contains window_results)
    test_results_path = os.path.join(save_path, 'test_results.json')
    if not os.path.exists(test_results_path):
        print(f"Warning: Test results file not found at {test_results_path}. Skipping plot generation.")
        return

    with open(test_results_path, 'r') as f:
        test_data = json.load(f)

    # Extract test_results from the loaded JSON
    if 'test_results' not in test_data:
        print("Warning: No test results found in test_results.json. Skipping plot generation.")
        return

    test_results = test_data['test_results']

    # Check if we have window results
    if 'window_results' not in test_results:
        print("Warning: No window results found. Skipping plot generation.")
        return

    window_results = test_results['window_results']

    # Extract window data
    window_numbers = [w['window'] for w in window_results]
    window_accuracies = [w['window_accuracy'] for w in window_results]
    window_end_indices = [w['end_idx'] for w in window_results]

    # Create figure with multiple subplots
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Accuracy by Window
    ax1.plot(window_numbers, window_accuracies, 'b-o', linewidth=2, markersize=8, label='Window Accuracy')
    ax1.axhline(y=test_results['accuracy'], color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy: {test_results["accuracy"]:.3f}')
    ax1.set_xlabel('Window Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Online Mode: Accuracy by Training Window', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.0])

    # Add value labels on points
    for x, y in zip(window_numbers, window_accuracies):
        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 2: Accuracy by Sample Index (cumulative)
    ax2.plot(window_end_indices, window_accuracies, 'g-s', linewidth=2, markersize=8, label='Accuracy')
    ax2.axhline(y=test_results['accuracy'], color='r', linestyle='--', linewidth=2, label=f'Overall Accuracy: {test_results["accuracy"]:.3f}')
    ax2.set_xlabel('Sample Index (End of Window)', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Online Mode: Accuracy by Sample Progress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()

    # Create plots subfolder
    plots_dir = os.path.join(save_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plots_dir, 'online_performance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {plot_path}")
    plt.close()

    # Also save the data as CSV for external plotting
    csv_path = os.path.join(plots_dir, 'online_performance_data.csv')
    with open(csv_path, 'w') as f:
        f.write("window,window_accuracy,start_idx,end_idx,samples_in_window\n")
        for w in window_results:
            f.write(f"{w['window']},{w['window_accuracy']},{w['start_idx']},{w['end_idx']},{w['window_total']}\n")
    print(f"Performance data saved to: {csv_path}")


def plot_training_progress(save_path, mode='online'):
    """
    Generate training progress plots showing pre/post train accuracy and playbook growth.

    Args:
        save_path: Path where results are saved and where plot will be saved
        mode: Run mode (should be 'online', default: 'online')
    """
    if mode != 'online':
        print(f"Skipping training plot generation - only available for online mode (current mode: {mode})")
        return

    # Load training results from the saved JSON file
    train_results_path = os.path.join(save_path, 'train_results.json')
    if not os.path.exists(train_results_path):
        print(f"Warning: Training results file not found at {train_results_path}. Skipping training plot generation.")
        return

    with open(train_results_path, 'r') as f:
        train_data = json.load(f)

    # Extract train_results from the loaded JSON
    if 'train_results' not in train_data:
        print("Warning: No train results found in train_results.json. Skipping training plot generation.")
        return

    train_results = train_data['train_results']

    if not train_results:
        print("Warning: Empty train results. Skipping training plot generation.")
        return

    # Extract data from train_results
    windows = [r['window'] for r in train_results]
    pre_train_acc = [r['train_result']['pre_train_accuracy'] for r in train_results]
    post_train_acc = [r['train_result']['post_train_accuracy'] for r in train_results]
    cumulative_test_acc = [r['cumulative_test_accuracy'] for r in train_results]
    playbook_tokens = [r['playbook_num_tokens'] for r in train_results]
    playbook_length = [r['playbook_length'] for r in train_results]

    # Calculate improvement per window
    improvement = [post - pre for pre, post in zip(pre_train_acc, post_train_acc)]

    # Create figure with multiple subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Pre-train vs Post-train Accuracy by Window
    ax1.plot(windows, pre_train_acc, 'r-o', linewidth=2, markersize=8, label='Pre-train Accuracy')
    ax1.plot(windows, post_train_acc, 'g-s', linewidth=2, markersize=8, label='Post-train Accuracy')
    ax1.plot(windows, cumulative_test_acc, 'b--^', linewidth=2, markersize=8, label='Cumulative Test Accuracy')
    ax1.set_xlabel('Window Number', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Progress: Pre-train vs Post-train Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1.0])

    # Plot 2: Training Improvement per Window
    colors = ['green' if x >= 0 else 'red' for x in improvement]
    ax2.bar(windows, improvement, color=colors, alpha=0.6, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Window Number', fontsize=12)
    ax2.set_ylabel('Accuracy Improvement', fontsize=12)
    ax2.set_title('Training Improvement per Window (Post - Pre)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (w, imp) in enumerate(zip(windows, improvement)):
        ax2.text(w, imp, f'{imp:+.3f}', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9)

    # Plot 3: Playbook Token Growth
    ax3.plot(windows, playbook_tokens, 'purple', marker='D', linewidth=2, markersize=8, label='Playbook Tokens')
    ax3.set_xlabel('Window Number', fontsize=12)
    ax3.set_ylabel('Number of Tokens', fontsize=12)
    ax3.set_title('Playbook Growth: Token Count', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    # Add value labels
    for w, tokens in zip(windows, playbook_tokens):
        ax3.annotate(f'{tokens}', (w, tokens), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Plot 4: Playbook Character Length Growth
    ax4.plot(windows, playbook_length, 'orange', marker='D', linewidth=2, markersize=8, label='Playbook Length (chars)')
    ax4.set_xlabel('Window Number', fontsize=12)
    ax4.set_ylabel('Character Count', fontsize=12)
    ax4.set_title('Playbook Growth: Character Length', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)

    # Add value labels
    for w, length in zip(windows, playbook_length):
        ax4.annotate(f'{length}', (w, length), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout()

    # Create plots subfolder
    plots_dir = os.path.join(save_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save plot
    plot_path = os.path.join(plots_dir, 'training_progress_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining progress plot saved to: {plot_path}")
    plt.close()

    # Also save the data as CSV for external plotting
    csv_path = os.path.join(plots_dir, 'training_progress_data.csv')
    with open(csv_path, 'w') as f:
        f.write("window,pre_train_accuracy,post_train_accuracy,cumulative_test_accuracy,improvement,playbook_tokens,playbook_length\n")
        for i, w in enumerate(windows):
            f.write(f"{w},{pre_train_acc[i]},{post_train_acc[i]},{cumulative_test_acc[i]},{improvement[i]},{playbook_tokens[i]},{playbook_length[i]}\n")
    print(f"Training progress data saved to: {csv_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Generate performance plots for ACE online training runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot results from a specific run directory
  python stream-bench/plot.py --run_dir results/ace_run_20260115_213336_bird_online

  # Specify mode explicitly (default is 'online')
  python stream-bench/plot.py --run_dir results/ace_run_20260115_213336_bird_online --mode online
        """
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='Path to the ACE run directory containing test_results.json'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='online',
        choices=['online', 'offline', 'eval_only'],
        help='Run mode (default: online). Only online mode supports plotting.'
    )

    args = parser.parse_args()

    # Validate run directory exists
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Check if test_results.json exists
    test_results_path = os.path.join(args.run_dir, 'test_results.json')
    if not os.path.exists(test_results_path):
        print(f"Error: test_results.json not found in {args.run_dir}")
        print(f"Expected path: {test_results_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"GENERATING PLOTS FOR ACE RUN")
    print(f"{'='*60}")
    print(f"Run directory: {args.run_dir}")
    print(f"Mode: {args.mode}")
    print(f"{'='*60}\n")

    # Generate test performance plots
    print("Generating test performance plots...")
    plot_online_performance(args.run_dir, args.mode)

    # Generate training progress plots
    print("\nGenerating training progress plots...")
    plot_training_progress(args.run_dir, args.mode)

    print(f"\n{'='*60}")
    print(f"PLOTTING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
