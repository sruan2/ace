#!/usr/bin/env python3
"""
Stream Bench task runner using ACE system.
"""
import os
import sys
import json
import time
import matplotlib.pyplot as plt
from .data_processor import DataProcessor

from ace import ACE
from finance.run import parse_args, load_initial_playbook, load_data


def plot_online_performance(results, save_path, mode):
    """
    Generate performance plots for online mode showing how accuracy changes over steps.

    Args:
        results: Results dictionary from ACE run
        save_path: Path to save the plot
        mode: Run mode (should be 'online')
    """
    if mode != 'online':
        print(f"Skipping plot generation - only available for online mode (current mode: {mode})")
        return

    # Extract data from results
    if 'online_test_results' not in results:
        print("Warning: No online test results found. Skipping plot generation.")
        return

    test_results = results['online_test_results']

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

    # Save plot
    plot_path = os.path.join(save_path, 'online_performance_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance plot saved to: {plot_path}")
    plt.close()

    # Also save the data as CSV for external plotting
    csv_path = os.path.join(save_path, 'online_performance_data.csv')
    with open(csv_path, 'w') as f:
        f.write("window,window_accuracy,start_idx,end_idx,samples_in_window\n")
        for w in window_results:
            f.write(f"{w['window']},{w['window_accuracy']},{w['start_idx']},{w['end_idx']},{w['window_total']}\n")
    print(f"Performance data saved to: {csv_path}")


class TeeLogger:
    """Logger that writes to both terminal and file simultaneously with auto-flush."""

    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, 'w', buffering=1)  # Line buffering
        self.log_file_path = log_file_path

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        # Force flush to ensure immediate write
        self.terminal.flush()
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        self.log_file.close()

def preprocess_data(task_name, config, mode):
    """
    Load training and test data for the specified task.

    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths and settings
        mode: Run mode ('offline', 'online', or 'eval_only')

    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
        - For offline mode: all three are loaded
        - For online mode: only test_samples
        - For eval_only mode: only test_samples
    """
    # Get max_samples from config, default to None (no limit)
    max_samples = config.get("max_samples", None)

    # Get bird_db_root from config, with default
    bird_db_root = config.get("bird_db_root", "stream-bench/data/bird/dev_databases")

    # Get db_name from config, with default
    db_name = config.get("db_name", "financial")

    processor = DataProcessor(
        bird_db_root=bird_db_root,
        max_samples=max_samples,
        db_name=db_name
    )


    # For online and eval_only modes, only load test data
    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            raise ValueError(f"{mode} mode requires test data in config.")

        if mode == "online":
            print(f"Online mode: Training and testing on {len(test_samples)} examples")
        else:
            print(f"Eval only mode: Testing on {len(test_samples)} examples")

    # For offline mode, load train, val, and optionally test data
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(f"Offline mode: Training on {len(train_samples)} examples, "
              f"validating on {len(val_samples)}, testing on {len(test_samples)}")

    return train_samples, val_samples, test_samples, processor


def main():
    """Main execution function."""
    # Start total timing
    total_start_time = time.time()

    args = parse_args()

    # Print initial banner (before logger setup)
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM - Stream Bench")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Data Config: {args.data_config}")
    print(f"{'='*60}\n")

    logger = None
    try:

        # Load data configuration
        with open(args.data_config, 'r') as f:
            data_config = json.load(f)

        # Get task-specific config
        if args.task_name not in data_config:
            raise ValueError(f"Task '{args.task_name}' not found in config file: {args.data_config}")

        task_config = data_config[args.task_name]

        # Print max_samples if specified in config
        if "max_samples" in task_config:
            print(f"Max samples (from config): {task_config['max_samples']}\n")
        else:
            print(f"Max samples: No limit\n")

        train_samples, val_samples, test_samples, data_processor = preprocess_data(
            args.task_name,
            task_config,
            args.mode
        )

        # Load initial playbook (or use empty if None provided)
        initial_playbook = load_initial_playbook(args.initial_playbook_path)
        if initial_playbook:
            print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
        else:
            print("Using empty playbook as initial playbook\n")

        # Create ACE system
        ace_system = ACE(
            api_provider=args.api_provider,
            generator_model=args.generator_model,
            reflector_model=args.reflector_model,
            curator_model=args.curator_model,
            max_tokens=args.max_tokens,
            initial_playbook=initial_playbook,
            use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
            bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold
        )

        # Extract config filename (without extension) from config path
        config_filename = os.path.splitext(os.path.basename(args.data_config))[0]

        # Prepare configuration
        config = {
            'num_epochs': args.num_epochs,
            'max_num_rounds': args.max_num_rounds,
            'curator_frequency': args.curator_frequency,
            'eval_steps': args.eval_steps,
            'online_eval_frequency': args.online_eval_frequency,
            'save_steps': args.save_steps,
            'playbook_token_budget': args.playbook_token_budget,
            'task_name': args.task_name,
            'mode': args.mode,
            'json_mode': args.json_mode,
            'no_ground_truth': args.no_ground_truth,
            'save_dir': args.save_path,
            'test_workers': args.test_workers,
            'initial_playbook_path': args.initial_playbook_path,
            'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
            'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
            'api_provider': args.api_provider,
            'config_name': config_filename
        }

        # Execute using the unified run method
        print(f"Starting ACE run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        run_start_time = time.time()

        results = ace_system.run(
            mode=args.mode,
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            data_processor=data_processor,
            config=config
        )

        run_elapsed_time = time.time() - run_start_time
        print(f"\nACE run completed in {run_elapsed_time/60:.2f} minutes ({run_elapsed_time:.2f} seconds)")

        # Save preprocessed data to individual run folder
        run_save_path = results.get('save_path', args.save_path)

        # Now set up logger to capture remaining output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(run_save_path, f"terminal_output_{timestamp}.txt")
        logger = TeeLogger(log_file_path)
        sys.stdout = logger

        print(f"Logging terminal output to: {log_file_path}\n")

        # Create processed_data subfolder
        processed_data_dir = os.path.join(run_save_path, "processed_data")
        os.makedirs(processed_data_dir, exist_ok=True)

        if train_samples is not None:
            train_path = os.path.join(processed_data_dir, "train_samples.json")
            with open(train_path, 'w') as f:
                json.dump(train_samples, f, indent=2)
            print(f"Saved train samples to {train_path}")

        if val_samples is not None:
            val_path = os.path.join(processed_data_dir, "val_samples.json")
            with open(val_path, 'w') as f:
                json.dump(val_samples, f, indent=2)
            print(f"Saved val samples to {val_path}")

        if test_samples is not None:
            test_path = os.path.join(processed_data_dir, "test_samples.json")
            with open(test_path, 'w') as f:
                json.dump(test_samples, f, indent=2)
            print(f"Saved test samples to {test_path}")

        # Calculate and display total timing
        total_elapsed_time = time.time() - total_start_time

        print(f"\n{'='*60}")
        print(f"TOTAL EXECUTION TIME")
        print(f"{'='*60}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")
        print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {total_elapsed_time/60:.2f} minutes ({total_elapsed_time:.2f} seconds)")
        print(f"ACE run time: {run_elapsed_time/60:.2f} minutes ({run_elapsed_time:.2f} seconds)")
        print(f"{'='*60}\n")

        # Generate performance plot if requested (online mode only)
        if args.plot_online_performance:
            print(f"\n{'='*60}")
            print(f"GENERATING PERFORMANCE PLOT")
            print(f"{'='*60}\n")
            plot_online_performance(results, run_save_path, args.mode)

        # Close the logger
        if logger:
            logger.close()
            print(f"Terminal output saved to {log_file_path}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: An exception occurred")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise
    finally:
        # Ensure logger is closed even if there's an error
        if 'logger' in locals():
            logger.close()


if __name__ == "__main__":
    main()
