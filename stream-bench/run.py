#!/usr/bin/env python3
"""
Stream Bench task runner using ACE system.
"""
import os
import sys
import json
import time
import shutil
from .data_processor import DataProcessor

from ace import ACE
from finance.run import parse_args, load_initial_playbook, load_data


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
        config: Configuration dictionary with data paths
        mode: Run mode ('offline', 'online', or 'eval_only')

    Returns:
        Tuple of (train_samples, val_samples, test_samples, data_processor)
        - For offline mode: all three are loaded
        - For online mode: only test_samples
        - For eval_only mode: only test_samples
    """
    processor = DataProcessor(
        bird_db_root="stream-bench/data/bird/dev_databases",  # where dev_databases.zip is extracted
        max_samples=100,              # default anyway
        db_name="financial"
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

    # Create a timestamped log file in the save directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_log_dir = os.path.join(args.save_path, "logs")
    os.makedirs(temp_log_dir, exist_ok=True)
    log_file_path = os.path.join(temp_log_dir, f"run_log_{timestamp}.txt")

    # Set up the logger to capture all stdout
    logger = TeeLogger(log_file_path)
    sys.stdout = logger

    try:
        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - Stream Bench")
        print(f"{'='*60}")
        print(f"Task: {args.task_name}")
        print(f"Mode: {args.mode.upper().replace('_', ' ')}")
        print(f"Generator Model: {args.generator_model}")
        print(f"Log file: {log_file_path}")
        print(f"{'='*60}\n")

        # Load data
        with open("./stream-bench/data/sample_config.json", 'r') as f:
            task_config = json.load(f)

        train_samples, val_samples, test_samples, data_processor = preprocess_data(
            args.task_name,
            task_config[args.task_name],
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
            'api_provider': args.api_provider
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

        if train_samples is not None:
            train_path = os.path.join(run_save_path, "train_samples.json")
            with open(train_path, 'w') as f:
                json.dump(train_samples, f, indent=2)
            print(f"Saved train samples to {train_path}")

        if val_samples is not None:
            val_path = os.path.join(run_save_path, "val_samples.json")
            with open(val_path, 'w') as f:
                json.dump(val_samples, f, indent=2)
            print(f"Saved val samples to {val_path}")

        if test_samples is not None:
            test_path = os.path.join(run_save_path, "test_samples.json")
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

        # Move log file to the actual run folder
        final_log_path = os.path.join(run_save_path, f"terminal_output_{timestamp}.txt")
        logger.close()
        shutil.move(log_file_path, final_log_path)

        # Print to original terminal
        print(f"Terminal output saved to {final_log_path}")

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
