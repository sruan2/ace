#!/usr/bin/env python3
"""
Stream Bench task runner using ACE system.
"""
import os
import sys
import json
import re
import time
import traceback

from ace import ACE
from .data_processor import DataProcessor
from .plot import plot_online_performance, plot_training_progress
from finance.run import get_base_parser, load_initial_playbook, load_data


def parse_args():
    """Parse command line arguments for stream-bench."""
    # Get base parser with all common arguments
    parser = get_base_parser(description='ACE System - Stream Bench')

    # Add stream-bench specific arguments
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to data configuration JSON file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate performance plot for online mode (shows accuracy vs steps)")
    parser.add_argument("--db_name", type=str, default=None,
                        help="Database name to filter data (optional, overrides config)")
    parser.add_argument("--curriculum", type=str, default=None, choices=["easy_to_hard", "hard_to_easy", "random"],
                        help="Curriculum ordering strategy: easy_to_hard, hard_to_easy, random")

    return parser.parse_args()


class TeeLogger:
    """Logger that writes to both terminal and file simultaneously with auto-flush."""

    def __init__(self, log_file_path, mode='w'):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, mode, buffering=1)  # Line buffering
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
        

def preprocess_data(task_name, config, mode, db_name=None, curriculum=None):
    """
    Load training and test data for the specified task.

    Args:
        task_name: Name of the task
        config: Configuration dictionary with data paths and settings
        mode: Run mode ('offline', 'online', or 'eval_only')
        db_name: Database name from command line args
        curriculum: Curriculum ordering from command line args

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

    # Get difficulty_filter from config (dataset-level selection)
    difficulty_filter = config.get("difficulty_filter", None)

    processor = DataProcessor(
        bird_db_root=bird_db_root,
        max_samples=max_samples,
        db_name=db_name,
        difficulty_filter=difficulty_filter,
        curriculum=curriculum
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

    # Create temporary log directory to capture all output from the start
    temp_log_dir = os.path.join(args.save_path, "temp_logs")
    os.makedirs(temp_log_dir, exist_ok=True)
    log_timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_log_path = os.path.join(temp_log_dir, f"terminal_output_{log_timestamp}.txt")

    # Set up logger immediately to capture ALL output
    logger = TeeLogger(temp_log_path)
    sys.stdout = logger

    # Print initial banner (now captured by logger)
    print(f"\n{'='*60}")
    print(f"ACE SYSTEM - Stream Bench")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper().replace('_', ' ')}")
    print(f"Generator Model: {args.generator_model}")
    print(f"Data Config: {args.data_config}")
    print(f"Logging all terminal output to: {temp_log_path}")
    print(f"{'='*60}\n")

    try:

        # Load data configuration
        with open(args.data_config, 'r') as f:
            data_config = json.load(f)

        # Get task-specific config
        if args.task_name not in data_config:
            raise ValueError(f"Task '{args.task_name}' not found in config file: {args.data_config}")

        task_config = data_config[args.task_name]

        # Print config settings
        if "max_samples" in task_config:
            print(f"Max samples (from config): {task_config['max_samples']}")
        else:
            print(f"Max samples: No limit")

        if args.db_name:
            print(f"Database filter: {args.db_name}")
        else:
            print(f"Database filter: None (using mixed databases)")

        if "difficulty_filter" in task_config:
            print(f"Difficulty filter (from config): {task_config['difficulty_filter']}")
        else:
            print(f"Difficulty filter: None (no filtering)")

        if args.curriculum:
            print(f"Curriculum ordering: {args.curriculum}")
        else:
            print(f"Curriculum ordering: None (original order)")

        print()  # blank line

        train_samples, val_samples, test_samples, data_processor = preprocess_data(
            args.task_name,
            task_config,
            args.mode,
            db_name=args.db_name,
            curriculum=args.curriculum
        )

        # Load initial playbook (or use empty if None provided)
        initial_playbook = load_initial_playbook(args.initial_playbook_path)
        if initial_playbook:
            print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
        else:
            print("Using empty playbook as initial playbook\n")

        # Create ACE system with a custom wrapper to intercept path creation
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
            'save_dir': args.save_path,  # Pass parent directory
            'test_workers': args.test_workers,
            'initial_playbook_path': args.initial_playbook_path,
            'use_bulletpoint_analyzer': args.use_bulletpoint_analyzer,
            'bulletpoint_analyzer_threshold': args.bulletpoint_analyzer_threshold,
            'pass_sql_eval_results': args.pass_sql_eval_results,
            'api_provider': args.api_provider,
            'config_name': config_filename,
            'db_name': args.db_name,
            'curriculum': args.curriculum
        }

        # Create a save hook to intercept when ACE creates the save path
        original_setup_paths = ace_system._setup_paths
        run_save_path_container = {'path': None}

        def setup_paths_with_data_save(*args, **kwargs):
            """Wrapper that saves processed data right after path creation."""
            result = original_setup_paths(*args, **kwargs)
            # Extract save_path from result (first element of tuple)
            save_path = result[0] if isinstance(result, tuple) else result
            run_save_path_container['path'] = save_path

            # Save processed data immediately after folder creation
            print(f"\nSaving preprocessed data to: {save_path}")
            processed_data_dir = os.path.join(save_path, "processed_data")
            os.makedirs(processed_data_dir, exist_ok=True)

            if train_samples is not None:
                train_path = os.path.join(processed_data_dir, "train_samples.json")
                with open(train_path, 'w') as f:
                    json.dump(train_samples, f, indent=2)
                print(f"  - Saved train samples ({len(train_samples)} samples)")

            if val_samples is not None:
                val_path = os.path.join(processed_data_dir, "val_samples.json")
                with open(val_path, 'w') as f:
                    json.dump(val_samples, f, indent=2)
                print(f"  - Saved val samples ({len(val_samples)} samples)")

            if test_samples is not None:
                test_path = os.path.join(processed_data_dir, "test_samples.json")
                with open(test_path, 'w') as f:
                    json.dump(test_samples, f, indent=2)
                print(f"  - Saved test samples ({len(test_samples)} samples)")

            print()  # blank line
            return result

        # Replace the method temporarily
        ace_system._setup_paths = setup_paths_with_data_save

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

        # Get the actual save path that was created
        run_save_path = run_save_path_container['path']
        if not run_save_path:
            # Fallback to results if something went wrong with the hook
            run_save_path = results.get('save_path', args.save_path)

        # Extract timestamp from the ace_run folder name to match it exactly
        # Folder format: ace_run_YYYYMMDD_HHMMSS_task_name_...
        folder_name = os.path.basename(run_save_path)
        timestamp_match = re.search(r'ace_run_(\d{8}_\d{6})', folder_name)
        if timestamp_match:
            ace_run_timestamp = timestamp_match.group(1)
        else:
            # Fallback to original timestamp if extraction fails
            ace_run_timestamp = log_timestamp

        # Move the log file from temp location to final location
        final_log_path = os.path.join(run_save_path, f"terminal_output_{ace_run_timestamp}.txt")

        # Close current logger before moving file
        logger.close()

        # Move the log file to final location
        import shutil
        shutil.move(temp_log_path, final_log_path)

        # Reopen logger with final path in APPEND mode to continue logging
        logger = TeeLogger(final_log_path, mode='a')
        sys.stdout = logger

        print(f"\nMoved terminal output log to: {final_log_path}")

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

        # Generate performance plots if requested (online mode only)
        if args.plot:
            print(f"\n{'='*60}")
            print(f"GENERATING PERFORMANCE PLOTS")
            print(f"{'='*60}\n")
            plot_online_performance(run_save_path, args.mode)
            plot_training_progress(run_save_path, args.mode)

        # Close the logger
        if logger:
            # Print before closing since we're using the logger
            final_message = f"Terminal output saved to {final_log_path}"
            logger.close()
            # Print to terminal after logger is closed
            print(final_message)

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: An exception occurred")
        print(f"{'='*60}")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise
    finally:
        # Ensure logger is closed even if there's an error
        if 'logger' in locals() and logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
