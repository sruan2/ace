#!/usr/bin/env python3
"""
Script to run a playbook from intermediate_playbooks folder on processed data
and evaluate accuracy.

Usage:
    python run_playbook.py --results_dir <path> --playbook_file <filename>

Example:
    python run_playbook.py --results_dir results/ace_run_20260119_234301_bird_all_hard_to_easy_online --playbook_file window_4_final_playbook.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add stream-bench to path
stream_bench_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, stream_bench_dir)

from ace.core import Generator
from utils import initialize_clients
from data_processor import DataProcessor


def load_playbook(playbook_path: str) -> str:
    """Load playbook content from file."""
    with open(playbook_path, 'r') as f:
        return f.read()




def extract_sql_from_response(response: str) -> str:
    """
    Extract SQL query from generator response.

    The response may be:
    1. JSON format with reasoning and final_answer
    2. Plain SQL query

    Args:
        response: Generator response

    Returns:
        Extracted SQL query
    """
    # Try to parse as JSON first
    try:
        response_json = json.loads(response)
        # Try to get final_answer field
        if 'final_answer' in response_json:
            sql = response_json['final_answer']
            # Remove any markdown code blocks
            sql = sql.replace('```sql', '').replace('```', '').strip()
            return sql
        # If no final_answer, try to get the whole response
        return response.strip()
    except (json.JSONDecodeError, TypeError):
        # If not JSON, assume it's plain SQL
        # Remove any markdown code blocks
        sql = response.replace('```sql', '').replace('```', '').strip()
        return sql


def generate_predictions_parallel(
    generator: Generator,
    samples: List[Dict[str, Any]],
    playbook: str,
    num_workers: int = 4
) -> List[str]:
    """
    Generate predictions for all samples using the playbook in parallel.

    Args:
        generator: Generator instance
        samples: List of samples with context and question
        playbook: Playbook content to use
        num_workers: Number of parallel workers

    Returns:
        List of predicted SQL queries
    """
    predictions = [None] * len(samples)

    def process_sample(idx: int, sample: Dict[str, Any]) -> tuple:
        """Process a single sample and return (idx, prediction)."""
        try:
            context = sample['context']
            question = sample['question']

            # Generate prediction using playbook
            # Generator.generate returns (response, bullet_ids, call_info)
            response, bullet_ids, call_info = generator.generate(
                question=question,
                playbook=playbook,
                context=context,
                use_json_mode=True  # Use JSON mode to get structured response
            )

            # Extract SQL from response
            predicted_sql = extract_sql_from_response(response)

            return idx, predicted_sql
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a placeholder SQL that will fail evaluation
            return idx, "SELECT 1"

    print(f"\nGenerating predictions with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_sample, i, sample): i
            for i, sample in enumerate(samples)
        }

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            idx, prediction = future.result()
            predictions[idx] = prediction
            completed += 1

            if completed % 10 == 0 or completed == len(samples):
                print(f"  Progress: {completed}/{len(samples)} samples completed")

    return predictions


def evaluate_test_samples(
    predictions: List[str],
    test_samples: List[Dict[str, Any]],
    bird_db_root: str
) -> Dict[str, Any]:
    """
    Evaluate predictions using test_samples.json (which already has ground truth).

    Args:
        predictions: List of predicted SQL queries
        test_samples: List of test samples with ground truth SQL
        bird_db_root: Path to BIRD database root

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n" + "="*70)
    print("EVALUATING ON TEST SAMPLES")
    print("="*70)
    print(f"Total test samples: {len(test_samples)}")
    print("="*70)

    if len(predictions) != len(test_samples):
        print(f"\nError: Mismatch between predictions ({len(predictions)}) and samples ({len(test_samples)})")
        return {
            'accuracy': 0.0,
            'total_samples': 0,
            'correct': 0,
            'error': 'Prediction count mismatch'
        }

    # Initialize DataProcessor for evaluation
    data_processor = DataProcessor(bird_db_root=bird_db_root)

    # Evaluate each sample
    correct = 0
    results = []

    print("\nEvaluating predictions...")
    for i, (pred, sample) in enumerate(zip(predictions, test_samples)):
        try:
            gt = sample['target']
            meta = sample.get('others', {})

            is_correct = data_processor.answer_is_correct(pred, gt, meta)

            if is_correct:
                correct += 1

            results.append({
                'question': sample['question'],
                'db_name': meta.get('db_name', ''),
                'predicted_sql': pred,
                'ground_truth_sql': gt,
                'is_correct': is_correct
            })
        except Exception as e:
            print(f"  Error evaluating sample {i}: {e}")
            results.append({
                'question': sample.get('question', ''),
                'db_name': sample.get('others', {}).get('db_name', ''),
                'predicted_sql': pred,
                'ground_truth_sql': sample.get('target', ''),
                'is_correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(test_samples):
            print(f"  Progress: {i + 1}/{len(test_samples)} samples evaluated (correct: {correct})")

    accuracy = correct / len(test_samples) if len(test_samples) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'total_samples': len(test_samples),
        'correct': correct,
        'results': results
    }




def load_run_config(results_dir: str) -> Dict[str, Any]:
    """Load run_config.json from results directory."""
    config_path = os.path.join(results_dir, 'run_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run a playbook on processed data and evaluate accuracy'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/ace_run_20260119_234301_bird_all_hard_to_easy_online)'
    )
    parser.add_argument(
        '--playbook_file',
        type=str,
        required=True,
        help='Playbook file path relative to results_dir (e.g., intermediate_playbooks/window_4_final_playbook.txt)'
    )
    parser.add_argument(
        '--bird_db_root',
        type=str,
        default='stream-bench/data/bird/dev_databases',
        help='Path to BIRD database root directory (for SQL execution during evaluation)'
    )
    parser.add_argument(
        '--api_provider',
        type=str,
        default=None,
        choices=['sambanova', 'together', 'openai'],
        help='API provider for LLM calls (defaults to value from run_config.json)'
    )
    parser.add_argument(
        '--generator_model',
        type=str,
        default=None,
        help='Model name for generator (defaults to generator_model from run_config.json)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel workers for generation'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Optional output file to save detailed results (JSON)'
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return 1

    # Join playbook_file with results_dir
    playbook_path = os.path.join(args.results_dir, args.playbook_file)
    if not os.path.exists(playbook_path):
        print(f"Error: Playbook file not found: {playbook_path}")
        return 1

    # Load run config to get default model and API provider
    run_config = load_run_config(args.results_dir)

    # Use config values if args not provided
    if args.api_provider is None:
        args.api_provider = run_config.get('config', {}).get('api_provider') or run_config.get('api_provider', 'sambanova')
        print(f"Using API provider from run_config.json: {args.api_provider}")

    if args.generator_model is None:
        args.generator_model = run_config.get('generator_model', 'DeepSeek-V3.1')
        print(f"Using generator_model from run_config.json: {args.generator_model}")

    # Load playbook
    print(f"\nLoading playbook from: {playbook_path}")
    playbook = load_playbook(playbook_path)
    print(f"Playbook loaded ({len(playbook)} characters)")

    # Load test samples from processed_data (has everything we need)
    test_samples_path = os.path.join(args.results_dir, 'processed_data', 'test_samples.json')

    if not os.path.exists(test_samples_path):
        print(f"\nError: Test samples file not found: {test_samples_path}")
        print("This file should be created during the ACE training run.")
        return 1

    print(f"\nLoading test samples from: {test_samples_path}")
    print("  (This file contains the test data with ground truth SQL)")
    with open(test_samples_path, 'r') as f:
        samples = json.load(f)
    print(f"  Loaded {len(samples)} test samples")

    # Initialize generator
    print(f"\nInitializing generator with {args.api_provider} API...")
    generator_client, _, _ = initialize_clients(args.api_provider)
    generator = Generator(generator_client, args.api_provider, args.generator_model, max_tokens=4096)

    # Generate predictions
    predictions = generate_predictions_parallel(
        generator, samples, playbook, num_workers=args.num_workers
    )

    # Evaluate
    print(f"\nEvaluating predictions using execution-based evaluation...")
    eval_results = evaluate_test_samples(predictions, samples, args.bird_db_root)

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Playbook: {args.playbook_file}")
    print(f"Total samples evaluated: {eval_results['total_samples']}")
    print(f"Correct: {eval_results['correct']}")
    print(f"Accuracy: {eval_results['accuracy']:.2%}")
    print("="*70)

    # Save detailed results if requested
    if args.output_file:
        # Save output file under the results_dir directory
        output_path = os.path.join(args.results_dir, args.output_file)

        # Warn if file already exists
        if os.path.exists(output_path):
            print(f"\nWarning: Output file already exists and will be overwritten: {output_path}")

        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump({
                    'playbook_file': args.playbook_file,
                    'playbook_path': playbook_path,
                    'accuracy': eval_results['accuracy'],
                    'total_samples': eval_results['total_samples'],
                    'correct': eval_results['correct'],
                    'results': eval_results['results']
                }, f, indent=2)
            print(f"\nDetailed results saved to: {output_path}")
        except Exception as e:
            print(f"\nError saving results to {output_path}: {e}")
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
