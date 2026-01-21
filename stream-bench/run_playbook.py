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


def load_processed_data(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load processed data from bullet_usage_log.jsonl.

    Returns list of samples with context, question, target, and metadata.
    """
    log_file = os.path.join(results_dir, "bullet_usage_log.jsonl")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"No bullet_usage_log.jsonl found in {results_dir}")

    samples = []
    seen_questions = set()

    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)

                # Extract question to deduplicate (same question appears multiple times during training)
                question = entry.get('sample_question', '')

                # Skip duplicates - only keep first occurrence
                if question in seen_questions:
                    continue
                seen_questions.add(question)

                # Parse the context to extract db_schema and other info
                context_raw = entry.get('sample_context', '')

                # Extract database schema and other metadata
                # The context format is standardized in data_processor.py
                samples.append({
                    'context': context_raw,
                    'question': question,
                    'target': None,  # We don't have ground truth in the log
                    'others': {
                        'db_name': extract_db_name_from_context(context_raw),
                        'question_id': entry.get('sample_id'),
                    }
                })

    print(f"Loaded {len(samples)} unique samples from {log_file}")
    return samples


def extract_db_name_from_context(context: str) -> str:
    """
    Extract database name from context string.

    The context contains db_schema as a dictionary string.
    We need to parse it to get db_id.
    """
    try:
        # Look for db_id in the context
        if "'db_id':" in context:
            start = context.index("'db_id':") + len("'db_id':")
            # Find the next quote
            start = context.index("'", start) + 1
            end = context.index("'", start)
            return context[start:end]
    except:
        pass
    return ""


def load_bird_dev_data(bird_jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load BIRD dev data from JSONL file.

    This provides ground truth SQL for evaluation.
    """
    if not os.path.exists(bird_jsonl_path):
        raise FileNotFoundError(f"BIRD dev JSONL not found: {bird_jsonl_path}")

    samples = []
    with open(bird_jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from BIRD dev data")
    return samples


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


def evaluate_with_data_processor(
    predictions: List[str],
    bird_data: List[Dict[str, Any]],
    samples: List[Dict[str, Any]],
    bird_db_root: str
) -> Dict[str, Any]:
    """
    Evaluate predictions using DataProcessor's execution-based evaluation.

    Args:
        predictions: List of predicted SQL queries
        bird_data: List of BIRD data items with ground truth SQL
        samples: List of samples (for matching questions)
        bird_db_root: Path to BIRD database root

    Returns:
        Dictionary with evaluation results
    """
    # Create a mapping from question to ground truth SQL
    question_to_gt = {}
    question_to_db = {}
    for item in bird_data:
        q = item.get('question', '')
        question_to_gt[q] = item.get('sql', '') or item.get('SQL', '')
        question_to_db[q] = item.get('db_name', '') or item.get('db_id', '')

    # Match predictions with ground truth
    matched_predictions = []
    matched_ground_truths = []
    matched_metadata = []

    for i, sample in enumerate(samples):
        question = sample['question']
        if question in question_to_gt:
            matched_predictions.append(predictions[i])
            matched_ground_truths.append(question_to_gt[question])
            matched_metadata.append({
                'db_name': question_to_db[question],
                'question': question
            })

    print(f"\nMatched {len(matched_predictions)} samples with ground truth")

    if len(matched_predictions) == 0:
        return {
            'accuracy': 0.0,
            'total_samples': 0,
            'correct': 0,
            'error': 'No samples matched with ground truth'
        }

    # Initialize DataProcessor for evaluation
    data_processor = DataProcessor(bird_db_root=bird_db_root)

    # Evaluate each sample
    correct = 0
    results = []

    print("\nEvaluating predictions...")
    for i, (pred, gt, meta) in enumerate(zip(matched_predictions, matched_ground_truths, matched_metadata)):
        try:
            is_correct = data_processor.answer_is_correct(pred, gt, meta)

            if is_correct:
                correct += 1

            results.append({
                'question': meta['question'],
                'db_name': meta['db_name'],
                'predicted_sql': pred,
                'ground_truth_sql': gt,
                'is_correct': is_correct
            })
        except Exception as e:
            print(f"  Error evaluating sample {i}: {e}")
            results.append({
                'question': meta['question'],
                'db_name': meta['db_name'],
                'predicted_sql': pred,
                'ground_truth_sql': gt,
                'is_correct': False,
                'error': str(e)
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(matched_predictions):
            print(f"  Progress: {i + 1}/{len(matched_predictions)} samples evaluated (correct: {correct})")

    accuracy = correct / len(matched_predictions)

    return {
        'accuracy': accuracy,
        'total_samples': len(matched_predictions),
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
        help='Name of playbook file in intermediate_playbooks folder (e.g., window_4_final_playbook.txt)'
    )
    parser.add_argument(
        '--bird_dev_jsonl',
        type=str,
        default='stream-bench/data/streambench_bird_test.jsonl',
        help='Path to BIRD dev JSONL file with ground truth'
    )
    parser.add_argument(
        '--bird_db_root',
        type=str,
        default='stream-bench/data/bird/dev_databases',
        help='Path to BIRD database root directory'
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
        '--reflector_model',
        type=str,
        default=None,
        help='Model name for reflector (defaults to reflector_model from run_config.json)'
    )
    parser.add_argument(
        '--curator_model',
        type=str,
        default=None,
        help='Model name for curator (defaults to curator_model from run_config.json)'
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

    playbook_path = os.path.join(args.results_dir, 'intermediate_playbooks', args.playbook_file)
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

    if args.reflector_model is None:
        args.reflector_model = run_config.get('reflector_model', args.generator_model)
        print(f"Using reflector_model from run_config.json: {args.reflector_model}")

    if args.curator_model is None:
        args.curator_model = run_config.get('curator_model', args.generator_model)
        print(f"Using curator_model from run_config.json: {args.curator_model}")

    # Load playbook
    print(f"\nLoading playbook from: {playbook_path}")
    playbook = load_playbook(playbook_path)
    print(f"Playbook loaded ({len(playbook)} characters)")

    # Load processed data
    print(f"\nLoading processed data from: {args.results_dir}")
    samples = load_processed_data(args.results_dir)

    # Load BIRD dev data for ground truth
    print(f"\nLoading BIRD dev data from: {args.bird_dev_jsonl}")
    bird_data = load_bird_dev_data(args.bird_dev_jsonl)

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
    eval_results = evaluate_with_data_processor(
        predictions, bird_data, samples, args.bird_db_root
    )

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
        output_path = args.output_file

        # Warn if file already exists
        if os.path.exists(output_path):
            print(f"\nWarning: Output file already exists and will be overwritten: {output_path}")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
