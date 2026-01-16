import os
import re
import sqlite3
import time
from typing import List, Dict, Any, Optional, Tuple

class DataProcessor:
    """
    BIRD-only DataProcessor with thread-safe evaluation.

    Evaluation mode: execute predicted & gold on sqlite DB and compare result sets

    Thread-safety: db_name metadata flows through sample['others'] dict to enable
    safe parallel evaluation across multiple workers.
    """

    def __init__(
        self,
        bird_db_root: Optional[str] = None,
        exec_timeout_ms: int = 20000,
        exec_max_rows: int = 20000,
        max_samples: Optional[int] = None,  # None = use all samples
        db_name: Optional[str] = None,  # None = use mixed databases (no filter)
        curriculum: Optional[str] = None,  # None = no curriculum filtering
    ):
        """
        Initialize DataProcessor.

        Args:
            curriculum: Strategy for selecting samples by difficulty. Options:
                - None: No filtering, use all samples
                - "simple-only": Only simple difficulty samples
                - "moderate-only": Only moderate difficulty samples
                - "challenging-only": Only challenging difficulty samples
                - "balanced": Equal distribution (1/3 from each difficulty)
                - "balanced-s2m2c": Balanced simple -> moderate -> challenging order
                - "balanced-c2m2s": Balanced challenging -> moderate -> simple order
        """
        self.bird_db_root = bird_db_root
        self.exec_timeout_ms = exec_timeout_ms
        self.exec_max_rows = exec_max_rows
        self.max_samples = max_samples
        self.db_name = db_name
        self.curriculum = curriculum

        # Validate curriculum option
        valid_curricula = [
            None, "simple-only", "moderate-only", "challenging-only",
            "balanced", "balanced-s2m2c", "balanced-c2m2s"
        ]
        if self.curriculum not in valid_curricula:
            raise ValueError(
                f"Invalid curriculum '{self.curriculum}'. "
                f"Valid options: {[c for c in valid_curricula if c is not None]}"
            )

    # -------------------------
    # REQUIRED SIGNATURES
    # -------------------------

    def process_task_data(self, raw_data):
        """
        Convert your BIRD JSONL rows into standardized format:
          [{"context": ..., "question": ..., "target": ..., "others": {...}}]

        db_name is stored in each sample's 'others' dict for thread-safe evaluation.
        """
        processed = []

        # Filter by db_name if specified
        if self.db_name is not None:
            raw_data = [
                item for item in raw_data
                if (item.get("db_name") or item.get("db_id") or "") == self.db_name
            ]

        # Apply curriculum-based filtering and ordering
        if self.curriculum is not None:
            raw_data = self._apply_curriculum(raw_data)

        # Cap samples
        raw_data = raw_data[: self.max_samples] if self.max_samples is not None else raw_data

        for item in raw_data:
            db_name = item.get("db_name") or item.get("db_id") or ""
            question = item.get("question", "")
            target_sql = item.get("sql", "") or item.get("SQL", "")
            db_schema = item.get("db_schema", "")

            context = f"""You are given a database schema and a question.

        INSTRUCTIONS:
        - Output ONLY a valid SQL query.
        - Do NOT include explanations, comments, markdown, or any extra text.
        - You may ONLY reference tables and columns that appear in the schema below.
        - Do NOT hallucinate tables, columns, or relationships.

        DATABASE SCHEMA:
        {db_schema}
        """

            processed.append({
                "context": context,
                "question": question,
                "target": target_sql,
                "others": {
                    "question_id": item.get("question_id"),
                    "difficulty": item.get("difficulty"),
                    "db_name": db_name,
                    "task": "bird",
                    "data_source": "streambench_bird",
                }
            })

        return processed

    def answer_is_correct(self, predicted, ground_truth, sample_metadata=None):
        """
        Compare predicted vs ground_truth using exec mode.

        Args:
            predicted: Predicted SQL query
            ground_truth: Ground truth SQL query
            sample_metadata: Optional dict containing 'db_name' and other metadata

        Returns:
            bool: True if execution results match, False otherwise
        """
        # Extract db_name from metadata
        db_name = ""
        if sample_metadata:
            db_name = sample_metadata.get("db_name", "")

        # If db_name not available, return False
        if not db_name:
            print(f"Warning: No db_name available in sample metadata")
            return False

        print(f"\n[EVAL START] Evaluating on DB: {db_name}")
        print(f"[EVAL START] Predicted SQL: {predicted[:200]}...")  # First 200 chars
        print(f"[EVAL START] Ground truth SQL: {ground_truth[:200]}...")

        try:
            result = self._exec_match(predicted, ground_truth, db_name)
            print(f"[EVAL DONE] Result: {result}")
            return result
        except Exception as e:
            print(f"[EVAL ERROR] Exception during evaluation: {e}")
            return False

    def evaluate_accuracy(self, predictions, ground_truths, samples=None):
        """
        Calculate accuracy using execution-based evaluation.

        For parallel test evaluation: The actual correctness is determined by
        answer_is_correct() in worker threads. This method re-evaluates using
        sample metadata if available, or falls back to string comparison.

        Args:
            predictions: List of predicted SQL queries
            ground_truths: List of ground truth SQL queries
            samples: Optional list of sample dicts with 'others' metadata

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if len(predictions) == 0:
            return 0.0

        correct = 0
        for i, (p, g) in enumerate(zip(predictions, ground_truths)):
            # If we have sample metadata, use execution-based evaluation
            if samples and i < len(samples):
                sample_metadata = samples[i].get("others", None) if isinstance(samples[i], dict) else None
                if self.answer_is_correct(p, g, sample_metadata):
                    correct += 1
            else:
                # Fallback to string comparison for training (no metadata available)
                if p.strip().lower() == g.strip().lower():
                    correct += 1

        return correct / len(predictions)

    # -------------------------
    # CURRICULUM LOGIC
    # -------------------------

    def _apply_curriculum(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply curriculum-based filtering and ordering to raw data.

        Args:
            raw_data: List of raw data items with 'difficulty' field

        Returns:
            Filtered and/or reordered list based on curriculum strategy
        """
        if not self.curriculum:
            return raw_data

        # Categorize samples by difficulty
        simple_samples = []
        moderate_samples = []
        challenging_samples = []
        unknown_samples = []

        for item in raw_data:
            difficulty = (item.get("difficulty") or "").lower()
            if difficulty == "simple":
                simple_samples.append(item)
            elif difficulty == "moderate":
                moderate_samples.append(item)
            elif difficulty == "challenging":
                challenging_samples.append(item)
            else:
                unknown_samples.append(item)

        # Apply curriculum strategy
        if self.curriculum == "simple-only":
            result = simple_samples
            print(f"Curriculum 'simple-only': Selected {len(result)} simple samples")

        elif self.curriculum == "moderate-only":
            result = moderate_samples
            print(f"Curriculum 'moderate-only': Selected {len(result)} moderate samples")

        elif self.curriculum == "challenging-only":
            result = challenging_samples
            print(f"Curriculum 'challenging-only': Selected {len(result)} challenging samples")

        elif self.curriculum == "balanced":
            # Equal distribution from each difficulty (1/3 each)
            min_count = min(len(simple_samples), len(moderate_samples), len(challenging_samples))
            result = (
                simple_samples[:min_count] +
                moderate_samples[:min_count] +
                challenging_samples[:min_count]
            )
            print(f"Curriculum 'balanced': Selected {min_count} from each difficulty "
                  f"(total: {len(result)} samples)")

        elif self.curriculum == "balanced-s2m2c":
            # Balanced: simple -> moderate -> challenging
            min_count = min(len(simple_samples), len(moderate_samples), len(challenging_samples))
            result = (
                simple_samples[:min_count] +
                moderate_samples[:min_count] +
                challenging_samples[:min_count]
            )
            print(f"Curriculum 'balanced-s2m2c': {min_count} simple -> {min_count} moderate -> "
                  f"{min_count} challenging (total: {len(result)} samples)")

        elif self.curriculum == "balanced-c2m2s":
            # Balanced: challenging -> moderate -> simple
            min_count = min(len(simple_samples), len(moderate_samples), len(challenging_samples))
            result = (
                challenging_samples[:min_count] +
                moderate_samples[:min_count] +
                simple_samples[:min_count]
            )
            print(f"Curriculum 'balanced-c2m2s': {min_count} challenging -> {min_count} moderate -> "
                  f"{min_count} simple (total: {len(result)} samples)")

        else:
            # Should not reach here due to validation in __init__
            result = raw_data

        if unknown_samples:
            print(f"Warning: {len(unknown_samples)} samples with unknown difficulty were excluded")

        return result

    # -------------------------
    # EXECUTION EVAL INTERNALS
    # -------------------------

    def _exec_match(self, predicted_sql: str, gold_sql: str, db_name: str) -> bool:
        sqlite_path = self._find_sqlite_path(db_name)
        if not sqlite_path:
            # DB not found -> fall back to exact
            print(f"SQLite DB for {db_name} not found under {self.bird_db_root}. Falling back to exact match.")
            return False

        try:
            print(f"[EXEC] Running PREDICTED SQL on {db_name}")
            pred_res = self._run_sql(sqlite_path, predicted_sql)

            print(f"[EXEC] Running GROUND TRUTH SQL on {db_name}")
            gold_res = self._run_sql(sqlite_path, gold_sql)

            # Print execution results
            print(f"\n--- Execution Results ---")
            print(f"DB: {db_name}")
            print(f"\nPredicted SQL:\n{predicted_sql}")
            print(f"\nPredicted Result ({len(pred_res)} rows):")
            for row in pred_res[:10]:  # Print first 10 rows
                print(f"  {row}")
            if len(pred_res) > 10:
                print(f"  ... ({len(pred_res) - 10} more rows)")

            print(f"\nGround Truth SQL:\n{gold_sql}")
            print(f"\nGround Truth Result ({len(gold_res)} rows):")
            for row in gold_res[:10]:  # Print first 10 rows
                print(f"  {row}")
            if len(gold_res) > 10:
                print(f"  ... ({len(gold_res) - 10} more rows)")
            print("-" * 50)

            print(f"[EXEC] Normalizing and comparing results...")
            match = self._normalize_result(pred_res) == self._normalize_result(gold_res)
            print(f"[EXEC] Match result: {match}")
            return match

        except Exception as e:
            print(f"\n--- Execution Error ---")
            print(f"DB: {db_name}")
            print(f"Error: {e}")
            print("-" * 50)
            return False

    def _find_sqlite_path(self, db_name: str) -> Optional[str]:
        """
        Typical layout after unzipping dev_databases.zip:
          <bird_db_root>/<db_name>/<db_name>.sqlite
        """
        if not self.bird_db_root:
            return None

        p = os.path.join(self.bird_db_root, db_name, f"{db_name}.sqlite")
        if os.path.exists(p):
            return p

        return None

    def _run_sql(self, sqlite_path: str, sql: str) -> List[Tuple[Any, ...]]:
        sql = (sql or "").strip().rstrip(";")
        if not sql:
            raise ValueError("Empty SQL")

        # guard against writes/DDL in evaluation
        lowered = sql.lower()
        if any(k in lowered for k in ["insert ", "update ", "delete ", "drop ", "alter ", "create ", "pragma ", "attach "]):
            raise ValueError("Unsafe SQL in evaluation")

        print(f"[DEBUG] Connecting to database: {sqlite_path}")
        conn = sqlite3.connect(sqlite_path, timeout=self.exec_timeout_ms / 1000.0)

        # Set up timeout mechanism using progress handler
        start_time = time.time()
        timeout_seconds = self.exec_timeout_ms / 1000.0

        def progress_handler():
            if time.time() - start_time > timeout_seconds:
                print(f"[TIMEOUT] Query exceeded {timeout_seconds}s timeout")
                return 1  # Non-zero return aborts the operation
            return 0

        try:
            # Call progress handler every 1000 VM instructions
            conn.set_progress_handler(progress_handler, 1000)

            print(f"[DEBUG] Executing SQL query...")
            cur = conn.cursor()
            cur.execute(sql)

            print(f"[DEBUG] Fetching results (max {self.exec_max_rows} rows)...")
            rows = cur.fetchmany(self.exec_max_rows)

            elapsed = time.time() - start_time
            print(f"[DEBUG] Query completed in {elapsed:.2f}s, returned {len(rows)} rows")

            return [tuple(r) for r in rows]
        except sqlite3.OperationalError as e:
            elapsed = time.time() - start_time
            if "interrupted" in str(e).lower():
                raise TimeoutError(f"SQL query timed out after {elapsed:.2f}s (limit: {timeout_seconds}s)")
            raise
        finally:
            conn.close()

    @staticmethod
    def _normalize_result(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
        def norm(v):
            if isinstance(v, float):
                return round(v, 6)
            return v

        normed = [tuple(norm(v) for v in row) for row in rows]
        return sorted(normed)