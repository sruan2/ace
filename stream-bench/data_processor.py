import os
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

class DataProcessor:
    """
    BIRD-only DataProcessor with fixed public method signatures.

    Evaluation mode: execute predicted & gold on sqlite DB and compare result sets

    IMPORTANT: Since we cannot pass per-sample metadata into answer_is_correct(),
    we store db_name in self._db_names (aligned with processed samples) during process_task_data(),
    and use an internal index while evaluating.
    """

    def __init__(
        self,
        bird_db_root: Optional[str] = None,
        exec_timeout_ms: int = 20000,
        exec_max_rows: int = 20000,
        max_samples: int = 10,     # default cap
        db_name: Optional[str] = None,  # filter by specific database name
    ):
        self.bird_db_root = bird_db_root
        self.exec_timeout_ms = exec_timeout_ms
        self.exec_max_rows = exec_max_rows
        self.max_samples = max_samples
        self.db_name = db_name

        # Stored per-sample metadata from the last processed dataset
        self._db_names: List[str] = []

        # Internal cursor used during evaluation to align db_name with sample
        self._eval_i: int = 0

    # -------------------------
    # REQUIRED SIGNATURES
    # -------------------------

    def process_task_data(self, raw_data):
        """
        Convert your BIRD JSONL rows into standardized format:
          [{"context": ..., "question": ..., "target": ..., "others": {...}}]

        Also stores db_name list in self._db_names for exec evaluation.
        """
        processed = []
        db_names = []

        # Filter by db_name if specified
        if self.db_name is not None:
            raw_data = [
                item for item in raw_data
                if (item.get("db_name") or item.get("db_id") or "") == self.db_name
            ]

        # Cap samples (default 10)
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

            db_names.append(db_name)

        # store for exec-mode evaluation
        self._db_names = db_names
        return processed

    def answer_is_correct(self, predicted, ground_truth):
        """
        Compare predicted vs ground_truth using exec mode.
        Uses internal self._eval_i to pick the db_name for exec-mode.
        """
        # get db_name for this sample (aligned with evaluation index)
        db_name = ""
        # print(self._db_names)
        # print(self._eval_i)
        if 0 <= self._eval_i < len(self._db_names):
            db_name = self._db_names[self._eval_i]

        print(f"Evaluating sample {self._eval_i} on DB: {db_name}")

        # If db_name not available, return False
        if not db_name:
            print(f"Warning: No db_name available for sample {self._eval_i}")
            return False

        return self._exec_match(predicted, ground_truth, db_name)

    def evaluate_accuracy(self, predictions, ground_truths):
        """
        Calculate accuracy using answer_is_correct for each sample.
        Keeps internal index in sync so exec-mode uses correct db_name.
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have the same length")
        if len(predictions) == 0:
            return 0.0

        correct = 0
        for i, (p, g) in enumerate(zip(predictions, ground_truths)):
            self._eval_i = i  # set current sample index for answer_is_correct()
            if self.answer_is_correct(p, g):
                correct += 1

        # reset cursor
        self._eval_i = 0
        return correct / len(predictions)

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
            pred_res = self._run_sql(sqlite_path, predicted_sql)
            gold_res = self._run_sql(sqlite_path, gold_sql)

            # Print execution results
            print(f"\n--- Execution Results (Sample {self._eval_i}) ---")
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

        except Exception as e:
            print(f"\n--- Execution Error (Sample {self._eval_i}) ---")
            print(f"DB: {db_name}")
            print(f"Error: {e}")
            print("-" * 50)
            return False

        return self._normalize_result(pred_res) == self._normalize_result(gold_res)

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

        conn = sqlite3.connect(sqlite_path)
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(self.exec_timeout_ms)};")
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchmany(self.exec_max_rows)
            return [tuple(r) for r in rows]
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