from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.scripts.evaluate_directional_objectives import main


class EvaluateDirectionalObjectivesTests(unittest.TestCase):
    def test_main_writes_output_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.csv"
            output_path = tmp_path / "out.json"
            frame = pd.DataFrame(
                {
                    "p_up": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
                    "y": [1, 0, 1, 0, 1, 0],
                    "regime_state": ["trend_ignition", "trend_ignition", "neutral", "neutral", "chop", "chop"],
                    "horizon_hours": [1, 1, 4, 4, 8, 8],
                }
            )
            frame.to_csv(input_path, index=False)

            import sys

            argv_backup = list(sys.argv)
            try:
                sys.argv = [
                    "evaluate_directional_objectives",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--min-rows",
                    "4",
                    "--group-min-rows",
                    "2",
                    "--max-ece",
                    "1.0",
                    "--min-f1",
                    "0.4",
                ]
                exit_code = main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())

    def test_main_accepts_sparse_regime_override_for_min_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.csv"
            output_path = tmp_path / "out.json"
            frame = pd.DataFrame(
                {
                    "p_up": [0.8, 0.2, 0.7, 0.3, 0.9],
                    "y": [1, 0, 1, 0, 1],
                    "regime_state": ["trend_ignition", "trend_ignition", "neutral", "neutral", "chop"],
                    "horizon_hours": [1, 1, 1, 1, 1],
                }
            )
            frame.to_csv(input_path, index=False)

            import sys

            argv_backup = list(sys.argv)
            try:
                sys.argv = [
                    "evaluate_directional_objectives",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--min-rows",
                    "5",
                    "--group-min-rows",
                    "2",
                    "--min-rows-by-regime",
                    "chop:1",
                    "--max-ece",
                    "1.0",
                    "--min-f1",
                    "0.0",
                ]
                exit_code = main()
            finally:
                sys.argv = argv_backup

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
