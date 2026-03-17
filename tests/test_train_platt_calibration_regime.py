from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from src.scripts.train_platt_calibration import (
    _fit_base_horizon_calibration_from_labeled_csv,
    _fit_regime_calibration_from_labeled_csv,
    _summarize_regime_calibration_coverage_from_labeled_csv,
)


class TrainPlattCalibrationRegimeTests(unittest.TestCase):
    def test_fit_regime_calibration_emits_horizon_regime_keys(self) -> None:
        rows = ["p_up,y_true,regime_state,horizon"]
        for value in range(20):
            rows.append(f"0.{55 + (value % 4)},1,neutral,1h")
            rows.append(f"0.{35 + (value % 4)},0,neutral,1h")
        for value in range(10):
            rows.append(f"0.{60 + (value % 3)},1,chop,1h")
            rows.append(f"0.{30 + (value % 3)},0,chop,1h")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labeled.csv"
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            payload = _fit_regime_calibration_from_labeled_csv(
                str(csv_path),
                regime_col="regime_state",
                min_rows=20,
                method="platt",
            )

        self.assertIn("1h@neutral", payload)
        self.assertIn("1h@chop", payload)
        self.assertEqual(payload["1h@neutral"]["method"], "platt")
        self.assertIn("a", payload["1h@neutral"])
        self.assertIn("b", payload["1h@neutral"])

    def test_coverage_defaults_to_1h_when_horizon_column_missing(self) -> None:
        rows = ["p_up,y_true,regime_state"]
        for value in range(20):
            rows.append(f"0.{55 + (value % 4)},1,neutral")
            rows.append(f"0.{35 + (value % 4)},0,neutral")
        for value in range(10):
            rows.append(f"0.{60 + (value % 3)},1,chop")
            rows.append(f"0.{30 + (value % 3)},0,chop")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labeled.csv"
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            coverage = _summarize_regime_calibration_coverage_from_labeled_csv(
                str(csv_path),
                regime_col="regime_state",
                min_rows=20,
            )

        self.assertFalse(coverage["has_horizon_col"])
        self.assertTrue(coverage["default_horizon_applied"])
        self.assertEqual(coverage["eligible_entry_count"], 2)
        self.assertEqual(
            {entry["horizon"] for entry in coverage["eligible_entries"]},
            {"1h"},
        )

    def test_cli_writes_coverage_output(self) -> None:
        rows = ["p_up,y_true,regime_state"]
        for value in range(20):
            rows.append(f"0.{55 + (value % 4)},1,neutral")
            rows.append(f"0.{35 + (value % 4)},0,neutral")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labeled.csv"
            output_path = Path(tmp_dir) / "platt.json"
            coverage_path = Path(tmp_dir) / "coverage.json"
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.train_platt_calibration",
                    "--horizons",
                    "1",
                    "--output-path",
                    str(output_path),
                    "--labeled-input",
                    str(csv_path),
                    "--regime-col",
                    "regime_state",
                    "--min-regime-rows",
                    "20",
                    "--coverage-output-path",
                    str(coverage_path),
                ],
                check=True,
                cwd="/workspaces/btc",
            )

            payload = json.loads(coverage_path.read_text(encoding="utf-8"))

        self.assertTrue(payload["enabled"])
        self.assertFalse(payload["has_horizon_col"])
        self.assertTrue(payload["default_horizon_applied"])

    def test_cli_can_lower_regime_floor_for_controlled_experiments(self) -> None:
        rows = ["p_up,y_true,regime_state,horizon"]
        for value in range(9):
            rows.append(f"0.{60 + (value % 3)},1,chop,4h")
            rows.append(f"0.{30 + (value % 3)},0,chop,4h")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labeled.csv"
            output_path = Path(tmp_dir) / "platt.json"
            coverage_path = Path(tmp_dir) / "coverage.json"
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.scripts.train_platt_calibration",
                    "--horizons",
                    "4",
                    "--output-path",
                    str(output_path),
                    "--labeled-input",
                    str(csv_path),
                    "--regime-col",
                    "regime_state",
                    "--min-regime-rows",
                    "15",
                    "--min-regime-rows-floor",
                    "15",
                    "--coverage-output-path",
                    str(coverage_path),
                ],
                check=True,
                cwd="/workspaces/btc",
            )

            coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
            calibration_payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(coverage_payload["min_rows"], 15)
        self.assertEqual(coverage_payload["eligible_entry_count"], 1)
        self.assertIn("4h@chop", calibration_payload)

    def test_fit_base_horizon_calibration_from_labeled_csv_emits_base_keys(self) -> None:
        rows = ["p_up,y_true,horizon"]
        for value in range(20):
            rows.append(f"0.{60 + (value % 3)},1,1h")
            rows.append(f"0.{30 + (value % 3)},0,1h")
            rows.append(f"0.{65 + (value % 3)},1,4h")
            rows.append(f"0.{25 + (value % 3)},0,4h")

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "labeled.csv"
            csv_path.write_text("\n".join(rows), encoding="utf-8")

            payload = _fit_base_horizon_calibration_from_labeled_csv(
                str(csv_path),
                min_rows=20,
                method="isotonic",
            )

        self.assertIn("1h", payload)
        self.assertIn("4h", payload)
        self.assertEqual(payload["1h"]["method"], "isotonic")
        self.assertIn("x", payload["1h"])
        self.assertIn("y", payload["1h"])


if __name__ == "__main__":
    unittest.main()