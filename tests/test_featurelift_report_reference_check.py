from __future__ import annotations

import unittest
from pathlib import Path

from src.scripts.check_featurelift_report_references import find_stale_reference_hits


class FeatureliftReportReferenceCheckTests(unittest.TestCase):
    def test_find_stale_reference_hits_skips_allowed_paths(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        file_paths = [
            Path("src/scripts/generate_featurelift_comparison_report.py"),
            Path("docs/featurelift_notes.md"),
        ]

        docs_path = repo_root / "docs/featurelift_notes.md"
        docs_path.write_text(
            "See artifacts/analysis/featurelift_20260331/comparison_report.json for the old report.\n",
            encoding="utf-8",
        )
        try:
            hits = find_stale_reference_hits(
                file_paths,
                "artifacts/analysis/featurelift_20260331/comparison_report.json",
                repo_root=repo_root,
                allowed_paths={Path("src/scripts/generate_featurelift_comparison_report.py")},
            )
        finally:
            docs_path.unlink(missing_ok=True)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["path"], "docs/featurelift_notes.md")

    def test_find_stale_reference_hits_ignores_allowed_reference_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        file_paths = [Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.json")]

        hits = find_stale_reference_hits(
            file_paths,
            "artifacts/analysis/featurelift_20260331/comparison_report.json",
            repo_root=repo_root,
            allowed_paths={Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.json")},
        )

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()