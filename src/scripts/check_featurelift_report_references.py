from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPERSEDED_REPORT = Path("artifacts/analysis/featurelift_20260331/comparison_report.json")
CURRENT_REPORT = Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.json")
CURRENT_REPORT_MARKDOWN = Path("artifacts/analysis/featurelift_20260331_rerun/comparison_report.md")
DEFAULT_ALLOWED_REFERENCES = {
    SUPERSEDED_REPORT,
    CURRENT_REPORT,
    Path("src/scripts/generate_featurelift_comparison_report.py"),
    Path("src/scripts/check_featurelift_report_references.py"),
}


def _iter_tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    entries = [entry for entry in result.stdout.decode("utf-8").split("\0") if entry]
    return [Path(entry) for entry in entries]


def _looks_like_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return False
    return b"\0" not in sample


def find_stale_reference_hits(
    file_paths: Iterable[Path],
    stale_reference: str,
    *,
    repo_root: Path,
    allowed_paths: set[Path] | None = None,
) -> list[dict[str, object]]:
    allowed = {path.as_posix() for path in (allowed_paths or set())}
    hits: list[dict[str, object]] = []
    for relative_path in file_paths:
        rel_posix = relative_path.as_posix()
        if rel_posix in allowed:
            continue
        absolute_path = repo_root / relative_path
        if not absolute_path.is_file() or not _looks_like_text(absolute_path):
            continue
        try:
            content = absolute_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            if stale_reference in line:
                hits.append({
                    "path": rel_posix,
                    "line": line_number,
                    "line_text": line.strip(),
                })
    return hits


def _format_hits(hits: list[dict[str, object]]) -> str:
    formatted = []
    for hit in hits:
        formatted.append(f"{hit['path']}:{hit['line']}: {hit['line_text']}")
    return "\n".join(formatted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail when the superseded feature-lift comparison report is referenced outside the allow list."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--stale-reference",
        default=SUPERSEDED_REPORT.as_posix(),
        help="Repository-relative stale report path to reject.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    tracked_files = _iter_tracked_files(repo_root)
    hits = find_stale_reference_hits(
        tracked_files,
        args.stale_reference,
        repo_root=repo_root,
        allowed_paths=DEFAULT_ALLOWED_REFERENCES,
    )
    if hits:
        print("Found disallowed references to superseded feature-lift report:", file=sys.stderr)
        print(_format_hits(hits), file=sys.stderr)
        raise SystemExit(1)

    print("No disallowed references to superseded feature-lift report were found.")


if __name__ == "__main__":
    main()