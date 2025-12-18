import numpy as np

from src.scripts import qa_datasets


def _write_npz(path, **arrays):
    np.savez(path, **arrays)


def test_main_passes_on_clean_dataset(tmp_path, capsys):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    clean_path = datasets_dir / "clean_dataset.npz"
    _write_npz(
        clean_path,
        X_train=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        X_val=np.array([[5.0, 6.0]], dtype=np.float64),
        X_test=np.array([[7.0, 8.0]], dtype=np.float64),
        y_train=np.array([0, 1], dtype=np.int64),
        y_val=np.array([1], dtype=np.int64),
        y_test=np.array([0], dtype=np.int64),
        feature_names=np.array(["feat_a", "feat_b"]),
        ts_train=np.array(["2024-01-01T00:00:00", "2024-01-01T01:00:00"], dtype="datetime64[s]"),
        ts_val=np.array(["2024-01-02T00:00:00"], dtype="datetime64[s]"),
        ts_test=np.array(["2024-01-03T00:00:00"], dtype="datetime64[s]"),
    )

    exit_code = qa_datasets.main([
        "--datasets-dir",
        str(datasets_dir),
        "--datasets",
        "clean_dataset",
    ])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "# Dataset QA Summary" in captured.out
    assert "| clean_dataset.npz | train |" in captured.out
    assert "## Issues" not in captured.out


def test_main_flags_dataset_with_issues(tmp_path, capsys):
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()

    dirty_path = datasets_dir / "dirty_dataset.npz"
    _write_npz(
        dirty_path,
        X_train=np.array([[1.0, 1.0], [np.nan, 1.0]], dtype=np.float64),
        feature_names=np.array(["feat_const", "feat_nan"]),
        ts_train=np.array(["2024-01-01T00:00:00", "2024-01-01T01:00:00"], dtype="datetime64[s]"),
    )

    exit_code = qa_datasets.main([
        "--datasets-dir",
        str(datasets_dir),
        "--datasets",
        "dirty_dataset.npz",
    ])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "# Dataset QA Summary" in captured.out
    assert "dirty_dataset.npz" in captured.out
    assert "## Issues" in captured.out
    assert "contains 1 NaN feature values" in captured.out
    assert "has zero-variance features" in captured.out
