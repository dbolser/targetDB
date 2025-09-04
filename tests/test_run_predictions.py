import sqlite3
import pandas as pd

import sys
from pathlib import Path

# Ensure modules inside the package directory can be imported directly
sys.path.append(str(Path(__file__).resolve().parent.parent / "targetDB"))
import predict_all_targets as pat


def test_run_predictions_minimal_db(monkeypatch, tmp_path):
    # Create temporary SQLite database with minimal Targets table
    db_path = tmp_path / "temp.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE Targets (Target_id TEXT, Gene_name TEXT)")
    conn.execute(
        "INSERT INTO Targets (Target_id, Gene_name) VALUES (?, ?)", ("T1", "GeneA")
    )
    conn.commit()
    conn.close()

    # Dummy sequential executor to avoid multiprocessing complexity in tests
    class DummyExecutor:
        def __init__(self, max_workers=None, initializer=None, initargs=()):
            if initializer:
                initializer(*initargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            for item in iterable:
                yield func(item)

    monkeypatch.setattr(pat, "ProcessPoolExecutor", DummyExecutor)

    # Patch machine learning utilities
    class DummyModel:
        classes_ = [0, 1]

    monkeypatch.setattr(pat.dml, "generate_model", lambda: DummyModel())
    monkeypatch.setattr(pat.dml, "predict_prob", lambda model, comps: [[0.2, 0.8]])
    monkeypatch.setattr(pat.dml, "in_training_set", lambda comps: [True])

    # Capture mode passed to target_scores to ensure programmatic operation
    mode_used = {}

    def dummy_get_descriptors_list(target_id, targetdb=None):
        return pd.DataFrame({"Target_id": [target_id]})

    class DummyScore:
        def __init__(self, data):
            self.scores = data[["Target_id"]].copy()
            self.score_components = pd.DataFrame({"feat": [0]})

    def dummy_target_scores(data, mode="list"):
        mode_used["mode"] = mode
        return DummyScore(data)

    monkeypatch.setattr(pat.td, "get_descriptors_list", dummy_get_descriptors_list)
    monkeypatch.setattr(pat.td, "target_scores", dummy_target_scores)

    # Run predictions
    result = pat.run_predictions(str(db_path), workers=1)

    # Verify returned DataFrame structure and contents
    assert list(result["Target_id"]) == ["T1"]
    assert list(result["Gene_name"]) == ["GeneA"]
    assert "Tractability_probability" in result.columns
    assert "Tractable" in result.columns
    assert "In_training_set" in result.columns
    assert result.loc[0, "Tractable"] == "Tractable"
    assert result.loc[0, "Tractability_probability"] == 80.0
    assert mode_used["mode"] == "programmatic"
