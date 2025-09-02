from pathlib import Path
import sys

import pandas as pd

# Ensure the package is importable when tests are executed from different CWDs
sys.path.append(str(Path(__file__).resolve().parents[1]))
from targetDB.utils import druggability_ml


def _mock_training_df():
    """Small training set for tests."""
    return pd.DataFrame(
        {
            "feature1": [0, 1],
            "feature2": [1, 0],
            "DRUGGABLE": [1, 0],
        },
        index=["t1", "t2"],
    )


def _mock_score_df():
    """Mimic score_components with all columns required by predict functions."""
    return pd.DataFrame(
        {
            "Target_id": ["t1", "t3"],
            "feature1": [0, 1],
            "feature2": [1, 1],
            "OT_max_association_score": [0.1, 0.2],
            "Heart_alert": [True, False],
            "Liver_alert": [0, 0],
            "Kidney_alert": [0, 1],
            "dis_AScore": [0.5, 0.6],
            "bio_EScore": [0.5, 0.7],
            "safe_EScore": [0.2, 0.1],
            "chembl_selective_M": [0, 0],
            "chembl_selective_G": [1, 1],
            "chembl_selective_E": [0, 0],
            "bindingDB_phase2": [1, 0],
            "commercial_potent": [True, False],
            "information_score": [0.3, 0.4],
            "gen_AQualScore": [0.5, 0.6],
            "genetic_NORM": [0.2, 0.3],
        }
    )


def test_predict_and_prob_prune_features(monkeypatch):
    training_df = _mock_training_df()
    score_df = _mock_score_df()

    # Mock reading of the training dataset
    monkeypatch.setattr(druggability_ml.pd, "read_json", lambda *a, **k: training_df)

    model = druggability_ml.generate_model()

    # Capture columns sent to the model to ensure pruning
    captured = {}
    orig_predict = model.predict
    orig_proba = model.predict_proba

    def capture_predict(x):
        captured["predict_cols"] = list(x.columns)
        return orig_predict(x)

    def capture_proba(x):
        captured["proba_cols"] = list(x.columns)
        return orig_proba(x)

    monkeypatch.setattr(model, "predict", capture_predict)
    monkeypatch.setattr(model, "predict_proba", capture_proba)

    preds = druggability_ml.predict(model, score_df)
    probs = druggability_ml.predict_prob(model, score_df)

    assert preds.shape == (2,)
    assert probs.shape == (2, 2)
    assert captured["predict_cols"] == ["feature1", "feature2"]
    assert captured["proba_cols"] == ["feature1", "feature2"]


def test_in_training_set(monkeypatch):
    training_df = _mock_training_df()
    monkeypatch.setattr(druggability_ml.pd, "read_json", lambda *a, **k: training_df)
    df = pd.DataFrame({"Target_id": ["t1", "t3"]})
    result = druggability_ml.in_training_set(df)
    assert list(result) == ["Yes", "No"]
