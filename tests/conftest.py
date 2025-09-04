import sys
import types
import pathlib
import pytest


@pytest.fixture(scope="module")
def dr_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.extend([str(root), str(root / "targetDB")])
    dml_stub = types.SimpleNamespace(
        generate_model=lambda: types.SimpleNamespace(classes_=[0, 1]),
        predict=lambda *a, **k: 0,
        predict_prob=lambda *a, **k: [0.0, 1.0],
        in_training_set=lambda *a, **k: False,
    )
    sys.modules["utils.druggability_ml"] = dml_stub
    sys.modules["utils.targetDB_gui"] = types.ModuleType("targetDB_gui")
    import targetDB.druggability_report as dr

    return dr
