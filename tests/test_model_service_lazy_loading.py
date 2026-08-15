import importlib
import sys


def test_model_service_import_does_not_load_model(monkeypatch):
    import services.model_artifact as model_artifact

    monkeypatch.setattr(model_artifact, "ensure_model_artifact", lambda path: None)
    sys.modules.pop("services.model_service", None)

    module = importlib.import_module("services.model_service")

    assert module.model is None
    assert module.MODEL_PATH is not None
