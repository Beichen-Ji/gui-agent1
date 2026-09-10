import sys
from types import SimpleNamespace

import pytest

from gui_agent.model_loading import load_multimodal_model, load_processor


class LoaderProbe:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, model_name: str, **kwargs: object) -> str:
        self.calls.append((model_name, kwargs))
        return self.result


def test_shared_model_loaders_forward_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = LoaderProbe("processor")
    model = LoaderProbe("model")
    fake = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=processor),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    assert load_processor("model-id", revision="abc") == "processor"
    assert load_multimodal_model("model-id", dtype="bf16") == "model"
    assert processor.calls == [("model-id", {"revision": "abc"})]
    assert model.calls == [("model-id", {"dtype": "bf16"})]
