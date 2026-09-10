import pytest
from pydantic import ValidationError

from gui_agent._models import StrictFrozenModel


class ExampleModel(StrictFrozenModel):
    value: int


def test_strict_frozen_model_preserves_the_repository_schema_policy() -> None:
    model = ExampleModel(value=1)
    field_name = "value"

    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"value": "1"})
    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"value": 1, "extra": True})
    with pytest.raises(ValidationError):
        setattr(model, field_name, 2)
