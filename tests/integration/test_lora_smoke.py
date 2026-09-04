import json
import os
from pathlib import Path

import pytest

from gui_agent.training.cli import main


@pytest.mark.integration
def test_real_qwen_qlora_check_saves_and_reloads_adapter() -> None:
    if os.environ.get("GUI_AGENT_RUN_LORA_INTEGRATION") != "1":
        pytest.skip("set GUI_AGENT_RUN_LORA_INTEGRATION=1 for the real GPU/model check")
    data_dir = Path(os.environ["GUI_AGENT_LORA_DATA"])
    output = Path("artifacts/week5/integration-lora-smoke")

    result = main(
        [
            "check",
            "--config",
            "configs/week5_qwen3vl_qlora.toml",
            "--data",
            str(data_dir),
            "--output",
            str(output),
        ]
    )

    manifest = json.loads((output / "run-manifest.json").read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["adapter_reloaded"] is True
    assert manifest["structured_generation"] is True
    assert manifest["peak_memory_bytes"] > 0
