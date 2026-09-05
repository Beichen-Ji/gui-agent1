import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from examples import agent_demo, gui_testbed
from gui_agent import cli
from gui_agent.agent.loop import AgentRunResult
from gui_agent.agent.types import (
    AgentDecision,
    StepResult,
    TypeTextAction,
)
from gui_agent.types import ScreenRegion


class RuntimeProbe:
    def __init__(self, result: AgentRunResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None, int]] = []

    def run(
        self,
        goal: str,
        *,
        success_criteria: str | None = None,
        max_steps: int = 10,
    ) -> AgentRunResult:
        self.calls.append((goal, success_criteria, max_steps))
        return self.result


def run_result(
    *,
    status: str = "succeeded",
    decisions: tuple[AgentDecision, ...] = (),
    results: tuple[StepResult, ...] = (),
) -> AgentRunResult:
    return AgentRunResult(
        goal="Open Browser",
        status=status,  # type: ignore[arg-type]
        message="completed",
        plan=None,
        observation=None,
        decisions=decisions,
        results=results,
    )


@pytest.mark.parametrize("command", [[], ["run", "--help"]])
def test_cli_help_lists_commands_and_run_safety_options(
    command: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(command if command else ["--help"])

    assert captured.value.code == 0
    output = capsys.readouterr().out
    if not command:
        for name in ("dataset", "model-smoke", "run"):
            assert name in output
    else:
        for option in (
            "--task",
            "--task-id",
            "--provider",
            "--monitor",
            "--region",
            "--max-steps",
            "--execute",
            "--allow-remote-image",
            "--trace-dir",
        ):
            assert option in output


@pytest.mark.parametrize("command", ["dataset", "model-smoke"])
def test_delegated_help_works_outside_repository_directory(
    command: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from gui_agent.cli import main; "
                f"raise SystemExit(main(['{command}', '--help']))"
            ),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--provider", "fake"],
        ["run", "--task", "Open Browser", "--provider", "invalid"],
        ["run", "--task", "Open Browser", "--max-steps", "0"],
        ["run", "--task", "Open Browser", "--monitor", "0"],
    ],
)
def test_cli_rejects_missing_task_invalid_provider_and_non_positive_limits(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(argv)

    assert captured.value.code == 2


def test_cli_defaults_to_dry_run_and_passes_validated_runtime_options() -> None:
    configs: list[cli.RunConfig] = []
    runner = RuntimeProbe(run_result())

    def runtime_factory(
        config: cli.RunConfig,
        _input_fn: Callable[[str], str],
    ) -> RuntimeProbe:
        configs.append(config)
        return runner

    exit_code = cli.main(
        [
            "run",
            "--task",
            "Open Browser",
            "--provider",
            "fake",
            "--max-steps",
            "4",
        ],
        runtime_factory=runtime_factory,
    )

    assert exit_code == 0
    assert len(configs) == 1
    assert configs[0].execute is False
    assert configs[0].provider == "fake"
    assert configs[0].monitor == 1
    assert configs[0].region is None
    assert runner.calls == [("Open Browser", None, 4)]


def test_cli_passes_configured_success_criteria_but_not_for_free_text() -> None:
    task_runner = RuntimeProbe(run_result())
    free_runner = RuntimeProbe(run_result())

    assert cli.main(
        ["run", "--task-id", "open-browser", "--provider", "fake"],
        runtime_factory=lambda _config, _input: task_runner,
    ) == 0
    assert cli.main(
        ["run", "--task", "Open Browser", "--provider", "fake"],
        runtime_factory=lambda _config, _input: free_runner,
    ) == 0

    assert task_runner.calls[0][1] == "The Browser area is visible and marked open."
    assert free_runner.calls[0][1] is None


@pytest.mark.parametrize(
    "capture_args",
    [
        ["--monitor", "1", "--region", "0", "0", "760", "520"],
        ["--region", "0", "0", "0", "520"],
        ["--region", "0", "0", "760", "-1"],
    ],
)
def test_cli_rejects_conflicting_or_non_positive_capture_regions(
    capture_args: list[str],
) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(
            [
                "run",
                "--task",
                "Open Browser",
                "--provider",
                "qwen",
                *capture_args,
            ]
        )

    assert captured.value.code == 2


def test_cli_passes_absolute_capture_region_without_a_monitor() -> None:
    configs: list[cli.RunConfig] = []
    runner = RuntimeProbe(run_result())

    def runtime_factory(
        config: cli.RunConfig,
        _input_fn: Callable[[str], str],
    ) -> RuntimeProbe:
        configs.append(config)
        return runner

    exit_code = cli.main(
        [
            "run",
            "--task",
            "Open Browser",
            "--provider",
            "qwen",
            "--region",
            "-60",
            "-20",
            "760",
            "520",
        ],
        runtime_factory=runtime_factory,
    )

    assert exit_code == 0
    assert len(configs) == 1
    assert configs[0].monitor is None
    assert configs[0].region == ScreenRegion(-60, -20, 760, 520)


def test_cli_passes_optional_adapter_only_to_qwen_runtime(tmp_path: Path) -> None:
    configs: list[cli.RunConfig] = []
    runner = RuntimeProbe(run_result())

    def runtime_factory(
        config: cli.RunConfig,
        _input_fn: Callable[[str], str],
    ) -> RuntimeProbe:
        configs.append(config)
        return runner

    adapter = tmp_path / "adapter-output"
    assert cli.main(
        [
            "run",
            "--task",
            "Open Browser",
            "--provider",
            "qwen",
            "--adapter",
            str(adapter),
        ],
        runtime_factory=runtime_factory,
    ) == 0

    assert configs[0].adapter_path == adapter

    with pytest.raises(SystemExit) as captured:
        cli.main(
            [
                "run",
                "--task",
                "Open Browser",
                "--provider",
                "fake",
                "--adapter",
                str(adapter),
            ]
        )
    assert captured.value.code == 2


def test_cli_requires_explicit_remote_image_permission() -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(
            [
                "run",
                "--task",
                "Open Browser",
                "--provider",
                "openai-compatible",
            ]
        )

    assert captured.value.code == 2


def test_fake_execute_path_denies_wrong_confirmation_without_desktop_input(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(
        [
            "run",
            "--task",
            "Open Browser",
            "--provider",
            "fake",
            "--execute",
        ],
        input_fn=lambda _prompt: "no",
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert '"failure_stage": "policy"' in output


def test_cli_loads_exactly_the_five_week4_task_ids() -> None:
    tasks = cli.load_task_definitions(cli.DEFAULT_TASKS_PATH)

    assert set(tasks) == {
        "open-browser",
        "search-content",
        "open-file",
        "send-message",
        "close-app",
    }
    assert all(task.actions for task in tasks.values())


def test_cli_trace_redacts_typed_text_and_writes_only_when_requested(
    tmp_path: Path,
) -> None:
    secret = "potential-password-value"
    action = TypeTextAction(text=secret)
    planned = AgentDecision(
        current_step_id="step-1",
        rationale_summary="Type into the local testbed",
        action=action,
        expected_outcome="The local inbox updates",
    )
    completed = StepResult(
        step_index=0,
        action=action,
        status="dry_run",
        message="type_text recorded without desktop input",
    )
    runner = RuntimeProbe(run_result(decisions=(planned,), results=(completed,)))

    def runtime_factory(
        _config: cli.RunConfig,
        _input_fn: Callable[[str], str],
    ) -> RuntimeProbe:
        return runner

    trace_dir = tmp_path / "trace"
    exit_code = cli.main(
        [
            "run",
            "--task",
            "Send a local message",
            "--provider",
            "fake",
            "--trace-dir",
            str(trace_dir),
        ],
        runtime_factory=runtime_factory,
    )

    trace = (trace_dir / "run-summary.json").read_text(encoding="utf-8")
    assert exit_code == 0
    assert secret not in trace
    assert '"action_kinds": [\n    "type_text"' in trace


def test_agent_demo_forwards_arguments_to_run_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def cli_probe(argv: list[str]) -> int:
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(
        agent_demo,
        "cli_main",
        cli_probe,
    )

    assert agent_demo.main(["--task", "Open Browser", "--provider", "fake"]) == 0
    assert calls == [["run", "--task", "Open Browser", "--provider", "fake"]]


def test_testbed_state_keeps_messages_in_memory_and_files_in_sandbox(
    tmp_path: Path,
) -> None:
    root = tmp_path / "testbed"
    state = gui_testbed.TestbedState(root)

    state.open_browser()
    search_result = state.search("week4 safe search")
    file_content = state.open_file("week4-demo.txt")
    state.send_message("week4 test message")
    snapshot = state.snapshot()

    assert search_result == "Search result: week4 safe search"
    assert "WEEK4_DEMO_READY" in file_content
    assert snapshot["browser_open"] is True
    assert snapshot["search_query"] == "week4 safe search"
    assert snapshot["opened_file"] == "week4-demo.txt"
    assert snapshot["messages"] == ("week4 test message",)
    assert list(root.iterdir()) == [root / "week4-demo.txt"]


def test_testbed_rejects_file_traversal_and_tracks_its_own_close_state(
    tmp_path: Path,
) -> None:
    state = gui_testbed.TestbedState(tmp_path / "testbed")

    with pytest.raises(ValueError, match="inside the testbed directory"):
        state.open_file("../outside.txt")

    state.close()
    assert state.snapshot()["closed"] is True
