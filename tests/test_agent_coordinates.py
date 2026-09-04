import pytest

from gui_agent.agent.types import (
    ClickAction,
    DragAction,
    HotkeyAction,
    ScrollAction,
    TypeTextAction,
)
from gui_agent.types import ScreenRegion


def _pointer_values(action: object) -> tuple[int, ...]:
    if isinstance(action, ClickAction):
        return (action.x, action.y)
    if isinstance(action, ScrollAction):
        assert action.x is not None and action.y is not None
        return (action.x, action.y)
    if isinstance(action, DragAction):
        return (action.start_x, action.start_y, action.end_x, action.end_y)
    raise AssertionError(f"not a pointer action: {action!r}")


@pytest.mark.parametrize(
    "action",
    [
        ClickAction(x=-1919, y=1079),
        ScrollAction(clicks=-4, x=-960, y=540),
        DragAction(start_x=-1920, start_y=0, end_x=-1, end_y=1079),
    ],
)
def test_pointer_actions_round_trip_on_negative_origin_multimonitor_bounds(
    action: ClickAction | ScrollAction | DragAction,
) -> None:
    from gui_agent.agent.coordinates import action_from_grid, action_to_grid

    bounds = ScreenRegion(left=-1920, top=0, width=1920, height=1080)

    grid_action = action_to_grid(action, bounds=bounds)
    restored = action_from_grid(grid_action, bounds=bounds)

    assert all(
        abs(actual - expected) <= 1
        for actual, expected in zip(
            _pointer_values(restored),
            _pointer_values(action),
            strict=True,
        )
    )


def test_coordinate_grid_maps_the_first_and_last_image_pixels_exactly() -> None:
    from gui_agent.agent.coordinates import action_from_grid, action_to_grid

    bounds = ScreenRegion(left=-100, top=20, width=800, height=600)

    assert action_to_grid(ClickAction(x=-100, y=20), bounds=bounds) == ClickAction(
        x=0,
        y=0,
    )
    assert action_to_grid(ClickAction(x=699, y=619), bounds=bounds) == ClickAction(
        x=999,
        y=999,
    )
    assert action_from_grid(ClickAction(x=999, y=999), bounds=bounds) == ClickAction(
        x=699,
        y=619,
    )


def test_coordinate_conversion_does_not_rewrite_non_pointer_actions() -> None:
    from gui_agent.agent.coordinates import action_from_grid, action_to_grid

    bounds = ScreenRegion(left=0, top=0, width=800, height=600)
    actions = (TypeTextAction(text="hello"), HotkeyAction(keys=("ctrl", "a")))

    for action in actions:
        assert action_to_grid(action, bounds=bounds) is action
        assert action_from_grid(action, bounds=bounds) is action


@pytest.mark.parametrize(
    "action",
    [ClickAction(x=-1, y=20), ClickAction(x=100, y=20)],
)
def test_coordinate_conversion_rejects_pixels_outside_bounds(action: ClickAction) -> None:
    from gui_agent.agent.coordinates import action_to_grid

    with pytest.raises(ValueError, match="outside bounds"):
        action_to_grid(action, bounds=ScreenRegion(0, 0, 100, 50))


def test_coordinate_conversion_rejects_partial_scroll_coordinates() -> None:
    from gui_agent.agent.coordinates import action_to_grid

    with pytest.raises(ValueError, match="provided together"):
        action_to_grid(
            ScrollAction(clicks=2, x=10),
            bounds=ScreenRegion(0, 0, 100, 50),
        )
