from gui_agent.agent.types import AgentAction, ClickAction, DragAction, ScrollAction
from gui_agent.types import ScreenRegion

DEFAULT_COORDINATE_GRID_SIZE = 1000


def _validate_grid_size(grid_size: int) -> None:
    if isinstance(grid_size, bool) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")


def _pixel_to_grid(value: int, *, origin: int, extent: int, grid_size: int) -> int:
    offset = value - origin
    if not 0 <= offset < extent:
        raise ValueError("pointer coordinate is outside bounds")
    if extent == 1:
        return 0
    return (offset * (grid_size - 1) + (extent - 1) // 2) // (extent - 1)


def _grid_to_pixel(value: int, *, origin: int, extent: int, grid_size: int) -> int:
    if not 0 <= value < grid_size:
        raise ValueError(f"grid pointer coordinates must be between 0 and {grid_size - 1}")
    if extent == 1:
        return origin
    offset = (value * (extent - 1) + (grid_size - 1) // 2) // (grid_size - 1)
    return origin + offset


def _convert_action(
    action: AgentAction,
    *,
    bounds: ScreenRegion,
    grid_size: int,
    to_grid: bool,
) -> AgentAction:
    _validate_grid_size(grid_size)

    def x(value: int) -> int:
        if to_grid:
            return _pixel_to_grid(
                value,
                origin=bounds.left,
                extent=bounds.width,
                grid_size=grid_size,
            )
        return _grid_to_pixel(
            value,
            origin=bounds.left,
            extent=bounds.width,
            grid_size=grid_size,
        )

    def y(value: int) -> int:
        if to_grid:
            return _pixel_to_grid(
                value,
                origin=bounds.top,
                extent=bounds.height,
                grid_size=grid_size,
            )
        return _grid_to_pixel(
            value,
            origin=bounds.top,
            extent=bounds.height,
            grid_size=grid_size,
        )

    if isinstance(action, ClickAction):
        return action.model_copy(update={"x": x(action.x), "y": y(action.y)})
    if isinstance(action, DragAction):
        return action.model_copy(
            update={
                "start_x": x(action.start_x),
                "start_y": y(action.start_y),
                "end_x": x(action.end_x),
                "end_y": y(action.end_y),
            }
        )
    if isinstance(action, ScrollAction):
        if action.x is None and action.y is None:
            return action
        if action.x is None or action.y is None:
            raise ValueError("scroll x and y must be provided together")
        return action.model_copy(update={"x": x(action.x), "y": y(action.y)})
    return action


def action_to_grid(
    action: AgentAction,
    *,
    bounds: ScreenRegion,
    grid_size: int = DEFAULT_COORDINATE_GRID_SIZE,
) -> AgentAction:
    """Convert absolute image/desktop pixels to an inclusive relative grid."""
    return _convert_action(action, bounds=bounds, grid_size=grid_size, to_grid=True)


def action_from_grid(
    action: AgentAction,
    *,
    bounds: ScreenRegion,
    grid_size: int = DEFAULT_COORDINATE_GRID_SIZE,
) -> AgentAction:
    """Convert an inclusive relative grid action to absolute image/desktop pixels."""
    return _convert_action(action, bounds=bounds, grid_size=grid_size, to_grid=False)


__all__ = ["DEFAULT_COORDINATE_GRID_SIZE", "action_from_grid", "action_to_grid"]
