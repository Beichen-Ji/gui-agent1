"""Headless deterministic desktop simulation with no real input emission."""

from gui_agent.simulation.apps import (
    APP_IDS,
    APPLICATIONS,
    AppId,
    ApplicationDefinition,
    ControlDefinition,
    RelativeBox,
)
from gui_agent.simulation.harness import (
    ObservationMode,
    SimulatedActionExecutor,
    SimulatedDesktop,
    SimulatedObservationSource,
    SimulationPolicy,
)
from gui_agent.simulation.render import LayoutControl, layout_desktop, render_desktop
from gui_agent.simulation.state import (
    DEFAULT_TESTBED_ROOT,
    DEMO_CONTENT,
    DEMO_FILENAME,
    FAULT_PROFILES,
    FaultProfile,
    TestbedState,
)

__all__ = [
    "APPLICATIONS",
    "APP_IDS",
    "DEFAULT_TESTBED_ROOT",
    "DEMO_CONTENT",
    "DEMO_FILENAME",
    "FAULT_PROFILES",
    "AppId",
    "ApplicationDefinition",
    "ControlDefinition",
    "FaultProfile",
    "LayoutControl",
    "ObservationMode",
    "RelativeBox",
    "SimulatedActionExecutor",
    "SimulatedDesktop",
    "SimulatedObservationSource",
    "SimulationPolicy",
    "TestbedState",
    "layout_desktop",
    "render_desktop",
]
