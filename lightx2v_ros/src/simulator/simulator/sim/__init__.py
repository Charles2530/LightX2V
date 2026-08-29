from typing import TYPE_CHECKING

from .base_env import BaseSimEnv, Observation

if TYPE_CHECKING:
    from .node import SimulatorNode

__all__ = [
    "BaseSimEnv",
    "Observation",
    "SimulatorNode",
    "rgb_to_image_msg",
    "run_simulator_node",
]


def __getattr__(name):
    if name in {"SimulatorNode", "rgb_to_image_msg", "run_simulator_node"}:
        from . import node

        return getattr(node, name)
    raise AttributeError(name)
