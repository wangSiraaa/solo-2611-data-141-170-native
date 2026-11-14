"""Offline adaptive inspection-policy analyzer.

Loads and validates the decision model, enumerates *all* reachable
observation histories exactly (arbitrary-precision rational arithmetic),
and solves for a policy that minimises

    E[ inspection fees actually paid + terminal action loss ]

subject to (a) each test bought at most once on a path, (b) ``requires``
satisfied before a test may be bought, (c) cumulative fees along every
branch within the budget, and (d) posterior consistency with the frozen
joint distribution of test outcomes.

This package deliberately does **not** train models, learn online, or act
as a generic decision platform.
"""

from .model import Model, ModelError, load_model
from .solver import solve, solve_model, Node
from .replay import replay, ReplayError, replay_model
from .report import static_report, render_report, render_tree

__all__ = [
    "Model",
    "ModelError",
    "load_model",
    "solve",
    "solve_model",
    "Node",
    "replay",
    "replay_model",
    "ReplayError",
    "static_report",
    "render_report",
    "render_tree",
]

__version__ = "1.0.0"
