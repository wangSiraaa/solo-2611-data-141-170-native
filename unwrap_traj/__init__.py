"""Periodic molecular-trajectory unwrapping.

The package reconstructs finite molecules from folded coordinates by searching
integer lattice images that satisfy the bond graph, then continues each
molecule's centre-of-mass trajectory across frames only when the lattice
translation is unique.
"""

from .core import Cell, AnalysisOptions, analyze_dataset
from .validate import InputError, load_inputs

__all__ = [
    "Cell",
    "AnalysisOptions",
    "analyze_dataset",
    "InputError",
    "load_inputs",
]

__version__ = "1.0.0"
