"""Exception hierarchy.

Two failure modes are deliberately kept separate:

* :class:`StructureError` -- the inputs themselves are illegal (wrong FCS
  version/datatype, unparseable plan, unknown channels, gate cycles, ...).
  No per-event analysis is possible and the CLI exits non-zero.

* :class:`SolverError` -- the inputs are legal but the LP backend returned
  something other than optimal/infeasible for a particular event.  The event
  is reported as ``SOLVER_ERROR`` instead of being silently dropped.
"""


class SatflowError(Exception):
    """Base class for all satflow errors."""


class StructureError(SatflowError):
    """Illegal FCS file or illegal analysis plan."""


class SolverError(SatflowError):
    """The linear-programming backend failed on a well-formed problem."""
