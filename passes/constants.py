"""Physical and model constants.

All lengths in the geometry kernel are kilometres.  Station height is read in
metres and converted once, at input time, so that km/m mixing is impossible
later in the chain.
"""

from __future__ import annotations

SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0
CENTURIES_PER_DAY = 1.0 / 36525.0

# WGS84
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B_KM = WGS84_A_KM * (1.0 - WGS84_F)

# Numerical defaults.  The public interface also reads these from the plan.
DEFAULT_BOUNDARY_TOLERANCE_S = 0.2
DEFAULT_SEARCH_STEP_S = 0.25
DEFAULT_VERIFY_STEP_S = 0.02
DEFAULT_OBSERVATION_STEP_S = 5.0
DEFAULT_ROOT_XTOL_S = 1.0e-4
DEFAULT_EXTREMUM_XTOL_S = 1.0e-4

# Roots closer than this are treated as the same geometric crossing.  This is
# far below the plan tolerance and only suppresses floating-point duplicates.
ROOT_MERGE_TOL_S = 1.0e-6

# A raw interval shorter than this is a numerical tangency, not a window.
TANGENCY_EPS_S = 1.0e-6
