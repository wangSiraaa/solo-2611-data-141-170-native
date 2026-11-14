"""Visible pass/sub-window solver for the synthetic ground-station fixture.

The model is deliberately the one specified by the task README:

* SGP4 propagates the TLE and returns TEME position in kilometres.
* UTC is treated as UT1, polar motion is zero, and the supplied GMST
  polynomial rotates TEME into ECEF.
* The WGS84 ellipsoid and the station ellipsoid height (metres) define the
  site vector; ENU is obtained with the exact local basis.
* Visibility is geometric elevation >= a periodic piecewise-linear horizon.
* No refraction, no leap-second table, no EOP download.
"""

__version__ = "1.0.0"
