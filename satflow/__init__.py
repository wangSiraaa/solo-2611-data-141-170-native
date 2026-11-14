"""satflow: feasibility-aware analysis of saturated flow-cytometry FCS data.

The package never treats the detector saturation ceiling as a precise
measurement.  For every event it builds the *joint* feasible set of
non-negative marker vectors ``x`` consistent with ``y = S x`` and the
observation bands (two-sided for normal detectors, lower-bound-only for
saturated ones), then classifies closed-rectangle gates against that whole
set rather than against a single compensated point estimate.
"""

__version__ = "1.0.0"
