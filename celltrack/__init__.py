"""celltrack: adjacent-frame lineage tracking over pre-segmented label TIFFs.

Pipeline
--------
1. ``series``   -- parse series.json, validate frames, load label images.
2. ``features`` -- per-object centroid / area / morphology in um, with
   provenance back to (frame_id, label) and (x, y) pixels.
3. ``candidates`` -- enumerate hypothesised continuation / division / birth /
   disappearance events between each adjacent frame pair, each with a support
   gate and an explicit cost.
4. ``solve``    -- jointly select one event per object (continuation, division,
   disappearance) or one birth per observation, using scipy MILP; enumerate
   near-equivalent alternative optima and flag undecided objects.
5. ``lineage``  -- build global contiguous track ids and mother/daughter links,
   verifying observation-uniqueness, parent-uniqueness, and acyclicity.
6. ``report``   -- write object / event / lineage / mother-daughter tables.

Only adjacent frames are linked.  Births and disappearances are legitimate
events, never "failures" to be papered over with forced links.
"""

__version__ = "0.1.0"
