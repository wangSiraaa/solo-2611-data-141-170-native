# Comparable-month price index — `single_bridge`

Base month **2025-01** (index = 100). Additive log-price model with persistent SKU levels and supplier levels; families are **not** equivalence keys.

## Month index

| month | index (base 100) | status | 95% interval | replicates identified | interval reportable |
|---|---:|---|---|---:|---|
| 2025-01 | 100.00 | identified | [100.00, 100.00] | 500/500 (100.0%) | True |
| 2025-02 | 102.13 | identified | [101.27, 102.71] | 500/500 (100.0%) | True |
| 2025-03 | 105.28 | identified | [104.40, 105.82] | 500/500 (100.0%) | True |
| 2025-04 | 106.82 | insufficient_replicate_support | withheld | 342/500 (68.4%) | False |
| 2025-05 | 109.15 | insufficient_replicate_support | withheld | 342/500 (68.4%) | False |
| 2025-06 | 113.25 | insufficient_replicate_support | withheld | 342/500 (68.4%) | False |

## Identification evidence

- Design: 18 parameters, 18 exactly independent columns, 0 null direction(s).
- Normalization: month effect at base = 0; supplier level `supplier-1` = 0. These are the only normalizations.
- Item–month graph components (edges are real quotes only):
  - component 0: items ['A-new', 'A-old', 'B-new', 'B-old', 'C-new', 'C-old']; months ['2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06']; suppliers ['supplier-1', 'supplier-2', 'supplier-3', 'supplier-4', 'supplier-5', 'supplier-6', 'supplier-7', 'supplier-8']

- **2025-04** is identified only through critical bridge supplier(s) ['supplier-1'].
  - bridge quotes for `supplier-1`: q0008 (2025-03 A-new 122.62), q0010 (2025-03 B-new 171.06), q0012 (2025-04 A-old 101.41), q0014 (2025-04 B-old 160.69)
- **2025-05** is identified only through critical bridge supplier(s) ['supplier-1'].
  - bridge quotes for `supplier-1`: q0008 (2025-03 A-new 122.62), q0010 (2025-03 B-new 171.06), q0012 (2025-04 A-old 101.41), q0014 (2025-04 B-old 160.69)
- **2025-06** is identified only through critical bridge supplier(s) ['supplier-1'].
  - bridge quotes for `supplier-1`: q0008 (2025-03 A-new 122.62), q0010 (2025-03 B-new 171.06), q0012 (2025-04 A-old 101.41), q0014 (2025-04 B-old 160.69)

## SKU level spreads within families (measured, not equality claims)

| family | pair | ratio B/A | 95% interval | status |
|---|---|---:|---|---|
| family-A | A-new / A-old | 1.2194 | — | insufficient_replicate_support |
| family-B | B-new / B-old | 1.0866 | — | insufficient_replicate_support |
| family-C | C-new / C-old | 0.9613 | — | insufficient_replicate_support |

## Leave-one-supplier-out stability

| removed | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 |
|---|---:|---:|---:|---:|---:|---:|
| supplier-1 | 100.00 | 102.18 | 105.44 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-2 | 100.00 | 102.09 | 105.21 | 106.77 | 109.29 | 113.22 |
| supplier-3 | 100.00 | 102.05 | 105.19 | 106.69 | 108.89 | 112.95 |
| supplier-4 | 100.00 | 102.35 | 105.51 | 107.13 | 109.48 | 113.73 |
| supplier-5 | 100.00 | 102.12 | 105.28 | 106.75 | 109.21 | 113.14 |
| supplier-6 | 100.00 | 101.97 | 105.14 | 106.55 | 108.76 | 112.76 |
| supplier-7 | 100.00 | 102.28 | 105.36 | 106.92 | 109.48 | 113.60 |
| supplier-8 | 100.00 | 102.00 | 105.13 | 106.74 | 108.71 | 113.00 |

## Warnings

- month 2025-04: identified only because supplier(s) ['supplier-1'] bridge otherwise disconnected SKU groups
- month 2025-05: identified only because supplier(s) ['supplier-1'] bridge otherwise disconnected SKU groups
- month 2025-06: identified only because supplier(s) ['supplier-1'] bridge otherwise disconnected SKU groups
- month 2025-04: point index is identified in the full sample but only 68.4% of resampled trajectory sets retain identification; interval withheld (evidence insufficient)
- month 2025-05: point index is identified in the full sample but only 68.4% of resampled trajectory sets retain identification; interval withheld (evidence insufficient)
- month 2025-06: point index is identified in the full sample but only 68.4% of resampled trajectory sets retain identification; interval withheld (evidence insufficient)

## Limitations

- The model is additive on log prices with persistent item and supplier levels; if relative SKU prices drift within the six-month window the month effect is biased toward the average drift.
- Families only record a replacement relationship. Any old/new level difference reported here is a measured persistent level difference from overlap, not a quality or equivalence claim.
- Identification is exact and conditional on the observed quote pattern; absence of overlap is reported as insufficient support rather than bridged with external product prices.
- Supplier-level disturbance dependence is handled only by whole-trajectory resampling; no parametric covariance model is fitted.
- Percentile bootstrap intervals are descriptive and may be optimistic when the bridge rests on a single supplier, which is shown explicitly by the delete-one stability table.
- No forecasting, hedonic regression, rolling analysis or automatic product matching is performed.
