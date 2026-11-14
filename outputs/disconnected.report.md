# Comparable-month price index — `disconnected`

Base month **2025-01** (index = 100). Additive log-price model with persistent SKU levels and supplier levels; families are **not** equivalence keys.

## Month index

| month | index (base 100) | status | 95% interval | replicates identified | interval reportable |
|---|---:|---|---|---:|---|
| 2025-01 | 100.00 | identified | [100.00, 100.00] | 500/500 (100.0%) | True |
| 2025-02 | 102.13 | identified | [101.27, 102.69] | 500/500 (100.0%) | True |
| 2025-03 | 105.28 | identified | [104.41, 105.81] | 500/500 (100.0%) | True |
| 2025-04 | — | insufficient_support | withheld | 0/500 (0.0%) | False |
| 2025-05 | — | insufficient_support | withheld | 0/500 (0.0%) | False |
| 2025-06 | — | insufficient_support | withheld | 0/500 (0.0%) | False |

## Identification evidence

- Design: 18 parameters, 17 exactly independent columns, 1 null direction(s).
- Normalization: month effect at base = 0; supplier level `supplier-1` = 0. These are the only normalizations.
- Item–month graph components (edges are real quotes only):
  - component 0: items ['A-new', 'B-new', 'C-new']; months ['2025-04', '2025-05', '2025-06']; suppliers ['supplier-1', 'supplier-2', 'supplier-3', 'supplier-4', 'supplier-5', 'supplier-6', 'supplier-7', 'supplier-8']
  - component 1: items ['A-old', 'B-old', 'C-old']; months ['2025-01', '2025-02', '2025-03']; suppliers ['supplier-1', 'supplier-2', 'supplier-3', 'supplier-4', 'supplier-5', 'supplier-6', 'supplier-7', 'supplier-8']

- **2025-04 is not identifiable.** Confounding null direction: item levels {'A-new': '-1', 'B-new': '-1', 'C-new': '-1'}, month effects {'2025-04': '1', '2025-05': '1', '2025-06': '1'}. Witness quotes: ['q0013', 'q0015', 'q0016', 'q0017', 'q0018', 'q0019', 'q0020', 'q0021', 'q0022', 'q0032', 'q0034', 'q0035', 'q0036', 'q0037', 'q0038', 'q0051', 'q0053', 'q0054', 'q0055', 'q0056', 'q0057', 'q0058', 'q0059', 'q0060', 'q0070', 'q0072', 'q0073', 'q0074', 'q0075', 'q0076', 'q0089', 'q0091', 'q0092', 'q0093', 'q0094', 'q0095', 'q0096', 'q0097', 'q0098', 'q0108', 'q0110', 'q0111', 'q0112', 'q0113', 'q0114', 'q0127', 'q0129', 'q0130', 'q0131', 'q0132', 'q0133', 'q0134', 'q0135', 'q0136', 'q0146', 'q0148', 'q0149', 'q0150', 'q0151', 'q0152'].
- **2025-05 is not identifiable.** Confounding null direction: item levels {'A-new': '-1', 'B-new': '-1', 'C-new': '-1'}, month effects {'2025-04': '1', '2025-05': '1', '2025-06': '1'}. Witness quotes: ['q0013', 'q0015', 'q0016', 'q0017', 'q0018', 'q0019', 'q0020', 'q0021', 'q0022', 'q0032', 'q0034', 'q0035', 'q0036', 'q0037', 'q0038', 'q0051', 'q0053', 'q0054', 'q0055', 'q0056', 'q0057', 'q0058', 'q0059', 'q0060', 'q0070', 'q0072', 'q0073', 'q0074', 'q0075', 'q0076', 'q0089', 'q0091', 'q0092', 'q0093', 'q0094', 'q0095', 'q0096', 'q0097', 'q0098', 'q0108', 'q0110', 'q0111', 'q0112', 'q0113', 'q0114', 'q0127', 'q0129', 'q0130', 'q0131', 'q0132', 'q0133', 'q0134', 'q0135', 'q0136', 'q0146', 'q0148', 'q0149', 'q0150', 'q0151', 'q0152'].
- **2025-06 is not identifiable.** Confounding null direction: item levels {'A-new': '-1', 'B-new': '-1', 'C-new': '-1'}, month effects {'2025-04': '1', '2025-05': '1', '2025-06': '1'}. Witness quotes: ['q0013', 'q0015', 'q0016', 'q0017', 'q0018', 'q0019', 'q0020', 'q0021', 'q0022', 'q0032', 'q0034', 'q0035', 'q0036', 'q0037', 'q0038', 'q0051', 'q0053', 'q0054', 'q0055', 'q0056', 'q0057', 'q0058', 'q0059', 'q0060', 'q0070', 'q0072', 'q0073', 'q0074', 'q0075', 'q0076', 'q0089', 'q0091', 'q0092', 'q0093', 'q0094', 'q0095', 'q0096', 'q0097', 'q0098', 'q0108', 'q0110', 'q0111', 'q0112', 'q0113', 'q0114', 'q0127', 'q0129', 'q0130', 'q0131', 'q0132', 'q0133', 'q0134', 'q0135', 'q0136', 'q0146', 'q0148', 'q0149', 'q0150', 'q0151', 'q0152'].

## SKU level spreads within families (measured, not equality claims)

| family | pair | ratio B/A | 95% interval | status |
|---|---|---:|---|---|
| family-A | A-new / A-old | — | — | insufficient_support |
| family-B | B-new / B-old | — | — | insufficient_support |
| family-C | C-new / C-old | — | — | insufficient_support |

## Leave-one-supplier-out stability

| removed | 2025-01 | 2025-02 | 2025-03 | 2025-04 | 2025-05 | 2025-06 |
|---|---:|---:|---:|---:|---:|---:|
| supplier-1 | 100.00 | 102.18 | 105.44 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-2 | 100.00 | 102.09 | 105.21 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-3 | 100.00 | 102.05 | 105.19 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-4 | 100.00 | 102.35 | 105.51 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-5 | 100.00 | 102.11 | 105.28 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-6 | 100.00 | 101.95 | 105.14 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-7 | 100.00 | 102.26 | 105.34 | UNIDENT. | UNIDENT. | UNIDENT. |
| supplier-8 | 100.00 | 102.01 | 105.15 | UNIDENT. | UNIDENT. | UNIDENT. |

## Warnings

- month 2025-04: month effect is not identifiable relative to 2025-01; see confounding_direction
- month 2025-05: month effect is not identifiable relative to 2025-01; see confounding_direction
- month 2025-06: month effect is not identifiable relative to 2025-01; see confounding_direction
- month 2025-04: index not reported; no observed overlap supports it and no external bridge is used
- month 2025-05: index not reported; no observed overlap supports it and no external bridge is used
- month 2025-06: index not reported; no observed overlap supports it and no external bridge is used

## Limitations

- The model is additive on log prices with persistent item and supplier levels; if relative SKU prices drift within the six-month window the month effect is biased toward the average drift.
- Families only record a replacement relationship. Any old/new level difference reported here is a measured persistent level difference from overlap, not a quality or equivalence claim.
- Identification is exact and conditional on the observed quote pattern; absence of overlap is reported as insufficient support rather than bridged with external product prices.
- Supplier-level disturbance dependence is handled only by whole-trajectory resampling; no parametric covariance model is fitted.
- Percentile bootstrap intervals are descriptive and may be optimistic when the bridge rests on a single supplier, which is shown explicitly by the delete-one stability table.
- No forecasting, hedonic regression, rolling analysis or automatic product matching is performed.
