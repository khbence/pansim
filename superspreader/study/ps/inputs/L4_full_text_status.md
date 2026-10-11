# Full-text status for the starter evidence table

13 of the 18 starter rows were checked against the full text. Five were not, and they stay `verified_full_text = False`.

Verified, with the locator written into `published_evidence_verified.csv`: E01, E02, E03, E04, E05, E06, E07, E10, E13, E14, E16, E17, E18.

Not verified:

| row | paper | why |
|---|---|---|
| E08, E09 | Nikolay 2019, NEJM, Nipah reproduction numbers | Europe PMC full text for PMC6547369 returned HTTP 500, and the PDF render returned HTTP 403 |
| E11, E12 | Pan 2026, IJID, Nipah superspreading model | Europe PMC lists it as not open access. The ScienceDirect PDF returned HTTP 403, so no k was added |
| E15 | Garg 2025, NEJM, US H5N1 household contacts | no open full text could be fetched |

Two checks that change the starter, both from full text:

- E03. Adam 2020 does report k. The observed-offspring estimate is 0.43 (95% CI 0.29 to 0.67), with R 0.58 (0.45 to 0.72). A cluster-size sensitivity analysis gives k 0.33 (0.14 to 0.98). The 19% (15 to 24%) share is confirmed.
- E16. The viral-load slope is per log10 copies per mL. Table 3 gives an adjusted odds ratio of 1.29 (1.10 to 1.50); the abstract rounds this to 1.3 (1.1 to 1.5). The 12% stratum is index viral load below 1e6 copies/mL and the 24% stratum is 1e10 copies/mL or higher.

`published_k_from_R7.csv` is one row per study from Wegehaupt 2023 Tables 1 and 2. Where a study reports several populations, the row used in a figure is the one named in `which_estimate`, and the others are in `other_estimates`. `published_k_by_setting_R6.csv` has both of Chen 2025's tables. The cluster-ascertainment table is the one behind the abstract (retail 0.05, household 1.11). The individual-offspring table is a sensitivity analysis and is not the same quantity.
