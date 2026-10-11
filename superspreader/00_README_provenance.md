# ACT-049 README and provenance

Work packet: ACT-049 superspreader research brief. Run date 9 October 2026 (single bounded run). Output folder: Management/work-packages/ACT-049/.

## Files

| File | Content |
|---|---|
| 01_literature_review.md | PRIMARY deliverable: central question, definitions, per-pathogen graded conclusions, thin-evidence flags, annotated reading list (30 items) |
| 02_brief.md | Brief (about 2 pages): answer first, then secondary sections, PanSim recommendation, experiments |
| 03_pansim_options_table.md | Implementation options table (pros, cons, data needs, calibration, identifiability) |
| 04_pansim_experiments.md | Four proposed PanSim experiments |
| 05_k_estimates_and_modelling.md | Secondary sections (a)-(c): definitions recap, published k per pathogen, modelling approaches |
| ACT-049_citation_verification.csv | For each of the 30 reading-list items: DOI, whether doi.org redirects, whether Crossref has a matching record, Crossref first authors, year, title, journal, and whether the abstract was read |

## Citation verification (what "verified" means here)

For each reading-list item: (1) the DOI was returned by a Crossref record whose title, first authors, year and journal match the cited work; (2) doi.org returns a redirect for the DOI; (3) the abstract was read in PubMed (all 30); (4) full texts were NOT read. Claims in annotations are limited to what the abstracts state, plus clearly labelled interpretation. Items whose DOI failed verification during the run (wrong work or no record) were corrected or dropped and are not in the list. Several DOIs first recalled from memory were wrong (for example one that resolved to a different H5N1 paper and one that resolved to a paper on a different disease) and were replaced after lookup. Versioned Wellcome Open Research DOIs (R3, R9, R10) resolve to specific versions; newer versions of those articles exist.

Seven further verified papers are named but not annotated in the review (six in section 10 and one, Gupta et al. 2026, in section 5).

Two web-sourced situation updates (WHO Western Pacific avian influenza weekly updates, July 2026; UN India and other relays of the WHO statement on the West Bengal Nipah event, January 2026) were seen as search results and not independently re-fetched. They are context, not reading-list items, and carry no DOI. One was labelled as conflicting with media accounts of case numbers; recheck before relying on it.

## Constraint check

| Constraint | Status |
|---|---|
| Public literature only | Met. All sources are published papers or public health agency updates |
| No unpublished partner Results or Background | Met. The packet text was read only for scope and requirements; no partner results or background were used. PanSim internals, ACT-004 outputs and ACT-047 results were not consulted, so PanSim feasibility statements are assumptions |
| No outbound communications | Met. No emails, messages, posts, shares or author contact. Files were only written to the ACT-049 folder |
| No compute jobs | Met in substance. Only literature and DOI lookups, plus one tiny deterministic calculation (gamma-share and negative-binomial zero-probability values for the reference tables in file 05). No simulation, PanSim run or job. If István reads even this as outside the constraint, delete the "Reference table computed here" subsection of 05 and the Nipah zero-offspring comparison in 01 and 05; nothing else depends on them |
| Write only to ACT-049 folder | Met. No other Drive folder was modified and no registers were edited |
| No em-dashes or en-dashes | Met. All outputs were scanned for these characters before upload, including in imported titles |

## Deviations and blockers

1. The Drive file id given in the instruction message (1QwpKafcqfrpSCAJyEZh4R8BNCIxUIsju) returned "not found". A search found a file with the stated name, ACT-049-superspreader-research.md, in the parent work-packages folder (a different file id, created 9 October 2026, matching the "revision v2" text). That version was used, together with the instruction message, which takes precedence. If a later revision exists beyond this one, differences are not reflected here.
2. Nipah primary evidence is minimal and the one explicit superspreading model (Pan et al. 2026) and the systematic review (Naidoo et al. 2026) were read from abstracts only because full texts were not available through the connectors used. The k value in the Pan paper was not verified.
3. The literature search was bounded and not exhaustive (PubMed plus Crossref; no Embase, Web of Science or preprint servers). It is not a registered systematic review.

## Suggested spot checks for aurorai-reviewer

- Open ACT-049_citation_verification.csv and check any five rows against the DOI landing pages; items R22, R23 (2026) and R3, R9, R10 (versioned DOIs) are the most useful to check.
- Check the numbers quoted in annotations for R4, R6, R14, R16, R21 against the abstracts.
- Balance check: section 4.2-4.4 each give "for" and "against or qualifying" evidence; section 7 grades; section 8 lists contested points.
- Confirm the claim in section 5 of the review that the 72% Poisson zero-offspring figure is a worker calculation, not a published value.
