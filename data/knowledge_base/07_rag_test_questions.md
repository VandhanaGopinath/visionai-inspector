# RAG Evaluation Question Set

Use this set to test retrieval quality, multi-document reasoning, and hallucination avoidance. Expected grounding documents are noted for each question so you can check retrieval precision, not just answer quality.

## 1. Basic Retrieval

1. What is contamination? *(→ DOC-002)*
2. What is a broken_small defect? *(→ DOC-002)*
3. What is a broken_large defect? *(→ DOC-002)*
4. What equipment is required for bottle inspection? *(→ DOC-001)*
5. What is the definition of "Critical" severity? *(→ DOC-003)*
6. What should be logged after every inspection? *(→ DOC-001)*
7. What is the first step in the CAPA process after a defect is confirmed? *(→ DOC-004)*

## 2. Inspection Procedure Questions

8. How should a bottle be visually inspected, step by step? *(→ DOC-001)*
9. What lighting conditions are required for inspection? *(→ DOC-001)*
10. What conditions require a bottle to be rejected? *(→ DOC-003)*
11. What should an inspector do after finding contamination? *(→ DOC-002, DOC-001)*
12. When should a bottle be reinspected rather than immediately accepted or rejected? *(→ DOC-003)*

## 3. Reasoning Questions

13. A bottle has a small visible chip at the rim. What should happen to it? *(→ DOC-002 broken_small, DOC-003)*
14. If the automated model detects contamination but the image is unclear, what should the inspector do? *(→ DOC-005, DOC-002)*
15. When should a suspected defect be escalated instead of simply rejected? *(→ DOC-003 Section 5, DOC-001 Section 11)*
16. Why might a broken_large defect require escalation even though the reject decision itself is obvious? *(→ DOC-002, DOC-004)*
17. If a human inspector overrides the automated system's "anomalous" flag to "accept," what should happen? *(→ DOC-003 Section 6)*

## 4. Multi-Document Questions

18. A bottle has suspected contamination. What is the defect definition, what makes it rejectable, and what corrective action should be taken? *(→ DOC-002, DOC-003, DOC-004)*
19. A bottle shows a large structural break. Walk through the full process from detection to CAPA documentation. *(→ DOC-001, DOC-002, DOC-003, DOC-004)*
20. The automated system and a human inspector disagree about whether a bottle has a defect — what does the SOP say to do, what does the QC document say about escalation, and how should this be documented for CAPA purposes? *(→ DOC-001, DOC-003, DOC-005, DOC-004)*
21. What is the relationship between defect severity (Document 3) and the specific rejection criteria for contamination (Document 2)?

## 5. Hallucination / Unanswerable Questions

These should produce an "the provided documentation does not contain enough information to answer this" response, not an invented number or procedure:

22. What exact temperature should the inspection facility be maintained at?
23. What conveyor belt speed should be used during inspection?
24. What chemical concentration is used to clean the filling equipment?
25. What is the exact percentage sampling rate required for reinspection of accepted bottles?
26. What ISO standard number governs this inspection process?
27. How many bottles per hour can this inspection line process?
28. What is the exact anomaly-score threshold used by the vision model? *(This lives in your vision system's config, not in these documents — good test that the LLM doesn't blend vision-system implementation detail with document content.)*
29. What brand of camera or lighting equipment should be used?
30. What is the shelf life of the bottled product?

## How to Use This Set

- For questions 1–17: check that retrieval returns the correct `document_id`/`section` combination, and that the answer doesn't mix in facts from an unrelated defect type.
- For questions 18–21: check that retrieval pulls chunks from *all* the noted documents, not just the first match — this is the best test of your chunking/metadata design.
- For questions 22–30: check that the LLM explicitly declines rather than guessing a plausible-sounding number. This is your key hallucination-resistance demonstration for the portfolio writeup.
