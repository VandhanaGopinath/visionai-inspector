---
document_id: DOC-003
document_type: qc_criteria
title: Quality Control and Acceptance Criteria
version: 1.0
created_for: "Portfolio AI/ML project — simulated industrial documentation"
source: project-created (general severity-tier concept loosely modeled on common manufacturing QC practice, not a specific named standard)
---

> **Note:** This is an example / simulated industrial inspection document created for an academic AI/ML portfolio project. The severity-tier concept (Minor/Major/Critical) used below is a general, widely-used manufacturing QC convention and is not attributed to any specific ISO, ASTM, or FDA standard. It is included to make the acceptance-decision logic realistic, not to represent an official requirement.

# Quality Control and Acceptance Criteria

This document defines the decision layer that sits above individual defect definitions: how severity is classified, what conditions make a bottle acceptable or unacceptable overall, and when a case must be escalated or reinspected.

## 1. Acceptable Product Conditions

**Section: Acceptable Conditions**

A bottle is considered acceptable when:

- No defect is detected by the automated vision system above the calibrated anomaly-score threshold, and
- No visible defect is identified in any human spot-check review, and
- Any minor cosmetic variation present (e.g., very slight, non-structural surface marks) is confirmed as within normal manufacturing variation rather than a listed defect category (see Document 2).

## 2. Unacceptable Product Conditions

**Section: Unacceptable Conditions**

A bottle is considered unacceptable when it matches the Rejection Criteria of any defect category defined in Document 2 (contamination, broken_small, broken_large), or when the automated system flags an anomaly that cannot be confidently resolved as a normal variation upon human review.

## 3. Defect Severity Levels

**Section: Severity Level Definitions**

*Simulated severity scale used for this project:*

- **Minor** — A cosmetic or superficial issue that does not affect product safety, function, or structural integrity. May be acceptable depending on defect-specific criteria (Document 2).
- **Major** — A defect that affects product quality, cleanliness, or safety perception but does not represent an immediate structural failure. Generally results in rejection.
- **Critical** — A defect that represents a clear safety hazard or structural failure. Always results in rejection and escalation.

## 4. Reject Criteria

**Section: Reject Criteria**

A bottle is rejected when it meets any of the following:

- Matches the Rejection Criteria for contamination, broken_small, or broken_large as defined in Document 2.
- Is assessed as Major or Critical severity under Section 3 above.
- Cannot be confidently classified after human review (defaults to reject rather than accept, per a conservative quality policy).

## 5. Escalation Criteria

**Section: Escalation Criteria**

A rejected (or ambiguous) bottle should be escalated to a supervisor or quality lead when:

- The defect is classified as Critical severity (e.g., broken_large).
- The automated vision system and the human inspector disagree on the outcome.
- The same defect type is observed repeatedly within a short time window or batch, suggesting a possible systemic/process issue rather than an isolated unit defect (see Document 4, CAPA Guide, Section on Investigation).
- Image quality or inspection conditions were insufficient to make a confident determination (see Document 5, Troubleshooting Guide).

## 6. Reinspection Requirements

**Section: Reinspection Requirements**

- Any bottle initially flagged as anomalous by the automated system but overridden to "accept" by a human inspector should be logged and subject to a secondary spot-check, since automated-system overrides are a quality-assurance risk point.
- A sampling-based reinspection of automated "accept" outcomes is recommended at a rate defined by the facility's quality plan (this project does not define a specific numeric sampling rate, since that would require a real facility's throughput and risk-tolerance data).

## 7. How This Document Relates to the Others

This document defines *thresholds and decision logic*. For what a specific defect *is*, see Document 2. For *how* an inspection is physically carried out, see Document 1. For *what happens after* a reject decision, see Document 4. For *ambiguous or low-confidence* cases, see Document 5.
