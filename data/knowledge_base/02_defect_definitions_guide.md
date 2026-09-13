---
document_id: DOC-002
document_type: defect_guide
title: Defect Definitions and Classification Guide
version: 1.0
created_for: "Portfolio AI/ML project — simulated industrial documentation"
source: project-created
---

> **Note:** This is an example / simulated industrial inspection document created for an academic AI/ML portfolio project (MVTec AD bottle defect detection). Defect categories correspond to those present in the MVTec AD "bottle" category. Definitions and criteria below are simulated for this project and do not represent an official industry standard.

# Defect Definitions and Classification Guide

This document defines each defect category used in the bottle inspection system, so that a detected defect can be explained consistently. Each defect follows the same structure — Definition, Visual Characteristics, Typical Location, Severity, Acceptance Criteria, Rejection Criteria, Example, Inspector Action — so that any single section can be retrieved and understood independently.

---

## Defect: Contamination

**document_id:** DOC-002 · **defect_type:** contamination

**Definition**

Contamination refers to the presence of foreign material — such as particulate matter, residue, or discoloration — on or within the bottle that should not be present in a normal, defect-free unit.

**Visual Characteristics**

Contamination typically appears as small dark or light-colored spots, smears, or irregular patches on the bottle surface or, in some cases, visible material lodged against the inner or outer wall. It is usually irregular in shape and does not follow the bottle's structural lines, which distinguishes it from a manufacturing seam or a lighting artifact.

**Typical Location**

Contamination can appear anywhere on the bottle surface, but is most frequently observed on the body or lower-middle region, consistent with where residue may settle during filling or handling.

**Severity**

*Simulated severity classification for this project:* Contamination is generally classified as **Major** severity, since it raises product-safety and cleanliness concerns, unless the affected area is extremely small and clearly cosmetic (e.g., a single tiny surface speck with no depth), in which case it may be classified as **Minor** (see Document 3 for severity level definitions).

**Acceptance Criteria**

A bottle may be accepted only if the contamination is confirmed to be a superficial, removable surface artifact (e.g., an external smudge verified to wipe away) and not embedded material. In practice, for this project's simulated criteria, any contamination detected by the vision system should default to a "reject" or "escalate" recommendation rather than an automatic accept, pending human confirmation.

**Rejection Criteria**

A bottle should be rejected if contamination is confirmed to be embedded material, discoloration that does not wipe away, or any particulate visible inside the bottle wall.

**Example**

A bottle detected with a dark, irregular patch in the lower-middle region of the body, not aligned with any mold seam, and not removable by wiping — classified as contamination, Major severity, reject.

**Inspector Action**

Confirm whether the contamination is surface-level or embedded. If uncertain, escalate per the Escalation Procedure (Document 1, Section 11) and the Troubleshooting Guide (Document 5). Log the finding per Document 1, Section 12.

---

## Defect: Broken (Small)

**document_id:** DOC-002 · **defect_type:** broken_small

**Definition**

A "broken_small" defect refers to a minor structural break, chip, or crack in the bottle — typically limited to a small section of the rim, neck, or edge — that does not compromise the overall structural integrity of the bottle but represents a visible defect.

**Visual Characteristics**

Appears as a small chipped or fractured edge, a short crack line, or a small missing fragment, generally limited in size relative to the overall bottle dimensions. Edges at the break point are typically irregular and sharp rather than smooth (which would suggest an intentional design feature).

**Typical Location**

Most frequently observed at the rim, neck opening, or base edge, where the bottle is most exposed to impact during handling.

**Severity**

*Simulated severity classification for this project:* broken_small is generally classified as **Major** severity due to safety concerns (sharp edges, structural compromise), even though the visible defect area is small.

**Acceptance Criteria**

For this project's simulated criteria, no broken_small defect is considered acceptable for shipment, since even a small break can create a sharp edge or a point of further fracture. Acceptance criteria here default to "no acceptance" — any confirmed broken_small defect proceeds to rejection.

**Rejection Criteria**

Any confirmed chip, crack, or missing fragment at the rim, neck, or edge — regardless of size — meets rejection criteria.

**Example**

A bottle with a small chip at the rim, roughly a few millimeters across, with a sharp irregular edge — classified as broken_small, Major severity, reject.

**Inspector Action**

Confirm the break is not an imaging artifact (e.g., glare or reflection mimicking an edge defect — see Document 5). If confirmed, reject and log per Document 1, Section 12. No escalation is required unless the break is unusually large or the automated and human assessments disagree.

---

## Defect: Broken (Large)

**document_id:** DOC-002 · **defect_type:** broken_large

**Definition**

A "broken_large" defect refers to a significant structural break — a large missing section, major crack, or substantial fracture — that clearly compromises the bottle's structural integrity or overall shape.

**Visual Characteristics**

Appears as a large missing section of the bottle body, neck, or base, a long crack spanning a significant portion of the bottle, or a fracture that visibly alters the bottle's silhouette. This is typically detectable even from the overall shape/contour of the bottle, not only from close inspection of an edge.

**Typical Location**

Can occur at any location but is most visually apparent on the body or neck, where the deviation from the expected bottle silhouette is most obvious.

**Severity**

*Simulated severity classification for this project:* broken_large is classified as **Critical** severity, since it represents a significant safety hazard and a clear structural failure.

**Acceptance Criteria**

No broken_large defect is acceptable under any circumstance in this project's simulated criteria.

**Rejection Criteria**

Any confirmed large fracture, major missing section, or significant deviation from the expected bottle silhouette meets rejection criteria and should be treated as an automatic reject.

**Example**

A bottle with a large section of the body missing, altering its overall shape — classified as broken_large, Critical severity, reject, escalate.

**Inspector Action**

Reject immediately. Given the Critical severity, this defect should also be escalated per Document 1, Section 11, so that the batch or production run can be reviewed for a possible systemic cause (see Document 4, CAPA Guide).

---

## Defect Category Cross-Reference Table

| defect_type | Typical Severity | Default Decision | Escalate? |
|---|---|---|---|
| contamination | Major (Minor if trivial/surface-only) | Reject / Escalate for confirmation | If embedded or unclear |
| broken_small | Major | Reject | Only if disputed or unusually large |
| broken_large | Critical | Reject | Always |

*This table is a simulated summary for this project and should be read together with Document 3 for full severity-level definitions.*
