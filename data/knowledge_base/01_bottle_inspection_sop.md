---
document_id: DOC-001
document_type: sop
title: Bottle Visual Inspection — Standard Operating Procedure
version: 1.0
created_for: "Portfolio AI/ML project — simulated industrial documentation"
source: project-created
---

> **Note:** This is an example / simulated industrial inspection document created for an academic AI/ML portfolio project (MVTec AD bottle defect detection). It does not represent an official standard, regulation, or the procedures of any real manufacturing facility.

# Bottle Visual Inspection — Standard Operating Procedure (SOP)

## 1. Purpose

This procedure defines how bottles are visually inspected for surface and structural defects before packaging, and how an automated visual inspection system (vision model) is integrated into that inspection process. It exists to ensure defective bottles are identified consistently, whether by a human inspector or an AI-assisted vision system.

## 2. Scope

This SOP applies to glass and rigid plastic bottles moving through the final visual inspection stage of the production line, prior to labeling and packing. It covers both fully manual inspection and inspection assisted by an automated anomaly-detection system.

## 3. Inspection Environment

**Section: Inspection Environment**

Inspection should take place in an area with controlled, consistent lighting and a neutral, non-reflective background. Direct glare on the bottle surface should be avoided, since glare can obscure both real defects and cause an automated vision system to misclassify a normal bottle as anomalous. The inspection station should be kept free of dust and debris that could be mistaken for contamination.

## 4. Equipment Required

**Section: Equipment**

- Fixed-position camera or inspection light box (for automated inspection)
- Diffuse, uniform white lighting source
- Neutral gray or white backdrop
- Reference sample bottles (known-good and known-defective examples) for calibration
- Inspection log (physical or digital) for recording results

## 5. Inspection Procedure

**Section: Inspection Procedure — Overview**

1. Bottle is positioned at the inspection station (manually or via conveyor).
2. The automated vision system captures an image and produces an anomaly score.
3. If the automated system flags the bottle as anomalous, it should also indicate, when possible, the suspected region (localization) and, when a classifier is available, the likely defect category.
4. A human inspector confirms or reviews any automated flag before a final accept/reject decision is recorded, particularly for borderline or low-confidence cases.
5. The result (accept, reject, or escalate) is logged along with any defect category identified.

## 6. Visual Inspection Steps

**Section: Visual Inspection Steps**

When a human inspector performs or reviews an inspection, the following sequence should be followed:

1. Examine the overall bottle silhouette for irregularities in shape (may indicate breakage).
2. Examine the body and neck surfaces for foreign material, discoloration, or particulate matter (may indicate contamination).
3. Examine edges, rim, and base for chips, cracks, or missing material (may indicate breakage).
4. Cross-reference any automated system flag and localization region with the human visual check.
5. Record the outcome.

## 7. Lighting and Positioning Requirements

**Section: Lighting and Positioning**

- Lighting should be diffuse and shadow-minimizing; harsh directional lighting increases both false positives and false negatives.
- The bottle should be presented in a consistent orientation matching the orientation used to capture training images for the vision system, since inspection accuracy depends on consistent framing.
- Bottles should be rotated or imaged from multiple angles where the inspection process allows, since a single viewpoint may not reveal defects on the opposite side.

## 8. Inspection Frequency

**Section: Inspection Frequency**

Every bottle passing the final inspection station should receive at least one automated inspection pass. A sampling-based additional human re-check (for example, a percentage of the automated "accept" outcomes) is recommended as a quality-assurance measure, at a rate determined by the facility's quality plan (see Document 3, Quality Control / Acceptance Criteria, for severity-based sampling guidance).

## 9. Defect Identification Handoff

**Section: Defect Identification Handoff**

When the automated system flags a bottle as anomalous:

- If a defect category is available (e.g., contamination, broken_small, broken_large), that category should be used to retrieve the relevant defect definition (see Document 2).
- If no category is available, the bottle should be treated as "unclassified anomaly" and routed to the Troubleshooting Guide (Document 5) for handling guidance.

## 10. Acceptance / Rejection Process

**Section: Acceptance / Rejection Process**

The accept/reject decision itself is governed by Document 3 (Quality Control / Acceptance Criteria) and the per-defect criteria in Document 2 (Defect Definitions and Classification Guide). This SOP defines *how* the inspection is carried out; it does not itself set the pass/fail thresholds.

## 11. Escalation Procedure

**Section: Escalation Procedure**

A bottle should be escalated to a supervisor or quality lead rather than given a simple accept/reject decision when:

- The automated system and a human inspector disagree on the outcome.
- The defect does not clearly match any defined category.
- The defect, if confirmed, would be classified as high severity (see Document 3).
- Image quality was insufficient to make a confident determination (see Document 5).

## 12. Documentation Requirements

**Section: Documentation Requirements**

Each inspection outcome should be logged with: timestamp, bottle/batch identifier, automated anomaly score (if available), defect category (if identified), localization region (if available), final decision (accept/reject/escalate), and inspector identifier (if a human reviewed the case). This record supports the corrective and preventive action process described in Document 4.
