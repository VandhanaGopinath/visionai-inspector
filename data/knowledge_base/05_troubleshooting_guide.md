---
document_id: DOC-005
document_type: troubleshooting_guide
title: Visual Inspection Troubleshooting Guide
version: 1.0
created_for: "Portfolio AI/ML project — simulated industrial documentation"
source: project-created
---

> **Note:** This is an example / simulated industrial inspection document created for an academic AI/ML portfolio project. It addresses ambiguous, low-confidence, or edge-case situations arising from the interaction between the automated vision system and human inspection.

# Visual Inspection Troubleshooting Guide

This document covers situations where the inspection outcome is unclear, disputed, or affected by conditions outside normal operation — cases that don't map cleanly onto a single defect definition.

## 1. Suspected Contamination — Unclear Evidence

**Section: Suspected Contamination, Unclear**

If the automated system flags possible contamination but the flagged region is small, low-contrast, or ambiguous: a human inspector should visually re-examine the specific localized region under improved lighting before making a final determination. If still unclear, default to "escalate" rather than "accept," per the conservative decision policy in Document 3, Section 4.

## 2. Suspected Breakage — Unclear Evidence

**Section: Suspected Breakage, Unclear**

If the flagged region resembles a chip or crack but could also be a lighting reflection or shadow artifact: check whether the same feature is visible from a second image or angle, if available. A genuine break should remain visible across viewing angles/lighting changes, whereas a reflection artifact typically will not. If a second view is not available, escalate for physical inspection.

## 3. Unclear Visual Evidence (General)

**Section: Unclear Visual Evidence**

When neither the automated system's confidence score nor the human inspector's visual assessment is decisive, the case defaults to "escalate" rather than a forced accept/reject decision, consistent with the conservative default described in Document 3.

## 4. Poor Lighting Conditions

**Section: Poor Lighting**

Symptoms of poor lighting include inconsistent shadows, glare spots, or underexposed regions on the captured image. If poor lighting is suspected as the cause of a flag, the inspection station lighting should be checked against the requirements in Document 1, Section 7, and the bottle re-imaged if possible before a final decision is recorded.

## 5. Image Quality Problems

**Section: Image Quality Problems**

Blurring, partial occlusion, or sensor artifacts can cause both false positive and false negative outcomes. If image quality is suspect, the bottle should be re-imaged rather than judged from a degraded capture. If re-imaging is not possible, the case should be escalated rather than decided from low-quality input.

## 6. False Positive Suspicion

**Section: False Positive Suspicion**

A false positive is suspected when the automated system flags an anomaly but a human inspector, under good conditions, cannot identify any corresponding visible defect. Such cases should still be logged (per Document 1, Section 12) even if ultimately accepted, since a pattern of false positives on a particular defect type may indicate the automated system's threshold needs recalibration (see Document 4, Preventive Action).

## 7. Disagreement Between Automated System and Human Inspector

**Section: System/Human Disagreement**

When the automated system and the human inspector reach different conclusions:

- The case is never resolved by simply trusting one side by default; it is treated as ambiguous.
- Per Document 3, Section 5, disagreement between the automated system and a human inspector is itself an escalation trigger.
- The case should be reviewed by a supervisor or quality lead, and the outcome (and which side was correct, if determinable) should be logged to help evaluate and improve the automated system over time.

## 8. Summary Decision Rule for Ambiguous Cases

**Section: Ambiguous Case Default Rule**

As a general rule throughout this project's simulated inspection process: **when in doubt, escalate rather than accept.** This conservative default appears consistently across Documents 1, 3, and 5, and should be treated as the fallback behavior whenever a specific rule does not clearly resolve a case.
