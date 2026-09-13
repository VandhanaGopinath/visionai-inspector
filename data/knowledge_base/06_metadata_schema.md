# Metadata Schema for the RAG Knowledge Base

This defines the metadata fields attached to each chunk when the five documents are embedded, and which fields should be indexed/filterable in the vector database versus kept only for display and citation.

## Field Definitions

| Field | Type | Example | Purpose |
|---|---|---|---|
| `document_id` | string | `DOC-002` | Identifies which of the 5 source documents the chunk came from |
| `document_type` | enum | `sop`, `defect_guide`, `qc_criteria`, `capa_guide`, `troubleshooting_guide` | Coarse routing — lets you filter to "only QC criteria" or "only troubleshooting" |
| `title` | string | `Defect Definitions and Classification Guide` | Human-readable document name, for citation display |
| `section` | string | `Contamination — Acceptance Criteria` | The specific subsection this chunk represents |
| `defect_type` | enum, nullable | `contamination`, `broken_small`, `broken_large`, `null` | Directly maps to your vision model's output category — the single most important field for vision→RAG grounding |
| `severity` | enum, nullable | `minor`, `major`, `critical`, `null` | Enables severity-aware retrieval/filtering |
| `source` | string | `project-created` or `project-created (general QC concept, not a named standard)` | Preserves your provenance/credibility labeling at the chunk level |
| `version` | string | `1.0` | Document version, for reproducibility as you iterate |
| `created_for` | string | `Portfolio AI/ML project — simulated industrial documentation` | Carries the "this is simulated" disclaimer into every chunk, so the LLM can repeat it if asked whether this is a real standard |
| `keywords` | list[string] | `["contamination", "acceptance", "reject", "surface defect"]` | Optional lexical-search boost / hybrid search support |

## What to Index vs. Filter vs. Display

**Indexed as filterable metadata (used in query-time filters, e.g. hybrid semantic+filter search):**
- `document_type`
- `defect_type`
- `severity`

These three are the fields your vision pipeline output maps directly onto. When your vision model outputs `defect_type=contamination`, your retrieval query should apply a metadata filter `defect_type == "contamination" OR defect_type == null` (the `null` case includes general documents like the SOP and Troubleshooting Guide that apply regardless of defect type) combined with the semantic query text.

**Stored but not filtered (returned with results for citation/display only):**
- `document_id`
- `title`
- `section`
- `version`
- `created_for`
- `keywords` (unless you specifically want lexical/BM25 hybrid search, in which case index this too)

## Example Chunk Record

```json
{
  "chunk_id": "DOC-002-contamination-acceptance",
  "text": "Acceptance Criteria (Contamination): A bottle may be accepted only if the contamination is confirmed to be a superficial, removable surface artifact...",
  "metadata": {
    "document_id": "DOC-002",
    "document_type": "defect_guide",
    "title": "Defect Definitions and Classification Guide",
    "section": "Contamination — Acceptance Criteria",
    "defect_type": "contamination",
    "severity": "major",
    "source": "project-created",
    "version": "1.0",
    "created_for": "Portfolio AI/ML project — simulated industrial documentation",
    "keywords": ["contamination", "acceptance criteria", "surface defect"]
  }
}
```

## Chunking Recommendation

Chunk at the **subsection level** (each "Section: ..." marker in the source documents), not at a fixed token count. This preserves the self-contained, independently-understandable property you asked for, and keeps each chunk aligned to exactly one `section` / `defect_type` / `severity` metadata combination. Expect chunk sizes of roughly 60–200 words, which is small enough for precise retrieval and large enough to remain meaningful on its own.
