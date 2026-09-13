"""
rag_documents.py
=================
Parses the project's markdown knowledge base into chunk records matching
the metadata schema defined in data/knowledge_base/06_metadata_schema.md:
document_id, document_type, title, section, defect_type, severity, source,
version, created_for.

Two document shapes exist in this knowledge base, so two parsing strategies:

1. The defect guide (DOC-002) tags each defect block inline
   (`**document_id:** DOC-002 · **defect_type:** contamination`) and then
   uses bold field headers (**Definition**, **Severity**, etc.) as
   subsection boundaries within each defect block.
2. Every other procedural document (SOP, QC criteria, CAPA, troubleshooting)
   uses `## N. Heading` sections, each containing one or more explicit
   `**Section: X**` markers that define the actual chunk boundaries — this
   matches the chunking recommendation in the metadata schema doc exactly.

Chunking at the granularity the documents already declare (rather than a
fixed token count) is what keeps each chunk self-contained and independently
retrievable, per the schema doc's stated goal.
"""

import re
from pathlib import Path

import yaml

SEVERITY_WORDS = ["Minor", "Major", "Critical"]


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown file into (frontmatter dict, remaining body text)."""
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    frontmatter = yaml.safe_load(match.group(1))
    return frontmatter, match.group(2)


def _extract_severity(text: str) -> str | None:
    """Find the severity word (Minor/Major/Critical) that appears EARLIEST in the text
    (by text position, not by a fixed word-priority) — a defect's primary severity is
    usually stated before any caveat/exception mentioning a different level."""
    positions = []
    for word in SEVERITY_WORDS:
        match = re.search(rf"\*\*{word}\*\*", text)
        if match:
            positions.append((match.start(), word.lower()))
    if not positions:
        return None
    return min(positions, key=lambda p: p[0])[1]


def _infer_defect_type_from_section_name(section_name: str) -> str | None:
    name = section_name.lower()
    if "small or large" in name or "small, large" in name:
        return None  # explicitly covers multiple defect types -- don't attribute to just one
    if "contamination" in name:
        return "contamination"
    if "broken (small" in name or "broken_small" in name:
        return "broken_small"
    if "broken (large" in name or "broken_large" in name:
        return "broken_large"
    return None  # general section, applies regardless of defect type


def parse_defect_guide(path: Path, frontmatter: dict, body: str) -> list[dict]:
    """Parser for DOC-002's structure: '## Defect: X' blocks, each with inline
    document_id/defect_type tags and bold-field subsections."""
    chunks = []

    defect_blocks = re.split(r"\n## Defect: ", body)
    for block in defect_blocks[1:]:  # first split part is intro text before the first defect block
        header_line, _, rest = block.partition("\n")
        defect_name = header_line.strip()

        tag_match = re.search(r"\*\*defect_type:\*\*\s*(\w+)", rest)
        defect_type = tag_match.group(1) if tag_match else _infer_defect_type_from_section_name(defect_name)

        severity = _extract_severity(rest)

        field_matches = list(re.finditer(r"\*\*([A-Za-z ]+)\*\*\n", rest))
        for i, field_match in enumerate(field_matches):
            field_name = field_match.group(1).strip()
            start = field_match.end()
            end = field_matches[i + 1].start() if i + 1 < len(field_matches) else len(rest)
            field_text = rest[start:end].strip().rstrip("-").strip()
            if not field_text:
                continue
            chunks.append({
                "chunk_id": f"{frontmatter['document_id']}-{defect_name.lower().replace(' ', '_')}-{field_name.lower().replace(' ', '_')}",
                "text": f"{defect_name} — {field_name}: {field_text}",
                "document_id": frontmatter["document_id"],
                "document_type": frontmatter["document_type"],
                "title": frontmatter["title"],
                "section": f"{defect_name} — {field_name}",
                "defect_type": defect_type,
                "severity": severity,
                "source": frontmatter.get("source", ""),
                "version": frontmatter.get("version", ""),
                "created_for": frontmatter.get("created_for", ""),
            })

    # Trailing cross-reference table: keep as one general chunk.
    table_match = re.search(r"## Defect Category Cross-Reference Table\n(.*)", body, re.DOTALL)
    if table_match:
        chunks.append({
            "chunk_id": f"{frontmatter['document_id']}-cross-reference-table",
            "text": "Defect Category Cross-Reference Table:\n" + table_match.group(1).strip(),
            "document_id": frontmatter["document_id"],
            "document_type": frontmatter["document_type"],
            "title": frontmatter["title"],
            "section": "Defect Category Cross-Reference Table",
            "defect_type": None,
            "severity": None,
            "source": frontmatter.get("source", ""),
            "version": frontmatter.get("version", ""),
            "created_for": frontmatter.get("created_for", ""),
        })

    return chunks


def parse_section_marked_doc(path: Path, frontmatter: dict, body: str) -> list[dict]:
    """Parser for SOP/QC/CAPA/Troubleshooting docs: chunk at each '**Section: X**' marker."""
    chunks = []

    h2_blocks = re.split(r"\n## ", body)
    for block in h2_blocks[1:]:  # skip intro text before the first h2
        heading_line, _, rest = block.partition("\n")
        h2_heading = heading_line.strip()

        section_matches = list(re.finditer(r"\*\*Section:\s*(.+?)\*\*\n", rest))

        if not section_matches:
            # No explicit Section marker in this h2 block (e.g. QC doc's closing
            # cross-reference section) — treat the whole block as one chunk.
            text = rest.strip()
            if text:
                chunks.append(_make_chunk(frontmatter, h2_heading, text))
            continue

        for i, section_match in enumerate(section_matches):
            section_name = section_match.group(1).strip()
            start = section_match.end()
            end = section_matches[i + 1].start() if i + 1 < len(section_matches) else len(rest)
            section_text = rest[start:end].strip()
            if section_text:
                chunks.append(_make_chunk(frontmatter, section_name, section_text))

    return chunks


def _make_chunk(frontmatter: dict, section_name: str, text: str) -> dict:
    return {
        "chunk_id": f"{frontmatter['document_id']}-{re.sub(r'[^a-z0-9]+', '-', section_name.lower()).strip('-')}",
        "text": f"{section_name}: {text}",
        "document_id": frontmatter["document_id"],
        "document_type": frontmatter["document_type"],
        "title": frontmatter["title"],
        "section": section_name,
        "defect_type": _infer_defect_type_from_section_name(section_name),
        "severity": _extract_severity(text),
        "source": frontmatter.get("source", ""),
        "version": frontmatter.get("version", ""),
        "created_for": frontmatter.get("created_for", ""),
    }


def parse_evaluation_questions(path: Path) -> list[dict]:
    """
    Parse 07_rag_test_questions.md into structured records for programmatic
    testing: {number, question, expected_docs, category, is_hallucination_test}.

    expected_docs is None for the hallucination/unanswerable category (by
    design, those questions have no correct grounding document).
    """
    text = path.read_text()
    records = []

    category_blocks = re.split(r"\n## \d+\.\s*", text)
    for block in category_blocks[1:]:
        category_line, _, rest = block.partition("\n")
        category = category_line.strip()
        is_hallucination = "hallucination" in category.lower() or "unanswerable" in category.lower()

        for match in re.finditer(r"^\d+\.\s+(.+?)(?:\s*\*\(→\s*(.+?)\)\*)?\s*$", rest, re.MULTILINE):
            question_text = match.group(1).strip()
            expected_docs_raw = match.group(2)
            expected_docs = None
            if expected_docs_raw:
                expected_docs = [d.strip() for d in re.split(r",", expected_docs_raw) if d.strip().startswith("DOC-")]
                expected_docs = [d.split()[0] for d in expected_docs]  # strip trailing "Section N" text if present

            records.append({
                "question": question_text,
                "expected_docs": expected_docs,
                "category": category,
                "is_hallucination_test": is_hallucination,
            })

    return records


def load_knowledge_base(knowledge_base_dir: Path, filenames: list[str] | None = None) -> list[dict]:
    """
    Parse every knowledge document in knowledge_base_dir into chunk records.

    Parameters
    ----------
    filenames : restrict parsing to these files only (used to exclude the
        metadata-schema doc and the evaluation question set, which aren't
        meant to be retrievable knowledge themselves).
    """
    all_chunks = []
    for path in sorted(knowledge_base_dir.iterdir()):
        if filenames is not None and path.name not in filenames:
            continue
        if path.suffix != ".md":
            continue

        text = path.read_text()
        frontmatter, body = _parse_frontmatter(text)
        if not frontmatter:
            continue  # not a knowledge document (e.g. no frontmatter -> metadata schema/question set)

        if frontmatter["document_id"] == "DOC-002":
            chunks = parse_defect_guide(path, frontmatter, body)
        else:
            chunks = parse_section_marked_doc(path, frontmatter, body)

        all_chunks.extend(chunks)

    return all_chunks
