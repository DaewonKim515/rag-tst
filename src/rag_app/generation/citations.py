"""
Citation parser, verifier, and formatter for RAG generation output.
"""

import re
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass

from ..domain.models import SearchHit
from ..observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CitationSource:
    """Verified citation source metadata."""
    citation_id: str
    source_path: str
    file_name: str
    location: str
    section_title: str

    def format_citation(self) -> str:
        """Format citation line for CLI / user output."""
        display_name = self.file_name or Path(self.source_path).name
        sec = f", \"{self.section_title}\"" if self.section_title and self.section_title != "본문" else ""
        return f"[{self.citation_id}] {display_name} ({self.location}){sec}"


class CitationVerifier:
    """Parses and verifies citation markers in LLM generated answers."""

    CITATION_REGEX = re.compile(r"\[S(\d+)\]")

    def parse_citations(self, text: str) -> List[str]:
        """Extract unique citation IDs from text (e.g. ['S1', 'S2'])."""
        matches = self.CITATION_REGEX.findall(text)
        seen = set()
        citations = []
        for match in matches:
            cid = f"S{match}"
            if cid not in seen:
                seen.add(cid)
                citations.append(cid)
        return citations

    def verify_and_clean(
        self,
        answer_text: str,
        sources: List[SearchHit],
    ) -> Tuple[str, List[CitationSource]]:
        """
        Verify citations in answer_text against provided sources.
        Removes invalid/hallucinated citations and formats valid citation metadata.

        Args:
            answer_text: Generated answer text.
            sources: List of SearchHit objects provided in context (ordered S1, S2...).

        Returns:
            Tuple of (cleaned_answer_text, list_of_verified_CitationSource_objects).
        """
        valid_map: Dict[str, SearchHit] = {}
        for idx, hit in enumerate(sources, start=1):
            valid_map[f"S{idx}"] = hit

        found_citations = self.parse_citations(answer_text)
        verified_sources = []
        invalid_citations = []

        for cid in found_citations:
            if cid in valid_map:
                hit = valid_map[cid]
                # Format location
                loc_parts = []
                if hit.page_number is not None:
                    loc_parts.append(f"{hit.page_number}페이지")
                if hit.sheet_name:
                    loc_parts.append(f"{hit.sheet_name} 시트")
                if hit.slide_number is not None:
                    loc_parts.append(f"{hit.slide_number} 슬라이드")
                loc_str = ", ".join(loc_parts) if loc_parts else "위치 정보 없음"

                c_source = CitationSource(
                    citation_id=cid,
                    source_path=hit.source_path,
                    file_name=hit.file_name,
                    location=loc_str,
                    section_title=hit.section_title or "본문",
                )
                verified_sources.append(c_source)
            else:
                invalid_citations.append(cid)

        # Handle invalid/hallucinated citations (strip invalid ones from text)
        cleaned_text = answer_text
        if invalid_citations:
            logger.warning(
                "invalid_citations_detected",
                f"Removed hallucinated citations: {invalid_citations}",
                invalid_citations=invalid_citations,
            )
            for inv_id in invalid_citations:
                pattern = f"\\[{inv_id}\\]"
                cleaned_text = re.sub(pattern, "", cleaned_text)

        return cleaned_text, verified_sources
