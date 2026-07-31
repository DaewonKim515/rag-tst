"""
Context builder formatting search hits into bounded SOURCE blocks with token management.
"""

from typing import List, Tuple, Optional
from ..config.schema import AppConfig
from ..domain.models import SearchHit
from ..processing.chunker import TokenCounter
from ..observability.logging import get_logger

logger = get_logger(__name__)


class ContextBuilder:
    """
    Constructs bounded prompt context from search hits with citation IDs
    and token budget enforcement.
    """

    def __init__(self, config: AppConfig, token_counter: Optional[TokenCounter] = None):
        self.config = config
        self.token_counter = token_counter or TokenCounter(config.models.embedding.model_id)
        self.max_context_tokens = config.models.llm.max_context_tokens

    def build_context(
        self,
        hits: List[SearchHit],
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, List[SearchHit]]:
        """
        Build structured context string from search hits.

        Args:
            hits: List of SearchHit candidates (in rank order).
            max_tokens: Optional token budget limit for context.

        Returns:
            Tuple of (formatted_context_string, list_of_included_hits).
        """
        if not hits:
            return "", []

        budget = max_tokens or (self.max_context_tokens - 1024)  # Leave budget for prompt instructions & system
        if budget <= 0:
            budget = 4096

        context_blocks = []
        included_hits = []
        accumulated_tokens = 0

        for idx, hit in enumerate(hits, start=1):
            source_id = f"S{idx}"
            
            # Format location metadata
            location_parts = []
            if hit.page_number is not None:
                location_parts.append(f"page {hit.page_number}")
            if hit.sheet_name:
                location_parts.append(f"sheet {hit.sheet_name}")
            if hit.slide_number is not None:
                location_parts.append(f"slide {hit.slide_number}")
            
            location_str = ", ".join(location_parts) if location_parts else "N/A"
            section_str = hit.section_title or "N/A"

            # Format source block with strict isolation boundaries
            block_str = (
                f"[SOURCE {source_id}]\n"
                f"file: {hit.source_path}\n"
                f"location: {location_str}\n"
                f"section: {section_str}\n"
                f"content:\n{hit.content}\n"
                f"[/SOURCE {source_id}]"
            )

            block_tokens = self.token_counter.count(block_str)

            if accumulated_tokens + block_tokens > budget:
                # Token budget exceeded, stop adding more hits
                logger.info(
                    "context_budget_reached",
                    f"Context budget reached ({accumulated_tokens} + {block_tokens} > {budget}). Used {len(included_hits)} hits.",
                )
                break

            context_blocks.append(block_str)
            included_hits.append(hit)
            accumulated_tokens += block_tokens

        formatted_context = "\n\n".join(context_blocks)
        logger.info(
            "context_built",
            f"Built context with {len(included_hits)} sources ({accumulated_tokens} tokens)",
            included_sources=len(included_hits),
            total_tokens=accumulated_tokens,
        )

        return formatted_context, included_hits
