"""
Answer service orchestrating the full Retrieval, Reranking, Context Building, LLM Generation, and Citation pipeline.
"""

import time
from typing import Optional, List
from dataclasses import dataclass
from uuid import UUID, uuid4

from ..config.schema import AppConfig
from ..domain.models import GenerationRequest, GenerationResult, SearchHit
from ..domain.exceptions import PromptInjectionDetectedError, RetrievalError
from ..retrieval.retriever import DenseRetriever
from ..models.reranker import BgeReranker, get_reranker
from ..retrieval.context_builder import ContextBuilder
from ..models.llm import get_llm, PromptInjectionDetector
from ..models.llm import LanguageModel
from .citations import CitationVerifier, CitationSource
from ..persistence.sqlite import DatabaseManager, get_database_manager
from ..observability.logging import get_logger, new_job_context, RequestContext

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """Full RAG Answer Result with timing and citation metadata."""
    request_id: UUID
    question: str
    answer: str
    citations: List[CitationSource]
    sources: List[SearchHit]
    retrieval_time_ms: float
    rerank_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    status: str
    error_message: Optional[str] = None


class AnswerService:
    """End-to-End RAG Answer Pipeline Orchestrator."""

    def __init__(
        self,
        config: AppConfig,
        retriever: Optional[DenseRetriever] = None,
        reranker: Optional[BgeReranker] = None,
        context_builder: Optional[ContextBuilder] = None,
        llm: Optional[LanguageModel] = None,
        db_manager: Optional[DatabaseManager] = None,
    ):
        self.config = config
        self.db_manager = db_manager or get_database_manager(config)
        self.retriever = retriever or DenseRetriever(config=config, db_manager=self.db_manager)
        self.reranker = reranker or get_reranker(config)
        self.context_builder = context_builder or ContextBuilder(config)
        self.llm = llm or get_llm(config)
        self.citation_verifier = CitationVerifier()
        self.injection_detector = PromptInjectionDetector()

    def answer_question(self, question: str, request_id: Optional[UUID] = None) -> AnswerResult:
        """
        Process a user question through the complete RAG pipeline.

        Args:
            question: User question string.
            request_id: Optional request correlation ID.

        Returns:
            AnswerResult containing answer, citations, timing, and sources.
        """
        req_id = request_id or uuid4()
        start_total = time.perf_counter()

        with RequestContext(req_id):
            logger.info("answer_service_started", f"Processing question: {question}", request_id=req_id)

            # 1. Security Check: Prompt Injection Defense [1.14.3]
            try:
                self.injection_detector.validate_or_raise(question)
            except PromptInjectionDetectedError as e:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                return AnswerResult(
                    request_id=req_id,
                    question=question,
                    answer="보안 정책 위반: 허용되지 않은 프로젝션 또는 탈옥 명령 패턴이 감지되었습니다.",
                    citations=[],
                    sources=[],
                    retrieval_time_ms=0.0,
                    rerank_time_ms=0.0,
                    generation_time_ms=0.0,
                    total_time_ms=total_ms,
                    status="rejected_security",
                    error_message=str(e),
                )

            # 2. Retrieval Phase
            start_retrieval = time.perf_counter()
            try:
                retrieved_hits = self.retriever.retrieve(question)
            except Exception as e:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                logger.error("answer_service_retrieval_failed", f"Retrieval failed: {e}", request_id=req_id)
                return AnswerResult(
                    request_id=req_id,
                    question=question,
                    answer="검색 시스템 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    citations=[],
                    sources=[],
                    retrieval_time_ms=0.0,
                    rerank_time_ms=0.0,
                    generation_time_ms=0.0,
                    total_time_ms=total_ms,
                    status="failed_retrieval",
                    error_message=str(e),
                )

            retrieval_ms = (time.perf_counter() - start_retrieval) * 1000.0

            # 3. Check Context Sufficiency (If no hits, return rejection without calling LLM)
            if not retrieved_hits:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                logger.info("answer_service_no_hits", "No relevant chunks found for question", request_id=req_id)
                return AnswerResult(
                    request_id=req_id,
                    question=question,
                    answer="제공된 문서에서 답변을 찾을 수 없습니다.",
                    citations=[],
                    sources=[],
                    retrieval_time_ms=retrieval_ms,
                    rerank_time_ms=0.0,
                    generation_time_ms=0.0,
                    total_time_ms=total_ms,
                    status="no_context",
                )

            # 4. Reranking Phase
            start_rerank = time.perf_counter()
            reranked_hits = self.reranker.rerank(question, retrieved_hits)
            rerank_ms = (time.perf_counter() - start_rerank) * 1000.0

            # 5. Context Building Phase
            context_text, included_sources = self.context_builder.build_context(reranked_hits)

            if not included_sources or not context_text:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                return AnswerResult(
                    request_id=req_id,
                    question=question,
                    answer="제공된 문서에서 답변을 찾을 수 없습니다.",
                    citations=[],
                    sources=[],
                    retrieval_time_ms=retrieval_ms,
                    rerank_time_ms=rerank_ms,
                    generation_time_ms=0.0,
                    total_time_ms=total_ms,
                    status="insufficient_context",
                )

            # 6. LLM Generation Phase
            start_gen = time.perf_counter()
            gen_request = GenerationRequest(
                prompt=question,
                context=context_text,
                temperature=self.config.models.llm.temperature,
                max_tokens=self.config.models.llm.max_output_tokens,
            )

            try:
                gen_result = self.llm.generate(gen_request)
            except Exception as e:
                total_ms = (time.perf_counter() - start_total) * 1000.0
                logger.error("answer_service_generation_failed", f"Generation failed: {e}", request_id=req_id)
                return AnswerResult(
                    request_id=req_id,
                    question=question,
                    answer="답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    citations=[],
                    sources=included_sources,
                    retrieval_time_ms=retrieval_ms,
                    rerank_time_ms=rerank_ms,
                    generation_time_ms=0.0,
                    total_time_ms=total_ms,
                    status="failed_generation",
                    error_message=str(e),
                )

            gen_ms = (time.perf_counter() - start_gen) * 1000.0

            # 7. Citation Verification Phase
            cleaned_answer, verified_citations = self.citation_verifier.verify_and_clean(
                gen_result.answer, included_sources
            )

            total_ms = (time.perf_counter() - start_total) * 1000.0

            logger.info(
                "answer_service_completed",
                f"Question answered in {total_ms:.2f}ms (sources: {len(included_sources)}, citations: {len(verified_citations)})",
                request_id=req_id,
                total_time_ms=total_ms,
                sources_count=len(included_sources),
                citations_count=len(verified_citations),
            )

            return AnswerResult(
                request_id=req_id,
                question=question,
                answer=cleaned_answer,
                citations=verified_citations,
                sources=included_sources,
                retrieval_time_ms=retrieval_ms,
                rerank_time_ms=rerank_ms,
                generation_time_ms=gen_ms,
                total_time_ms=total_ms,
                status="success",
            )
