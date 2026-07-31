"""
CLI interface for Document RAG System (ask, scan, status, rebuild, validate-config).
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

from .config.loader import load_config
from .config.schema import AppConfig
from .generation.answer_service import AnswerService
from .ingestion.coordinator import IndexCoordinator
from .indexing.qdrant_store import QdrantStore
from .persistence.sqlite import get_database_manager
from .persistence.repositories import MetadataRepository
from .observability.logging import setup_logging, get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create main CLI argument parser with subcommands."""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON configuration file (e.g. config/config.production.json)",
    )

    parser = argparse.ArgumentParser(
        prog="rag_app",
        description="Document RAG System - High Precision Document Search & Q&A CLI",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a question against indexed documents", parents=[parent_parser])
    ask_parser.add_argument("question", type=str, help="User question string")

    # Command: scan
    subparsers.add_parser("scan", help="Run an immediate manual document scan & indexing job", parents=[parent_parser])

    # Command: status
    subparsers.add_parser("status", help="Show system indexing and database status", parents=[parent_parser])

    # Command: rebuild
    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild entire vector index from scratch", parents=[parent_parser])
    rebuild_parser.add_argument("--force", action="store_true", help="Force rebuild without confirmation prompt")

    # Command: validate-config
    subparsers.add_parser("validate-config", help="Validate JSON configuration file syntax and security rules", parents=[parent_parser])

    return parser


def handle_ask(config: AppConfig, question: str) -> int:
    """Handle 'ask' subcommand."""
    print(f"\n[질문]: {question}\n")
    print("검색 및 답변 생성 중...")

    # Early check: is Qdrant accessible before loading heavy ML models?
    is_available, err_msg = QdrantStore.check_availability(config)
    if not is_available:
        print("\n" + "=" * 60)
        print("답변:")
        print("벡터 저장소에 접근할 수 없어 검색이 불가능합니다.")
        print("=" * 60)
        print(f"\n[벡터 저장소 연결 실패]:")
        print(f"  {err_msg}")
        print("=" * 60 + "\n")
        return 1

    service = AnswerService(config)
    result = service.answer_question(question)

    print("\n" + "=" * 60)
    print(f"답변:\n{result.answer}")
    print("=" * 60)

    if result.citations:
        print("\n출처:")
        for c in result.citations:
            print(f"  {c.format_citation()}")
    elif result.sources:
        print("\n참조 문서 후보:")
        for idx, s in enumerate(result.sources, start=1):
            loc = f"{s.page_number}페이지" if s.page_number else (s.sheet_name or "본문")
            # 파일명 추출 (경로에서 파일명만)
            file_name = s.file_name or s.source_path.split("\\")[-1].split("/")[-1]
            print(f"  [{idx}] {file_name} ({loc}) - \"{s.section_title or '본문'}\"")
    else:
        print("\n출처: 없음 (검색된 문서 근거가 부족합니다)")
        init_err = QdrantStore._init_error_msg
        if init_err:
            print(f"\n[벡터 저장소 상태 경고]:")
            print(f"  {init_err}")

    print(f"\n처리 시간:")
    print(
        f"  검색: {result.retrieval_time_ms / 1000.0:.2f}초 | "
        f"리랭킹: {result.rerank_time_ms / 1000.0:.2f}초 | "
        f"생성: {result.generation_time_ms / 1000.0:.2f}초 | "
        f"전체: {result.total_time_ms / 1000.0:.2f}초"
    )
    print("=" * 60 + "\n")
    return 0


def handle_scan(config: AppConfig) -> int:
    """Handle 'scan' subcommand."""
    print("\n수동 문서 스캔 및 인덱싱을 시작합니다...")
    db_mgr = get_database_manager(config)
    coordinator = IndexCoordinator(config=config, db_manager=db_mgr)
    
    summary = coordinator.run_indexing_job(job_type="manual_scan")
    
    print("\n[스캔 결과 요약]:")
    print(f"  - 탐색된 파일 수: {summary.get('discovered', 0)}")
    print(f"  - 신규 파일 수:   {summary.get('new', 0)}")
    print(f"  - 변경 파일 수:   {summary.get('changed', 0)}")
    print(f"  - 삭제 파일 수:   {summary.get('deleted', 0)}")
    print(f"  - 미변경 파일 수: {summary.get('unchanged', 0)}")
    print(f"  - 성공 건수:     {summary.get('success_count', 0)}")
    print(f"  - 실패 건수:     {summary.get('failed_count', 0)}")
    print("\n스캔 작업이 완료되었습니다.\n")
    return 0


def handle_status(config: AppConfig) -> int:
    """Handle 'status' subcommand."""
    print("\n=== Document RAG System Status ===")
    print(f"환경 (Environment): {config.environment}")
    print(f"문서 루트 경로:      {config.paths.document_root.resolve()}")
    print(f"메타데이터 DB 경로:  {config.paths.metadata_db.resolve()}")
    print(f"Qdrant 컬렉션:      {config.qdrant.collection} ({config.qdrant.url})")

    db_mgr = get_database_manager(config)
    meta_repo = MetadataRepository(db_mgr)
    stats = meta_repo.get_stats()

    print("\n[문서 및 청크 통계]:")
    print(f"  - 총 문서 수:       {meta_repo.documents.count()}")
    print(f"  - 활성 문서 수:     {meta_repo.documents.count(status=None)}")
    print(f"  - 저장된 총 청크 수: {stats.get('total_chunks', 0)}")
    print(f"  - 인덱싱 작업 횟수: {stats.get('total_jobs', 0)}")

    doc_stats = stats.get("documents_by_status", {})
    if doc_stats:
        print("  - 문서 상태 분포:   " + ", ".join(f"{k}: {v}" for k, v in doc_stats.items()))

    print("\n시스템 상태가 정상입니다.\n")
    return 0


def handle_rebuild(config: AppConfig, force: bool = False) -> int:
    """Handle 'rebuild' subcommand."""
    if not force:
        confirm = input("\n모든 벡터 인덱스를 초기화하고 전체 재색인을 진행하시겠습니까? (y/N): ")
        if confirm.lower() != "y":
            print("전체 재구축 작업이 취소되었습니다.")
            return 0

    print("\n전체 인덱스 재구축을 시작합니다...")
    store = QdrantStore(config)
    store.clear_collection()

    db_mgr = get_database_manager(config)
    with db_mgr.transaction() as conn:
        conn.execute("DELETE FROM file_versions")
        conn.execute("DELETE FROM documents")
        conn.execute("DELETE FROM index_jobs")

    coordinator = IndexCoordinator(config=config, db_manager=db_mgr)
    summary = coordinator.run_indexing_job(job_type="rebuild")

    print(f"\n재구축 완료: 성공 {summary.get('success_count', 0)}개 문서, 실패 {summary.get('failed_count', 0)}개 문서\n")
    return 0


def handle_validate_config(config_path: Optional[str]) -> int:
    """Handle 'validate-config' subcommand."""
    print("\n설정 파일 검증 중...")
    path = Path(config_path) if config_path else None
    config = load_config(path)

    print("SUCCESS: JSON 설정 파일 규격 및 보안 검증 통과!")
    print(f"  - Environment: {config.environment}")
    print(f"  - Document Root: {config.paths.document_root.resolve()}")
    print(f"  - Embedding Model: {config.models.embedding.model_id}")
    print(f"  - LLM Model: {config.models.llm.model_id}\n")
    return 0


def main(args: Optional[list] = None) -> int:
    """CLI Main Entry Point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    # Validate config for subcommand
    if parsed_args.command == "validate-config":
        return handle_validate_config(parsed_args.config)

    config_path = Path(parsed_args.config) if parsed_args.config else None
    config = load_config(config_path)

    # Setup Logging
    setup_logging(config.logging, config.paths.log_dir)

    if parsed_args.command == "ask":
        return handle_ask(config, parsed_args.question)
    elif parsed_args.command == "scan":
        return handle_scan(config)
    elif parsed_args.command == "status":
        return handle_status(config)
    elif parsed_args.command == "rebuild":
        return handle_rebuild(config, force=parsed_args.force)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
