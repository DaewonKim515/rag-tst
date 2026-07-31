# Document RAG System

> **문서 검색 및 질의응답 시스템** - 문서 폴더의 문서를 자동 색인하고 근거 기반 답변을 제공하는 RAG(Retrieval-Augmented Generation) 시스템

---

## 프로젝트 개요

본 시스템은 로컬/사내 문서 저장소(`document/` 폴더)를 자동으로 탐색하여 변경된 문서만 증분 색인하고, 사용자 질문에 대해 **검색 → 리랭킹 → LLM 생성** 파이프라인을 통해 **근거 기반 답변**과 **출처 인용**을 제공하는 CLI 기반 RAG 시스템입니다.

---

### 주요 구성요소

| 구성요소 | 책임 |
|----------|------|
| **CLI** | 질문 입력, 답변/출처 출력, 인덱싱 상태 확인, 전체 재구축 명령 |
| **Config Loader** | JSON 로드, 스키마 검증, 경로 안전성 검사, 환경변수 주입 |
| **Index Scheduler** | 시작 시/주기적 자동 스캔 실행 |
| **File Scanner** | `document` 하위 폴더 탐색, 확장자 필터링 |
| **MD5 Comparator** | 스트리밍 MD5 계산, SQLite 기록 비교, 신규/변경/삭제 분류 |
| **Parser Registry** | 확장자별 파서 선택, 공통 `ParsedDocument` 형식 반환 |
| **OCR Adapter** | 스캔 PDF 판별 후 Unlimited OCR 호출 및 결과 병합 |
| **Text Normalizer/Chunker** | 문단/문장/페이지 경계 고려 청크 생성 (한국어 `kiwipiepy`) |
| **Embedder** | `BAAI/bge-m3` 배치 임베딩 및 벡터 정규화 |
| **Index Coordinator** | 문서 버전 스테이징, Qdrant 저장, SQLite 활성 버전 전환 |
| **Retriever** | 질의 임베딩, Qdrant 검색, 활성 버전 필터, 중복 억제 |
| **Reranker** | `BAAI/bge-reranker-v2-m3` 질문-청크 쌍 재정렬 |
| **Context Builder** | 토큰 예산 내 근거 구성, 인용 번호 부여, 문서 지시문 격리 |
| **LLM Gateway** | EXAONE 프롬프트 구성, 추론 호출, 제한시간/오류 처리 |
| **Metadata Repository** | SQLite 트랜잭션 및 파일/문서 버전/작업 상태 관리 |
| **Observability** | JSON Lines 로그, 처리 시간, 성공/실패/제외 건수 기록 |

---

## 기술 스택

| 구분 | 기술 | 용도 |
|------|------|------|
| **언어** | Python 3.13 | CLI, 문서 처리, 모델 연동 |
| **LLM** | `Qwen3.5:4B` | 로컬/추론 서버 답변 생성 |
| **임베딩** | `BAAI/bge-m3` | 문서 청크/질의 dense embedding |
| **리랭커** | `BAAI/bge-reranker-v2-m3` | 1차 검색 후보 재정렬 |
| **OCR** | Unlimited OCR | 스캔 PDF/이미지 OCR (어댑터 패턴) |
| **한국어 분석** | `kiwipiepy` | 문장 경계 및 형태소 기반 청킹 보조 |
| **벡터 DB** | Qdrant | 임베딩, 청크 본문, 메타데이터 저장·검색 |
| **메타데이터 DB** | SQLite (WAL 모드) | 파일 MD5, 활성 버전, 처리 상태, 인덱싱 이력 |
| **문서 파서** | PyMuPDF, python-docx, openpyxl, python-pptx | PDF/DOCX/XLSX/PPTX 파싱 |
| **설정** | Pydantic v2 + JSON | 스키마 검증, 환경별 설정, 환경변수 시크릿 |
| **테스트** | pytest | 단위/통합/회귀 테스트 |

---

## 설치 및 실행 환경

### 필수 요구사항

- **Python 3.13**
- **Qdrant** 벡터 데이터베이스 (로컬 실행 또는 컨테이너)
- **GPU 메모리** 16GB+ 권장 (모델 동시 로딩 시)
- **Unlimited OCR** 서비스 (OCR 기능 사용 시, 선택 사항)

### 의존성 설치

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
linux: source .venv/bin/activate 
Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### Qdrant 실행 (Docker) -> 이거 있긴한데 그냥 로컬로 돌려서 써보진 않았습니다. 그냥 쓰셔도 돼요

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 모델 다운로드

최초 실행 시 Hugging Face에서 모델이 자동 다운로드됩니다. 오프라인 환경인 경우 사전 다운로드 필요:

```bash
# 임베딩 모델
huggingface-cli download BAAI/bge-m3

# 리랭커 모델  
huggingface-cli download BAAI/bge-reranker-v2-m3

# LLM (EXAONE)
huggingface-cli download 
```

---

## 설정

### 설정 파일 구조

```
config/
├── config.development.json   # 개발 환경
├── config.test.json          # 테스트 환경
├── config.production.json    # 운영 환경
```

### 환경변수

```bash
# .env 파일 또는 시스템 환경변수
export QDRANT_API_KEY="your-qdrant-api-key"
export UNLIMITED_OCR_API_KEY="your-ocr-api-key"
```

---

## 사용법

### CLI 명령어

```bash
# 기본 실행 (src/ 경로가 PYTHONPATH에 포함되어 있어야 함)
python -m rag_app <명령어> [옵션]

# 또는 run.py 사용
python run.py <명령어> [옵션]
```

### 주요 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `ask` | 단일 질문 수행 | `python -m rag_app ask "제품 설치 방법은?"` |
| `chat` | 대화형 질의 루프 | `python -m rag_app chat` |
| `status` | 시스템 상태 확인 | `python -m rag_app status` |
| `scan` | 즉시 수동 스캔/색인 실행 | `python -m rag_app scan` |
| `rebuild` | 전체 벡터 인덱스 재구축 | `python -m rag_app rebuild --force` |
| `validate-config` | 설정 파일 검증만 수행 | `python -m rag_app validate-config` |

### 옵션

```bash
--config PATH    # 설정 파일 경로 지정 (기본: 환경별 자동 선택)
```

### 사용 예시

```bash
# 1. 설정 검증
python -m rag_app validate-config --config config/config.production.json

# 2. 시스템 상태 확인
python -m rag_app status --config config/config.production.json

# 3. 문서 스캔 및 색인 (자동으로 scan_on_start=true 시 시작 시 수행)
python -m rag_app scan --config config/config.production.json

# 4. 질문하기
python -m rag_app ask "암호 알고리즘 키 길이는 어떻게 설정하나요?" --config config/config.production.json

# 5. 대화형 모드
python -m rag_app chat --config config/config.production.json

# 6. 전체 재구축 (주의: 기존 인덱스 삭제)
python -m rag_app rebuild --config config/config.production.json --force
```

### 출력 예시 (ask 명령)

```text
[질문]: 암호 알고리즘 키 길이는 어떻게 설정하나요?

답변:
AES-256을 사용하는 경우 키 길이는 256비트(32바이트)로 설정해야 합니다. [S1]
RSA-2048의 경우 공개키 모듈러스는 2048비트여야 합니다. [S2]

출처:
[S1] 암호 알고리즘 및 키 길이 이용 안내서_2018.pdf | page 12 | section: 3.2 대칭키 알고리즘
[S2] 암호 알고리즘 및 키 길이 이용 안내서_2018.pdf | page 15 | section: 4.1 공개키 알고리즘

처리 시간:
검색 0.42초 / 리랭킹 0.85초 / 생성 6.20초 / 전체 7.47초
```

---

## 프로젝트 구조

```
rag-tst/
├── document/                      # 대상 문서 폴더 (사용자 관리)
│   ├── *.pdf, *.docx, *.txt, *.md
│   └── 하위 폴더 지원
├── config/                        # 환경별 설정 파일
│   ├── config.development.json
│   ├── config.test.json
│   └── config.production.json
├── data/                          # 런타임 데이터 (자동 생성)
│   ├── metadata.db                # SQLite 메타데이터
│   ├── logs/                      # JSON Lines 로그
│   └── qdrant_storage/            # Qdrant 데이터 (Docker 볼륨)
├── src/
│   └── rag_app/                   # 메인 애플리케이션 패키지
│       ├── __main__.py            # python -m rag_app 진입점
│       ├── cli.py                 # CLI 명령어 구현
│       ├── config/                # 설정 로드/검증
│       ├── domain/                # 도메인 모델/예외/열거형
│       ├── ingestion/             # 파일 스캔, MD5, 변경감지, 조정자
│       ├── parsing/               # 파서 레지스트리, PDF/DOCX/TXT/MD/OCR
│       ├── processing/            # 정규화, 문장분리, 청킹
│       ├── models/                # 임베딩, 리랭커, LLM 구현체
│       ├── indexing/              # Qdrant 스토어, 버전 매니저
│       ├── retrieval/             # 검색기, 컨텍스트 빌더, 필터
│       ├── generation/            # 답변 서비스, 인용 포맷터
│       ├── persistence/           # SQLite, 리포지토리, 마이그레이션
│       ├── observability/         # 구조화 로깅
│       └── evaluation/            # 평가 프레임워크 (확장 중)
├── tests/                         # 테스트 코드
│   ├── unit/                      # 단위 테스트
│   ├── integration/               # 통합 테스트
│   ├── regression/                # 회귀 테스트
│   └── fixtures/                  # 테스트 픽스처
├── development_document/          # 설계 문서
│   ├── architecture-design.md     # 아키텍처 설계서
│   └── 요구사항정의서.md          # 요구사항 정의서
├── main.py                        # 진입점 (run.py로 위임)
├── run.py                         # 실행 스크립트 (경로 설정 후 cli.main 호출)
├── requirements.txt               # Python 의존성
├── pytest.ini                     # pytest 설정
└── todo.md                        # 구현 진행 체크리스트
```

---

## 테스트

```bash
# 전체 테스트 실행
pytest

# 단위 테스트만
pytest tests/unit/

# 통합 테스트만
pytest tests/integration/

# 커버리지 포함
pytest --cov=src/rag_app --cov-report=html

# 특정 테스트 파일
pytest tests/unit/test_chunker.py -v
```

### 테스트 카테고리

| 카테고리 | 대상 |
|----------|------|
| **단위 테스트** | 설정 검증, 경로 정규화, MD5 재현성, 변경 감지, 파서 출력, 한국어 문장 분리, 청킹, 버전 상태 전이, 인용 검증 |
| **통합 테스트** | 파일 추가→SQLite+Qdrant 색인, 동일 MD5 재스캔 시 모델 미호출, 파일 수정→새 버전만 검색, 재색인 실패 시 이전 버전 유지, 파일 삭제→Qdrant 포인트 제거, 스캔 PDF+OCR, Qdrant 장애 시 LLM 미호출, 전체 ask 파이프라인 |
| **회귀/품질 테스트** | Recall@5, nDCG, MRR, 근거 일치율, 출처 정확도, 미답변 질문 거절률, 다국어(ko/ja/en), 상충 문서/프롬프트 인젝션 처리 |

---

## 모니터링 및 운영

### 구조화 로그 (JSON Lines)

```json
{
  "timestamp": "2026-07-27T12:00:00+09:00",
  "level": "INFO",
  "event": "document_indexed",
  "job_id": "uuid",
  "document_id": "uuid",
  "version_id": "uuid",
  "source_path": "manual/product.pdf",
  "duration_ms": 3520,
  "chunk_count": 42,
  "status": "ACTIVE"
}
```

### 주요 지표

- 탐색/신규/변경/삭제/건너뜀/실패 파일 수
- 파일별 파싱/OCR/청킹/임베딩/저장 시간
- 질의별 임베딩/검색/리랭킹/LLM/전체 지연 시간
- 검색 결과 없음 비율, 정보 부족 응답 비율
- 모델/외부 서비스 실패율

### 백업 및 복구

- **SQLite**: 일관된 스냅샷 또는 SQLite 백업 API
- **Qdrant**: 컬렉션 스냅샷 기능
- **동일 백업 세대 ID** 부여로 정합성 보장
- 복구 후 `version_id`와 활성 버전 검증, 불일치 시 원본 문서 기준 전체 재구축