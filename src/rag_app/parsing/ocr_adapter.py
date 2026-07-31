"""
OCR Adapter for scanning PDFs and image-based documents.

Provides interface to Unlimited OCR service with configurable
page-level routing, retry logic, and result merging.
"""

import base64
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import httpx

from ..config.schema import AppConfig, OCRConfig
from ..domain.models import ParsedDocument, ParsedBlock
from ..domain.enums import BlockType, OCRStatus
from ..domain.exceptions import OCRProcessingError, OCRRequiredError
from ..ingestion.hasher import ZipBombDetector
from ..observability.logging import get_logger, get_current_job_id


logger = get_logger(__name__)


class OCRProvider(str, Enum):
    """Supported OCR providers."""
    UNLIMITED_OCR = "unlimited_ocr"
    TESSERACT = "tesseract"  # Local fallback


@dataclass(frozen=True, slots=True)
class OCRPageResult:
    """Result of OCR on a single page."""
    page_number: int
    text: str
    confidence: float
    language: str
    processing_time_ms: float


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Complete OCR result for a document."""
    pages: List[OCRPageResult]
    total_pages: int
    successful_pages: int
    failed_pages: List[int]
    total_time_ms: float
    provider: str


class OCRClient(ABC):
    """Abstract OCR client interface."""
    
    @abstractmethod
    def extract_text(self, pdf_path: Path, page_numbers: List[int]) -> OCRResult:
        """Extract text from specified PDF pages."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        pass


class UnlimitedOCRClient(OCRClient):
    """Client for Unlimited OCR API."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.endpoint = config.endpoint.rstrip("/")
        self.api_key = config.api_key_env  # Will be resolved from env
        self.languages = config.languages
        self.timeout = config.timeout_seconds
        self._client: Optional[httpx.Client] = None
    
    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            )
        return self._client
    
    def _resolve_api_key(self) -> str:
        """Resolve API key from environment (optional for local Unlimited OCR)."""
        import os
        if not self.config.api_key_env:
            return ""
        return os.getenv(self.config.api_key_env, "")
    
    def extract_text(self, pdf_path: Path, page_numbers: List[int]) -> OCRResult:
        """Extract text from PDF pages using Unlimited OCR API."""
        job_id = get_current_job_id()
        start_time = time.time()
        
        api_key = self._resolve_api_key()
        
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            # Encode PDF as base64
            pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            
            # Prepare request
            payload = {
                "pdf_base64": pdf_b64,
                "pages": page_numbers,
                "languages": self.languages,
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            client = self._get_client()
            
            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.post(
                        f"{self.endpoint}/ocr",
                        json=payload,
                        headers=headers,
                        timeout=self.timeout,
                    )
                    response.raise_for_status()
                    break
                except httpx.TimeoutException as e:
                    if attempt == max_retries - 1:
                        raise OCRProcessingError(pdf_path, page_numbers, e) from e
                    wait_time = 2 ** attempt
                    logger.warning("ocr_timeout_retry", f"OCR timeout, retrying in {wait_time}s",
                                 job_id=job_id, attempt=attempt + 1, page_count=len(page_numbers))
                    time.sleep(wait_time)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code >= 500:
                        if attempt == max_retries - 1:
                            raise OCRProcessingError(pdf_path, page_numbers, e) from e
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                    else:
                        # Client error - don't retry
                        raise OCRProcessingError(pdf_path, page_numbers, e) from e
            
            result_data = response.json()
            
            # Parse response
            pages = []
            failed_pages = []
            
            for page_result in result_data.get("pages", []):
                page_num = page_result.get("page_number")
                if page_result.get("success", False):
                    pages.append(OCRPageResult(
                        page_number=page_num,
                        text=page_result.get("text", ""),
                        confidence=page_result.get("confidence", 0.0),
                        language=page_result.get("language", "unknown"),
                        processing_time_ms=page_result.get("processing_time_ms", 0.0),
                    ))
                else:
                    failed_pages.append(page_num)
            
            total_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                pages=pages,
                total_pages=len(page_numbers),
                successful_pages=len(pages),
                failed_pages=failed_pages,
                total_time_ms=total_time,
                provider="unlimited_ocr",
            )
            
        except OCRProcessingError:
            raise
        except Exception as e:
            raise OCRProcessingError(pdf_path, page_numbers, e) from e
    
    def is_available(self) -> bool:
        """Check if OCR service is reachable."""
        try:
            client = self._get_client()
            response = client.get(f"{self.endpoint}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            self._client.close()


class TesseractOCRClient(OCRClient):
    """Local Tesseract OCR fallback (if available)."""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.languages = config.languages
        self._available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if tesseract is installed."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def extract_text(self, pdf_path: Path, page_numbers: List[int]) -> OCRResult:
        """Extract text using local Tesseract."""
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
        import io
        
        job_id = get_current_job_id()
        start_time = time.time()
        
        pages = []
        failed_pages = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in page_numbers:
                if page_num > len(doc):
                    failed_pages.append(page_num)
                    continue
                
                try:
                    page = doc[page_num - 1]
                    
                    # Render page as image
                    pix = page.get_pixmap(dpi=300)
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # OCR with Tesseract
                    lang_str = "+".join(self.languages)
                    text = pytesseract.image_to_string(img, lang=lang_str)
                    
                    # Get confidence
                    data = pytesseract.image_to_data(img, lang=lang_str, output_type=pytesseract.Output.DICT)
                    confidences = [int(c) for c in data["conf"] if int(c) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    pages.append(OCRPageResult(
                        page_number=page_num,
                        text=text.strip(),
                        confidence=avg_confidence / 100.0,
                        language=self.languages[0] if self.languages else "unknown",
                        processing_time_ms=0,  # Not tracked for local
                    ))
                    
                except Exception as e:
                    logger.error("tesseract_page_failed", f"Tesseract failed for page {page_num}: {e}",
                               job_id=job_id, page=page_num)
                    failed_pages.append(page_num)
            
            doc.close()
            
        except Exception as e:
            raise OCRProcessingError(pdf_path, page_numbers, e) from e
        
        total_time = (time.time() - start_time) * 1000
        
        return OCRResult(
            pages=pages,
            total_pages=len(page_numbers),
            successful_pages=len(pages),
            failed_pages=failed_pages,
            total_time_ms=total_time,
            provider="tesseract",
        )
    
    def is_available(self) -> bool:
        return self._available


class OCRAdapter:
    """High-level OCR adapter with page detection and result merging."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.ocr_config = config.ocr
        self.client = self._create_client()
        self.text_density_threshold = 0.001  # chars per pixel
        self.min_text_length = 50
    
    def _create_client(self) -> OCRClient:
        """Create OCR client based on configuration."""
        if self.ocr_config.provider == "unlimited_ocr":
            return UnlimitedOCRClient(self.ocr_config)
        elif self.ocr_config.provider == "tesseract":
            return TesseractOCRClient(self.ocr_config)
        else:
            raise ValueError(f"Unknown OCR provider: {self.ocr_config.provider}")
    
    def detect_scan_pages(self, pdf_path: Path) -> List[int]:
        """
        Detect which pages in a PDF are scanned (image-only).
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of page numbers that need OCR.
        """
        import fitz
        
        scan_pages = []
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                text_length = len(text.strip())
                
                # Calculate text density
                page_area = page.rect.width * page.rect.height
                text_density = text_length / page_area if page_area > 0 else 0
                
                # Check for images on page
                images = page.get_images()
                has_images = len(images) > 0
                
                # Heuristic: low text density OR has images but little text
                if text_density < self.text_density_threshold and text_length < self.min_text_length:
                    scan_pages.append(page_num)
                elif has_images and text_length < 100:
                    # Page has images but very little text - might be scanned with some OCR already
                    scan_pages.append(page_num)
            
            doc.close()
            
        except Exception as e:
            logger.error("scan_detection_failed", f"Failed to detect scan pages: {e}",
                        job_id=get_current_job_id(), file_path=str(pdf_path))
            # On error, assume no scan pages
            return []
        
        return scan_pages
    
    def process_document(
        self, 
        pdf_path: Path, 
        parsed_doc: ParsedDocument
    ) -> ParsedDocument:
        """
        Process a document with OCR if needed.
        
        Args:
            pdf_path: Path to PDF file.
            parsed_doc: Already parsed document (may have some text).
            
        Returns:
            Updated ParsedDocument with OCR results merged.
        """
        job_id = get_current_job_id()
        
        if not self.ocr_config.enabled:
            # Check if OCR is required but disabled
            scan_pages = self.detect_scan_pages(pdf_path)
            if scan_pages:
                raise OCRRequiredError(pdf_path, scan_pages)
            return parsed_doc
        
        # Detect pages needing OCR
        scan_pages = self.detect_scan_pages(pdf_path)
        
        if not scan_pages:
            # No OCR needed
            return ParsedDocument(
                **parsed_doc.__dict__,
                ocr_status=OCRStatus.NOT_NEEDED,
                ocr_pages=(),
            )
        
        logger.info("ocr_starting", f"OCR processing {len(scan_pages)} pages",
                   job_id=job_id, file_path=str(pdf_path), pages=scan_pages)
        
        # Run OCR on scan pages
        try:
            ocr_result = self.client.extract_text(pdf_path, scan_pages)
        except OCRProcessingError as e:
            # Partial failure handling
            if self.ocr_config.enabled:
                # Log and continue with what we have
                logger.error("ocr_failed", f"OCR failed: {e}",
                           job_id=job_id, file_path=str(pdf_path))
                # Return original doc with OCR_FAILED status
                return ParsedDocument(
                    **parsed_doc.__dict__,
                    ocr_status=OCRStatus.FAILED,
                    ocr_pages=tuple(scan_pages),
                )
            raise
        
        # Merge OCR results with existing blocks
        merged_blocks = self._merge_ocr_results(parsed_doc.blocks, ocr_result)
        
        # Determine final OCR status
        if ocr_result.failed_pages:
            if len(ocr_result.failed_pages) == len(scan_pages):
                ocr_status = OCRStatus.FAILED
            else:
                ocr_status = OCRStatus.PARTIAL_FAILED
        else:
            ocr_status = OCRStatus.COMPLETED
        
        return ParsedDocument(
            **parsed_doc.__dict__,
            blocks=tuple(merged_blocks),
            ocr_status=ocr_status,
            ocr_pages=tuple(scan_pages),
        )
    
    def _merge_ocr_results(
        self, 
        existing_blocks: Tuple[ParsedBlock, ...], 
        ocr_result: OCRResult
    ) -> List[ParsedBlock]:
        """Merge OCR results with existing parsed blocks."""
        merged = list(existing_blocks)
        
        # Group existing blocks by page
        blocks_by_page: Dict[int, List[ParsedBlock]] = {}
        for block in existing_blocks:
            if block.page_number is not None:
                blocks_by_page.setdefault(block.page_number, []).append(block)
        
        for page_result in ocr_result.pages:
            page_num = page_result.page_number
            
            if not page_result.text.strip():
                continue
            
            existing = blocks_by_page.get(page_num, [])
            
            if existing:
                # Check if existing text is substantially similar
                existing_text = "\n".join(b.text for b in existing)
                if self._text_similarity(existing_text, page_result.text) > 0.8:
                    # Text is similar, keep existing (likely already OCR'd or extracted)
                    continue
                
                # Add OCR text as additional block
                ocr_block = ParsedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=page_result.text,
                    page_number=page_num,
                    section_path=(f"Page {page_num} (OCR)",),
                    sequence=len(merged),
                    metadata={"ocr_confidence": page_result.confidence},
                )
                merged.append(ocr_block)
            else:
                # No existing text for this page, add OCR block
                ocr_block = ParsedBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=page_result.text,
                    page_number=page_num,
                    section_path=(f"Page {page_num} (OCR)",),
                    sequence=len(merged),
                    metadata={"ocr_confidence": page_result.confidence},
                )
                merged.append(ocr_block)
        
        # Sort by sequence
        merged.sort(key=lambda b: b.sequence)
        
        # Renumber sequences
        for idx, block in enumerate(merged):
            # Create new block with updated sequence (since frozen)
            merged[idx] = ParsedBlock(
                block_type=block.block_type,
                text=block.text,
                page_number=block.page_number,
                sheet_name=block.sheet_name,
                slide_number=block.slide_number,
                section_path=block.section_path,
                sequence=idx,
                metadata=block.metadata,
            )
        
        return merged
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using Jaccard index on words."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return self.client.is_available()
    
    def close(self):
        """Close client connections."""
        if hasattr(self.client, 'close'):
            self.client.close()