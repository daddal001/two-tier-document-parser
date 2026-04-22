"""Pydantic models for Fast Parser Service (ADR-0049)."""
from typing import List, Optional

from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """Extracted image metadata — unified shape with accurate parser (ADR-0049).

    Populated by fast-parser's direct PyMuPDF API extraction (service.extract_images).
    SHA-256 content hash (first 16 chars) serves as ``image_id`` so the same image
    referenced from multiple pages deduplicates to one record with ``pages=[...]``.
    """
    image_id: str = Field(..., description="First 16 chars of SHA-256 of image bytes")
    image_base64: str = Field(..., description="Base64-encoded image bytes")
    page: int = Field(..., description="First page where image appears")
    bbox: Optional[List[float]] = Field(None, description="[x0, y0, x1, y1]; None for masked/XObject images")
    pages: Optional[List[int]] = Field(default=None, description="All pages where this deduplicated image appears")
    mime: Optional[str] = Field(default=None, description="RFC 2046 media type, e.g. 'image/png'")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")


class ParseResponse(BaseModel):
    """Response model for full-document PDF parsing."""
    markdown: str = Field(..., description="Extracted markdown content")
    metadata: dict = Field(..., description="Parsing metadata")
    images: List[ImageData] = Field(
        default_factory=list,
        description="Extracted images (empty when extract_images=false). OWASP ASVS V12 caps apply.",
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    workers: int = Field(..., description="Number of ProcessPool workers")
    no_gil: bool = Field(..., description="Whether Python no-GIL mode is enabled")
    parser: str = Field(default="pymupdf4llm", description="Parser library name")
    version: str = Field(default="1.0.0", description="Service version")
    minio_connected: bool = Field(default=False, description="MinIO connectivity status")
    layout_enabled: bool = Field(default=False, description="Whether pymupdf-layout GNN is active (ADR-0049)")


class PageRangeMetadata(BaseModel):
    """Metadata for page-range parsing response."""
    filename: str = Field(..., description="Original filename")
    total_pages: int = Field(..., description="Total pages in document")
    pages_parsed: int = Field(..., description="Number of pages actually parsed")
    start_page: int = Field(..., description="Start page (0-indexed, inclusive)")
    end_page: int = Field(..., description="End page (exclusive)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    parser: str = Field(default="pymupdf4llm", description="Parser library name")
    version: str = Field(default="0.2.0", description="Parser version")


class PageRangeParseResponse(BaseModel):
    """Response model for page-range PDF parsing."""
    markdown: str = Field(..., description="Extracted markdown content from page range")
    metadata: PageRangeMetadata = Field(..., description="Page range parsing metadata")
    images: List[ImageData] = Field(
        default_factory=list,
        description="Extracted images from the requested page range (ADR-0049 Stage 1 + Stage 2 unified).",
    )
