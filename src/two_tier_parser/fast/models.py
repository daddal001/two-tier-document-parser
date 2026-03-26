"""
Pydantic models for Fast Parser Service.
"""
from pydantic import BaseModel, Field


class ParseResponse(BaseModel):
    """Response model for PDF parsing."""
    markdown: str = Field(..., description="Extracted markdown content")
    metadata: dict = Field(..., description="Parsing metadata")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    workers: int = Field(..., description="Number of ThreadPool workers")
    no_gil: bool = Field(..., description="Whether Python no-GIL mode is enabled")
    parser: str = Field(default="pymupdf4llm", description="Parser library name")
    version: str = Field(default="1.0.0", description="Service version")
    minio_connected: bool = Field(default=False, description="MinIO connectivity status")


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
