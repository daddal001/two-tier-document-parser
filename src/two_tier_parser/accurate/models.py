"""
Pydantic models for Accurate Parser Service.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class ImageData(BaseModel):
    """Model for extracted images (ADR-0049 unified shape — fast + accurate tiers)."""
    image_id: str = Field(..., description="First 16 chars of SHA-256 of image bytes")
    image_base64: str = Field(..., description="Base64-encoded image bytes")
    page: int = Field(..., description="First page where image appears")
    bbox: Optional[List[float]] = Field(None, description="[x0, y0, x1, y1]; None for masked/XObject images")
    # ADR-0049 additions — additive, backwards-compatible.
    pages: Optional[List[int]] = Field(default=None, description="All pages where this deduplicated image appears (ADR-0049)")
    mime: Optional[str] = Field(default=None, description="RFC 2046 media type, e.g. 'image/png'")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")


class TableData(BaseModel):
    """Model for extracted tables."""
    table_id: str = Field(..., description="Unique table identifier")
    markdown: str = Field(..., description="Table in markdown format")
    page: int = Field(..., description="Page number where table was found")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")


class FormulaData(BaseModel):
    """Model for extracted formulas."""
    formula_id: str = Field(..., description="Unique formula identifier")
    latex: str = Field(..., description="Formula in LaTeX format")
    page: int = Field(..., description="Page number where formula was found")
    bbox: Optional[List[float]] = Field(None, description="Bounding box [x0, y0, x1, y1]")


class ParsingMetadata(BaseModel):
    """Metadata for parsing results."""
    pages: int = Field(..., description="Number of pages processed")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    parser: str = Field(..., description="Parser library used")
    backend: str = Field(..., description="Backend used (transformers/pipeline)")
    gpu_used: bool = Field(..., description="Whether GPU was used")
    accuracy_tier: str = Field(..., description="Accuracy tier (high/very-high)")
    version: str = Field(..., description="MinerU version")
    filename: str = Field(..., description="Original filename")


class ParseResponse(BaseModel):
    """Response model for PDF parsing."""
    markdown: str = Field(..., description="Extracted markdown content")
    metadata: ParsingMetadata = Field(..., description="Parsing metadata")
    images: List[ImageData] = Field(default_factory=list, description="Extracted images")
    tables: List[TableData] = Field(default_factory=list, description="Extracted tables")
    formulas: List[FormulaData] = Field(default_factory=list, description="Extracted formulas")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    workers: int = Field(..., description="Number of ThreadPool workers")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    parser: str = Field(default="mineru", description="Parser library name")
    version: str = Field(default="1.0.0", description="Service version")
