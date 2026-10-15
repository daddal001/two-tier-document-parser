# API Documentation

Complete API reference for the Two-Tier Document Parser services.

## Base URLs

- **Fast Parser**: `http://localhost:8004`
- **Accurate Parser**: `http://localhost:8005`

Both services provide interactive API documentation via Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## Common Endpoints

### Health Check

**Endpoint**: `GET /health`

Check the health status of a parser service.

**Response** (Fast Parser):
```json
{
  "status": "healthy",
  "workers": 4,
  "no_gil": true,
  "parser": "pymupdf4llm",
  "version": "1.0.0"
}
```

**Response** (Accurate Parser):
```json
{
  "status": "healthy",
  "workers": 2,
  "gpu_available": true,
  "parser": "mineru",
  "version": "1.0.0"
}
```

---

## Fast Parser API

### Parse Document

**Endpoint**: `POST /parse`

Parse a PDF document using the fast CPU-based parser.

**Request**:
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): PDF file to parse

**cURL Example**:
```bash
curl -X POST "http://localhost:8004/parse" \
  -F "file=@document.pdf"
```

**Python Example**:
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8004/parse', files=files)
    result = response.json()
    print(result['markdown'])
```

**Response**:
```json
{
  "markdown": "# Document Title\n\nDocument content in markdown format...",
  "metadata": {
    "pages": 10,
    "processing_time_ms": 5234,
    "parser": "pymupdf4llm",
    "version": "1.0.0"
  }
}
```

**Response Fields**:
- `markdown` (string): Extracted content in markdown format
- `metadata` (object):
  - `pages` (integer): Number of pages in the document
  - `processing_time_ms` (integer): Processing time in milliseconds
  - `parser` (string): Parser library name
  - `version` (string): Service version

**Error Responses**:
- `400 Bad Request`: Invalid file format or missing file
- `413 Payload Too Large`: File exceeds size limit (100MB)
- `500 Internal Server Error`: Parsing failed

---

## Accurate Parser API

### Parse Document

**Endpoint**: `POST /parse`

Parse a PDF document using the accurate parser with automatic GPU detection and multimodal extraction.

**Automatic Backend Selection**:
- **GPU Available**: Uses Transformers backend with MinerU VLM model (95%+ accuracy, ~15-60s/page)
- **No GPU**: Uses Pipeline backend with traditional CV models (80-85% accuracy, ~5-15s/page, CPU-only)
- Detection is fully automatic - no configuration needed

**Request**:
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (required): PDF file to parse
- **Timeout**: Recommended 600 seconds for large documents

**cURL Example**:
```bash
curl -X POST "http://localhost:8005/parse" \
  -F "file=@document.pdf" \
  --max-time 600
```

**Python Example**:
```python
import requests

with open('document.pdf', 'rb') as f:
    files = {'file': f}
    # Increase timeout for processing (especially for GPU/VLM mode)
    response = requests.post(
        'http://localhost:8005/parse',
        files=files,
        timeout=600
    )
    result = response.json()

    # Check which backend was used
    backend = result['metadata']['backend']
    gpu_used = result['metadata']['gpu_used']
    print(f"Backend: {backend}, GPU: {gpu_used}")

    print(result['markdown'])
    print(f"Extracted {len(result['images'])} images")
    print(f"Extracted {len(result['tables'])} tables")
```

**Response**:
```json
{
  "markdown": "# Document Title\n\nContent with images, tables, and formulas...",
  "metadata": {
    "pages": 10,
    "processing_time_ms": 450000,
    "parser": "mineru",
    "backend": "transformers",
    "gpu_used": true,
    "accuracy_tier": "very-high",
    "version": "2.6.4",
    "filename": "document.pdf"
  },
  "images": [
    {
      "image_id": "page_1_img_0",
      "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
      "page": 1,
      "bbox": [100.0, 200.0, 300.0, 400.0]
    }
  ],
  "tables": [
    {
      "table_id": "page_2_table_0",
      "markdown": "| Column 1 | Column 2 |\n|----------|----------|\n| Value 1  | Value 2  |",
      "page": 2,
      "bbox": [50.0, 100.0, 500.0, 300.0]
    }
  ],
  "formulas": [
    {
      "formula_id": "page_3_formula_0",
      "latex": "E = mc^2",
      "page": 3,
      "bbox": [200.0, 150.0, 300.0, 200.0]
    }
  ]
}
```

**Response Fields**:
- `markdown` (string): Extracted content in markdown format with references to images/tables/formulas
- `metadata` (object): Parsing metadata with backend information
  - `pages` (integer): Number of pages processed
  - `processing_time_ms` (integer): Total processing time in milliseconds
  - `parser` (string): Parser name ("mineru")
  - `backend` (string): Backend used - `"transformers"` (GPU/VLM) or `"pipeline"` (CPU)
  - `gpu_used` (boolean): Whether GPU was used for processing
  - `accuracy_tier` (string): `"very-high"` (VLM) or `"high"` (Pipeline)
  - `version` (string): MinerU version
  - `filename` (string): Original filename
- `images` (array): Array of extracted images
  - `image_id` (string): Unique identifier
  - `image_base64` (string): Base64-encoded PNG image
  - `page` (integer): Page number where image was found
  - `bbox` (array, optional): Bounding box `[x0, y0, x1, y1]`
- `tables` (array): Array of extracted tables
  - `table_id` (string): Unique identifier
  - `markdown` (string): Table in markdown format
  - `page` (integer): Page number where table was found
  - `bbox` (array, optional): Bounding box `[x0, y0, x1, y1]`
- `formulas` (array): Array of extracted formulas
  - `formula_id` (string): Unique identifier
  - `latex` (string): Formula in LaTeX format
  - `page` (integer): Page number where formula was found
  - `bbox` (array, optional): Bounding box `[x0, y0, x1, y1]`

**Error Responses**:
- `400 Bad Request`: Invalid file format or missing file
- `413 Payload Too Large`: File exceeds size limit (500MB)
- `500 Internal Server Error`: Parsing failed
- `504 Gateway Timeout`: Processing timeout (increase client timeout)

---

## Rate Limits

Currently, no rate limits are enforced. However, consider:

- **Fast Parser**: Can handle high concurrency (CPU-bound)
- **Accurate Parser**: Limited by GPU memory (recommend queuing for production)

---

## Best Practices

1. **File Size**: Keep files under 100MB (fast) or 500MB (accurate)
2. **Timeouts**: Set appropriate timeouts (30s for fast, 600s for accurate)
3. **Error Handling**: Always check response status codes
4. **Concurrency**: Use connection pooling for multiple requests
5. **Health Checks**: Monitor `/health` endpoint for service availability

---

## Example: Complete Integration

```python
import requests
from typing import Optional

class DocumentParser:
    def __init__(self, fast_url: str = "http://localhost:8004",
                 accurate_url: str = "http://localhost:8005"):
        self.fast_url = fast_url
        self.accurate_url = accurate_url

    def parse_fast(self, pdf_path: str) -> dict:
        """Parse using fast parser."""
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.fast_url}/parse",
                files=files,
                timeout=30
            )
            response.raise_for_status()
            return response.json()

    def parse_accurate(self, pdf_path: str) -> dict:
        """Parse using accurate parser."""
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f"{self.accurate_url}/parse",
                files=files,
                timeout=600
            )
            response.raise_for_status()
            return response.json()

    def health_check(self) -> dict:
        """Check health of both services."""
        return {
            'fast': requests.get(f"{self.fast_url}/health").json(),
            'accurate': requests.get(f"{self.accurate_url}/health").json()
        }

# Usage
parser = DocumentParser()
result = parser.parse_fast('document.pdf')
print(result['markdown'])
```

---

For more information, see the [README](../README.md) or visit the interactive API docs at `/docs` on each service.
