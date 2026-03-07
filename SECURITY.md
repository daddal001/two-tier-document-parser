# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x.x   | ✅ Yes    |
| < 1.0   | ❌ No     |

## Reporting a Vulnerability

**DO NOT** open public issues for security vulnerabilities.

### Reporting Process

1. **Email:** Send details to the repository maintainers via the contact information in the package metadata
2. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### Response Timeline

| Stage | Timeframe |
|-------|-----------|
| Acknowledgment | 48 hours |
| Initial assessment | 5 business days |
| Fix development | Depends on severity |
| Public disclosure | After fix is released |

### Severity Classification

| Severity | Response Time | Examples |
|----------|--------------|----------|
| **Critical** | 24-48 hours | RCE, authentication bypass |
| **High** | 1 week | Data exposure, privilege escalation |
| **Medium** | 2 weeks | Information disclosure, DoS |
| **Low** | Next release | Minor issues, hardening |

## Security Considerations

### Input Validation

- PDF files are validated before processing
- File size limits are enforced
- Temporary files are cleaned up after processing

### Dependencies

This project depends on:
- **MinerU** (AGPL-3.0) - Document parsing with VLM
- **PyMuPDF** (AGPL-3.0) - PDF text extraction

Both are actively maintained open-source projects.

### Container Security

- Non-root user recommended for production
- Minimal base images used
- No sensitive data stored in images

## Best Practices for Users

1. **Network Isolation:** Run parsers in isolated network segments
2. **Input Validation:** Validate files before sending to parser
3. **Resource Limits:** Set appropriate memory/CPU limits
4. **Monitoring:** Monitor for unusual parsing patterns
5. **Updates:** Keep dependencies updated

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who help improve the project's security.
