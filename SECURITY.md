# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of LM Watermark Lab seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. Email us at **security@terragonlabs.com** with details
3. Include the word "VULNERABILITY" in the subject line
4. Provide as much information as possible about the vulnerability

### What to Include

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **24 hours**: Initial acknowledgment of your report
- **7 days**: Initial assessment and severity classification
- **30 days**: Target for providing a fix or mitigation plan
- **90 days**: Public disclosure (coordinated with reporter)

## Security Measures

### Code Security

- **Static Analysis**: All code is scanned with Bandit and CodeQL
- **Dependency Scanning**: Regular vulnerability scans with Safety and pip-audit
- **Container Scanning**: Docker images scanned with Trivy and Grype
- **Secrets Detection**: Automated scanning for exposed credentials
- **License Compliance**: Monitoring for restrictive or incompatible licenses

### Infrastructure Security

- **Principle of Least Privilege**: Minimal required permissions
- **Network Segmentation**: Isolated network zones for different components
- **Encryption**: TLS 1.3 for all external communications
- **Input Validation**: Comprehensive sanitization of all user inputs
- **Rate Limiting**: Protection against abuse and DoS attacks

### Data Protection

- **No Persistent Storage**: User data is not stored permanently
- **Memory Security**: Sensitive data cleared from memory after use
- **Audit Logging**: All security-relevant events are logged
- **Access Controls**: Multi-factor authentication for administrative access

### Model Security

- **Model Integrity**: Cryptographic verification of model files
- **Secure Loading**: Sandboxed model execution environment
- **Resource Limits**: Memory and CPU constraints to prevent abuse
- **Watermark Key Protection**: Secure handling of watermarking keys

## Security Configuration

### Environment Variables

Sensitive configuration should be provided via environment variables, not config files:

```bash
# Required for production
SECRET_KEY=<cryptographically-secure-random-string>
DATABASE_URL=<connection-string-with-credentials>
REDIS_URL=<redis-connection-string>

# Optional API keys
OPENAI_API_KEY=<openai-key>
ANTHROPIC_API_KEY=<anthropic-key>
HUGGINGFACE_TOKEN=<hf-token>
```

### Docker Security

When running with Docker, use these security practices:

```bash
# Run as non-root user
docker run --user 1000:1000 lm-watermark-lab

# Limit resources
docker run --memory=4g --cpus=2.0 lm-watermark-lab

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE lm-watermark-lab

# Use read-only filesystem where possible
docker run --read-only --tmpfs /tmp lm-watermark-lab
```

### Kubernetes Security

For Kubernetes deployments:

```yaml
apiVersion: v1
kind: SecurityContext
spec:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
  seccompProfile:
    type: RuntimeDefault
```

## Security Best Practices

### For Users

1. **Keep Updated**: Always use the latest version
2. **Secure Configuration**: Use strong passwords and API keys
3. **Network Security**: Deploy behind a firewall and use HTTPS
4. **Monitoring**: Enable logging and monitor for suspicious activity
5. **Backup Security**: Encrypt backups and store securely

### For Developers

1. **Input Validation**: Validate all inputs at API boundaries
2. **Error Handling**: Don't expose sensitive information in errors
3. **Logging**: Log security events but avoid logging sensitive data
4. **Dependencies**: Keep dependencies updated and scan for vulnerabilities
5. **Testing**: Include security tests in your test suite

## Vulnerability Disclosure

### Our Commitment

- We will acknowledge receipt of vulnerability reports within 24 hours
- We will provide regular updates on our progress
- We will credit reporters in our security advisories (unless they prefer to remain anonymous)
- We will not pursue legal action against security researchers who follow this policy

### Hall of Fame

We maintain a hall of fame for security researchers who have helped improve our security:

*No vulnerabilities reported yet*

## Security Contacts

- **General Security**: security@terragonlabs.com
- **Security Team Lead**: security-lead@terragonlabs.com
- **PGP Key**: [Available on our website]

## Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [CWE Top 25](https://cwe.mitre.org/top25/archive/2023/2023_top25_list.html)

## Security Advisories

Security advisories are published on our [GitHub Security Advisories page](https://github.com/terragon-labs/lm-watermark-lab/security/advisories).

## Compliance

LM Watermark Lab is designed to help meet various compliance requirements:

- **SOC 2**: Security controls and monitoring
- **GDPR**: Privacy by design and data minimization
- **HIPAA**: Security safeguards for healthcare data
- **ISO 27001**: Information security management

For specific compliance questions, contact our compliance team at compliance@terragonlabs.com.