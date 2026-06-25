# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Yes             |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do NOT open a public GitHub Issue**.

Instead, please report it privately:

1. Go to the [Security tab](https://github.com/Jemin29/AI-Fashion-Agent/security) of this repository
2. Click **"Report a vulnerability"**
3. Provide a detailed description including:
   - Type of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge your report within **48 hours** and provide a resolution timeline within **7 days**.

## Security Best Practices for Contributors

- Never commit API keys, tokens, or credentials to the repository
- Use `.env` files for local secrets (already in `.gitignore`)
- Audit third-party dependencies before adding them
- Keep model weights and dataset paths out of source code
