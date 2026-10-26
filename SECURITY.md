# Security

- Keep provider keys in environment variables; never commit API keys or populated local configuration files.
- Do not commit reference photos, model request/response payloads, generated `.blend` files, renders, logs, or run directories.
- If a credential is accidentally committed, revoke it with its provider before removing it from Git history.
