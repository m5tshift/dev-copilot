# shortlink

Minimal URL shortener used by the marketing team.

```python
from shortlink.api import ShortLinkService

svc = ShortLinkService()
code = svc.shorten("https://example.com/landing")
svc.resolve(code)  # -> "https://example.com/landing"
```

- Architecture: `docs/architecture.md`
- Decisions: `docs/adr/`
- Run tests: `python -m pytest`
