# Architecture

`shortlink` is a library wrapped by a thin HTTP layer (not in this repo).

```
HTTP handler -> ShortLinkService (shortlink/api.py)
                  |-- codes.py   base62 id <-> code
                  `-- store.py   SQLite persistence
```

- A short code is the base62 encoding of the auto-increment row id.
  Codes are therefore sequential and guessable; this is accepted for
  marketing links (see ADR 0001).
- The service is deployed as 4 gunicorn workers behind nginx.
- All timestamps are stored and compared in UTC.
