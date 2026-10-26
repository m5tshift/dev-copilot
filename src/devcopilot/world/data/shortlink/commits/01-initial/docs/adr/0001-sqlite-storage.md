# ADR 0001: SQLite as link storage

Status: accepted (2026-09-01)

## Context
Expected volume is under 100k links and a few hundred redirects per minute.

## Decision
Use a single SQLite file on the app server. Ids are auto-increment integers
and short codes are their base62 encoding.

## Consequences
- No extra infrastructure.
- Codes are guessable. Do not use shortlink for private URLs.
- Horizontal scaling beyond one host requires migrating to Postgres.
