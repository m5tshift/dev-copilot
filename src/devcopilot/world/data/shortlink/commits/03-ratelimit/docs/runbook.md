# Runbook

## Users get HTTP 429 when creating links
Rate limit is 30 creations per client per minute (`shortlink/ratelimit.py`).
Counters are per gunicorn worker, so the real limit varies between 30 and 120.
If a partner integration is blocked, ask them to batch requests; raising
`MAX_REQUESTS` requires a deploy.

## Resolved link returns 404
1. Check that the code decodes to an existing id (`shortlink.codes.decode`).
2. If the link has a TTL, check `created_at` and `ttl_hours` in the `links` table.
   Expired links intentionally resolve to 404.

## Resolve returns HTTP 500
Look at the traceback in the app logs. Recent deploys are listed in CI
(main branch). Roll back to the previous green build if the error started
right after a deploy.
