# fatihhekimoglu-api

FastAPI compute backend for the fatihhekimoglu.com tools platform. Wraps existing Python quant projects (`stock-dashboard`, `ma-crossover-backtest`, `markowitz-optimizer`) as HTTP endpoints consumed by the Next.js frontend.

## Architecture

- **FastAPI** on Python 3.12, served by uvicorn.
- One router per tool under `api/routers/`.
- Vendored compute libraries under `api/lib/` (one subpackage per source project). **Never modify in place** — these are imported byte-for-byte from their source repos.
- **Supabase Postgres** for shared OHLCV cache (`platform.ohlcv_cache`) and run history (`platform.tool_runs`).
- Deployed to **Fly.io** at region `lhr`.

## Local dev

```bash
uv sync --all-extras
cp .env.example .env  # fill in FH_SUPABASE_SERVICE_ROLE_KEY
uv run uvicorn api.main:app --reload --port 8080
```

Health: `curl http://localhost:8080/health`.

## Tests

```bash
uv run pytest                      # default — excludes @network and @slow
uv run pytest -m network           # hit live yfinance / Stooq
uv run pytest --cov=api            # with coverage
```

## Deploy

```bash
# One-time
fly auth login
fly apps create fatihhekimoglu-api --org personal
fly secrets set FH_SUPABASE_URL=https://rjzlehedbbcahwmfmdig.supabase.co
fly secrets set FH_SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>

# Every deploy
fly deploy
```

## Tool roster

| Slug | Router file | Source project |
|---|---|---|
| `stock-dashboard` | `api/routers/stock_dashboard.py` | `~/stock-dashboard/src` |
| `ma-crossover-backtest` | _(future)_ | `~/ma-crossover-backtest/src` |
| `markowitz-optimizer` | _(future)_ | `~/markowitz-optimizer/src` |
