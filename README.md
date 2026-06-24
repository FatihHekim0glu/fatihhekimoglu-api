# fatihhekimoglu-api

FastAPI compute backend for the fatihhekimoglu.com tools platform. It wraps a set of Python quant projects as HTTP endpoints that the Next.js frontend calls.

## Architecture

- FastAPI on Python 3.11+, served by uvicorn.
- One router per tool under `api/routers/`.
- Vendored compute libraries under `api/lib/`, one subpackage per source project. These are imported byte for byte from their source repos, so do not edit them in place. Re-vendor instead.
- Supabase Postgres backs a shared OHLCV cache (`platform.ohlcv_cache`) and run history (`platform.tool_runs`). Both are best effort: a Supabase outage degrades caching and logging but never breaks a request.
- Deployed to Fly.io in region `lhr`.

## Local dev

```bash
uv sync --all-extras
cp .env.example .env  # fill in FH_SUPABASE_SERVICE_ROLE_KEY
uv run uvicorn api.main:app --reload --port 8080
```

Health check: `curl http://localhost:8080/health`.

## Tests

The suite is offline by default. Tests that hit live data carry the `network`, `slow` or `integration` markers and are excluded unless you opt in.

```bash
uv run pytest                                      # default offline run
uv run pytest -m network                           # opt in to live yfinance / Stooq
uv run pytest --cov --cov-report=term-missing      # with coverage
```

Coverage is measured on the app layer (routers, config, deps). The vendored
libraries under `api/lib/` are tested in their own repos and are excluded from
the coverage gate.

## Lint and types

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Ruff and mypy skip `api/lib/`, since the vendored sources are owned upstream.

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

Each tool is one router under `api/routers/` wired in `api/main.py`. The table
below is generated from those `include_router(...)` calls so it can't drift:

```bash
python scripts/gen_tool_roster.py          # rewrite the table below in place
python scripts/gen_tool_roster.py --check  # CI guard: fail if stale
```

<!-- TOOL-ROSTER:START -->
_26 tools served. Auto-generated from `api/main.py` by `scripts/gen_tool_roster.py` — do not edit by hand._

| Slug | Router file | OpenAPI tag |
|---|---|---|
| `stock-dashboard` | `api/routers/stock_dashboard.py` | `stock-dashboard` |
| `markowitz-optimizer` | `api/routers/markowitz_optimizer.py` | `markowitz-optimizer` |
| `factorlab` | `api/routers/factorlab.py` | `factorlab` |
| `pairs-trading` | `api/routers/pairs_trading.py` | `pairs-trading` |
| `ma-crossover-backtest` | `api/routers/ma_crossover_backtest.py` | `ma-crossover-backtest` |
| `stock-price-forecast` | `api/routers/stock_price_forecast.py` | `stock-price-forecast` |
| `eigen-portfolios` | `api/routers/eigen_portfolios.py` | `eigen-portfolios` |
| `hrp-portfolio` | `api/routers/hrp_portfolio.py` | `hrp-portfolio` |
| `regime-hmm` | `api/routers/regime_hmm.py` | `regime-hmm` |
| `stock-clusters` | `api/routers/stock_clusters.py` | `stock-clusters` |
| `lendingclub-default` | `api/routers/lendingclub_default.py` | `lendingclub-default` |
| `crypto-arb-scanner` | `api/routers/crypto_arb_scanner.py` | `crypto-arb-scanner` |
| `anomaly-detector` | `api/routers/anomaly_detector.py` | `anomaly-detector` |
| `lstm-forecast` | `api/routers/lstm_forecast.py` | `lstm-forecast` |
| `mvts-forecast` | `api/routers/mvts_forecast.py` | `mvts-forecast` |
| `volforecast` | `api/routers/volforecast.py` | `volforecast` |
| `wsb-sentiment-signal` | `api/routers/wsb_sentiment.py` | `wsb-sentiment-signal` |
| `nn-vs-bs` | `api/routers/nn_vs_bs.py` | `nn-vs-bs` |
| `finbert-sentiment` | `api/routers/finbert_sentiment.py` | `finbert-sentiment` |
| `edgar-nlp` | `api/routers/edgar_nlp.py` | `edgar-nlp` |
| `gnn-stocks` | `api/routers/gnn_stocks.py` | `gnn-stocks` |
| `fed-causal` | `api/routers/fed_causal.py` | `fed-causal` |
| `rl-trader` | `api/routers/rl_trader.py` | `rl-trader` |
| `rl-allocator` | `api/routers/rl_allocator.py` | `rl-allocator` |
| `algo-system` | `api/routers/algo_system.py` | `algo-system` |
| `rag-10k` | `api/routers/rag_10k.py` | `rag-10k` |
<!-- TOOL-ROSTER:END -->

## Adding a tool

1. Vendor its `src/` under `api/lib/<slug>/`.
2. Create `api/routers/<slug>.py` with a router named `router`.
3. Import and include it in `api/main.py`.
