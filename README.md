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
