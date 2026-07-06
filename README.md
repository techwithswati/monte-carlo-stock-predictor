# 📈 Monte Carlo Stock Price Predictor

<p align="center">
  <img src="outputs/AAPL_gbm_simulation.png" alt="Monte Carlo Simulation — AAPL GBM" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/YOUR_GITHUB_USERNAME/monte-carlo-stock-predictor/actions">
    <img src="https://github.com/YOUR_GITHUB_USERNAME/monte-carlo-stock-predictor/actions/workflows/ci.yml/badge.svg" alt="CI Pipeline"/>
  </a>
  <a href="https://codecov.io/gh/YOUR_GITHUB_USERNAME/monte-carlo-stock-predictor">
    <img src="https://codecov.io/gh/YOUR_GITHUB_USERNAME/monte-carlo-stock-predictor/branch/main/graph/badge.svg" alt="Coverage"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.11%20|%203.12-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Docker-ready-2496ED?logo=docker" alt="Docker"/>
  <img src="https://img.shields.io/badge/Kubernetes-ready-326CE5?logo=kubernetes" alt="Kubernetes"/>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License"/>
</p>

> **Production-grade stochastic stock price simulation** built with a Private Equity DevOps mindset.  
> Three quantitative models · FastAPI REST service · Dockerised · Kubernetes-ready · Full CI/CD · 22 unit tests.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Monte Carlo Engine                       │
│                                                             │
│   Market Data (yfinance)  ──►  SimulationConfig             │
│                                      │                      │
│              ┌───────────────────────┼──────────────┐       │
│              ▼               ▼               ▼       │       │
│         GBM Model      Heston Model   Jump-Diffusion │       │
│     (Black-Scholes)  (Stoch. Vol.)   (Merton 1976)  │       │
│              └───────────────────────┴──────────────┘       │
│                                      │                      │
│                              SimulationResult               │
│                    (VaR · CVaR · Sharpe · Percentiles)      │
└─────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              ▼                                  ▼
      FastAPI REST API                    CLI + Charts
      /simulate · /health           run_simulation.py
              │
    ┌─────────┴──────────┐
    ▼                    ▼
 Prometheus           Grafana
 (metrics)          (dashboard)
```

---

## 🔢 Quantitative Models

| Model | Description | Best For |
|-------|-------------|----------|
| **GBM** | Geometric Brownian Motion (Black-Scholes) | Baseline forecasting, liquid large-caps |
| **Heston** | Stochastic Volatility — vol follows mean-reverting CIR | Volatility clustering, options pricing |
| **Jump-Diffusion** | Merton (1976) Poisson jumps + GBM | Crash risk, fat-tail modelling, high-beta stocks |

### Risk Metrics Computed

| Metric | Formula |
|--------|---------|
| **VaR 95%** | 5th percentile of simulated returns |
| **CVaR 95%** | Expected loss beyond VaR (Expected Shortfall) |
| **Sharpe Ratio** | `(μ − rf) / σ × √252`  (rf = 5.25%, 2026) |
| **Max Drawdown** | `min((S_t − max(S_0..t)) / max(S_0..t))` on mean path |
| **Prob. Profit** | `P(S_T > S_0)` across all simulated paths |

---

## 🚀 Quick Start

### Option 1 — Python CLI (fastest)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/monte-carlo-stock-predictor.git
cd monte-carlo-stock-predictor

pip install -r requirements.txt

# GBM — 10,000 paths, 1-year horizon
python run_simulation.py --ticker AAPL --model gbm --sims 10000

# Heston stochastic volatility
python run_simulation.py --ticker NVDA --model heston --sims 5000

# Merton jump-diffusion (fat tails)
python run_simulation.py --ticker TSLA --model jump_diffusion --sims 8000
```

**Sample output:**
```
════════════════════════════════════════════════════════════
  SIMULATION RESULTS — AAPL (GBM)
════════════════════════════════════════════════════════════
  Current Price  : $185.00
  Horizon        : 252 trading days
  Simulations    : 10,000

  P   5           : $    133.46  ▼ 27.9%
  P  25           : $    176.92  ▼  4.4%
  P  50           : $    212.40  ▲ 14.8%
  P  75           : $    255.67  ▲ 38.2%
  P  95           : $    335.43  ▲ 81.3%

  VaR 95%        : -27.86%
  CVaR 95%       : -35.45%
  Sharpe Ratio   : 0.448
  Prob. Profit   : 69.9%
  Elapsed        : 0.12s          ← 10,000 paths, pure NumPy
════════════════════════════════════════════════════════════
```

### Option 2 — Docker (recommended)

```bash
# Build & start full stack (API + Prometheus + Grafana)
make docker-build
make docker-up

# API docs
open http://localhost:8000/docs

# Grafana dashboard
open http://localhost:3000   # admin / montecarlo
```

### Option 3 — REST API

```bash
# Start server
make run-api

# Quick GET
curl "http://localhost:8000/simulate?ticker=AAPL&model=gbm&num_simulations=5000"

# Full POST
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA","model":"heston","trading_days":126,"num_simulations":10000}'
```

**Response:**
```json
{
  "ticker": "NVDA",
  "model": "heston",
  "current_price": 900.0,
  "percentiles": {
    "0.05": 606.94,
    "0.25": 1006.58,
    "0.5":  1347.78,
    "0.75": 1718.33,
    "0.95": 2377.68
  },
  "var_95": -0.3256,
  "cvar_95": -0.4582,
  "sharpe_ratio": 0.984,
  "prob_profit": 0.821,
  "elapsed_seconds": 0.138
}
```

---

## 📊 Simulation Outputs

### AAPL — Geometric Brownian Motion (10,000 paths)
<img src="outputs/AAPL_gbm_simulation.png" width="100%"/>

### NVDA — Heston Stochastic Volatility (5,000 paths)
<img src="outputs/NVDA_heston_simulation.png" width="100%"/>

### TSLA — Merton Jump-Diffusion (8,000 paths)
<img src="outputs/TSLA_jump_diffusion_simulation.png" width="100%"/>

---

## 🗂️ Project Structure

```
monte-carlo-stock-predictor/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint → Test → Security → Docker build
│       └── cd.yml              # Staging → Approval gate → Production
├── src/
│   ├── simulation/
│   │   └── monte_carlo.py      # GBM · Heston · Jump-Diffusion engines
│   ├── data/
│   │   └── fetcher.py          # yfinance market data + μ/σ computation
│   ├── api/
│   │   └── app.py              # FastAPI REST service
│   └── utils/
│       └── logger.py           # Structured JSON logging (ELK/Datadog)
├── tests/
│   └── test_monte_carlo.py     # 22 unit tests — statistical correctness
├── infrastructure/
│   ├── docker-compose.yml      # API + Prometheus + Grafana
│   └── k8s/
│       └── deployment.yaml     # Deployment · Service · HPA
├── monitoring/
│   └── prometheus.yml
├── run_simulation.py           # CLI entry-point + publication charts
├── Dockerfile                  # Multi-stage build — image < 200 MB
├── Makefile                    # Developer experience commands
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
# Run all 22 tests
make test

# With coverage report
make test-cov
```

```
PASSED  TestGBM::test_shape
PASSED  TestGBM::test_initial_price_exact
PASSED  TestGBM::test_all_prices_positive
PASSED  TestGBM::test_mean_path_within_range     ← validates log-normal mean = S₀·e^(μT)
PASSED  TestGBM::test_reproducibility
PASSED  TestGBM::test_var_negative
PASSED  TestGBM::test_cvar_le_var                ← CVaR ≤ VaR (coherent risk measure)
PASSED  TestHeston::test_positive_prices
PASSED  TestJumpDiffusion::test_fatter_tails_than_gbm
PASSED  TestEdgeCases::test_all_models_run
...
22 passed in 2.36s
```

---

## ⚙️ DevOps Pipeline

```
Push to main
    │
    ▼
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  🔍 Lint    │───►│  🧪 Test    │───►│  🔒 Security │───►│  🐳 Docker   │
│  ruff+black │    │  pytest     │    │  bandit+trivy│    │  multi-arch  │
│             │    │  py3.11+3.12│    │  SAST + CVE  │    │  amd64+arm64 │
└─────────────┘    └─────────────┘    └──────────────┘    └──────────────┘
                                                                  │
                                                    ┌─────────────┴──────────────┐
                                                    ▼                            ▼
                                             🚀 Staging                   ✅ Approved?
                                             (auto-deploy)                      │
                                                                    ┌───────────┘
                                                                    ▼
                                                             🏆 Production
                                                           (Blue/Green + HPA)
```

---

## 🛠️ Makefile Commands

```bash
make install          # Install production dependencies
make install-dev      # Install dev + test tools
make lint             # ruff check
make format           # black + ruff --fix
make test             # pytest
make test-cov         # pytest + HTML coverage report
make run-api          # FastAPI dev server (hot reload)
make run-sim          # CLI simulation (AAPL GBM)
make docker-build     # Build Docker image
make docker-up        # Start full monitoring stack
make docker-down      # Stop all containers
make k8s-apply        # Deploy to Kubernetes
make clean            # Remove caches and outputs
```

---

## 🏛️ Mathematical Foundation

### Geometric Brownian Motion
```
dS = μS dt + σS dW
S_T = S_0 · exp((μ − σ²/2)T + σ√T · Z),   Z ~ N(0,1)
```

### Heston Stochastic Volatility
```
dS = μS dt + √V · S dW_S
dV = κ(θ − V) dt + ξ√V dW_V
corr(dW_S, dW_V) = ρ
```

### Merton Jump-Diffusion
```
dS = (μ − λk̄)S dt + σS dW + S dJ
J ~ Compound Poisson(λ) with log-normal jump sizes N(μ_J, σ_J²)
```

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.12 |
| **Simulation** | NumPy (vectorised, zero Python loops) |
| **Data** | yfinance · pandas |
| **API** | FastAPI · uvicorn · pydantic v2 |
| **Visualisation** | matplotlib |
| **Testing** | pytest · pytest-cov |
| **Containers** | Docker (multi-stage) · Docker Compose |
| **Orchestration** | Kubernetes · HPA |
| **CI/CD** | GitHub Actions |
| **Security** | Bandit (SAST) · Trivy (CVE scan) · Safety |
| **Monitoring** | Prometheus · Grafana |
| **Logging** | Structured JSON (ELK / Datadog compatible) |

---

## ⚠️ Disclaimer

This project is built for **educational and portfolio purposes only**.  
It does not constitute financial advice. Past simulated performance is not indicative of future results.  
Always consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT © 2026 — see [LICENSE](LICENSE) for details.
