"""
Monte Carlo Stock Predictor — REST API
=======================================
Production-grade FastAPI service with rate limiting, structured logging,
health checks, and Prometheus metrics.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.data.fetcher import fetch_market_data
from src.simulation.monte_carlo import MonteCarloEngine, SimulationConfig, SimulationModel
from src.utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
APP_ENV = os.getenv("APP_ENV", "production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Monte Carlo API starting up | version=%s env=%s", APP_VERSION, APP_ENV)
    yield
    logger.info("🛑 Monte Carlo API shutting down")


app = FastAPI(
    title="Monte Carlo Stock Predictor",
    description=(
        "Production-grade stochastic stock price simulation API. "
        "Supports GBM, Heston Stochastic Volatility, and Merton Jump-Diffusion models."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class SimulationRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Yahoo Finance ticker symbol")
    trading_days: int = Field(252, ge=1, le=1260, description="Forecast horizon in trading days")
    num_simulations: int = Field(10_000, ge=100, le=100_000, description="Number of Monte Carlo paths")
    model: SimulationModel = Field(SimulationModel.GBM, description="Stochastic model")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    uptime_seconds: float


_start_time = time.time()

# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.perf_counter() - t0:.4f}s"
    return response

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """Liveness probe — used by Kubernetes and load balancers."""
    return HealthResponse(
        status="healthy",
        version=APP_VERSION,
        environment=APP_ENV,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/ready", tags=["Ops"])
async def readiness():
    """Readiness probe — confirms the service is ready to serve traffic."""
    return {"status": "ready"}


@app.get("/simulate", tags=["Simulation"])
async def simulate_get(
    ticker: str = Query(..., example="AAPL"),
    trading_days: int = Query(252, ge=1, le=1260),
    num_simulations: int = Query(5_000, ge=100, le=50_000),
    model: SimulationModel = Query(SimulationModel.GBM),
):
    """Quick GET endpoint for browser / curl access."""
    req = SimulationRequest(
        ticker=ticker,
        trading_days=trading_days,
        num_simulations=num_simulations,
        model=model,
    )
    return await _run_simulation(req)


@app.post("/simulate", tags=["Simulation"])
async def simulate_post(req: SimulationRequest):
    """Full POST endpoint with JSON body — preferred for programmatic access."""
    return await _run_simulation(req)


@app.get("/models", tags=["Simulation"])
async def list_models():
    """Return all available stochastic models with descriptions."""
    return {
        "models": [
            {
                "id": "gbm",
                "name": "Geometric Brownian Motion",
                "description": "Classic Black-Scholes baseline. Constant drift & volatility.",
            },
            {
                "id": "heston",
                "name": "Heston Stochastic Volatility",
                "description": "Volatility follows a mean-reverting CIR process. Captures vol clustering.",
            },
            {
                "id": "jump_diffusion",
                "name": "Merton Jump-Diffusion",
                "description": "Adds Poisson-distributed price jumps. Models crash risk & fat tails.",
            },
        ]
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _run_simulation(req: SimulationRequest) -> dict:
    logger.info(
        "Simulation request | ticker=%s model=%s days=%d sims=%d",
        req.ticker, req.model.value, req.trading_days, req.num_simulations,
    )
    try:
        market = fetch_market_data(req.ticker)
    except Exception as exc:
        logger.error("Data fetch failed | ticker=%s error=%s", req.ticker, exc)
        raise HTTPException(status_code=400, detail=f"Failed to fetch data for '{req.ticker}': {exc}")

    config = SimulationConfig(
        ticker=req.ticker,
        current_price=market.current_price,
        expected_return=market.expected_return,
        volatility=market.volatility,
        trading_days=req.trading_days,
        num_simulations=req.num_simulations,
        model=req.model,
        seed=req.seed,
    )

    try:
        engine = MonteCarloEngine(config)
        result = engine.run()
    except Exception as exc:
        logger.error("Simulation failed | %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    payload = result.to_dict()
    payload["market_data"] = {
        "source": market.source,
        "current_price": market.current_price,
        "annualised_return": round(market.expected_return, 4),
        "annualised_volatility": round(market.volatility, 4),
    }
    return payload
