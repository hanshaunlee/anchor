"""
Anchor API: FastAPI backend, Supabase, LangGraph pipelines.
Auth: Supabase Auth; household-scoped RLS.
"""
from pathlib import Path

# Load .env from repo root so ANTHROPIC_API_KEY etc. are available (run_api.sh cd's to root)
try:
    from dotenv import load_dotenv
    root = Path(__file__).resolve().parents[3]  # apps/api/api/main.py -> repo root
    load_dotenv(root / ".env")
except ImportError:
    pass

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.broadcast import add_subscriber, broadcast_risk_signal, remove_subscriber
from api.config import settings
from api.routers import agents, alerts, capabilities, connectors, device, explain, graph, households, incident_packets, ingest, investigation, maintenance, outreach, playbooks, protection, risk_signals, rings, sessions, summaries, system, watchlists


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title="Anchor API",
    description="Independence Graph backend: sessions, events, risk signals, watchlists, device sync",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(households.router)
app.include_router(graph.router)
app.include_router(sessions.router)
app.include_router(risk_signals.router)
app.include_router(alerts.router)
app.include_router(capabilities.router)
app.include_router(playbooks.router)
app.include_router(incident_packets.router)
app.include_router(connectors.router)
app.include_router(protection.router)
app.include_router(explain.router)
app.include_router(watchlists.router)
app.include_router(rings.router)
app.include_router(device.router)
app.include_router(ingest.router)
app.include_router(summaries.router)
app.include_router(investigation.router)
app.include_router(maintenance.router)
app.include_router(system.router)
app.include_router(agents.router)
app.include_router(outreach.router)


@app.websocket("/ws/risk_signals")
async def websocket_risk_signals(websocket: WebSocket) -> None:
    """Push new risk_signals to subscribed clients. UI: connect for realtime alerts."""
    await websocket.accept()
    add_subscriber(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        remove_subscriber(websocket)


@app.api_route("/", methods=["GET", "HEAD"])
def root() -> dict:
    """Root: links to docs and no-auth demo. Auth-required endpoints return 401 without a valid JWT."""
    return {
        "name": "Anchor API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "demo_no_auth": "/agents/financial/demo",
    }


def _health_body() -> dict:
    return {"status": "ok"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return _health_body()


@app.api_route("/healthz", methods=["GET", "HEAD"])
def healthz() -> dict:
    """Kubernetes/Render-style health check; same response as /health."""
    return _health_body()
