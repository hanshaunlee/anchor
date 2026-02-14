"""
Anchor API: FastAPI backend, Supabase, LangGraph pipelines.
Auth: Supabase Auth; household-scoped RLS.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routers import device, households, ingest, risk_signals, sessions, summaries, watchlists

# Optional: LangGraph pipeline router when implemented
# from api.routers import pipeline


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
app.include_router(sessions.router)
app.include_router(risk_signals.router)
app.include_router(watchlists.router)
app.include_router(device.router)
app.include_router(ingest.router)
app.include_router(summaries.router)


# In-memory broadcast for demo; production: use Supabase Realtime or Redis
_risk_signal_subscribers: set[WebSocket] = set()


@app.websocket("/ws/risk_signals")
async def websocket_risk_signals(websocket: WebSocket) -> None:
    """Push new risk_signals to subscribed clients. UI: connect for realtime alerts."""
    await websocket.accept()
    _risk_signal_subscribers.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _risk_signal_subscribers.discard(websocket)


def broadcast_risk_signal(payload: dict) -> None:
    """Called by worker/pipeline when a new risk_signal is created."""
    import asyncio
    for ws in list(_risk_signal_subscribers):
        try:
            asyncio.create_task(ws.send_json(payload))
        except Exception:
            _risk_signal_subscribers.discard(ws)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
