"""Tests for api.main: root, health, docs, WebSocket /ws/risk_signals."""
import pytest

from fastapi.testclient import TestClient

from api.main import app, lifespan


def test_root() -> None:
    """GET / returns name, version, links to docs and demo."""
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data.get("name") == "Anchor API"
    assert data.get("version") == "0.1.0"
    assert data.get("docs") == "/docs"
    assert data.get("redoc") == "/redoc"
    assert data.get("health") == "/health"
    assert data.get("demo_no_auth") == "/agents/financial/demo"


def test_health() -> None:
    """GET /health returns status ok."""
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_lifespan() -> None:
    """Lifespan context manager yields without error."""
    import asyncio
    async def run():
        async with lifespan(app):
            pass
    asyncio.run(run())


@pytest.mark.asyncio
async def test_websocket_risk_signals_accept_and_disconnect() -> None:
    """WebSocket /ws/risk_signals accepts connection, adds subscriber, removes on disconnect."""
    from fastapi.testclient import TestClient as SyncClient
    # TestClient supports websocket_context
    with SyncClient(app) as client:
        with client.websocket_connect("/ws/risk_signals") as ws:
            # Connection accepted
            ws.send_text("ping")
            # No reply expected (server just receives); disconnect closes
    # After block, subscriber should be removed (no exception)


def test_app_has_all_routers() -> None:
    """App includes expected route prefixes."""
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/" in routes
    assert "/health" in routes
    assert "/ws/risk_signals" in routes
    # Router prefixes
    assert any("/risk_signals" in p or p == "/risk_signals" for p in routes)
    assert any("/agents" in p for p in routes)
    assert any("/ingest" in p for p in routes)
    assert any("/device" in p for p in routes)
    assert any("/households" in p for p in routes)
    assert any("/sessions" in p for p in routes)
    assert any("/watchlists" in p for p in routes)
    assert any("/summaries" in p for p in routes)
