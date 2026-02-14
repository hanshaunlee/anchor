"""In-memory broadcast for risk signals. Production: use Supabase Realtime or Redis."""
from typing import Set

from fastapi import WebSocket

_risk_signal_subscribers: Set[WebSocket] = set()


def add_subscriber(ws: WebSocket) -> None:
    _risk_signal_subscribers.add(ws)


def remove_subscriber(ws: WebSocket) -> None:
    _risk_signal_subscribers.discard(ws)


def broadcast_risk_signal(payload: dict) -> None:
    """Called when a new risk_signal is created (persist or agent)."""
    import asyncio
    for ws in list(_risk_signal_subscribers):
        try:
            asyncio.create_task(ws.send_json(payload))
        except Exception:
            _risk_signal_subscribers.discard(ws)
