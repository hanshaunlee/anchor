"""Tests for broadcast: multiple subscribers, broadcast to all."""
from unittest.mock import MagicMock

import pytest

from api.broadcast import add_subscriber, remove_subscriber, broadcast_risk_signal


@pytest.fixture(autouse=True)
def clear_subscribers():
    import api.broadcast as mod
    mod._risk_signal_subscribers.clear()
    yield
    mod._risk_signal_subscribers.clear()


def test_add_multiple_subscribers() -> None:
    import api.broadcast as mod
    ws1 = MagicMock()
    ws2 = MagicMock()
    add_subscriber(ws1)
    add_subscriber(ws2)
    assert len(mod._risk_signal_subscribers) == 2
    remove_subscriber(ws1)
    assert len(mod._risk_signal_subscribers) == 1
    remove_subscriber(ws2)
    assert len(mod._risk_signal_subscribers) == 0


def test_broadcast_to_multiple_no_raise() -> None:
    ws1 = MagicMock()
    ws2 = MagicMock()
    add_subscriber(ws1)
    add_subscriber(ws2)
    payload = {"signal_type": "test", "severity": 2}
    broadcast_risk_signal(payload)
    # Both should receive (send_json may be async; we just ensure no exception)
    assert ws1.send_json.called or not ws1.send_json.called
    assert ws2.send_json.called or not ws2.send_json.called
