"""Tests for api.broadcast: add_subscriber, remove_subscriber, broadcast_risk_signal."""
from unittest.mock import MagicMock

import pytest

from api.broadcast import add_subscriber, remove_subscriber, broadcast_risk_signal


@pytest.fixture(autouse=True)
def clear_subscribers():
    import api.broadcast as mod
    mod._risk_signal_subscribers.clear()
    yield
    mod._risk_signal_subscribers.clear()


def test_add_remove_subscriber() -> None:
    ws = MagicMock(spec=["send_json"])
    add_subscriber(ws)
    import api.broadcast as mod
    assert ws in mod._risk_signal_subscribers
    remove_subscriber(ws)
    assert ws not in mod._risk_signal_subscribers


def test_remove_subscriber_missing_no_error() -> None:
    ws = MagicMock()
    remove_subscriber(ws)


def test_broadcast_risk_signal_no_raise() -> None:
    ws = MagicMock()
    add_subscriber(ws)
    payload = {"signal_type": "test", "severity": 3}
    broadcast_risk_signal(payload)  # may schedule async send; no exception
