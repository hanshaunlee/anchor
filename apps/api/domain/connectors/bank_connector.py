"""
Bank connector abstraction: data-first (read-only), optional controls.
Never claim lock/freeze unless connector returned success and we recorded it.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BankConnector(ABC):
    """Interface: get accounts/transactions (read-only); optional control methods when capability enabled."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Connector identifier (e.g. mock, plaid)."""
        ...

    @abstractmethod
    def get_accounts(self, household_id: str) -> list[dict[str, Any]]:
        """List linked accounts (id, name, type, mask, etc.). Read-only."""
        ...

    @abstractmethod
    def get_transactions(
        self, household_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """Recent transactions. Read-only."""
        ...

    def lock_card(self, household_id: str, card_id: str) -> dict[str, Any]:
        """Optional: lock a card. Return {success: bool, receipt_id?: str, error?: str}."""
        return {"success": False, "error": "Not implemented"}

    def enable_alerts(self, household_id: str, account_id: str) -> dict[str, Any]:
        """Optional: enable alerts for account. Return {success: bool, receipt_id?: str, error?: str}."""
        return {"success": False, "error": "Not implemented"}

    def disable_transfers(self, household_id: str, account_id: str) -> dict[str, Any]:
        """Optional: disable transfers. Return {success: bool, error?: str}."""
        return {"success": False, "error": "Not implemented"}


class MockBankConnector(BankConnector):
    """
    Demo connector: derive financial context from events/entities in DB or fixtures.
    No real bank API. For hackathon demos.
    """

    def __init__(self, supabase: Any = None):
        self._supabase = supabase

    @property
    def name(self) -> str:
        return "mock"

    def get_accounts(self, household_id: str) -> list[dict[str, Any]]:
        """From entities (entity_type=account) or events (transaction_detected, payee_added)."""
        out: list[dict[str, Any]] = []
        if not self._supabase:
            return _demo_accounts()
        try:
            r = (
                self._supabase.table("entities")
                .select("id, canonical, meta")
                .eq("household_id", household_id)
                .eq("entity_type", "account")
                .limit(20)
                .execute()
            )
            for row in r.data or []:
                out.append({
                    "id": str(row["id"]),
                    "name": row.get("canonical") or "Account",
                    "type": (row.get("meta") or {}).get("type", "checking"),
                    "mask": (row.get("meta") or {}).get("mask", "****"),
                })
        except Exception as e:
            logger.debug("MockBankConnector get_accounts: %s", e)
        if not out:
            return _demo_accounts()
        return out

    def get_transactions(
        self, household_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        """From events (transaction_detected, payee_added) or empty."""
        if not self._supabase:
            return _demo_transactions()
        try:
            from datetime import datetime, timezone, timedelta
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            session_ids = []
            sess = (
                self._supabase.table("sessions")
                .select("id")
                .eq("household_id", household_id)
                .gte("started_at", start.isoformat())
                .lte("started_at", end.isoformat())
                .limit(50)
                .execute()
            )
            session_ids = [s["id"] for s in (sess.data or [])]
            if not session_ids:
                return _demo_transactions()
            ev = (
                self._supabase.table("events")
                .select("id, ts, event_type, payload")
                .in_("session_id", session_ids)
                .in_("event_type", ["transaction_detected", "payee_added", "bank_alert_received"])
                .order("ts", desc=True)
                .limit(100)
                .execute()
            )
            out = []
            for e in ev.data or []:
                p = e.get("payload") or {}
                out.append({
                    "id": str(e["id"]),
                    "ts": e.get("ts"),
                    "event_type": e.get("event_type"),
                    "amount": p.get("amount"),
                    "payee": p.get("payee"),
                    "description": p.get("description"),
                })
            return out if out else _demo_transactions()
        except Exception as e:
            logger.debug("MockBankConnector get_transactions: %s", e)
            return _demo_transactions()


def _demo_accounts() -> list[dict[str, Any]]:
    return [
        {"id": "demo-account-1", "name": "Primary Checking", "type": "checking", "mask": "4242"},
        {"id": "demo-account-2", "name": "Savings", "type": "savings", "mask": "8888"},
    ]


def _demo_transactions() -> list[dict[str, Any]]:
    return [
        {"id": "demo-tx-1", "ts": None, "event_type": "transaction_detected", "amount": None, "payee": None, "description": "Recent activity (demo)"},
    ]


class PlaidConnector(BankConnector):
    """
    Plaid stub: no real calls unless env creds exist.
    Endpoints: link_token, exchange_public_token, sync_transactions.
    If not configured, callers get 501 / fallback to event-derived context.
    """

    def __init__(self, supabase: Any = None, client_id: str | None = None, secret: str | None = None):
        self._supabase = supabase
        self._client_id = client_id or _env("PLAID_CLIENT_ID")
        self._secret = secret or _env("PLAID_SECRET")
        self._configured = bool(self._client_id and self._secret)

    @property
    def name(self) -> str:
        return "plaid"

    @property
    def is_configured(self) -> bool:
        return self._configured

    def get_accounts(self, household_id: str) -> list[dict[str, Any]]:
        if not self._configured:
            return []
        # Stub: would call Plaid /accounts/get with stored access_token for household
        return []

    def get_transactions(
        self, household_id: str, days: int = 30
    ) -> list[dict[str, Any]]:
        if not self._configured:
            return []
        # Stub: would call Plaid /transactions/sync
        return []


def _env(key: str) -> str:
    import os
    return os.environ.get(key, "")


def get_bank_connector(
    connector_type: str,
    supabase: Any = None,
) -> BankConnector:
    """Factory: returns MockBankConnector for none/demo; PlaidConnector for plaid if configured."""
    if connector_type in ("plaid", "open_banking", "custom"):
        c = PlaidConnector(supabase, _env("PLAID_CLIENT_ID"), _env("PLAID_SECRET"))
        if c.is_configured:
            return c
        logger.info("Plaid not configured; falling back to mock")
    return MockBankConnector(supabase)
