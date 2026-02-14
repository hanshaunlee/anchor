"""Connectors: bank data and optional controls. Data-first; controls only when capability is enabled."""

from .bank_connector import (
    BankConnector,
    get_bank_connector,
    MockBankConnector,
    PlaidConnector,
)

__all__ = [
    "BankConnector",
    "get_bank_connector",
    "MockBankConnector",
    "PlaidConnector",
]
