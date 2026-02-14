"""Outbound notify: provider abstraction for SMS, email, voice."""

from domain.notify.providers import get_provider, send_via_provider

__all__ = ["get_provider", "send_via_provider"]
