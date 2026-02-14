"""
Provider abstraction for outbound caregiver contact: SMS, email, voice.
Interface: send_sms(to, body) -> SendResult, send_email(to, subject, body) -> SendResult.
MockProvider (default): marks sent immediately. Twilio/SendGrid/SMTP require env creds; missing creds -> failed gracefully.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Status string for outbound_actions and API: "sent" | "failed"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"


@dataclass
class SendResult:
    success: bool
    provider_message_id: str | None
    error: str | None
    status: str  # "sent" | "failed" for API/DB


def _result(success: bool, provider_message_id: str | None, error: str | None) -> SendResult:
    return SendResult(
        success=success,
        provider_message_id=provider_message_id,
        error=error,
        status=STATUS_SENT if success else STATUS_FAILED,
    )


def get_notify_provider_from_env() -> str:
    """Read ANCHOR_NOTIFY_PROVIDER (or ANCHOR_NOTIFY_provider). Default mock."""
    try:
        from config.settings import get_notify_settings
        return (get_notify_settings().provider or "mock").lower().strip()
    except Exception:
        return os.environ.get("ANCHOR_NOTIFY_PROVIDER", "mock").lower().strip() or "mock"


def get_provider(name: str | None = None) -> Any:
    """Return provider by name. If name is None, use get_notify_provider_from_env()."""
    n = (name or get_notify_provider_from_env()).lower().strip()
    if n == "twilio":
        return TwilioProvider()
    if n == "sendgrid":
        return SendGridProvider()
    if n == "smtp":
        return SmtpProvider()
    return MockProvider()


class BaseProvider:
    def send_sms(self, to: str, body: str, **kwargs: Any) -> SendResult:
        return _result(False, None, "not_implemented")

    def send_email(self, to: str, subject: str, body: str, **kwargs: Any) -> SendResult:
        return _result(False, None, "not_implemented")

    def send_voice(self, to: str, message: str, **kwargs: Any) -> SendResult:
        return _result(False, None, "not_implemented")


class MockProvider(BaseProvider):
    """Writes to DB + logs; no real send. For tests and dev."""

    def __init__(self, next_id: int = 0):
        self._next_id = next_id

    def _next_mock_id(self) -> str:
        self._next_id += 1
        return f"mock-{self._next_id}"

    def send_sms(self, to: str, body: str, **kwargs: Any) -> SendResult:
        mid = self._next_mock_id()
        logger.info("MockProvider.send_sms to=%s len=%s provider_message_id=%s", to[-4:] if len(to) >= 4 else "****", len(body), mid)
        return _result(True, mid, None)

    def send_email(self, to: str, subject: str, body: str, **kwargs: Any) -> SendResult:
        mid = self._next_mock_id()
        logger.info("MockProvider.send_email to=%s subject=%s provider_message_id=%s", to[:20] + "..." if len(to) > 20 else to, subject[:30], mid)
        return _result(True, mid, None)

    def send_voice(self, to: str, message: str, **kwargs: Any) -> SendResult:
        mid = self._next_mock_id()
        logger.info("MockProvider.send_voice to=%s provider_message_id=%s", to[-4:] if len(to) >= 4 else "****", mid)
        return _result(True, mid, None)


class TwilioProvider(BaseProvider):
    """Twilio SMS/Voice. Requires env TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM."""

    def send_sms(self, to: str, body: str, **kwargs: Any) -> SendResult:
        try:
            from twilio.rest import Client
            sid = os.environ.get("TWILIO_ACCOUNT_SID")
            token = os.environ.get("TWILIO_AUTH_TOKEN")
            from_num = os.environ.get("TWILIO_FROM") or kwargs.get("from_")
            if not sid or not token or not from_num:
                return _result(False, None, "Twilio not configured (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM)")
            client = Client(sid, token)
            msg = client.messages.create(body=body, from_=from_num, to=to)
            return _result(True, msg.sid, None)
        except Exception as e:
            return _result(False, None, str(e))

    def send_voice(self, to: str, message: str, **kwargs: Any) -> SendResult:
        return _result(False, None, "Twilio voice not implemented in this stub")


class SendGridProvider(BaseProvider):
    """SendGrid email. Requires env SENDGRID_API_KEY, SENDGRID_FROM."""

    def send_email(self, to: str, subject: str, body: str, **kwargs: Any) -> SendResult:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            key = os.environ.get("SENDGRID_API_KEY")
            from_addr = os.environ.get("SENDGRID_FROM") or kwargs.get("from_")
            if not key or not from_addr:
                return _result(False, None, "SendGrid not configured (SENDGRID_API_KEY, SENDGRID_FROM)")
            message = Mail(from_email=from_addr, to_emails=to, subject=subject, plain_text_content=body)
            client = SendGridAPIClient(key)
            r = client.send(message)
            return _result(True, r.headers.get("X-Message-Id"), None)
        except Exception as e:
            return _result(False, None, str(e))


class SmtpProvider(BaseProvider):
    """SMTP fallback. Requires env SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM."""

    def send_email(self, to: str, subject: str, body: str, **kwargs: Any) -> SendResult:
        try:
            import smtplib
            from email.mime.text import MIMEText
            host = os.environ.get("SMTP_HOST")
            user = os.environ.get("SMTP_USER")
            password = os.environ.get("SMTP_PASSWORD")
            from_addr = os.environ.get("SMTP_FROM") or user
            if not host:
                return _result(False, None, "SMTP not configured (SMTP_HOST)")
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = from_addr
            msg["To"] = to
            with smtplib.SMTP(host) as s:
                if user and password:
                    s.starttls()
                    s.login(user, password)
                s.sendmail(from_addr, [to], msg.as_string())
            return _result(True, f"smtp-{to[:20]}", None)
        except Exception as e:
            return _result(False, None, str(e))


def send_via_provider(
    provider_name: str,
    channel: str,
    to: str,
    *,
    subject: str | None = None,
    body: str | None = None,
    sms_body: str | None = None,
) -> SendResult:
    """Dispatch to the right provider method by channel (sms, email, voice_call)."""
    provider = get_provider(provider_name)
    if channel == "sms":
        return provider.send_sms(to, sms_body or body or "")
    if channel == "email":
        return provider.send_email(to, subject or "Anchor Alert", body or "")
    if channel == "voice_call":
        return provider.send_voice(to, body or sms_body or "")
    return _result(False, None, f"unknown channel {channel}")
