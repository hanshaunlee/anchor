"""
Caregiver Escalation & Outreach Agent.
Ten steps: load context, policy gate, choose channel/recipients, evidence bundle,
generate message (Claude when ANTHROPIC_API_KEY set, else template), create outbound_actions row,
dispatch, update risk_signal, broadcast, audit.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from domain.agents.base import AgentContext, persist_agent_run_ctx, step
from domain.consent import normalize_consent_state
from domain.notify.providers import get_provider, get_notify_provider_from_env, send_via_provider

logger = logging.getLogger(__name__)

CAREGIVER_MESSAGE_MAX_CHARS = 600
ELDER_SAFE_MAX_CHARS = 280

AGENT_NAME = "caregiver_outreach"


def _agent_settings():
    try:
        from config.settings import get_agent_settings
        return get_agent_settings()
    except Exception:
        return None


def _pipeline_settings():
    try:
        from config.settings import get_pipeline_settings
        return get_pipeline_settings()
    except Exception:
        return None


def _fetch_household_consent_defaults(supabase: Any, household_id: str) -> dict[str, Any]:
    """Get household_consent_defaults so Settings (Allow outbound contact) is used when session has no override."""
    if not supabase:
        return {}
    d = supabase.table("household_consent_defaults").select("*").eq("household_id", household_id).limit(1).execute()
    if not d.data or len(d.data) == 0:
        return {}
    row = d.data[0]
    return {
        "share_with_caregiver": row.get("share_with_caregiver", True),
        "share_text": row.get("share_text", True),
        "allow_outbound_contact": row.get("allow_outbound_contact", False),
        "escalation_threshold": row.get("escalation_threshold", 3),
    }


def _fetch_consent_for_session(supabase: Any, household_id: str, session_id: str | None) -> dict[str, Any]:
    if not supabase:
        return normalize_consent_state({})
    # Start with household defaults (Settings → Allow outbound contact); session overrides on top.
    raw = dict(_fetch_household_consent_defaults(supabase, household_id))
    if session_id:
        r = supabase.table("sessions").select("consent_state").eq("id", session_id).limit(1).execute()
        if r.data and len(r.data) > 0:
            session_state = r.data[0].get("consent_state") or {}
            raw.update({k: v for k, v in session_state.items() if v is not None})
            return normalize_consent_state(raw)
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        session_state = r.data[0].get("consent_state") or {}
        raw.update({k: v for k, v in session_state.items() if v is not None})
    return normalize_consent_state(raw)


def _fetch_risk_signal(supabase: Any, risk_signal_id: str, household_id: str) -> dict | None:
    if not supabase:
        return None
    r = (
        supabase.table("risk_signals")
        .select("*")
        .eq("id", risk_signal_id)
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    return r.data if r.data else None


def _fetch_caregiver_contacts(supabase: Any, household_id: str) -> list[dict]:
    if not supabase:
        return []
    r = (
        supabase.table("caregiver_contacts")
        .select("*")
        .eq("household_id", household_id)
        .order("priority")
        .execute()
    )
    return list(r.data or [])


def _existing_outbound_for_signal(supabase: Any, risk_signal_id: str) -> dict | None:
    """If an outbound_action already exists for this risk_signal with status queued/sent/delivered, return it (idempotency)."""
    if not supabase:
        return None
    r = (
        supabase.table("outbound_actions")
        .select("id, status")
        .eq("triggered_by_risk_signal_id", risk_signal_id)
        .in_("status", ["queued", "sent", "delivered"])
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        return r.data[0]
    return None


def _in_quiet_hours(quiet_hours: dict, now: datetime) -> bool:
    start = quiet_hours.get("start")  # e.g. "22:00"
    end = quiet_hours.get("end")      # e.g. "08:00"
    if not start or not end:
        return False
    try:
        from datetime import time
        t = now.time()
        start_parts = [int(x) for x in start.split(":")[:2]]
        end_parts = [int(x) for x in end.split(":")[:2]]
        start_t = time(start_parts[0], start_parts[1] if len(start_parts) > 1 else 0)
        end_t = time(end_parts[0], end_parts[1] if len(end_parts) > 1 else 0)
        if start_t < end_t:
            return start_t <= t <= end_t
        return t >= start_t or t <= end_t
    except Exception:
        return False


def _build_evidence_bundle(signal: dict, consent_share_text: bool) -> dict[str, Any]:
    expl = signal.get("explanation") or {}
    timeline_snippet = list(expl.get("timeline_snippet") or [])
    if not consent_share_text and timeline_snippet:
        timeline_snippet = [{**item, "text_preview": "(redacted)"} if isinstance(item, dict) else item for item in timeline_snippet]
    return {
        "timeline_snippet": timeline_snippet,
        "model_subgraph": expl.get("model_subgraph") if expl.get("model_available") else None,
        "motifs": list(expl.get("motif_tags") or expl.get("motifs") or []),
        "evidence_refs": [
            {"event_id": e.get("event_id"), "entity_id": e.get("entity_id"), "relationship_id": e.get("relationship_id")}
            for e in (expl.get("evidence_refs") or [])
        ],
        "recommended_actions": list((signal.get("recommended_action") or {}).get("checklist") or (signal.get("recommended_action") or {}).get("steps") or []),
    }


def _generate_message_claude(
    signal: dict,
    evidence: dict[str, Any],
    consent_share_text: bool,
) -> dict[str, Any] | None:
    """Use Claude to write caregiver and elder-safe messages. Returns same shape as _generate_message_template or None on failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    expl = signal.get("explanation") or {}
    summary = (expl.get("summary") or expl.get("narrative") or "")[:800]
    severity = signal.get("severity", 0)
    motifs = list(expl.get("motif_tags") or expl.get("motifs") or [])
    timeline = evidence.get("timeline_snippet") or []
    timeline_text = ""
    if consent_share_text and timeline:
        for i, item in enumerate(timeline[:5]):
            if isinstance(item, dict):
                timeline_text += (item.get("text_preview") or item.get("summary") or str(item))[:200] + "\n"
            else:
                timeline_text += str(item)[:200] + "\n"
    recommended = evidence.get("recommended_actions") or []
    context = (
        f"Alert summary: {summary or 'Activity may need attention.'}\n"
        f"Severity (1-5): {severity}. Priority: {'High' if severity >= 4 else 'Medium' if severity >= 3 else 'Normal'}.\n"
        f"Motifs/tags: {', '.join(motifs[:8]) if motifs else 'None'}.\n"
    )
    if timeline_text:
        context += f"Relevant context (respect privacy):\n{timeline_text[:600]}\n"
    if recommended:
        context += f"Suggested next steps (use or adapt): {recommended[:5]}.\n"
    if not consent_share_text:
        context += "Do not include specific details; consent limits what we can share.\n"

    prompt = f"""You are writing a short message to a family caregiver for an elder-safety app (Anchor). The caregiver will receive this by SMS or email.

{context}

Respond with a JSON object only, no other text:
{{
  "caregiver_message": "2-4 sentences for the caregiver. Be clear, calm, and actionable. Say what kind of concern and suggest they check the dashboard. Max 600 characters total.",
  "elder_safe_message": "One short sentence the elder can see: that we've notified their caregiver and they may check in. Max 280 characters.",
  "subject_line": "Short email subject, e.g. 'Anchor: alert may need attention'"
}}

Rules: caregiver_message must be under 600 characters. elder_safe_message under 280. No names or PII unless already in the context. Tone: supportive, not alarming."""

    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()
        if not text:
            return None
        # Strip markdown code block if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(text)
        caregiver_message = (parsed.get("caregiver_message") or "")[:CAREGIVER_MESSAGE_MAX_CHARS]
        elder_safe_message = (parsed.get("elder_safe_message") or "We've shared a brief summary with your caregiver. They may reach out to check in.")[:ELDER_SAFE_MAX_CHARS]
        subject_line = (parsed.get("subject_line") or "Anchor: alert may need attention")[:100]
        if not caregiver_message:
            return None
        do_next = recommended[:5] if recommended else ["Review alert in dashboard.", "Consider calling to check in."]
        return {
            "caregiver_message": caregiver_message,
            "caregiver_message_email": caregiver_message + "\n\nView full details in the Anchor dashboard.",
            "elder_safe_message": elder_safe_message,
            "subject_line": subject_line,
            "why_now": (summary or caregiver_message)[:200],
            "do_next": do_next,
        }
    except Exception as e:
        logger.warning("Claude caregiver message failed: %s", e)
        return None


def _generate_message_template(
    signal: dict,
    evidence: dict[str, Any],
    consent_share_text: bool,
) -> dict[str, Any]:
    """Template-based message (no LLM). Human-readable; SMS-safe <= 600 chars; elder_safe <= 280."""
    raw_summary = (signal.get("explanation") or {}).get("summary") or ""
    if not consent_share_text:
        summary = "There may be activity that needs your attention. (Details are limited by privacy settings.)"
    else:
        # Plain-language summary: avoid jargon like "severity N/5" in the main line
        summary = raw_summary.strip() if raw_summary else "There may be activity that needs your attention."
        if summary and not any(summary.startswith(p) for p in ("There ", "Your ", "A ", "Someone ", "We ")):
            summary = "Activity note: " + summary
    why_now = (raw_summary or summary)[:200]
    severity = signal.get("severity", 0)
    severity_line = f"Priority: {'High' if severity >= 4 else 'Medium' if severity >= 3 else 'Normal'}."
    caregiver_message = (
        f"{summary[:380]}\n\n{severity_line} "
        "Please check the Anchor dashboard when you can and consider reaching out."
    )[:CAREGIVER_MESSAGE_MAX_CHARS]
    elder_safe_message = (
        "We've shared a brief summary with your caregiver. They may reach out to check in."
    )[:ELDER_SAFE_MAX_CHARS]
    do_next = evidence.get("recommended_actions") or ["Review alert in dashboard.", "Consider calling to check in."]
    return {
        "caregiver_message": caregiver_message,
        "caregiver_message_email": caregiver_message + "\n\nView full details in the Anchor dashboard.",
        "elder_safe_message": elder_safe_message,
        "subject_line": "Anchor: alert may need attention",
        "why_now": why_now,
        "do_next": do_next[:5],
    }


def run_caregiver_outreach_agent(
    household_id: str,
    supabase: Any,
    *,
    risk_signal_id: str | None = None,
    dry_run: bool = True,
    consent_state: dict[str, Any] | None = None,
    user_role: str = "caregiver",
) -> dict[str, Any]:
    """
    Run the Caregiver Outreach playbook. Caller must enforce 403 for elder (API layer).
    Returns step_trace, summary_json, run_id, outbound_action (if created), status.
    """
    now = datetime.now(timezone.utc)
    ctx = AgentContext(
        household_id=household_id,
        supabase=supabase,
        dry_run=dry_run,
        now=now,
        consent_state=consent_state or {},
        user_role=user_role,
    )
    step_trace: list[dict] = []
    started_at = now.isoformat()
    summary_json: dict[str, Any] = {
        "headline": "Caregiver Outreach",
        "outbound_action_id": None,
        "status_reason": None,
        "suppressed": False,
        "sent": False,
    }
    outbound_action_id: str | None = None
    agent_run_id: str | None = None

    # Step 1 — Load incident context
    with step(ctx, step_trace, "load_incident_context"):
        if not risk_signal_id:
            step_trace[-1]["status"] = "skip"
            step_trace[-1]["notes"] = "no risk_signal_id"
            summary_json["status_reason"] = "no risk_signal_id"
            agent_run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "completed", step_trace, summary_json)
            return {"step_trace": step_trace, "summary_json": summary_json, "run_id": agent_run_id, "outbound_action_id": None, "status": "completed"}
        signal = _fetch_risk_signal(supabase, risk_signal_id, household_id)
        if not signal:
            step_trace[-1]["status"] = "error"
            step_trace[-1]["error"] = "risk_signal not found"
            summary_json["status_reason"] = "risk_signal not found"
            agent_run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "failed", step_trace, summary_json)
            return {"step_trace": step_trace, "summary_json": summary_json, "run_id": agent_run_id, "outbound_action_id": None, "status": "failed"}
        session_ids = (signal.get("explanation") or {}).get("session_ids") or []
        session_id = str(session_ids[0]) if session_ids else None
        consent_normalized = _fetch_consent_for_session(supabase, household_id, session_id)
        ctx.consent_state = consent_normalized
        contacts = _fetch_caregiver_contacts(supabase, household_id)
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["notes"] = f"signal loaded; consent_allow_outbound={consent_normalized.get('consent_allow_outbound_contact')}; contacts={len(contacts)}"

    # Step 2 — Policy gate
    with step(ctx, step_trace, "policy_gate"):
        allow_outbound = ctx.consent_for("consent_allow_outbound_contact") or consent_normalized.get("consent_allow_outbound_contact", False)
        if not allow_outbound:
            step_trace[-1]["notes"] = "consent_allow_outbound_contact=false -> suppressed"
            summary_json["suppressed"] = True
            summary_json["status_reason"] = "Consent does not allow outbound contact."
            payload_blank = {
                "caregiver_message": "",
                "elder_safe_message": "No outbound message was sent (consent settings).",
                "evidence_refs": [],
                "recommended_actions": [],
            }
            if not dry_run and supabase:
                try:
                    ins = supabase.table("outbound_actions").insert({
                        "household_id": household_id,
                        "triggered_by_risk_signal_id": risk_signal_id,
                        "action_type": "caregiver_notify",
                        "channel": "sms",
                        "recipient_name": None,
                        "recipient_contact": None,
                        "payload": payload_blank,
                        "status": "suppressed",
                        "provider": "mock",
                        "error": "consent_allow_outbound_contact=false",
                    }).execute()
                    if ins.data and len(ins.data) > 0:
                        outbound_action_id = ins.data[0].get("id")
                        summary_json["outbound_action_id"] = outbound_action_id
                except Exception as e:
                    logger.warning("Insert suppressed outbound_action failed: %s", e)
            agent_run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "completed", step_trace, summary_json)
            return {"step_trace": step_trace, "summary_json": summary_json, "run_id": agent_run_id, "outbound_action_id": outbound_action_id, "status": "completed"}

    # Step 3 — Determine escalation channel + recipients
    with step(ctx, step_trace, "determine_channel_recipients"):
        policy = consent_normalized.get("caregiver_contact_policy") or {}
        allowed_channels = policy.get("allowed_channels") or ["sms", "email"]
        quiet_hours = policy.get("quiet_hours") or {}
        in_quiet = _in_quiet_hours(quiet_hours, now)
        if in_quiet and "email" in allowed_channels:
            channel = "email"
            step_trace[-1]["notes"] = "quiet hours -> email"
        else:
            channel = allowed_channels[0] if allowed_channels else "sms"
        recipient = None
        recipient_name = None
        recipient_contact = None
        if contacts:
            recipient = contacts[0]
            recipient_name = recipient.get("name") or "Caregiver"
            ch = recipient.get("channels") or {}
            if channel == "sms" and ch.get("sms"):
                recipient_contact = ch["sms"].get("number") or ch["sms"].get("value")
            elif channel == "email" and ch.get("email"):
                recipient_contact = ch["email"].get("email") or ch["email"].get("value")
            elif channel == "voice_call" and ch.get("voice"):
                recipient_contact = ch["voice"].get("number") or ch["voice"].get("value")
            else:
                for c in ["sms", "email", "voice_call"]:
                    if c in ch and ch[c]:
                        v = ch[c]
                        recipient_contact = v.get("number") or v.get("email") or v.get("value")
                        channel = "sms" if "number" in str(v) else "email" if "email" in str(v) else "voice_call"
                        break
        if not recipient_contact:
            recipient_contact = "demo@example.com"
            step_trace[-1]["notes"] = "no contact -> demo placeholder"
        step_trace[-1]["outputs_count"] = 1
        step_trace[-1]["artifacts_refs"] = {"channel": channel, "recipient_last4": (recipient_contact or "")[-4:]}

    # Step 4 — Evidence packaging
    with step(ctx, step_trace, "evidence_packaging"):
        consent_share_text = consent_normalized.get("consent_share_text", True)
        evidence_bundle = _build_evidence_bundle(signal, consent_share_text)
        step_trace[-1]["outputs_count"] = len(evidence_bundle.get("evidence_refs") or [])

    # Step 5 — Generate message (Claude when ANTHROPIC_API_KEY set, else template)
    with step(ctx, step_trace, "generate_message"):
        message_payload = _generate_message_claude(signal, evidence_bundle, consent_share_text)
        if not message_payload:
            message_payload = _generate_message_template(signal, evidence_bundle, consent_share_text)
        else:
            step_trace[-1]["notes"] = "claude"
        payload_for_db = {
            "caregiver_message": message_payload["caregiver_message"],
            "caregiver_message_email": message_payload.get("caregiver_message_email"),
            "elder_safe_message": message_payload["elder_safe_message"],
            "subject_line": message_payload.get("subject_line"),
            "why_now": message_payload.get("why_now"),
            "do_next": message_payload.get("do_next"),
            "evidence_refs": evidence_bundle.get("evidence_refs") or [],
            "recommended_actions": evidence_bundle.get("recommended_actions") or [],
        }
        if not consent_share_text:
            payload_for_db["redacted"] = True
        step_trace[-1]["outputs_count"] = 1
        # Expose for preview (dry_run) and API response
        summary_json["caregiver_message"] = message_payload["caregiver_message"]
        summary_json["elder_safe_message"] = message_payload["elder_safe_message"]

    # Step 6 — Create outbound_actions row (queued); idempotency: skip if already sent for this signal
    with step(ctx, step_trace, "create_outbound_action"):
        provider_name = get_notify_provider_from_env()
        existing = _existing_outbound_for_signal(supabase, risk_signal_id) if not dry_run and supabase else None
        if existing:
            outbound_action_id = existing.get("id")
            summary_json["outbound_action_id"] = outbound_action_id
            summary_json["idempotent"] = True
            step_trace[-1]["notes"] = f"already exists: {existing.get('status')} -> skip create/send"
        elif not dry_run and supabase:
            try:
                recipient_last4 = (recipient_contact or "")[-4:] if (recipient_contact and len(recipient_contact) >= 4) else None
                row: dict[str, Any] = {
                    "household_id": household_id,
                    "triggered_by_risk_signal_id": risk_signal_id,
                    "action_type": "caregiver_notify",
                    "channel": channel,
                    "recipient_user_id": (recipient or {}).get("user_id"),
                    "recipient_name": recipient_name,
                    "recipient_contact": recipient_contact,
                    "payload": payload_for_db,
                    "status": "queued",
                    "provider": provider_name,
                }
                if recipient_last4 is not None:
                    row["recipient_contact_last4"] = recipient_last4
                ins = supabase.table("outbound_actions").insert(row).execute()
                if ins.data and len(ins.data) > 0:
                    outbound_action_id = ins.data[0].get("id")
                    summary_json["outbound_action_id"] = outbound_action_id
            except Exception as e:
                step_trace[-1]["status"] = "error"
                step_trace[-1]["error"] = str(e)
                logger.warning("Insert outbound_actions failed: %s", e)
        step_trace[-1]["notes"] = step_trace[-1].get("notes") or (outbound_action_id or "dry_run")

    # Step 7 — Dispatch action (skip if idempotent or dry_run)
    with step(ctx, step_trace, "dispatch_action"):
        if dry_run:
            step_trace[-1]["notes"] = "dry_run skip send"
        elif summary_json.get("idempotent"):
            step_trace[-1]["notes"] = "idempotent skip send"
        elif outbound_action_id and recipient_contact:
            provider_name = get_notify_provider_from_env()
            if channel == "sms":
                result = send_via_provider(provider_name, "sms", recipient_contact, sms_body=payload_for_db["caregiver_message"])
            elif channel == "email":
                result = send_via_provider(
                    provider_name, "email", recipient_contact,
                    subject=payload_for_db.get("subject_line") or "Anchor Alert",
                    body=payload_for_db.get("caregiver_message_email") or payload_for_db["caregiver_message"],
                )
            else:
                result = send_via_provider(provider_name, "voice_call", recipient_contact, body=payload_for_db["caregiver_message"])
            status = result.status  # "sent" | "failed"
            err = result.error
            try:
                supabase.table("outbound_actions").update({
                    "status": status,
                    "provider_message_id": result.provider_message_id,
                    "error": err,
                    "sent_at": now.isoformat() if result.success else None,
                }).eq("id", outbound_action_id).execute()
            except Exception as e:
                logger.warning("Update outbound_actions after send failed: %s", e)
            summary_json["sent"] = result.success
            summary_json["provider_status"] = status
            step_trace[-1]["notes"] = f"provider_message_id={result.provider_message_id}; status={status}"
        else:
            step_trace[-1]["notes"] = "no outbound_action_id or recipient"

    # Step 8 — Update risk_signal
    with step(ctx, step_trace, "update_risk_signal"):
        if not dry_run and supabase and outbound_action_id:
            try:
                expl = dict(signal.get("explanation") or {})
                expl["escalation"] = {
                    "sent_at": now.isoformat(),
                    "channel": channel,
                    "recipient_last4": (recipient_contact or "")[-4:],
                    "outbound_action_id": outbound_action_id,
                }
                supabase.table("risk_signals").update({
                    "status": "escalated",
                    "explanation": expl,
                    "updated_at": now.isoformat(),
                }).eq("id", risk_signal_id).eq("household_id", household_id).execute()
            except Exception as e:
                logger.warning("Update risk_signal escalation failed: %s", e)
        step_trace[-1]["notes"] = "status -> escalated"

    # Step 9 — Notify UI (broadcast)
    with step(ctx, step_trace, "notify_ui"):
        try:
            from api.broadcast import broadcast_risk_signal
            broadcast_risk_signal({"id": risk_signal_id, "household_id": household_id, "status": "escalated"})
        except Exception as e:
            logger.debug("broadcast_risk_signal failed: %s", e)
        step_trace[-1]["notes"] = "broadcast_risk_signal"

    # Step 10 — Audit + finish
    with step(ctx, step_trace, "audit_finish"):
        summary_json["headline"] = "Caregiver Outreach complete"
        summary_json["channel"] = channel
        summary_json["suppressed"] = summary_json.get("suppressed", False)
        summary_json["sent"] = summary_json.get("sent", False)
    ended_at = datetime.now(timezone.utc).isoformat()
    agent_run_id = persist_agent_run_ctx(ctx, AGENT_NAME, "completed", step_trace, summary_json, started_at=started_at, ended_at=ended_at)
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "run_id": agent_run_id,
        "outbound_action_id": outbound_action_id,
        "status": "completed",
    }
