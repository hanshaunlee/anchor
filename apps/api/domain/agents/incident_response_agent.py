"""
Incident Response / Account Lockdown Agent.
Capability-aware: only executes supported actions; otherwise produces bank-ready case file + guided playbook DAG.
Never claims lock/freeze unless connector returned success.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any
from uuid import UUID

from domain.agents.base import AgentContext, persist_agent_run_ctx, step, upsert_summary_ctx
from domain.action_dag import build_action_graph
from domain.capability_service import get_household_capabilities
from domain.connectors.bank_connector import get_bank_connector

logger = logging.getLogger(__name__)

AGENT_NAME = "incident_response"

# Demo bank directory (resolve contact endpoints)
DEMO_BANK_DIRECTORY = [
    {"name": "Primary Bank", "phone": "1-800-555-0100", "email": "fraud@primarybank.demo"},
    {"name": "Credit Union", "phone": "1-800-555-0200", "email": "security@creditunion.demo"},
]


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


def _consent_state(supabase: Any, household_id: str) -> dict[str, Any]:
    if not supabase:
        return {}
    r = (
        supabase.table("sessions")
        .select("consent_state")
        .eq("household_id", household_id)
        .order("started_at", desc=True)
        .limit(1)
        .execute()
    )
    if r.data and len(r.data) > 0:
        return r.data[0].get("consent_state") or {}
    # Fallback to household_consent_defaults if table exists
    try:
        d = (
            supabase.table("household_consent_defaults")
            .select("share_with_caregiver, share_text, allow_outbound_contact, escalation_threshold")
            .eq("household_id", household_id)
            .limit(1)
            .execute()
        )
        if d.data and len(d.data) > 0:
            row = d.data[0]
            return {
                "share_with_caregiver": row.get("share_with_caregiver", True),
                "share_text": row.get("share_text", True),
                "allow_outbound_contact": row.get("allow_outbound_contact", False),
                "escalation_threshold": row.get("escalation_threshold", 3),
            }
    except Exception:
        pass
    return {}


def _build_incident_packet(
    risk_signal: dict,
    financial_context: list[dict],
    entity_ids: list[str],
    event_ids: list[str],
) -> dict[str, Any]:
    """Bank-ready, evidence-cited packet: timeline, suspected vector, entities, amounts, evidence_refs only."""
    expl = risk_signal.get("explanation") or {}
    timeline = expl.get("timeline_snippet") or []
    # Only include fields backed by evidence_refs
    return {
        "risk_signal_id": risk_signal.get("id"),
        "signal_type": risk_signal.get("signal_type"),
        "severity": risk_signal.get("severity"),
        "timeline": timeline[:10],
        "suspected_vector": expl.get("motif_tags") or expl.get("motifs") or [],
        "entity_ids": entity_ids,
        "event_ids": event_ids,
        "amounts": [],  # Populate from financial_context if present
        "evidence_refs": [{"event_id": eid} for eid in event_ids[:20]] + [{"entity_id": eid} for eid in entity_ids[:20]],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _generate_bank_script_llm(
    incident_packet: dict,
    evidence_refs_only: list[str],
) -> dict[str, Any] | None:
    """Structured output: call_script, email_template, key_facts (evidence_refs only), verification_checklist."""
    try:
        from pydantic import BaseModel, Field
        from domain.langchain_utils import get_llm, run_structured_prompt

        class BankContactScript(BaseModel):
            call_script: str = Field(description="Short spoken script for bank hotline")
            email_template: str = Field(description="Brief email body for bank")
            key_facts: list[str] = Field(description="Bullet list of facts; only from evidence")
            verification_checklist: list[str] = Field(description="Steps for bank to verify identity")

        llm = get_llm()
        if not llm:
            return _fallback_bank_script(incident_packet, evidence_refs_only)
        prompt = (
            "Generate a bank contact script for this incident. "
            "key_facts must ONLY cite the following evidence refs (use them verbatim or paraphrase strictly): "
            + json.dumps(evidence_refs_only[:15])
            + "\n\nIncident summary: "
            + json.dumps({k: v for k, v in incident_packet.items() if k not in ("evidence_refs", "generated_at")}, default=str)[:1500]
        )
        result = run_structured_prompt(llm, prompt, BankContactScript)
        if result:
            return {
                "call_script": result.call_script,
                "email_template": result.email_template,
                "key_facts": result.key_facts,
                "verification_checklist": result.verification_checklist,
            }
    except Exception as e:
        logger.warning("Bank script LLM failed: %s", e)
    return _fallback_bank_script(incident_packet, evidence_refs_only)


def _fallback_bank_script(incident_packet: dict, evidence_refs_only: list[str]) -> dict[str, Any]:
    return {
        "call_script": "I'm calling regarding possible unauthorized activity. I have a case file with timeline and evidence refs to share.",
        "email_template": "Subject: Possible fraud – case file attached\n\nPlease find attached a case file with timeline and evidence references.",
        "key_facts": [f"Evidence ref: {r}" for r in evidence_refs_only[:5]] if evidence_refs_only else ["See incident packet timeline."],
        "verification_checklist": ["Verify account holder identity", "Confirm recent transactions", "Request dispute if applicable"],
    }


def run_incident_response_agent(
    household_id: str,
    risk_signal_id: str,
    supabase: Any = None,
    dry_run: bool = False,
    *,
    consent_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run Incident Response Agent: load signal, capabilities, build DAG, incident packet,
    persist playbook/tasks, execute safe actions, update risk_signal, broadcast, summary.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    ctx = AgentContext(household_id, supabase, dry_run=dry_run, consent_state=consent_state or {})
    step_trace: list[dict] = []
    summary_json: dict[str, Any] = {"headline": "Incident Response", "playbook_id": None, "incident_packet_id": None}

    # Step 1 — Load incident
    with step(ctx, step_trace, "load_incident", inputs_count=1):
        risk_signal = _fetch_risk_signal(ctx.supabase, risk_signal_id, ctx.household_id)
        if not risk_signal:
            raise ValueError(f"Risk signal {risk_signal_id} not found")
        expl = risk_signal.get("explanation") or {}
        entity_ids = list(expl.get("entity_ids") or [])
        event_ids = list(expl.get("event_ids") or [])
        step_trace[-1]["notes"] = f"signal_type={risk_signal.get('signal_type')}, entities={len(entity_ids)}, events={len(event_ids)}"

    # Step 2 — Load capabilities
    with step(ctx, step_trace, "load_capabilities"):
        capabilities = get_household_capabilities(ctx.supabase, ctx.household_id) if ctx.supabase else {}
        if not capabilities and ctx.supabase:
            from domain.capability_service import DEFAULT_CAPABILITIES
            capabilities = {"household_id": ctx.household_id, **DEFAULT_CAPABILITIES}
        consent = consent_state or _consent_state(ctx.supabase, ctx.household_id)
        # Support both canonical (consent_allow_outbound_contact) and legacy (allow_outbound_contact) keys
        consent_allow_outbound = consent.get("consent_allow_outbound_contact", consent.get("allow_outbound_contact", False))

    # Step 3 — Fetch financial context
    with step(ctx, step_trace, "fetch_financial_context"):
        connector_type = capabilities.get("bank_data_connector", "none")
        bank = get_bank_connector(connector_type, ctx.supabase)
        if connector_type != "none" and hasattr(bank, "is_configured") and getattr(bank, "is_configured", False):
            accounts = bank.get_accounts(ctx.household_id)
            transactions = bank.get_transactions(ctx.household_id, days=30)
            financial_context = {"accounts": accounts, "transactions": transactions}
        else:
            # Event-derived
            accounts = bank.get_accounts(ctx.household_id)
            transactions = bank.get_transactions(ctx.household_id, days=30)
            financial_context = {"accounts": accounts, "transactions": transactions, "source": "events_or_mock"}
        step_trace[-1]["outputs_count"] = len(financial_context.get("accounts", [])) + len(financial_context.get("transactions", []))

    # Step 4 — Build incident packet
    with step(ctx, step_trace, "build_incident_packet"):
        incident_packet = _build_incident_packet(
            risk_signal,
            financial_context.get("transactions", []),
            entity_ids,
            event_ids,
        )
        evidence_refs_only = [str(x) for x in incident_packet.get("evidence_refs", [])]

    # Step 5 — Build Action DAG (deterministic)
    with step(ctx, step_trace, "build_action_dag"):
        graph = build_action_graph(risk_signal, capabilities, consent_allow_outbound)
        step_trace[-1]["outputs_count"] = len(graph.get("nodes", []))
        step_trace[-1]["notes"] = f"{len(graph['nodes'])} nodes, {len(graph.get('edges', []))} edges"

    # Step 6 — Generate bank script (LLM or fallback)
    with step(ctx, step_trace, "generate_bank_script"):
        bank_script = _generate_bank_script_llm(incident_packet, evidence_refs_only)
        if not bank_script:
            bank_script = _fallback_bank_script(incident_packet, evidence_refs_only)

    # Step 7 — Resolve contact endpoints
    with step(ctx, step_trace, "resolve_contact_endpoints"):
        contact_endpoints = DEMO_BANK_DIRECTORY[:2]

    # Step 8 — Persist playbook + tasks + incident_packet
    playbook_id = None
    incident_packet_id = None
    with step(ctx, step_trace, "persist_playbook_tasks"):
        if not ctx.dry_run and ctx.supabase:
            now = datetime.now(timezone.utc).isoformat()
            # Insert incident_packet
            ins_p = ctx.supabase.table("incident_packets").insert({
                "household_id": ctx.household_id,
                "risk_signal_id": risk_signal_id,
                "packet_json": incident_packet,
            }).execute()
            incident_packet_id = str(ins_p.data[0]["id"]) if ins_p.data else None
            # Insert action_playbooks
            ins_pb = ctx.supabase.table("action_playbooks").insert({
                "household_id": ctx.household_id,
                "risk_signal_id": risk_signal_id,
                "playbook_type": "incident_response",
                "graph": graph,
                "status": "active",
                "updated_at": now,
            }).execute()
            playbook_id = str(ins_pb.data[0]["id"]) if ins_pb.data else None
            # Insert action_tasks from graph nodes
            if playbook_id:
                for node in graph.get("nodes", []):
                    task_type = node.get("task_type", "file_report")
                    details = {}
                    if task_type in ("call_bank", "email_bank") and contact_endpoints:
                        details["phone"] = contact_endpoints[0].get("phone")
                        details["email"] = contact_endpoints[0].get("email")
                        details["call_script"] = bank_script.get("call_script", "")
                        details["email_template"] = bank_script.get("email_template", "")
                        details["key_facts"] = bank_script.get("key_facts", [])
                        details["verification_checklist"] = bank_script.get("verification_checklist", [])
                    details["evidence_refs"] = incident_packet.get("evidence_refs", [])[:10]
                    ctx.supabase.table("action_tasks").insert({
                        "playbook_id": playbook_id,
                        "task_type": task_type,
                        "status": "ready" if node.get("status") == "ready" else "blocked",
                        "details": details,
                    }).execute()
            summary_json["playbook_id"] = playbook_id
            summary_json["incident_packet_id"] = incident_packet_id
        step_trace[-1]["outputs_count"] = 1 if playbook_id else 0

    # Step 9 — Execute safe automatic actions
    with step(ctx, step_trace, "execute_safe_actions"):
        if ctx.dry_run:
            step_trace[-1]["notes"] = "Dry run; no notify/device/connector calls"
        else:
            # Notify caregiver if consent + capability
            if consent_allow_outbound and (capabilities.get("notify_sms_enabled") or capabilities.get("notify_email_enabled")):
                try:
                    from domain.agents.caregiver_outreach_agent import run_caregiver_outreach_agent
                    run_caregiver_outreach_agent(
                        ctx.household_id, ctx.supabase,
                        risk_signal_id=risk_signal_id, dry_run=False,
                        consent_state=consent, user_role="caregiver",
                    )
                except Exception as e:
                    logger.warning("Outreach from incident response: %s", e)
            # Device high_risk_mode: add watchlist + payload for device sync
            if capabilities.get("device_policy_push_enabled") and ctx.supabase:
                try:
                    from domain.agents.base import upsert_watchlist
                    upsert_watchlist(ctx.supabase, ctx.household_id, {
                        "watch_type": "high_risk_mode",
                        "pattern": {"active": True, "reason": "incident_response", "risk_signal_id": risk_signal_id},
                        "reason": "High-risk mode active; do not share codes.",
                        "priority": 100,
                    }, False)
                except Exception as e:
                    logger.warning("Watchlist high_risk_mode: %s", e)
            # enable_alerts via connector if supported and explicitly allowed (here we do not auto-call; task stays ready for caregiver)
            step_trace[-1]["notes"] = "Notify/device applied per capability; bank controls left as tasks"

    # Step 10 — Update risk_signal recommended_action
    with step(ctx, step_trace, "update_risk_signal"):
        if not ctx.dry_run and ctx.supabase and playbook_id:
            rec = risk_signal.get("recommended_action") or {}
            rec["playbook_id"] = playbook_id
            rec["incident_packet_id"] = incident_packet_id
            rec["key_tasks"] = [n.get("task_type") for n in graph.get("nodes", [])[:8]]
            ctx.supabase.table("risk_signals").update({
                "recommended_action": rec,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", risk_signal_id).eq("household_id", ctx.household_id).execute()

    # Step 11 — Broadcast
    with step(ctx, step_trace, "broadcast"):
        if not ctx.dry_run and playbook_id:
            try:
                from api.broadcast import broadcast_risk_signal
                broadcast_risk_signal({"risk_signal_id": risk_signal_id, "playbook_id": playbook_id})
            except Exception as e:
                logger.debug("Broadcast: %s", e)

    # Step 12 — Post-run summary
    with step(ctx, step_trace, "post_run_summary"):
        period_end = ctx.now.isoformat()
        period_start = (ctx.now - timedelta(days=1)).isoformat()
        summary_text = f"Incident response: playbook {playbook_id or 'n/a'} created for signal {risk_signal_id}. Tasks ready for caregiver."
        summary_json["key_metrics"] = {"playbook_id": playbook_id, "tasks_count": len(graph.get("nodes", []))}
        summary_json["key_findings"] = ["Bank-ready case file created.", "Action plan DAG persisted."]
        if not ctx.dry_run and ctx.supabase:
            upsert_summary_ctx(ctx, "incident_response", period_start, period_end, summary_text, summary_json)

    run_id = persist_agent_run_ctx(
        ctx, AGENT_NAME, "completed", step_trace, summary_json,
        artifacts_refs={"playbook_id": playbook_id, "incident_packet_id": incident_packet_id},
    )
    return {
        "step_trace": step_trace,
        "summary_json": summary_json,
        "status": "ok",
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "playbook_id": playbook_id,
        "incident_packet_id": incident_packet_id,
    }
