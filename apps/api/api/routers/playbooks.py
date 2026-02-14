"""Playbooks & tasks API: GET playbook, POST task complete, GET risk_signal playbook shortcut."""
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from supabase import Client

from api.deps import get_supabase, require_user
from api.schemas import ActionTaskDetail, PlaybookDetail, PlaybookTaskCompleteRequest
from domain.ingest_service import get_household_id, get_user_role

router = APIRouter(prefix="/playbooks", tags=["playbooks"])


def _redact_task_details_for_elder(details: dict, task_type: str) -> dict:
    """Elder view: no phone/email; keep script and checklist labels only."""
    out = dict(details)
    out.pop("phone", None)
    out.pop("email", None)
    out.pop("recipient_contact", None)
    # Keep call_script, email_template, key_facts, verification_checklist but could redact sensitive
    return out


def _get_playbook_with_tasks(
    supabase: Client,
    playbook_id: str,
    household_id: str,
    user_role: str,
) -> dict | None:
    pb = (
        supabase.table("action_playbooks")
        .select("*")
        .eq("id", playbook_id)
        .eq("household_id", household_id)
        .single()
        .execute()
    )
    if not pb.data:
        return None
    tasks_r = (
        supabase.table("action_tasks")
        .select("*")
        .eq("playbook_id", playbook_id)
        .order("created_at")
        .execute()
    )
    tasks = []
    for t in tasks_r.data or []:
        details = t.get("details") or {}
        if user_role == "elder":
            details = _redact_task_details_for_elder(details, t.get("task_type", ""))
        tasks.append(ActionTaskDetail(
            id=UUID(t["id"]),
            playbook_id=UUID(t["playbook_id"]),
            task_type=t["task_type"],
            status=t["status"],
            details=details,
            completed_by_user_id=UUID(t["completed_by_user_id"]) if t.get("completed_by_user_id") else None,
            completed_at=datetime.fromisoformat(t["completed_at"].replace("Z", "+00:00")) if t.get("completed_at") else None,
            created_at=datetime.fromisoformat(t["created_at"].replace("Z", "+00:00")),
        ))
    row = pb.data
    return {
        **row,
        "tasks": tasks,
    }


@router.get("/{playbook_id}", response_model=PlaybookDetail)
def get_playbook(
    playbook_id: UUID,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Get playbook by id with tasks. Elder sees simplified task details (no contacts)."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    role = get_user_role(supabase, user_id) or "elder"
    data = _get_playbook_with_tasks(supabase, str(playbook_id), hh_id, role)
    if not data:
        raise HTTPException(status_code=404, detail="Playbook not found")
    return PlaybookDetail(
        id=UUID(data["id"]),
        household_id=UUID(data["household_id"]),
        risk_signal_id=UUID(data["risk_signal_id"]),
        playbook_type=data["playbook_type"],
        graph=data.get("graph") or {},
        status=data["status"],
        created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
        tasks=data["tasks"],
    )


@router.post("/{playbook_id}/tasks/{task_id}/complete")
def complete_task(
    playbook_id: UUID,
    task_id: UUID,
    body: PlaybookTaskCompleteRequest | None = None,
    user_id: str = Depends(require_user),
    supabase: Client = Depends(get_supabase),
):
    """Mark task as done. Caregiver/elder can mark; completed_by_user_id set to current user."""
    hh_id = get_household_id(supabase, user_id)
    if not hh_id:
        raise HTTPException(status_code=404, detail="Not onboarded")
    # Verify playbook belongs to household and task belongs to playbook
    pb = (
        supabase.table("action_playbooks")
        .select("id")
        .eq("id", str(playbook_id))
        .eq("household_id", hh_id)
        .single()
        .execute()
    )
    if not pb.data:
        raise HTTPException(status_code=404, detail="Playbook not found")
    now = datetime.utcnow().isoformat() + "Z"
    up = (
        supabase.table("action_tasks")
        .update({
            "status": "done",
            "completed_by_user_id": user_id,
            "completed_at": now,
        })
        .eq("id", str(task_id))
        .eq("playbook_id", str(playbook_id))
        .execute()
    )
    if not up.data:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "task_id": str(task_id), "status": "done"}
