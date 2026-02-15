"""Entity display labels for ring members and other UI."""
from __future__ import annotations

from typing import Any


def get_entity_display_map(
    supabase: Any,
    household_id: str,
    entity_ids: list[str],
) -> dict[str, str]:
    """
    Return mapping entity_id -> display_label for the given entity IDs.
    Label is canonical (e.g. phone number, email) or "entity_type: canonical" when useful.
    """
    if not supabase or not household_id or not entity_ids:
        return {}
    out: dict[str, str] = {}
    try:
        ids_dedup = list(dict.fromkeys(str(x) for x in entity_ids if x))
        if not ids_dedup:
            return {}
        r = (
            supabase.table("entities")
            .select("id, entity_type, canonical, canonical_hash")
            .eq("household_id", household_id)
            .in_("id", ids_dedup)
            .execute()
        )
        for row in (r.data or []):
            eid = str(row.get("id", ""))
            canonical = (row.get("canonical") or "").strip()
            canonical_hash = row.get("canonical_hash")
            etype = row.get("entity_type") or "entity"
            if canonical:
                out[eid] = canonical
            elif canonical_hash:
                out[eid] = f"{etype}: {canonical_hash[:16]}…" if len(str(canonical_hash)) > 16 else f"{etype}: {canonical_hash}"
            else:
                out[eid] = f"{etype} ({eid[:8]}…)"
        for eid in ids_dedup:
            if eid not in out:
                out[eid] = f"entity ({eid[:8]}…)"
    except Exception:
        for eid in entity_ids:
            if eid:
                out[str(eid)] = str(eid)[:8] + "…"
    return out
