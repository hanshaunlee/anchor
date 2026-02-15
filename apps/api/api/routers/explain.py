"""Explain opaque identifiers (IDs, entity refs) in plain language. Uses Claude when configured."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import require_user
from domain.explain_service import explain_opaque_items

router = APIRouter(prefix="/explain", tags=["explain"])


class ExplainItem(BaseModel):
    id: str = Field(..., description="Opaque identifier shown to the user")
    hint: str | None = Field(None, description="Optional type or label for context")


class ExplainRequest(BaseModel):
    context: str = Field(
        ...,
        description="One of: pattern_members, alert_ids, top_connectors, entity_list",
    )
    items: list[ExplainItem] = Field(..., max_length=20)


class ExplainEntry(BaseModel):
    original: str
    explanation: str


class ExplainResponse(BaseModel):
    explanations: list[ExplainEntry]


@router.post("", response_model=ExplainResponse)
def post_explain(
    body: ExplainRequest,
    user_id: str = Depends(require_user),
) -> ExplainResponse:
    """
    Get short plain-English explanations for opaque IDs (entity IDs, alert IDs, etc.).
    Uses Claude when ANTHROPIC_API_KEY is set; otherwise returns fallback text.
    """
    items_dict = [{"id": it.id, "hint": it.hint} for it in body.items]
    results = explain_opaque_items(body.context, items_dict)
    return ExplainResponse(
        explanations=[ExplainEntry(original=r["original"], explanation=r["explanation"]) for r in results]
    )
