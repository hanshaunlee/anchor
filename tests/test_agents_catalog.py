"""Tests for agents catalog (visibility by role and consent)."""
from __future__ import annotations


def test_agents_catalog_visibility_by_role_and_consent() -> None:
    """Catalog filters by role; runnable and visibility depend on required_roles and consent."""
    from domain.agents.registry import get_agents_catalog

    catalog_elder = get_agents_catalog(role="elder", consent={})
    catalog_caregiver = get_agents_catalog(role="caregiver", consent={"share_with_caregiver": True})
    catalog_admin = get_agents_catalog(role="admin", consent={})

    slugs = [c.get("slug") for c in catalog_elder]
    assert "financial" in slugs
    assert "outreach" in slugs
    # Elder cannot run financial (required_roles caregiver, admin)
    financial_elder = next((c for c in catalog_elder if c.get("slug") == "financial"), None)
    assert financial_elder is not None
    assert financial_elder.get("runnable") is False

    financial_caregiver = next((c for c in catalog_caregiver if c.get("slug") == "financial"), None)
    assert financial_caregiver is not None
    assert financial_caregiver.get("runnable") is True

    # Redteam in prod is not runnable
    redteam = next((c for c in catalog_admin if c.get("slug") == "redteam"), None)
    assert redteam is not None
    assert redteam.get("runnable") is False
    assert redteam.get("reason") == "redteam_disabled_in_production"
