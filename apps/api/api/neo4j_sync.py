"""
Optional Neo4j sync: mirror evidence subgraph (entities + relationships) into Neo4j
for visualization and investigative queries. Not used by ML pipeline.
Only runs when NEO4J_URI is set; no-op otherwise.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _driver():
    """Lazy import; returns None if neo4j not installed or URI not set."""
    try:
        from api.config import settings
        if not (getattr(settings, "neo4j_uri", None) or "").strip():
            return None
        from neo4j import GraphDatabase
        uri = settings.neo4j_uri.strip()
        user = getattr(settings, "neo4j_user", "neo4j") or "neo4j"
        password = getattr(settings, "neo4j_password", "") or ""
        return GraphDatabase.driver(uri, auth=(user, password))
    except ImportError:
        logger.debug("neo4j package not installed; Neo4j sync disabled")
        return None
    except Exception as e:
        logger.warning("Neo4j driver init failed: %s", e)
        return None


def sync_evidence_graph_to_neo4j(
    household_id: str,
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> bool:
    """
    Write entities and relationships to Neo4j for the given household.
    Uses MERGE so repeated syncs are idempotent. Returns True if sync ran, False if skipped/failed.
    """
    driver = _driver()
    if not driver:
        return False
    try:
        with driver.session() as session:
            # Clear this household's subgraph so we replace with current state
            session.run(
                "MATCH (e:Entity {household_id: $hh}) DETACH DELETE e",
                hh=household_id,
            )
            # Create Entity nodes
            for ent in entities:
                if ent.get("household_id") != household_id:
                    continue
                session.run(
                    """
                    MERGE (e:Entity {id: $id, household_id: $hh})
                    SET e.entity_type = $entity_type, e.canonical = $canonical, e.canonical_hash = $canonical_hash
                    """,
                    id=ent.get("id", ""),
                    hh=household_id,
                    entity_type=ent.get("entity_type", ""),
                    canonical=ent.get("canonical", ""),
                    canonical_hash=ent.get("canonical_hash", ""),
                )
            # Create Relationship edges (CO_OCCURS, TRIGGERED, etc.)
            for rel in relationships:
                session.run(
                    """
                    MATCH (a:Entity {id: $src, household_id: $hh}), (b:Entity {id: $dst, household_id: $hh})
                    MERGE (a)-[r:RELATION {rel_type: $rel_type}]->(b)
                    SET r.weight = $weight, r.count = $count, r.first_seen_at = $first_seen_at, r.last_seen_at = $last_seen_at
                    """,
                    src=rel.get("src_entity_id", ""),
                    dst=rel.get("dst_entity_id", ""),
                    hh=household_id,
                    rel_type=rel.get("rel_type", "CO_OCCURS"),
                    weight=float(rel.get("weight", 1.0)),
                    count=int(rel.get("count", 0)),
                    first_seen_at=rel.get("first_seen_at"),
                    last_seen_at=rel.get("last_seen_at"),
                )
        logger.info("Neo4j sync completed for household %s: %d entities, %d relationships", household_id, len(entities), len(relationships))
        return True
    except Exception as e:
        logger.exception("Neo4j sync failed for household %s: %s", household_id, e)
        return False
    finally:
        try:
            driver.close()
        except Exception:
            pass


def neo4j_enabled() -> bool:
    """Return True if Neo4j is configured (used for UI to show 'Open in Neo4j Browser')."""
    try:
        from api.config import settings
        return bool((getattr(settings, "neo4j_uri", None) or "").strip())
    except Exception:
        return False
