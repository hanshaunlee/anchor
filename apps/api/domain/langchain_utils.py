"""
LangChain helpers: structured outputs (Pydantic), evidence-only guardrails.
Use only for steps that benefit from LLM reasoning; do not force LLM for pure math.
"""
from __future__ import annotations

import logging
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_llm():
    """Return ChatOpenAI instance if OPENAI_API_KEY is set; else None."""
    try:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    except Exception as e:
        logger.debug("get_llm failed: %s", e)
        return None


def run_structured_prompt(llm: Any, prompt: str, schema: type) -> Any | None:
    """
    Run prompt with structured output bound to Pydantic schema.
    Returns schema instance or None on failure.
    """
    if llm is None:
        return None
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=schema)
        template = ChatPromptTemplate.from_messages([
            ("system", "You respond only with valid JSON that matches the required schema. No extra text."),
            ("human", "{prompt}\n\n{format_instructions}"),
        ])
        chain = template | llm | parser
        result = chain.invoke({"prompt": prompt, "format_instructions": parser.get_format_instructions()})
        return result
    except Exception as e:
        logger.warning("run_structured_prompt failed: %s", e)
        return None


def evidence_only_guard(
    output: Any,
    allowed_entity_ids: set[str] | None = None,
    allowed_event_ids: set[str] | None = None,
) -> tuple[bool, str | None]:
    """
    Validate that output does not reference entity/event ids outside the allowed sets.
    Returns (valid, error_message). If valid, error_message is None.
    """
    allowed_entity_ids = allowed_entity_ids or set()
    allowed_event_ids = allowed_event_ids or set()
    if output is None:
        return True, None

    def collect_ids(obj: Any, into: set[str], key: str) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == key and isinstance(v, str):
                    into.add(v)
                elif k in ("entity_id", "event_id", "id") and isinstance(v, str):
                    into.add(v)
                else:
                    collect_ids(v, into, key)
        elif isinstance(obj, list):
            for x in obj:
                collect_ids(x, into, key)

    ref_entities: set[str] = set()
    ref_events: set[str] = set()
    collect_ids(output, ref_entities, "entity_id")
    collect_ids(output, ref_events, "event_id")
    if allowed_entity_ids and ref_entities and not ref_entities.issubset(allowed_entity_ids):
        return False, f"Output references entities not in evidence: {ref_entities - allowed_entity_ids}"
    if allowed_event_ids and ref_events and not ref_events.issubset(allowed_event_ids):
        return False, f"Output references events not in evidence: {ref_events - allowed_event_ids}"
    return True, None
