"""Tests for config.graph_policy: no graph mutation when ASR/intent confidence is low."""
import pytest

from config.graph_policy import allow_graph_mutation_for_event


def test_allow_mutation_final_asr_high_confidence() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 0, "event_type": "final_asr", "payload": {"confidence": 0.9}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.5) is True


def test_allow_mutation_final_asr_low_confidence() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 0, "event_type": "final_asr", "payload": {"confidence": 0.3}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.5) is False


def test_allow_mutation_final_asr_missing_confidence_fail_safe() -> None:
    """Missing confidence is treated as low: no graph mutation (fail safe)."""
    ev = {"session_id": "s1", "device_id": "d1", "seq": 0, "event_type": "final_asr", "payload": {}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.5) is False


def test_allow_mutation_intent_high_confidence() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 1, "event_type": "intent", "payload": {"name": "call", "confidence": 0.85}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.5) is True


def test_allow_mutation_intent_low_confidence() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 1, "event_type": "intent", "payload": {"name": "call", "confidence": 0.2}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.5) is False


def test_allow_mutation_wake_no_confidence_gate() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 0, "event_type": "wake", "payload": {}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.9) is True


def test_allow_mutation_zero_threshold_allows_all() -> None:
    ev = {"session_id": "s1", "device_id": "d1", "seq": 0, "event_type": "final_asr", "payload": {}}
    assert allow_graph_mutation_for_event(ev, confidence_min=0.0) is True
