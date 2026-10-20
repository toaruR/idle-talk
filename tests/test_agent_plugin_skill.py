"""Static contract tests for the skills-only idle-talk onboarding skill."""

import re
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).parents[1] / "agent-plugins" / "idle-talk"
SKILL_PATH = PLUGIN_DIR / "skills" / "idle-talk" / "SKILL.md"
CANONICAL_TIMEOUT_SECONDS = 30
TERMINAL_CODES = {
    "IDLE_TALK_UNAVAILABLE",
    "IDLE_TALK_PROTOCOL_MISMATCH",
    "IDLE_TALK_INVALID_MODE",
    "IDLE_TALK_BUSY",
    "IDLE_TALK_CANCELLED",
    "IDLE_TALK_TIMEOUT",
    "IDLE_TALK_NOT_FOUND",
    "IDLE_TALK_CANCEL_TOO_LATE",
}


def skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def documented_codes(text: str) -> set[str]:
    return set(re.findall(r"`(IDLE_TALK_[A-Z_]+)`", text))


def test_skill_exists_at_manifest_onboarding_path() -> None:
    assert SKILL_PATH.is_file()


def test_skill_keeps_lazy_user_directed_workflow() -> None:
    text = skill_text()
    assert "zero connection probes, mode-resource\nloads, and application requests" in text
    assert "only after this workflow is selected" in text
    assert "host-configured current idle-talk connection" in text
    assert "Never start a process" in text
    assert "another local service" in text
    assert "Ask for a mode when the user has not stated an intent" in text
    assert "conversation intent" in text
    assert "`radio` always requires an explicit\nuser choice" in text


def test_skill_documents_exact_provider_request_and_success_shapes() -> None:
    text = skill_text()
    for required in (
        "`idle_talk.status` with `{}`",
        '`{ "state": "ready" | "busy", "modes": string[], "connectionId": string }`',
        "`idle_talk.next` with",
        '`{ "mode": "radio" | "conversation", "requestId": string }`',
        '`{ "requestId": string, "text": string, "mode": string, "cancelled": false }`',
        "`idle_talk.cancel` only",
        '`{ "requestId": string, "cancelled": true }`',
    ):
        assert required in text


def test_skill_enforces_request_id_and_terminal_error_contract() -> None:
    text = skill_text()
    assert "missing or\nmismatched request ID is `IDLE_TALK_PROTOCOL_MISMATCH`" in text
    assert "preserve the supplied `requestId`" in text
    assert documented_codes(text) == TERMINAL_CODES
    assert "only the error\ncode and a concise recovery action" in text


def test_skill_enforces_canonical_timeout_and_cancellation_race() -> None:
    text = skill_text()
    assert re.findall(r"\b30 seconds\b", text) == ["30 seconds", "30 seconds"]
    assert "below 30 seconds" in text
    assert "At or after 30 seconds, make one\ncancellation attempt" in text
    assert "at most one cancellation request" in text
    assert "before cancellation is one terminal success and wins the race" in text
    assert "only if no matching terminal result\narrives after that cancellation attempt" in text


@pytest.mark.parametrize(
    "forbidden",
    [
        "filesystem authority",
        "shell authority",
        "arbitrary URL authority",
        "credential-reading authority",
        "process-management authority",
    ],
)
def test_skill_denies_forbidden_authority(forbidden: str) -> None:
    text = skill_text()
    assert f"no {forbidden}" in text


def test_skill_prohibits_unplanned_retries_queues_and_fallbacks() -> None:
    text = skill_text()
    for required in (
        "Allow at most one\nin-flight request",
        "keep no hidden queue",
        "Do not automatically retry",
        "parse fallback shapes",
        "make fallback calls",
    ):
        assert required in text
