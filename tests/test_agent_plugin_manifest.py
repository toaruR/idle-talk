"""Static contract tests for the skills-only idle-talk plugin manifest."""

import json
import re
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).parents[1] / "agent-plugins" / "idle-talk"
MANIFEST_PATH = PLUGIN_DIR / "plugin.json"
EXPECTED_MANIFEST = {
    "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
    "name": "idle-talk",
    "version": "1.0.0",
    "description": "Guide connected idle-talk application through safe, user-directed idle conversation.",
}
EXPECTED_ONBOARDING_SKILL = "./skills/idle-talk/SKILL.md"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def validate_identity(manifest: dict) -> None:
    """Accept exactly the published skills-only package identity."""
    for field, expected in EXPECTED_MANIFEST.items():
        if manifest.get(field) != expected:
            raise ValueError(f"invalid plugin {field}")


def onboarding_path(manifest: dict, package_dir: Path) -> Path:
    """Resolve an onboarding skill only when it stays inside its package."""
    try:
        skill = manifest["extensions"]["com.openai"]["onboardingSkill"]
    except (KeyError, TypeError) as error:
        raise ValueError("missing onboardingSkill") from error

    if not isinstance(skill, str):
        raise TypeError("onboardingSkill must be a string")

    package_root = package_dir.resolve()
    resolved = (package_root / skill).resolve()
    try:
        resolved.relative_to(package_root)
    except ValueError as error:
        raise ValueError("onboardingSkill must stay inside plugin package") from error
    return resolved


def test_manifest_has_designed_identity() -> None:
    manifest = load_manifest()

    validate_identity(manifest)
    assert (
        manifest["extensions"]["com.openai"]["onboardingSkill"]
        == EXPECTED_ONBOARDING_SKILL
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", None),
        ("name", "other-plugin"),
        ("version", "2.0.0"),
        ("description", "Changed description"),
        ("$schema", "https://example.invalid/plugin.schema.json"),
    ],
)
def test_manifest_rejects_missing_or_altered_identity(
    field: str, value: str | None
) -> None:
    manifest = load_manifest()
    if value is None:
        manifest.pop(field)
    else:
        manifest[field] = value

    with pytest.raises(ValueError, match=re.escape(f"invalid plugin {field}")):
        validate_identity(manifest)


def test_onboarding_skill_resolves_relative_to_plugin() -> None:
    resolved = onboarding_path(load_manifest(), PLUGIN_DIR)

    assert resolved == (PLUGIN_DIR / "skills" / "idle-talk" / "SKILL.md").resolve()


def test_onboarding_skill_rejects_path_outside_plugin() -> None:
    manifest = load_manifest()
    manifest["extensions"]["com.openai"]["onboardingSkill"] = "../outside/SKILL.md"

    with pytest.raises(ValueError, match="inside plugin package"):
        onboarding_path(manifest, PLUGIN_DIR)


def test_skills_only_package_has_no_root_mcp_manifest() -> None:
    assert not (PLUGIN_DIR / "mcp.json").exists()
