#!/usr/bin/env python3
"""
Tests for per-task model selection and profile identity in delegate_task.

Two opt-in features, both gated behind config flags (default off):

1. **Per-task model selection** (``delegation.allow_model_selection``):
   The agent names a model per task ("opus", "gpt-5", "glm") when fanning
   work out; resolution reuses the shared ``model_switch`` pipeline so names
   are matched leniently and the provider is resolved (not dictated).

2. **Per-task profile identity** (``delegation.allow_profile_identity``):
   The agent names a Hermes profile per task; the child loads that profile's
   SOUL.md, IDENTITY.md, and AGENTS.md as its system prompt identity, and
   reads model/provider from its config.yaml when not explicitly overridden.
   The child "becomes" the named bot rather than a generic subagent.

Both flags are off by default to preserve the "subagents inherit the parent
model and use a generic identity" contract. The schema fields only appear
when the corresponding flag is enabled (keeping the tool surface minimal).

Run with:  python3 -m pytest tests/tools/test_delegate_per_task_overrides.py -v
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_dynamic_schema_overrides,
    _get_allow_model_selection,
    _get_allow_profile_identity,
    _resolve_task_model_creds,
    _load_profile_identity,
    _build_child_system_prompt,
)


class _FakeParent:
    """Minimal parent agent for credential anchoring."""

    provider = "openrouter"
    model = "anthropic/claude-opus-4.8"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test"


_BASE_CREDS = {
    "model": None,
    "provider": None,
    "base_url": None,
    "api_key": None,
    "api_mode": None,
    "command": None,
    "args": None,
}


# ---------------------------------------------------------------------------
# Schema gating: fields only appear when the flag is on
# ---------------------------------------------------------------------------

class TestSchemaGating(unittest.TestCase):
    """Per-task fields only appear when the corresponding flag is enabled."""

    def test_both_flags_off_no_extra_fields(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertNotIn("model", props)
        self.assertNotIn("profile", props)
        task_props = props["tasks"]["items"]["properties"]
        self.assertNotIn("model", task_props)
        self.assertNotIn("profile", task_props)

    def test_model_flag_on_adds_model_field(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertEqual(props["model"]["type"], "string")
        self.assertIn("model", props["tasks"]["items"]["properties"])

    def test_profile_flag_on_adds_profile_field(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("profile", props)
        self.assertEqual(props["profile"]["type"], "string")
        self.assertIn("profile", props["tasks"]["items"]["properties"])

    def test_both_flags_on_adds_both_fields(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_model_selection": True,
                "allow_profile_identity": True,
            },
        ):
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        self.assertIn("model", props)
        self.assertIn("profile", props)
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("model", task_props)
        self.assertIn("profile", task_props)

    def test_static_schema_never_mutated(self):
        """Dynamic overrides must not leak into the static schema."""
        with patch(
            "tools.delegate_tool._load_config",
            return_value={
                "allow_model_selection": True,
                "allow_profile_identity": True,
            },
        ):
            _build_dynamic_schema_overrides()
        static_props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertNotIn("model", static_props)
        self.assertNotIn("profile", static_props)
        self.assertNotIn(
            "model", static_props["tasks"]["items"]["properties"]
        )
        self.assertNotIn(
            "profile", static_props["tasks"]["items"]["properties"]
        )


# ---------------------------------------------------------------------------
# Flag getters
# ---------------------------------------------------------------------------

class TestFlagGetters(unittest.TestCase):
    def test_model_selection_default_off(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_allow_model_selection())

    def test_model_selection_truthy_on(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_model_selection": True},
        ):
            self.assertTrue(_get_allow_model_selection())

    def test_profile_identity_default_off(self):
        with patch("tools.delegate_tool._load_config", return_value={}):
            self.assertFalse(_get_allow_profile_identity())

    def test_profile_identity_truthy_on(self):
        with patch(
            "tools.delegate_tool._load_config",
            return_value={"allow_profile_identity": True},
        ):
            self.assertTrue(_get_allow_profile_identity())


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

class TestModelResolution(unittest.TestCase):
    """`_resolve_task_model_creds` reuses the model_switch pipeline."""

    def test_empty_name_returns_base_unchanged(self):
        out = _resolve_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertIs(out, _BASE_CREDS)

    def test_base_creds_not_mutated(self):
        before = dict(_BASE_CREDS)
        _resolve_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertEqual(_BASE_CREDS, before)


# ---------------------------------------------------------------------------
# Profile identity loading
# ---------------------------------------------------------------------------

class TestProfileIdentityLoading(unittest.TestCase):
    """`_load_profile_identity` reads identity files from a profile directory."""

    def test_nonexistent_profile_returns_none(self):
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the profiles root but not the named profile
            pathlib.Path(tmpdir, "profiles").mkdir(parents=True)
            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("nonexistent-profile")
        self.assertIsNone(result)

    def test_loads_identity_files_and_config(self):
        """A well-formed profile returns soul, identity, agents, model, provider."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = pathlib.Path(tmpdir) / "profiles" / "test-bot"
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text("You are a test bot.")
            (profile_dir / "IDENTITY.md").write_text("Name: TestBot")
            (profile_dir / "AGENTS.md").write_text("# Test Agent Rules")
            (profile_dir / "config.yaml").write_text(
                "model:\n  default: glm-5.3\n  provider: ollama-cloud\n"
            )

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("test-bot")

        self.assertIsNotNone(result)
        self.assertEqual(result["soul"], "You are a test bot.")
        self.assertEqual(result["identity"], "Name: TestBot")
        self.assertEqual(result["agents"], "# Test Agent Rules")
        self.assertEqual(result["model"], "glm-5.3")
        self.assertEqual(result["provider"], "ollama-cloud")

    def test_partial_profile_still_loads(self):
        """A profile with only SOUL.md (no IDENTITY) still returns a valid dict."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_dir = pathlib.Path(tmpdir) / "profiles" / "partial-bot"
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text("You are partial.")

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                result = _load_profile_identity("partial-bot")

        self.assertIsNotNone(result)
        self.assertEqual(result["soul"], "You are partial.")
        self.assertIsNone(result["identity"])
        self.assertIsNone(result["agents"])
        self.assertIsNone(result["model"])
        self.assertIsNone(result["provider"])

    def test_path_traversal_rejected(self):
        """Profile names with slashes or dots are rejected to prevent traversal."""
        import pathlib

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a directory outside profiles that a traversal would hit
            secret_dir = pathlib.Path(tmpdir) / "secret"
            secret_dir.mkdir()
            (secret_dir / "SOUL.md").write_text("SECRET DATA")

            with patch(
                "hermes_constants.get_default_hermes_root",
                return_value=pathlib.Path(tmpdir),
            ):
                # All of these should return None, never reading the secret
                for bad_name in [
                    "../../secret",
                    "../secret",
                    "foo/../../secret",
                    ".../.../secret",
                    "foo/../bar",
                    "a/b",
                    "a.b",
                    ".hidden",
                    "trailing/",
                    "/absolute",
                ]:
                    with self.subTest(name=bad_name):
                        result = _load_profile_identity(bad_name)
                        self.assertIsNone(
                            result,
                            f"Path traversal with '{bad_name}' should return None",
                        )


# ---------------------------------------------------------------------------
# System prompt with profile identity
# ---------------------------------------------------------------------------

class TestChildSystemPromptWithProfile(unittest.TestCase):
    """`_build_child_system_prompt` uses profile identity when provided."""

    def test_profile_identity_replaces_generic_preamble(self):
        identity = {
            "soul": "You are a code reviewer.",
            "identity": "Name: ReviewerBot",
            "agents": "# Review Rules",
            "model": None,
            "provider": None,
        }
        prompt = _build_child_system_prompt(
            "Review the PR", profile_identity=identity
        )
        self.assertIn("You are a code reviewer.", prompt)
        self.assertIn("Name: ReviewerBot", prompt)
        self.assertIn("# Review Rules", prompt)
        self.assertIn("YOUR TASK:\nReview the PR", prompt)
        self.assertNotIn("You are a focused subagent", prompt)

    def test_no_profile_identity_uses_generic_preamble(self):
        prompt = _build_child_system_prompt("Fix the tests")
        self.assertIn("You are a focused subagent", prompt)
        self.assertIn("YOUR TASK:\nFix the tests", prompt)

    def test_partial_profile_warns_and_uses_agents(self):
        """Profile with only AGENTS.md (no SOUL/IDENTITY) falls back gracefully."""
        identity = {
            "soul": None,
            "identity": None,
            "agents": "# Agent Rules Only",
            "model": None,
            "provider": None,
        }
        with patch("tools.delegate_tool.logger") as mock_logger:
            prompt = _build_child_system_prompt(
                "Do the work", profile_identity=identity
            )
        mock_logger.warning.assert_called_once()
        self.assertIn("You are a focused subagent", prompt)
        self.assertIn("# Agent Rules Only", prompt)


if __name__ == "__main__":
    unittest.main()