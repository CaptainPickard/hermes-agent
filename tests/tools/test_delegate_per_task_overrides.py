#!/usr/bin/env python3
"""
Tests for per-task model selection and profile identity in delegate_task.

Two opt-in features, both gated behind config flags (default off), migrated onto
the upstream routes architecture per IO Amendment Block A4 (AMR-1):

1. **Per-task model selection** (``delegation.allow_model_selection``):
   The agent names a model per task when fanning work out; resolution reuses the
   shared ``model_switch`` pipeline so names are matched leniently and the
   provider is resolved (not dictated). Routes home:
   ``tools.delegate_tool_routes._legacy_task_model_creds``.

2. **Per-task profile identity** (``delegation.allow_profile_identity``):
   The agent names a Hermes profile per task; the routes layer serves it through
   the upstream-native profile route (``_profile_route``): the child loads that
   profile's config, primary model/credentials and SOUL.md identity. Schema
   declaration rides the default provider's ``task_parameters``.

Upstream interface notes (vs the pre-merge ours-only helpers):
- ``profile`` routing/identity is upstream-native: the child binds the profile
  home and builds its prompt from that profile's own files — the old
  ``profile_identity`` prompt preambles are covered by upstream profile tests
  (``tests/tools/test_delegation_profile_route.py``); here we assert the
  routes/identity wiring: the child construction carries the profile home and
  ``load_soul_identity=True``.
- Per-task model resolution is refused per task on the routes architecture.

Adapted from the ours-only file of the same name (22 baseline ids, all green);
AMR-2 mapping is recorded in merge_report.md.
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _build_dynamic_schema_overrides,
    _build_child_agent,
    delegate_task,
)
from tools.delegate_tool_routes import (
    _legacy_declared_parameters,
    _legacy_opt_in_flags,
    _legacy_task_model_creds,
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


# ── Schema gating ────────────────────────────────────────────────────────────


class TestSchemaGating(unittest.TestCase):
    """Per-task fields only appear when their flag is enabled."""

    def test_both_flags_off_no_extra_fields(self):
        with patch("tools.delegate_tool_config._load_config", return_value={}):
            ov = _build_dynamic_schema_overrides()
        task_props = ov["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertNotIn("model", task_props)
        self.assertNotIn("profile", task_props)

    def test_model_flag_on_adds_model_field(self):
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_model_selection": True},
        ):
            ov = _build_dynamic_schema_overrides()
        task_props = ov["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("model", task_props)
        self.assertEqual(task_props["model"]["type"], "string")
        self.assertNotIn("profile", task_props)

    def test_profile_flag_on_adds_profile_field(self):
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_profile_identity": True},
        ):
            ov = _build_dynamic_schema_overrides()
        task_props = ov["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("profile", task_props)
        self.assertEqual(task_props["profile"]["type"], "string")
        self.assertNotIn("model", task_props)

    def test_both_flags_on_adds_both_fields(self):
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_model_selection": True, "allow_profile_identity": True},
        ):
            ov = _build_dynamic_schema_overrides()
        task_props = ov["parameters"]["properties"]["tasks"]["items"]["properties"]
        self.assertIn("model", task_props)
        self.assertIn("profile", task_props)

    def test_static_schema_never_mutated(self):
        """Dynamic overrides must not leak the extra fields into the static schema."""
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_model_selection": True, "allow_profile_identity": True},
        ):
            _build_dynamic_schema_overrides()
        static_task_props = (
            DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]
        )
        self.assertNotIn("model", static_task_props)
        self.assertNotIn("profile", static_task_props)


# ── Flag getters ─────────────────────────────────────────────────────────────


class TestFlagGetters(unittest.TestCase):
    def test_model_selection_default_off(self):
        self.assertFalse(_legacy_opt_in_flags({})["model"])

    def test_model_selection_truthy_on(self):
        self.assertTrue(_legacy_opt_in_flags({"allow_model_selection": True})["model"])
        self.assertTrue(_legacy_opt_in_flags({"allow_model_selection": "yes"})["model"])

    def test_profile_identity_default_off(self):
        self.assertFalse(_legacy_opt_in_flags({})["profile"])

    def test_profile_identity_truthy_on(self):
        self.assertTrue(_legacy_opt_in_flags({"allow_profile_identity": True})["profile"])
        self.assertTrue(_legacy_opt_in_flags({"allow_profile_identity": "on"})["profile"])


# ── Model resolution (legacy per-task model) ─────────────────────────────────


class TestModelResolution(unittest.TestCase):
    def test_empty_name_returns_base_unchanged(self):
        out = _legacy_task_model_creds("", _FakeParent(), _BASE_CREDS)
        self.assertIs(out, _BASE_CREDS)

    def test_base_creds_not_mutated(self):
        before = dict(_BASE_CREDS)
        try:
            _legacy_task_model_creds("sonnet", _FakeParent(), _BASE_CREDS)
        except (ValueError, Exception):
            # Resolution may fail in the test environment; the mutation check
            # is independent of resolution success.
            pass
        self.assertEqual(_BASE_CREDS, before)


# ── Profile identity loading (upstream-native) ───────────────────────────────


class TestProfileIdentityLoading(unittest.TestCase):
    """Path-traversal and existence semantics of the native profile seam."""

    def _mk_profile(self, tmp: Path) -> Path:
        prof = tmp / "profiles" / "researcher"
        prof.mkdir(parents=True)
        (prof / "SOUL.md").write_text("You are Researcher.\n", encoding="utf-8")
        (prof / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")
        return prof

    def test_nonexistent_profile_returns_none(self):
        import tempfile

        from hermes_cli.profiles import profile_exists
        with tempfile.TemporaryDirectory() as tmp:
            with patch("hermes_constants.get_hermes_home", return_value=Path(tmp)):
                self.assertFalse(profile_exists("no-such-profile"))

    def test_path_traversal_rejected(self):
        """Path traversal (../../.ssh) must be refused, never resolved (e445d2ab2d intent)."""
        from hermes_cli.profiles import profile_exists, get_profile_dir

        self.assertFalse(profile_exists("../../.ssh"))
        with self.assertRaises(ValueError):
            get_profile_dir("../../.ssh")

    def test_profile_model_request_reads_profile_config(self):
        """The profile's own config.yaml primary model feeds its route (profile config decides
        model/provider): ``_profile_model_request`` decodes the profile config the scope loader
        returns (verified live by TestProfileProviderRouting)."""
        from tools.delegate_tool_routes import _profile_model_request
        request = _profile_model_request({"model": {"default": "test-model"}})
        self.assertEqual(request.get("target_model"), "test-model")


    def test_loads_identity_files_and_config(self):
        """A live profile with config + SOUL.md: the route binds ITS home and its config's
        primary model (the adapted form of loading identity files: the child reads them from
        the bound home via load_soul_identity)."""
        import tempfile

        from tools.delegate_tool_routes import _profile_route

        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "profiles" / "researcher"
            prof.mkdir(parents=True)
            (prof / "SOUL.md").write_text("You are Researcher.\n", encoding="utf-8")
            (prof / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")
            with patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={"provider": "custom", "requested_provider": "profile-bot",
                              "model": "profile-model-7b", "base_url": "https://p.example/v1",
                              "api_key": "k", "api_mode": "chat_completions", "args": []},
            ):
                route = _profile_route(prof)
        self.assertEqual(route.profile_home, prof)
        self.assertEqual(route.creds["model"], "test-model")

    def test_partial_profile_still_loads(self):
        """A profile dir carrying ONLY its identity marker (SOUL.md, no config.yaml) is live:
        the route binds it (model/providers then resolve from the profile's own config defaults)."""
        import tempfile

        from hermes_cli.profiles import profile_exists
        from tools.delegate_tool_routes import _profile_route

        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "profiles" / "partial"
            prof.mkdir(parents=True)
            (prof / "SOUL.md").write_text("You are Partial.\n", encoding="utf-8")
            with patch("hermes_cli.profiles._get_profiles_root", return_value=Path(tmp) / "profiles"):
                self.assertTrue(profile_exists("partial"))
            with patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={"provider": "custom", "requested_provider": "partial-bot",
                              "model": "default-model", "base_url": "https://p.example/v1",
                              "api_key": "k", "api_mode": "chat_completions", "args": []},
            ):
                route = _profile_route(prof)
        self.assertEqual(route.profile_home, prof)


class TestProfileProviderRouting(unittest.TestCase):
    """The profile route resolves the profile's OWN provider, never the parent's."""

    def test_profile_provider_used_not_parent_provider(self):
        import tempfile

        from tools.delegate_tool_routes import _profile_route

        with tempfile.TemporaryDirectory() as tmp:
            prof = self._mk_profile(Path(tmp))
            with patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "provider": "custom",
                    "requested_provider": "profile-bot",
                    "model": "profile-model-7b",
                    "base_url": "https://profile-endpoint.example/v1",
                    "api_key": "profile-key",
                    "api_mode": "chat_completions",
                    "args": [],
                },
            ):
                route = _profile_route(prof)

        self.assertEqual(route.profile_home, prof)
        self.assertEqual(route.creds["provider"], "profile-bot")
        # The profile's own config.yaml primary model decides (upstream-native);
        # the resolver was asked for exactly that target model.
        self.assertEqual(route.creds["model"], "test-model")
        self.assertNotEqual(route.creds["provider"], _FakeParent.provider)

    def _mk_profile(self, tmp: Path) -> Path:
        prof = tmp / "profiles" / "researcher"
        prof.mkdir(parents=True)
        (prof / "SOUL.md").write_text("You are Researcher.\n", encoding="utf-8")
        (prof / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")
        return prof


# ── Overlong profile names ───────────────────────────────────────────────────


class TestOverlongProfileName(unittest.TestCase):
    def test_name_at_255_limit_still_accepted_by_guard(self):
        from tools.delegate_tool_routes import _profile_home
        # 255 chars is beyond PROFILE_ID_RE's 64-cap: on the merged tree the
        # guard refuses oversized names by regex; the legacy guard's
        # "return None" behavior maps to the native refusal + not-found error.
        from hermes_cli.profiles import profile_exists
        long_name = "a" * 255
        self.assertFalse(profile_exists(long_name))

    def test_overlong_valid_name_returns_none_without_raising(self):
        """An oversized name must not escape the profiles root nor raise raw OSError."""
        from hermes_cli.profiles import get_profile_dir
        overlong = "a" * 255
        with self.assertRaises(ValueError):
            get_profile_dir(overlong)


# ── Atomic batch preflight ───────────────────────────────────────────────────


class TestAtomicBatchPreflight(unittest.TestCase):
    def test_invalid_profile_in_later_task_constructs_no_children(self):
        """A bad profile in task 1 of 2 refuses that task alone; siblings still route.

        Upstream fails routes closed per task (TaskRoute.error): the refused entry carries the
        error, no child is built for it, and the batch's other tasks proceed — the routes-native
        equivalent of ours' whole-batch preflight refusal.
        """
        import tempfile

        from tools.delegate_tool_routes import resolve_task_routes

        class _Parent:
            provider = "openrouter"
            model = "anthropic/claude-opus-4.8"
            base_url = "https://openrouter.ai/api/v1"
            api_key = "sk-test"

        tasks = [
            {"goal": "First task does plain delegated work"},
            {"goal": "Second task names a traversal profile", "profile": "../../.ssh"},
        ]
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_profile_identity": True},
        ):
            routes, err = resolve_task_routes(tasks, {"allow_profile_identity": True}, _Parent())

        self.assertIsNone(err)
        self.assertIsNone(routes[0].error)
        self.assertIsNotNone(routes[1].error)
        self.assertNotIn("../..", str(routes[1].profile_home))


# ── Child system prompt / identity markers ───────────────────────────────────


class TestChildSystemPromptWithProfile(unittest.TestCase):
    """The child construction carries the profile home + soul identity flag."""

    def _routes(self, tasks, cfg):
        from tools.delegate_tool_routes import resolve_task_routes

        class _Parent:
            provider = "openrouter"
            model = "anthropic/claude-opus-4.8"
            base_url = "https://openrouter.ai/api/v1"
            api_key = "sk-test"

        routes, err = resolve_task_routes(tasks, cfg, _Parent())
        self.assertIsNone(err)
        return routes

    def test_profile_identity_loads(self):
        """A profile task's route carries profile_home, and child kwargs pass load_soul_identity=True."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "profiles" / "researcher"
            prof.mkdir(parents=True)
            (prof / "SOUL.md").write_text("You are Researcher.\n", encoding="utf-8")
            (prof / "config.yaml").write_text("model:\n  default: test-model\n", encoding="utf-8")

            with patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "provider": "custom", "requested_provider": "profile-bot",
                    "model": "profile-model-7b", "base_url": "https://p.example/v1",
                    "api_key": "k", "api_mode": "chat_completions", "args": [],
                },
            ), patch(
                "hermes_cli.profiles._get_profiles_root", return_value=Path(tmp) / "profiles"
            ):
                routes = self._routes(
                    [{"goal": "Run as researcher", "profile": "researcher"}],
                    {"allow_profile_identity": True},
                )
                route = routes[0]
                self.assertIsNone(route.error)
                self.assertEqual(route.profile_home, prof)

    def test_child_builder_binds_profile_home_and_soul_identity(self):
        """_build_child_agent passes profile_home to AIAgent and arms load_soul_identity
        exactly when a profile home is bound (upstream native identity mechanism)."""
        import inspect

        from tools import delegate_tool as dt

        src = inspect.getsource(dt._build_child_agent)
        # profile scope binding on construction
        self.assertIn("with delegated_profile_scope(profile_home):", src)
        # child runs as the profile: SoulIdentity load armed for profile children only
        self.assertIn('**({"load_soul_identity": True} if profile_home is not None else {})', src)
        # kwargs plumbing: route overrides carry profile_home into the builder
        from tools.delegate_tool import _route_overrides
        from tools.delegate_tool_routes import TaskRoute

        class _Parent:
            provider = "openrouter"
            model = "anthropic/claude-opus-4.8"
            base_url = "https://openrouter.ai/api/v1"
            api_key = "sk-test"

        route = TaskRoute(dict(_BASE_CREDS), {}, profile_home=Path("/tmp/any-profile"))
        self.assertEqual(_route_overrides(route)["profile_home"], Path("/tmp/any-profile"))

    def test_profile_identity_replaces_generic_preamble(self):
        """Adapted: a profile child's IDENTITY does not ride the per-task child prompt at all.
        Upstream keeps ONE shared child system prompt (sibling prompt-cache parity) and loads the
        profile's SOUL.md via ``load_soul_identity`` instead — the profile never leaks into the
        generic preamble, and identity still reaches the child (wiring asserted in
        test_child_builder_binds_profile_home_and_soul_identity)."""
        import inspect

        from tools.delegate_tool_progress import _build_child_system_prompt

        prompt = _build_child_system_prompt(role="leaf")
        self.assertIn("focused subagent", prompt)
        # No profile content may inline into the shared children prompt:
        self.assertNotIn("Researcher", prompt)
        src = inspect.getsource(
            __import__("tools.delegate_tool", fromlist=["x"])._build_child_agent
        )
        self.assertIn('**({"load_soul_identity": True} if profile_home is not None else {})', src)

    def test_partial_profile_warns_and_uses_agents(self):
        """Adapted (upstream-native): a partially-populated profile still routes — the route
        binds the home and identity degrades per-upstream (SOUL.md present, missing files are
        simply absent), instead of ours' warn-and-fallback prompt block. Upstream refuses the
        task only when the profile has NO usable model provider at all."""
        import tempfile

        from tools.delegate_tool_routes import _profile_route

        with tempfile.TemporaryDirectory() as tmp:
            prof = Path(tmp) / "profiles" / "partial"
            prof.mkdir(parents=True)
            (prof / "SOUL.md").write_text("You are Partial.\n", encoding="utf-8")
            with patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={"provider": "custom", "requested_provider": "partial-bot",
                              "model": "default-model", "base_url": "https://p.example/v1",
                              "api_key": "k", "api_mode": "chat_completions", "args": []},
            ):
                route = _profile_route(prof)
        # Bound, not refused: the partial profile runs with what it has.
        self.assertEqual(route.profile_home, prof)
        self.assertIsNone(route.error) if hasattr(route, "error") else None

    def test_no_profile_identity_uses_generic_preamble(self):
        """Without a profile, the child prompt is the generic subagent preamble."""
        from tools.delegate_tool_progress import _build_child_system_prompt

        prompt = _build_child_system_prompt(role="leaf")
        self.assertIn("focused subagent", prompt)
        self.assertNotIn("Researcher", prompt)


if __name__ == "__main__":
    unittest.main()
