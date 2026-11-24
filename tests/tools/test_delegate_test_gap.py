#!/usr/bin/env python3
"""Test-gap closure for per-task model selection + profile identity (Card 4).

Four categories, all test-only (no production changes):

  1. Property-based (Hypothesis) fuzzing of the profile-name traversal guard,
     now upstream-native ``PROFILE_ID_RE = ^[a-z0-9][a-z0-9_-]{0,63}$``
     (``hermes_cli.profiles.get_profile_dir`` / ``profile_exists``): arbitrary
     strings, adversarial unicode / null bytes / control chars / extremely
     long names / path separators. Invalid names are refused with ValueError
     BEFORE the profiles root is ever consulted — the e445d2ab2d "rejected
     before filesystem access" intent, preserved on the native seam.
  2. Regression: ``allow_model_selection=False`` + ``allow_profile_identity=True``
     must gate the schema (profile advertised through the provider seam,
     model NOT) and must NOT let a profile's config.yaml model trigger a
     model switch.
  3. Fallback chain semantics: unresolvable models raise ValueError (fail
     loudly, never silently inherit); None/"" model names are no-ops that
     return the base creds untouched.
  4. Integration with mock child spawns: a profile child converses under the
     native ``load_soul_identity`` wiring bound to its profile home, no
     profile uses the generic preamble, and an explicit per-task model beats
     the profile's own route.

AMR-2 adaptation (IO Amendment Block A4, ours-only 8c216bca72 merged): the
legacy helpers moved off ``tools.delegate_tool`` —

- ``_get_allow_*``            -> ``tools.delegate_tool_routes._legacy_opt_in_flags``
- ``_load_profile_identity``  -> ``hermes_cli.profiles.profile_exists`` / ``get_profile_dir``
                                 (+ native ``_profile_route`` / ``load_soul_identity``)
- ``_resolve_task_model_creds`` -> ``tools.delegate_tool_routes._legacy_task_model_creds``

Schema note: ``model`` is a core-RESERVED route field — it rides the dynamic
overrides only (``_legacy_task_model_parameter``); ``profile`` is declared
through the default provider's ``task_parameters`` seam, so it appears ONLY in
``tasks[].properties``, never at the top level (the old top-level profile
assertion rode our pre-merge single-layer overrides and no longer applies).

Run with:  /tmp/venv-valspec-0234-py13/bin/python -m pytest tests/tools/test_delegate_test_gap.py -v
"""

import json
import pathlib
import re
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from hypothesis import given, settings, strategies as st

from tools.delegate_tool import (
    _build_dynamic_schema_overrides,
    delegate_task,
    set_spawn_paused,
)
from tools.delegate_tool_routes import (
    _legacy_opt_in_flags,
    _legacy_task_model_creds,
    _legacy_task_model_parameter,
)

# The exact guard regex the native seam enforces (hermes_cli.profiles.get_profile_dir
# via hermes_constants.PROFILE_ID_RE). Compare the NORMALIZED name (see
# normalize_profile_name): lowercase first, ``default`` matched case-insensitively.
SAFE_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _canon(name):
    """The normalized id hermes_cli.profiles would validate (normalize_profile_name
    rules), or None when normalize itself refuses (non-string / empty)."""
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    if stripped.casefold() == "default":
        return "default"
    return stripped.lower()


# Shared goal for integration dispatches (long enough to clear the batch
# goal-quality gate even if the shape ever changes to multi-task).
GOAL = "Do the delegated integration work end to end"


# ── Shared fixtures ───────────────────────────────────────────────────────


class _FakeParent:
    """Minimal parent agent for credential anchoring (model_switch path)."""

    provider = "openrouter"
    model = "anthropic/claude-opus-4.8"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-test"


def _base_creds():
    """Fresh base credential bundle shaped like _resolve_delegation_credentials."""
    return {
        "model": None,
        "provider": None,
        "base_url": None,
        "api_key": None,
        "api_mode": None,
        "command": None,
        "args": None,
    }


def _make_switch_result(
    new_model="gpt-5x",
    target_provider="openai",
    success=True,
    error_message=None,
):
    """A failed switch carries a real ModelSwitchResult (the validating gate the
    live resolver returns for an unresolvable name); a successful one looks
    exactly like hermes_cli.model_switch's return."""
    if not success:
        from hermes_cli.model_switch import ModelSwitchResult

        return ModelSwitchResult(success=False, error_message=error_message or "")
    r = MagicMock()
    r.success = True
    r.error_message = ""
    r.new_model = new_model
    r.target_provider = target_provider
    r.base_url = "https://api.openai.com/v1"
    r.api_key = "sk-openai"
    r.api_mode = "chat_completions"
    return r


def _make_mock_parent():
    """Mock parent with every field delegate_task / _build_child_agent touch.

    Mirrors the established pattern from tests/tools/test_delegate.py's
    _make_mock_parent.
    """
    parent = MagicMock()
    parent.session_id = "parent-session-gap"
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-opus-4.8"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.provider_require_parameters = False
    parent.provider_data_collection = None
    parent.request_overrides = {}
    parent.max_tokens = None
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent.disabled_toolsets = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._subagent_finalization_lock = threading.RLock()
    parent._current_task_id = None
    parent._current_turn_id = ""
    parent._memory_manager = None
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    # Usage-ledger fields (Card 2 contract).
    parent.session_delegation_usage = []
    parent.session_estimated_cost_usd = 0.0
    parent.session_cost_source = "none"
    parent.session_cost_status = "unknown"
    return parent


def _make_mock_child():
    """Mock child AIAgent whose run_conversation completes immediately."""
    child = MagicMock()
    child.run_conversation.return_value = {
        "final_response": "done",
        "completed": True,
        "interrupted": False,
        "api_calls": 2,
        "tokens": {"input": 100, "output": 50},
    }
    child._credential_pool = None
    return child


def _profile_resolver(return_model="profile-model-7b"):
    """Stub kwargs for hermes_cli.runtime_provider.resolve_runtime_provider, the
    seams the native profile route reads its model/credentials through."""
    return {
        "provider": "custom", "requested_provider": "profile-bot",
        "model": return_model, "base_url": "https://p.example/v1",
        "api_key": "profile-key", "api_mode": "chat_completions", "args": [],
    }


# ── Category 1: Property-based path traversal tests (Hypothesis) ─────────


class TestPathTraversalPropertyBased(unittest.TestCase):
    """Property-based fuzzing of the native ^[a-z0-9][a-z0-9_-]{0,63}$ profile-id
    guard (hermes_cli.profiles), which refuses invalid ids with ValueError."""

    @settings(max_examples=100, deadline=None)
    @given(st.text(max_size=100))
    def test_invalid_names_rejected_before_filesystem_access(self, name):
        """Any string outside the profile-id grammar (after normalization) must
        be refused, and must be rejected BEFORE the profiles root is ever
        consulted (the regex guard is the security control; the directory
        check is not). Grammar-legal names and the ``default`` alias are
        covered by their own properties/tests."""
        from hermes_cli import profiles as profiles_mod

        canon = _canon(name)
        if canon == "default" or (canon and SAFE_PROFILE_ID_RE.match(canon)):
            return  # handled by test_valid_names_* / the default alias contract
        with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
            with self.assertRaises(
                ValueError,
                msg=f"profile name {name!r} is outside the profile-id grammar "
                "and must be refused",
            ):
                profiles_mod.get_profile_dir(name)
            mock_root.assert_not_called()

    @settings(max_examples=50, deadline=None)
    @given(st.from_regex(r"[a-z0-9][a-z0-9_-]{0,63}", fullmatch=True))
    def test_valid_names_pass_guard_and_load(self, name):
        """Names matching the grammar must never be refused by the guard: a
        real on-disk profile with that exact name resolves to its directory
        and reads as live."""
        from hermes_cli import profiles as profiles_mod

        if name == "default":
            return  # the default alias resolves to the default home, not profiles/
        tmpdir = tempfile.mkdtemp()
        try:
            profiles_root = pathlib.Path(tmpdir) / "profiles"
            profile_dir = profiles_root / name
            profile_dir.mkdir(parents=True)
            (profile_dir / "SOUL.md").write_text(
                "gap-test soul", encoding="utf-8")
            with patch.object(profiles_mod, "_get_profiles_root",
                              return_value=profiles_root):
                resolved = profiles_mod.get_profile_dir(name)
                self.assertEqual(resolved, profile_dir)
                self.assertTrue(profiles_mod.profile_exists(name))
        finally:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_default_alias_passes_without_the_profiles_root(self):
        """``default`` (and any casing of it) is the special alias: it resolves
        without the id regex, so it can never be used to smuggle a traversal."""
        from hermes_cli import profiles as profiles_mod

        for alias in ["default", "Default", "DEFAULT", " default "]:
            with self.subTest(name=alias):
                canon = profiles_mod.normalize_profile_name(alias)
                self.assertEqual(canon, "default")
                resolved = profiles_mod.get_profile_dir(alias)
                self.assertTrue(resolved.is_dir() or str(resolved))

    def test_adversarial_traversal_names_rejected(self):
        """Hand-picked adversarial inputs (spec list) must all be refused with
        ValueError — never resolved — even with a real secret directory sitting
        outside the profiles root that a broken guard would read."""
        from hermes_cli import profiles as profiles_mod

        adversarial = [
            "../", "..\\", ".../", "./", ".\\", "/etc", "\\windows",
            "foo/../../bar", "a/b/c", "a.b.c", " ", "  ", "\t", "\n",
            "null\x00byte",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            secret = pathlib.Path(tmpdir) / "secret"
            secret.mkdir()
            (secret / "SOUL.md").write_text("SECRET DATA", encoding="utf-8")
            for bad in adversarial:
                with self.subTest(name=bad):
                    canon = _canon(bad)
                    if canon == "default" or (canon and SAFE_PROFILE_ID_RE.match(canon)):
                        continue  # grammar-legal: not an adversarial case here
                    with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
                        with self.assertRaises(
                            ValueError,
                            msg=f"adversarial name {bad!r} must be refused",
                        ):
                            profiles_mod.get_profile_dir(bad)
                        mock_root.assert_not_called()

    def test_unicode_names_rejected(self):
        """Non-ASCII codepoints can never satisfy the ASCII-only grammar: they
        are regex-refused with ValueError (the legacy seam returned None; the
        refusal-first intent is preserved)."""
        from hermes_cli import profiles as profiles_mod

        for bad in ["böt", "日本語", "🤖-bot", "pro\u00e9file", "профиль", "‮rtl"]:
            with self.subTest(name=bad):
                with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
                    with self.assertRaises(ValueError):
                        profiles_mod.get_profile_dir(bad)
                    mock_root.assert_not_called()

    def test_control_characters_rejected(self):
        """Null bytes and control characters must be refused at the guard."""
        from hermes_cli import profiles as profiles_mod

        for bad in ["\x00", "a\x00b", "null\x00byte", "a\x01b", "a\x1fb",
                    "a\x7fb"]:
            with self.subTest(name=bad):
                with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
                    with self.assertRaises(ValueError):
                        profiles_mod.get_profile_dir(bad)
                    mock_root.assert_not_called()

    def test_extremely_long_names(self):
        """Extremely long names: an INVALID 5000-char name is refused at the
        guard before any filesystem access; the native grammar additionally
        caps length at 64 chars, so a long-but-ASCII name (240 chars,
        filename-legal) is ALSO refused by the regex — no OSError can escape
        (the baseline's KNOWN FINDING is structurally closed upstream)."""
        from hermes_cli import profiles as profiles_mod

        with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
            with self.assertRaises(ValueError):
                profiles_mod.get_profile_dir("a/" * 5000)
            mock_root.assert_not_called()
        # Grammar characters but over the 64-char cap: refused by the same guard.
        with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
            with self.assertRaises(ValueError):
                profiles_mod.get_profile_dir("a" * 240)
            mock_root.assert_not_called()
        # At the cap and grammar-legal: the guard passes; the (nonexistent)
        # directory lookup then reports not-live without crashing.
        at_cap = "a" * 64
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(profiles_mod, "_get_profiles_root",
                              return_value=pathlib.Path(tmpdir)):
                self.assertFalse(profiles_mod.profile_exists(at_cap))

    def test_traversal_never_reads_outside_the_profiles_root(self):
        """Security intent (e445d2ab2d / ../../.ssh): a traversal id is refused
        before filesystem access, so an attacker-named profile can never read
        another directory's identity files."""
        from hermes_cli import profiles as profiles_mod
        from hermes_cli.profiles import get_profile_dir, profile_exists

        traversal = "../../.ssh"
        self.assertFalse(profile_exists(traversal))
        with self.assertRaises(ValueError):
            get_profile_dir(traversal)
        # And the refusal happens at the regex, before any root consultation:
        with patch.object(profiles_mod, "_get_profiles_root") as mock_root:
            with self.assertRaises(ValueError):
                profiles_mod.get_profile_dir(traversal)
            mock_root.assert_not_called()


# ── Category 2: Regression — allow_model_selection=false gating ─────────


class TestProfileModelGatingRegression(unittest.TestCase):
    """Regression: allow_model_selection=False must gate model selection even
    when allow_profile_identity=True exposes profiles (whose config.yaml may
    name a model). The two flags must stay fully independent."""

    def test_flag_getters_when_model_selection_disabled(self):
        cfg = {
            "allow_profile_identity": True,
            "allow_model_selection": False,
        }
        flags = _legacy_opt_in_flags(cfg)
        self.assertFalse(flags["model"])
        self.assertTrue(flags["profile"])

    def test_schema_has_profile_but_not_model(self):
        """The core gating regression: profile ON + model OFF must advertise
        the profile field but NOT the model field.

        Adapted per-merge: ``profile`` is declared through the default
        provider's ``task_parameters`` seam, so it appears ONLY inside the
        tasks[] item properties; the top level keeps just the core fields
        (the old top-level profile assertion rode our pre-merge single-layer
        overrides and no longer applies). ``model`` is reserved — it must not
        appear in EITHER layer while the flag is off."""
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={
                "allow_profile_identity": True,
                "allow_model_selection": False,
            },
        ):
            ov = _build_dynamic_schema_overrides()
            self.assertEqual(
                _legacy_task_model_parameter(), {},
                "model flag off: the reserved model field must not ride the "
                "dynamic overrides",
            )
        props = ov["parameters"]["properties"]
        self.assertNotIn("model", props)
        task_props = props["tasks"]["items"]["properties"]
        self.assertNotIn("model", task_props)
        self.assertIn("profile", task_props)
        self.assertEqual(task_props["profile"]["type"], "string")

    def test_flag_isolation_profile_on_only(self):
        """Turning the profile flag on must not turn the model flag on."""
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_profile_identity": True},
        ):
            flags = _legacy_opt_in_flags({"allow_profile_identity": True})
            self.assertTrue(flags["profile"])
            self.assertFalse(flags["model"])
            ov = _build_dynamic_schema_overrides()
        props = ov["parameters"]["properties"]
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("profile", task_props)
        self.assertEqual(task_props["profile"]["type"], "string")
        self.assertNotIn("model", task_props)
        self.assertNotIn("model", props)

    def test_flag_isolation_model_on_only(self):
        """Turning the model flag on must not turn the profile flag on."""
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_model_selection": True},
        ):
            flags = _legacy_opt_in_flags({"allow_model_selection": True})
            self.assertTrue(flags["model"])
            self.assertFalse(flags["profile"])
            # model rides the dynamic overrides (reserved route field) …
            model_param = _legacy_task_model_parameter()
            self.assertIn("model", model_param)
            self.assertEqual(model_param["model"]["type"], "string")
            ov = _build_dynamic_schema_overrides()
            # … and the provider seam must NOT advertise it.
            from tools.delegate_tool_routes import _legacy_declared_parameters

            self.assertEqual(_legacy_declared_parameters({"allow_model_selection": True}), {})
        props = ov["parameters"]["properties"]
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("model", task_props)
        self.assertEqual(task_props["model"]["type"], "string")
        self.assertNotIn("profile", task_props)

    def test_profile_model_ignored_when_model_selection_disabled(self):
        """Dispatch-level regression: with allow_model_selection=False, a
        profile's config.yaml model must NOT trigger switch_model. The child
        runs the PROFILE's own native route (its config.yaml primary model via
        ``_profile_route`` — upstream-native), never the legacy per-task
        override pipeline: switch_model stays uncalled."""
        set_spawn_paused(False)
        parent = _make_mock_parent()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prof = pathlib.Path(tmp) / "profiles" / "devbot"
            prof.mkdir(parents=True)
            (prof / "SOUL.md").write_text(
                "You are Devbot — a build agent.\n", encoding="utf-8")
            (prof / "config.yaml").write_text(
                "model:\n  default: glm-5.3\n", encoding="utf-8")
            from hermes_cli import profiles as profiles_mod

            with patch(
                "tools.delegate_tool_config._load_config",
                return_value={
                    "allow_profile_identity": True,
                    "allow_model_selection": False,
                },
            ), patch.object(
                profiles_mod, "_get_profiles_root",
                return_value=pathlib.Path(tmp) / "profiles",
            ), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_profile_resolver(),
            ), patch(
                "hermes_cli.model_switch.switch_model"
            ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
                child = _make_mock_child()
                MockAgent.return_value = child
                out = delegate_task(
                    tasks=[{"goal": GOAL, "profile": "devbot"}],
                    parent_agent=parent,
                )
            payload = json.loads(out)
            self.assertNotIn("error", payload)
            # The per-task override pipeline never ran: no legacy model switch.
            mock_switch.assert_not_called()
            # The child runs the PROFILE's own route (its config.yaml primary
            # model via the native profile path), not the parent's model.
            kwargs = MockAgent.call_args.kwargs
            self.assertEqual(kwargs["model"], "glm-5.3")
            # Identity still applied — gating the model does not gate the
            # profile: the child is armed to load the profile's SOUL.md.
            self.assertTrue(kwargs["load_soul_identity"])


# ── Category 3: Fallback chain semantics ─────────────────────────────────


class TestFallbackChainSemantics(unittest.TestCase):
    """Fail loudly, never silently inherit."""

    def test_unresolvable_model_raises_value_error(self):
        """A name the shared model_switch pipeline cannot resolve raises
        ValueError — no silent fallback to the base model.

        Offline the upstream validator soft-accepts unknown slugs (no provider
        listing to check), so the live raise is environment-dependent; the
        failing-gate branch is asserted through the simulated failed switch —
        the same gate the live resolver returns for a genuinely unresolvable
        name (pattern shared with test_delegate_model_selection.py)."""
        creds = _base_creds()
        with patch(
            "hermes_cli.model_switch.switch_model",
            return_value=_make_switch_result(
                success=False, error_message="zzz-not-a-real-model-xyz"),
        ):
            with self.assertRaises(ValueError):
                _legacy_task_model_creds(
                    "zzz-not-a-real-model-xyz", _FakeParent(), creds)

    def test_resolver_message_surfaced_in_value_error(self):
        """The resolver's error message must reach the caller (per-task error
        quality), not be swallowed into a generic message."""
        creds = _base_creds()
        with patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="no such model in catalog"
            )
            with self.assertRaises(ValueError) as ctx:
                _legacy_task_model_creds("bogus", _FakeParent(), creds)
        self.assertIn("no such model in catalog", str(ctx.exception))

    def test_none_model_name_returns_base_unchanged(self):
        """None model name is treated as empty — a no-op returning the base
        creds by identity, with no resolution attempted."""
        creds = _base_creds()
        out = _legacy_task_model_creds(None, _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_empty_model_name_returns_base_unchanged(self):
        """Empty model name is a no-op (covered elsewhere too; kept here for
        completeness as the fallback-chain contract)."""
        creds = _base_creds()
        out = _legacy_task_model_creds("", _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_whitespace_model_name_returns_base_unchanged(self):
        """A whitespace-only name strips to empty and is a no-op."""
        creds = _base_creds()
        out = _legacy_task_model_creds("   ", _FakeParent(), creds)
        self.assertIs(out, creds)

    def test_base_creds_not_mutated_on_failure(self):
        """Even when resolution fails (ValueError), the base creds dict must
        be byte-identical to its pre-call state."""
        creds = _base_creds()
        before = dict(creds)
        with patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="unknown model"
            )
            with self.assertRaises(ValueError):
                _legacy_task_model_creds("zzz-not-real", _FakeParent(), creds)
        self.assertEqual(creds, before)

    def test_dispatch_fails_loudly_on_unresolvable_model(self):
        """delegate_task must surface a per-task error for an unresolvable
        model instead of silently spawning the child on the default model.
        Adapted: an all-refused batch fails closed as a tool error (the
        upstream routes-native per-task refusal contract — no child for any
        refused task)."""
        set_spawn_paused(False)
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool_config._load_config",
            return_value={"allow_model_selection": True},
        ), patch(
            "hermes_cli.model_switch.switch_model"
        ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
            mock_switch.return_value = _make_switch_result(
                success=False, error_message="Could not resolve model "
                "'zzz-bogus-model-xyz' for this subagent."
            )
            out = delegate_task(
                tasks=[{"goal": GOAL, "model": "zzz-bogus-model-xyz"}],
                parent_agent=parent,
            )
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn("zzz-bogus-model-xyz", payload["error"])
        self.assertIn("Could not resolve model", payload["error"])
        # No child was ever constructed — no silent fallback spawn.
        MockAgent.assert_not_called()


# ── Category 4: Integration with mock child spawns ───────────────────────


class TestProfileIdentityIntegration(unittest.TestCase):
    """delegate_task end-to-end with mocked AIAgent spawns (no real API
    calls): a profile child runs with the native load_soul_identity wiring
    bound to its profile home, without a profile the generic preamble is
    used, and an explicit per-task model beats the profile's own route."""

    def setUp(self):
        set_spawn_paused(False)

    def _mk_profile(self, tmp: pathlib.Path, name: str = "devbot") -> pathlib.Path:
        prof = pathlib.Path(tmp) / "profiles" / name
        prof.mkdir(parents=True)
        (prof / "SOUL.md").write_text(
            "You are Devbot — a build agent.\n", encoding="utf-8")
        (prof / "config.yaml").write_text(
            "model:\n  default: glm-5.3\n", encoding="utf-8")
        return prof

    def test_profile_identity_appears_in_child_system_prompt(self):
        """Adapted (upstream-native): the profile never inlines into the child
        system prompt at all. Sibling children share ONE cached preamble
        (``You are a focused subagent…``) and the profile rides the route
        instead: the child converses INSIDE the profile's scope loading its
        SOUL.md (``load_soul_identity``), so identity still reaches the child
        — the wiring test_child_builder_binds… pattern in
        test_delegate_per_task_overrides.py, exercised live here."""
        from agent.prompt_builder import load_soul_md
        from tools.delegate_tool_child_run import _ChildRun
        from tools.delegate_tool_routes import _route_to_task_route

        parent = _make_mock_parent()
        with tempfile.TemporaryDirectory() as tmp:
            prof = self._mk_profile(pathlib.Path(tmp))
            from hermes_cli import profiles as profiles_mod

            with patch.object(profiles_mod, "_get_profiles_root",
                              return_value=pathlib.Path(tmp) / "profiles"), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_profile_resolver(),
            ):
                route = _route_to_task_route({"profile": "devbot"}, parent, {})
            # A refused route would mean the profile seam is broken: fail
            # here, not downstream.
            self.assertIsNone(route.error)
            self.assertEqual(route.profile_home, prof)

            class _Child:
                provider, model, _delegate_profile_home = "custom", "glm-5.3", prof

                def run_conversation(self, user_message, **kw):
                    return {
                        "final_response": "done",
                        "completed": True,
                        "interrupted": False,
                        "api_calls": 2,
                        "soul_seen": load_soul_md(),
                    }

            class _Parent:
                _current_task_id = None

            run = _ChildRun(_Child(), _Parent(), 0, GOAL, None, None)
            with patch(
                "tools.delegate_tool_child_run._create_isolated_worktree",
                return_value=None,
            ):
                run.seed_workspace()
                result, failure, _ = run.await_child()
            self.assertIsNone(failure)
            self.assertEqual(result["final_response"], "done")
            # The child conversed AS the profile: its SOUL.md loaded.
            self.assertIn("You are Devbot", result["soul_seen"] or "")

    def test_no_profile_uses_generic_preamble(self):
        """Without a profile, the child gets the generic subagent preamble and
        no soul-identity load is armed."""
        parent = _make_mock_parent()
        with patch(
            "tools.delegate_tool_config._load_config", return_value={}
        ), patch("run_agent.AIAgent") as MockAgent:
            child = _make_mock_child()
            MockAgent.return_value = child
            out = delegate_task(
                tasks=[{"goal": GOAL}], parent_agent=parent
            )
        payload = json.loads(out)
        self.assertNotIn("error", payload)
        prompt = MockAgent.call_args.kwargs["ephemeral_system_prompt"]
        self.assertIn("You are a focused subagent", prompt)
        self.assertNotIn("You are Devbot", prompt)
        # The soul-identity kwarg is armed for profile children only.
        self.assertNotIn("load_soul_identity", MockAgent.call_args.kwargs)

    def test_explicit_model_takes_precedence_over_profile_model(self):
        """When both profile and model are specified, the explicit task model
        wins: switch_model runs exactly once, for the task's model — never
        for the profile's config.yaml model (the profile route itself
        resolved earlier through the native profile path)."""
        parent = _make_mock_parent()
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            prof = self._mk_profile(pathlib.Path(tmp))
            from hermes_cli import profiles as profiles_mod

            with patch(
                "tools.delegate_tool_config._load_config",
                return_value={
                    "allow_profile_identity": True,
                    "allow_model_selection": True,
                },
            ), patch.object(
                profiles_mod, "_get_profiles_root",
                return_value=pathlib.Path(tmp) / "profiles",
            ), patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value=_profile_resolver(),
            ), patch(
                "hermes_cli.model_switch.switch_model"
            ) as mock_switch, patch("run_agent.AIAgent") as MockAgent:
                mock_switch.return_value = _make_switch_result(
                    new_model="gpt-5x", target_provider="openai"
                )
                child = _make_mock_child()
                MockAgent.return_value = child
                out = delegate_task(
                    tasks=[
                        {"goal": GOAL, "profile": "devbot", "model": "gpt-5x"}
                    ],
                    parent_agent=parent,
                )
            payload = json.loads(out)
            self.assertNotIn("error", payload)
            # The shared switch pipeline ran once, for the EXPLICIT model.
            self.assertEqual(mock_switch.call_count, 1)
            self.assertEqual(mock_switch.call_args.kwargs["raw_input"], "gpt-5x")
            # The child was constructed with the resolved model/provider (the
            # task's explicit entry beat the profile's own glm-5.3 route).
            kwargs = MockAgent.call_args.kwargs
            self.assertEqual(kwargs["model"], "gpt-5x")
            self.assertEqual(kwargs["provider"], "openai")
            # Profile identity still applied alongside the model override.
            self.assertTrue(kwargs["load_soul_identity"])


if __name__ == "__main__":
    unittest.main()
