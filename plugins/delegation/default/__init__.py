"""Default delegation strategy (``delegation.backend: default``).

Every ``delegate_task`` task takes the route the ``delegation`` config section describes: a
``delegation.base_url`` endpoint, a ``delegation.provider`` pin, ``delegation.model`` and
``delegation.reasoning_effort``, or, when those are empty, the parent's own route. Bundled
``kind: backend``, so it is always loaded; other strategies are plugins of the same kind
(``agent/delegation_provider.py``).

Legacy opt-in per-task overrides (ours-only ``8c216bca72``, migrated onto the routes
architecture per IO Amendment Block A4 / AMR-1): when ``delegation.allow_model_selection``
(respectively ``delegation.allow_profile_identity``) is enabled, a task may name a per-task
``model`` (respectively ``profile``). The routes layer (``tools/delegate_tool_routes.py``)
owns the flags: it declares the schema fields through this provider's ``task_parameters``
and resolves the per-task route (parent-anchored model switch / the native profile route).
This provider only copies a declared entry into the route dict. Flag-off behavior is exactly
upstream's ``default``: the fields are not declared, so they never become model-facing and an
entry a client force-sends goes nowhere (legacy ``8c216bca72`` ignored them the same way).
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.delegation_provider import DelegationProvider


class DefaultDelegationProvider(DelegationProvider):
    """Route every task through the ``delegation`` section."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def display_name(self) -> str:
        return "Default"

    @property
    def summary(self) -> str:
        return "Every subagent uses delegation.model / delegation.provider, else the parent's model."

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """The legacy opt-in per-task schema fields (8c216bca72), declared through the standard
        provider seam; ``{}`` (upstream's original surface) while both flags are off."""
        from tools.delegate_tool_routes import _legacy_declared_parameters
        return _legacy_declared_parameters(config)

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        # The delegation section has no speed tier: an unknown key there never becomes one.
        base = {k: v for k, v in config.items() if k != "service_tier"}
        # Legacy opt-in overrides (8c216bca72): a task entry arrives here only when the routes
        # layer declared it (the legacy flags), so copy it into the route dict for the routes
        # layer's per-task resolution (_legacy_task_model_creds / the native profile path).
        for key in ("model", "profile"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                base[key] = value.strip()
        return base


def register(ctx) -> None:
    ctx.register_delegation_provider(DefaultDelegationProvider())
