# Spec: Per-Task Profile Identity for delegate_task (Phase 2)

## Summary

Add a `profile` field to individual tasks in `delegate_task` batch mode so each subagent loads a named Hermes profile's full identity (SOUL.md, IDENTITY.md, AGENTS.md) and model/provider config, becoming that bot instead of a generic subagent.

## Motivation

Currently, `delegate_task` builds a generic system prompt for every child: "You are a focused subagent working on a specific delegated task." The code explicitly skips SOUL.md (line 1208: "identity belongs to the parent"). This means every subagent is a blank slate that IO must manually inject personality into via the goal text.

The full vision: when a task specifies `profile: "designbot"`, the child agent loads that profile's complete identity and becomes Designbot — its persona, its operating philosophy, its model, its workspace guidelines. No manual "You are Designbot..." text needed in the goal.

## Usage Example

```
delegate_task(tasks=[
  {goal: "Design the login page UI", profile: "designbot"},
  {goal: "Implement the backend API", profile: "devbot"},
  {goal: "Validate the implementation", profile: "testerbot"}
])
```

Each child loads its profile's SOUL.md, IDENTITY.md, AGENTS.md, and config.yaml model/provider. The result: three agents that are actually Designbot, Devbot, and Testerbot — not generic subagents with role text pasted into the goal.

## File Manifest

All changes in the fork repo at `/workspace/hermes-agent-fork/`:

1. **`tools/delegate_tool.py`** — schema, profile resolution, system prompt builder
2. **`tests/tools/test_delegate.py`** — tests for per-task profile loading

## Implementation Detail

### Change 1: Add `profile` to the task schema

Location: `DELEGATE_TASK_SCHEMA` in `tools/delegate_tool.py` (~line 4858)

In the `tasks.items.properties` dict, add:

```python
"profile": {
    "type": "string",
    "description": (
        "Optional Hermes profile name for THIS task. When set, "
        "the child agent loads that profile's SOUL.md, "
        "IDENTITY.md, and AGENTS.md as its system prompt "
        "identity, and reads model/provider from the profile's "
        "config.yaml if not explicitly overridden. The child "
        "becomes the named bot, not a generic subagent."
    ),
},
```

### Change 2: Resolve profile path and load identity files

Add a new helper function `_load_profile_identity(profile_name: str) -> Optional[str]`:

```python
def _load_profile_identity(profile_name: str) -> Optional[dict]:
    """Load a Hermes profile's identity files and model config.
    
    Returns a dict with keys:
      - soul: str or None (SOUL.md content)
      - identity: str or None (IDENTITY.md content)
      - agents: str or None (AGENTS.md content)
      - model: str or None (from config.yaml model.default)
      - provider: str or None (from config.yaml model.provider)
    
    Returns None if the profile directory doesn't exist.
    """
    from pathlib import Path
    
    # Resolve profile path: ~/.hermes/profiles/<name>/
    # Use get_default_hermes_root() for the root
    from hermes_constants import get_default_hermes_root
    root = get_default_hermes_root()
    profile_path = root / "profiles" / profile_name
    
    if not profile_path.exists():
        return None
    
    result = {"soul": None, "identity": None, "agents": None, "model": None, "provider": None}
    
    # Load SOUL.md
    soul_path = profile_path / "SOUL.md"
    if soul_path.exists():
        try:
            result["soul"] = soul_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    
    # Load IDENTITY.md
    identity_path = profile_path / "IDENTITY.md"
    if identity_path.exists():
        try:
            result["identity"] = identity_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    
    # Load AGENTS.md
    agents_path = profile_path / "AGENTS.md"
    if agents_path.exists():
        try:
            result["agents"] = agents_path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    
    # Load model/provider from config.yaml
    config_path = profile_path / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_cfg = config.get("model", {})
            result["model"] = model_cfg.get("default")
            result["provider"] = model_cfg.get("provider")
        except Exception:
            pass
    
    return result
```

### Change 3: Modify `_build_child_system_prompt` to accept profile identity

Add a `profile_identity: Optional[dict] = None` parameter to `_build_child_system_prompt`.

When `profile_identity` is set, replace the generic "You are a focused subagent" opener with the profile's identity files:

```python
def _build_child_system_prompt(
    goal: str,
    context: Optional[str] = None,
    *,
    workspace_path: Optional[str] = None,
    role: str = "leaf",
    max_spawn_depth: int = 2,
    child_depth: int = 1,
    profile_identity: Optional[dict] = None,  # NEW
) -> str:
    if profile_identity and (profile_identity.get("soul") or profile_identity.get("identity")):
        # Profile-backed identity: use SOUL.md and IDENTITY.md as the opener
        parts = []
        if profile_identity.get("soul"):
            parts.append(profile_identity["soul"])
        if profile_identity.get("identity"):
            parts.append(profile_identity["identity"])
        if profile_identity.get("agents"):
            parts.append(profile_identity["agents"])
        # Then append the task
        parts.append(f"\nYOUR TASK:\n{goal}")
        if context and context.strip():
            parts.append(f"\nCONTEXT:\n{context}")
        # ... rest of the prompt builder (workspace, completion instructions)
    else:
        # Existing generic path (no profile)
        parts = ["You are a focused subagent working on a specific delegated task.", ...]
```

### Change 4: Wire profile loading into the child-building loop

In the `for i, t in enumerate(task_list):` loop (~line 3870):

```python
# Load profile identity if specified
_profile_name = t.get("profile")
_profile_identity = None
if _profile_name:
    _profile_identity = _load_profile_identity(_profile_name)
    if _profile_identity is None:
        return tool_error(
            f"Profile '{_profile_name}' not found. "
            f"Check that the profile exists in ~/.hermes/profiles/{_profile_name}/"
        )
    # If model/provider not explicitly set on the task, use profile config
    if not _task_model and _profile_identity.get("model"):
        _task_model = _profile_identity["model"]
    if not _task_provider and _profile_identity.get("provider"):
        _task_provider = _profile_identity["provider"]
```

This goes BEFORE the credential resolution block from Phase 1, so per-task model/provider override still takes precedence, but profile config.yaml acts as a fallback when those aren't explicitly set.

### Change 5: Pass profile_identity to _build_child_system_prompt

In `_build_child_agent` where the system prompt is built (~line 1731):

```python
child_prompt = _build_child_system_prompt(
    goal,
    context,
    workspace_path=workspace_hint,
    role=effective_role,
    max_spawn_depth=max_spawn,
    child_depth=child_depth,
    profile_identity=profile_identity,  # NEW - threaded through from the task dict
)
```

This requires threading `profile_identity` through `_build_child_preserving_parent_tools` and `_build_child_agent` as a new keyword argument.

### Change 6: Update the top-level description

In `_build_top_level_description()`, add:

```
"Tasks can optionally specify a profile name to load that "
"profile's SOUL.md, IDENTITY.md, AGENTS.md, and model/provider "
"config — the child becomes that bot, not a generic subagent."
```

## Precedence Rules

When resolving model/provider for a task with a profile:

1. Explicit per-task `model`/`provider` fields (highest priority)
2. Profile's `config.yaml` `model.default`/`model.provider` (if profile is set)
3. Global `delegation.model`/`delegation.provider` (lowest, backward compat)

When building the system prompt:

1. If `profile` is set AND profile has SOUL.md or IDENTITY.md: use those as identity
2. If `profile` is set but has no identity files: fall back to generic opener with a warning
3. If `profile` is not set: use generic opener (existing behavior, backward compat)

## Acceptance Criteria

1. A task with `profile: "devbot"` loads devbot's SOUL.md as the child's system prompt identity.
2. A task with `profile: "devbot"` that does NOT specify model/provider reads them from devbot's config.yaml.
3. A task with `profile: "devbot"` that DOES specify model/provider uses the explicit values (overrides profile config).
4. A task without `profile` uses the generic opener (backward compatible).
5. A task with a nonexistent profile returns a clear error message.
6. All existing tests pass unchanged.
7. The profile's AGENTS.md is included in the system prompt as workspace guidelines.
8. IDENTITY.md is included in the system prompt as role definition.

## Constraints

- Do NOT change the single-goal form behavior (per-task profile only applies to tasks=[...]).
- Do NOT add `profile` to the top-level schema properties.
- Do NOT load the profile's skills, memory, or session state — only identity files and config model/provider.
- Do NOT break prompt caching for existing calls (additive schema change only).
- Keep the implementation under 200 lines of actual code modification.
- Use `get_default_hermes_root()` for profile path resolution (NOT hardcoded paths).

## Test Plan (for Testerbot)

1. V1: Schema contains `profile` in tasks.items.properties (not top-level)
2. V2: Backward compat — task without profile uses generic opener
3. V3: Task with profile loads SOUL.md content into system prompt
4. V4: Task with profile reads model from profile config.yaml
5. V5: Task with profile + explicit model override uses explicit model
6. V6: Nonexistent profile returns clear error
7. V7: Profile with no identity files falls back to generic opener
8. V8: All existing tests pass
9. V9: _MODEL_HIDDEN_TASK_FIELDS does not strip profile