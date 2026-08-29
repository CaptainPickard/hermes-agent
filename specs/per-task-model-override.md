# Spec: Per-Task Model/Provider Override for delegate_task

## Summary

Add `model` and `provider` fields to individual tasks in `delegate_task` batch mode so each subagent in a batch can run on a different model/provider pair. When not specified, tasks fall back to the global `delegation.model` / `delegation.provider` config (backward compatible).

## Motivation

Currently all sub-agents dispatched via `delegate_task` use a single global `delegation.model` / `delegation.provider` from config.yaml. In multi-role workflows (e.g. IO dispatching Devbot + Testerbot + Designbot in one batch), every sub-agent runs on the same model. We need per-task model routing so Designbot can use `glm-5.3` via `ollama-cloud` while Devbot uses `gpt-5.6-terra` via `openai-codex` in the same `delegate_task` call.

## File Manifest

All changes in the fork repo at `/workspace/hermes-agent-fork/`:

1. **`tools/delegate_tool.py`** — main implementation (schema + credential resolution + child building loop)
2. **`tests/tools/test_delegate.py`** — add test for per-task model override ( Testerbot writes this, but Devbot should add the minimal test fixture )

## Implementation Detail

### Change 1: Add `model` and `provider` to the task schema (DELEGATE_TASK_SCHEMA)

Location: `DELEGATE_TASK_SCHEMA` in `tools/delegate_tool.py` (~line 4800-4840)

In the `tasks.items.properties` dict, add two new optional fields:

```python
"model": {
    "type": "string",
    "description": (
        "Optional model override for THIS task only. "
        "When set, the child runs on this model instead of "
        "delegation.model from config.yaml. Must be paired "
        "with provider if the model lives on a different "
        "provider than the global delegation.provider."
    ),
},
"provider": {
    "type": "string",
    "description": (
        "Optional provider override for THIS task only. "
        "When set, credentials are resolved via the runtime "
        "provider system for this provider. Falls back to "
        "delegation.provider when not set."
    ),
},
```

### Change 2: Resolve per-task credentials in the child-building loop

Location: The `for i, t in enumerate(task_list):` loop (~line 3870)

Currently:
```python
# All children use the same creds dict resolved once
child = _build_child_preserving_parent_tools(
    ...
    model=creds["model"],
    override_provider=creds["provider"],
    override_base_url=creds["base_url"],
    override_api_key=creds["api_key"],
    override_api_mode=creds["api_mode"],
    ...
)
```

Change to: resolve per-task credentials when the task has `model` or `provider` set:

```python
# Resolve per-task credentials: task-level model/provider
# overrides fall back to global delegation config.
_task_model = t.get("model")
_task_provider = t.get("provider")

if _task_model or _task_provider:
    # Build a per-task config dict merging global delegation
    # settings with per-task overrides
    _task_cfg = dict(cfg)  # shallow copy of delegation config
    if _task_model:
        _task_cfg["model"] = _task_model
    if _task_provider:
        _task_cfg["provider"] = _task_provider
    # Keep base_url/api_key/api_mode from global config unless
    # the provider changed (then _resolve_delegation_credentials
    # will re-resolve them via the runtime provider system)
    if _task_provider and _task_provider != str(cfg.get("provider") or "").strip():
        # New provider: clear base_url/api_key/api_mode so the
        # resolver does a full runtime provider resolution
        _task_cfg["base_url"] = ""
        _task_cfg["api_key"] = ""
        _task_cfg["api_mode"] = ""
    try:
        _task_creds = _resolve_delegation_credentials(_task_cfg, parent_agent)
    except ValueError as exc:
        return tool_error(str(exc))
else:
    _task_creds = creds

child = _build_child_preserving_parent_tools(
    ...
    model=_task_creds["model"],
    override_provider=_task_creds["provider"],
    override_base_url=_task_creds["base_url"],
    override_api_key=_task_creds["api_key"],
    override_api_mode=_task_creds["api_mode"],
    override_request_overrides=_task_creds.get("request_overrides"),
    override_max_tokens=_task_creds.get("max_output_tokens"),
    override_acp_command=_task_creds.get("command"),
    override_acp_args=_task_creds.get("args"),
    ...
)
```

### Change 3: Thread per-task credentials through the async dispatch path

Location: The `dispatch_async_delegation_batch` call (~line 4286)

Currently the async path uses a single `creds` for the batch metadata. The per-task credentials are already handled in the child-building loop above (which runs before the async dispatch), so the async dispatch's `model=creds["model"]` parameter is just metadata for the completion block. The actual per-child model is already set on each constructed child agent.

No change needed here — the children are already built with per-task credentials by the time the async dispatch runs. The `model=creds["model"]` in the async dispatch is batch-level metadata only.

### Change 4: Update the top-level description

Location: `_build_top_level_description()` (~line 4680)

Add a note about per-task model override:

```python
# In the description string, add:
"- Tasks can optionally specify model/provider to run each "
"subagent on a different model than the global delegation config. "
```

## Acceptance Criteria

1. A batch `delegate_task` call with `tasks=[{goal: "...", model: "glm-5.3", provider: "ollama-cloud"}, {goal: "...", model: "gpt-5.6-terra", provider: "openai-codex"}]` resolves each child's credentials independently.

2. A batch `delegate_task` call without per-task `model`/`provider` fields uses the global `delegation.model`/`delegation.provider` exactly as before (backward compatible).

3. A single-task `delegate_task` call (top-level `goal` form) still uses the global delegation config (no change to legacy form).

4. Per-task `model` without `provider` uses the global `delegation.provider` with the task-specific model.

5. Per-task `provider` without `model` uses the global `delegation.model` with the task-specific provider's credential resolution.

6. Invalid per-task `provider` raises a clear ValueError with a user-friendly message (same as the existing global provider validation).

7. All existing tests pass unchanged.

## Test Plan (for Testerbot)

1. **Backward compat:** Run the existing `tests/tools/test_delegate.py` suite. All must pass unchanged.

2. **Per-task model override:** Mock `_resolve_delegation_credentials` and verify it's called with per-task model/provider when those fields are present in a task dict.

3. **Fallback:** Verify that tasks without `model`/`provider` fields use the global `creds` dict (same resolution as before).

4. **Mixed batch:** Verify a batch with some tasks having per-task overrides and some not — the overridden tasks resolve independently while the non-overridden tasks use global creds.

5. **Schema validation:** Verify the tool schema includes `model` and `provider` in the `tasks.items.properties` dict.

## Constraints

- Do NOT change the single-goal form (top-level `goal`/`context`). Per-task overrides only apply to the `tasks=[...]` batch form.
- Do NOT add `model`/`provider` to the top-level schema properties (only in `tasks[].items.properties`).
- Do NOT change the `credentials_cfg` internal parameter behavior (that's for the /review engine, leave it alone).
- Do NOT break prompt caching: the schema changes are additive (new optional fields), so existing tool schemas remain valid.
- Keep the change under 150 lines of actual code modification.