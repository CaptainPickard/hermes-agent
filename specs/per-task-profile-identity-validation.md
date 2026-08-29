# Validation Spec: Per-Task Profile Identity (Phase 2)

## What to Validate

Testerbot must independently verify that the per-task profile identity implementation correctly loads profile identity files and model config, with proper backward compatibility and error handling.

## File Under Test

`/workspace/hermes-agent-fork/tools/delegate_tool.py`

## Validation Checks

### V1: Schema Contains profile in tasks.items.properties

Read `DELEGATE_TASK_SCHEMA`. Confirm `tasks.items.properties` includes `profile` (type: string). It must NOT be in the top-level `properties` dict.

### V2: Backward Compatibility - No Profile

Call `delegate_task` with tasks that do NOT include `profile`. Verify the child's system prompt starts with "You are a focused subagent" (the generic opener). No profile files are loaded.

### V3: Profile Loads SOUL.md

Call `delegate_task` with a task that has `profile: "devbot"`. Verify the child's system prompt contains the content of devbot's SOUL.md (not the generic opener). The system prompt should NOT start with "You are a focused subagent."

### V4: Profile Loads IDENTITY.md

Same test as V3, but verify IDENTITY.md content is also present in the system prompt.

### V5: Profile Reads Model from config.yaml

Call `delegate_task` with a task that has `profile: "devbot"` but no explicit `model` or `provider` fields. Verify the child agent is constructed with the model from devbot's config.yaml (`model.default`).

### V6: Explicit Model Overrides Profile Config

Call `delegate_task` with a task that has `profile: "devbot"` AND `model: "gpt-5.6-terra"`. Verify the child uses `gpt-5.6-terra` (the explicit value), not devbot's config model.

### V7: Nonexistent Profile Returns Error

Call `delegate_task` with a task that has `profile: "nonexistent-bot"`. Verify it returns a tool_error with a clear message mentioning the profile name. No child should be spawned.

### V8: Profile With No Identity Files Falls Back

Mock a profile directory that exists but has no SOUL.md, IDENTITY.md, or AGENTS.md. Verify the child falls back to the generic opener (no crash).

### V9: All Existing Tests Pass

Run `python3 -m pytest tests/tools/test_delegate.py -v` and confirm all pre-existing tests still pass.

### V10: _MODEL_HIDDEN_TASK_FIELDS Does Not Strip profile

Verify `_MODEL_HIDDEN_TASK_FIELDS` remains exactly `{"acp_command", "acp_args"}`. The new `profile` field must pass through.

### V11: Profile AGENTS.md Included in System Prompt

Call `delegate_task` with `profile: "devbot"`. Verify the child's system prompt includes content from devbot's AGENTS.md file.

### V12: Precedence - Per-Task Model > Profile Config > Global

Verify the full precedence chain:
- Task with model override + profile: uses task model
- Task with profile only (no model): uses profile config model
- Task with neither: uses global delegation config model

## Test Isolation

All tests must use isolated HERMES_HOME (tempdir). For profile loading tests, create mock profile directories under the temp HERMES_HOME with test SOUL.md, IDENTITY.md, AGENTS.md, and config.yaml files.

## PASS/FAIL Criteria

- V1-V12 all pass: PASS
- Any V1-V12 fails: FAIL with specific failure detail
- Existing tests broken: FAIL (critical)