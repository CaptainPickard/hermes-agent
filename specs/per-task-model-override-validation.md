# Validation Spec: Per-Task Model/Provider Override

## What to Validate

Testerbot must independently verify that the per-task model/provider override implementation in `tools/delegate_tool.py` is correct, backward compatible, and handles edge cases.

## File Under Test

`/workspace/hermes-agent-fork/tools/delegate_tool.py`

## Validation Checks

### V1: Schema Contains model and provider in tasks.items.properties

Read the `DELEGATE_TASK_SCHEMA` dict. Confirm that `tasks.items.properties` includes both `model` (type: string) and `provider` (type: string). Neither should be in the `required` list.

Command:
```bash
python3 -c "
import ast, sys
with open('tools/delegate_tool.py') as f:
    tree = ast.parse(f.read())
# Find DELEGATE_TASK_SCHEMA assignment and check tasks.items.properties
"
```

### V2: Backward Compatibility - No Per-Task Overrides

Call `delegate_task` with tasks that do NOT include `model` or `provider` fields. Verify that `_resolve_delegation_credentials` is called once with the global config (same as before). All children should use the same `creds` dict.

Test approach: Mock `_resolve_delegation_credentials` to return a known dict. Call delegate_task with 2 tasks without model/provider. Assert the mock was called once (not twice) and both children got the same model.

### V3: Per-Task Model Override

Call `delegate_task` with two tasks:
- Task 0: `{goal: "test a", model: "glm-5.3", provider: "ollama-cloud"}`
- Task 1: `{goal: "test b"}` (no override)

Verify that Task 0's child gets `model="glm-5.3"` and `override_provider="ollama-cloud"` while Task 1's child gets the global delegation model/provider.

Test approach: Mock `_resolve_delegation_credentials` to return different values based on input. Or mock `_build_child_preserving_parent_tools` and inspect the `model` and `override_provider` kwargs passed for each task.

### V4: Per-Task Provider Without Model

Call `delegate_task` with a task that has `provider: "ollama-cloud"` but no `model`. Verify the child uses the global `delegation.model` but with the per-task provider's credential resolution.

### V5: Per-Task Model Without Provider

Call `delegate_task` with a task that has `model: "glm-5.3"` but no `provider`. Verify the child uses `model="glm-5.3"` with the global `delegation.provider` credentials.

### V6: Invalid Provider Raises ValueError

Call `delegate_task` with a task that has `provider: "nonexistent-provider"`. Verify it returns a tool_error with a clear message. No child should be spawned.

### V7: Existing Tests Pass

Run `python3 -m pytest tests/tools/test_delegate.py -v` and confirm all pre-existing tests still pass.

### V8: No Top-Level model/provider in Schema

Verify the top-level `properties` dict in `DELEGATE_TASK_SCHEMA` does NOT include `model` or `provider` (they should only be in `tasks.items.properties`). This prevents the model from accidentally setting a global override on the wrong parameter.

### V9: _strip_model_hidden_task_fields Does Not Strip model/provider

Verify `_MODEL_HIDDEN_TASK_FIELDS` only contains `acp_command` and `acp_args`. The new `model` and `provider` fields must pass through to the handler.

## Test Isolation

All tests must use isolated HERMES_HOME (tempdir) and mock `parent_agent` with the required attributes (`model`, `provider`, `base_url`, `api_key`, `session_id`, `_delegate_depth`, `enabled_toolsets`, etc.). Follow the existing test patterns in `tests/tools/test_delegate.py`.

## PASS/FAIL Criteria

- V1-V9 all pass: PASS
- Any V1-V9 fails: FAIL with specific failure detail
- Existing tests broken: FAIL (critical)