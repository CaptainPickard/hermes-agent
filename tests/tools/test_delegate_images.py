"""Per-task image forwarding on delegate_task: a task's ``images`` (local paths, http(s) or data: URLs) reach a
vision-capable child as native ``image_url`` content parts on its goal turn; non-vision children get
``[Image attached …]`` hints for ``vision_analyze``; any failure degrades to the text-only goal."""

import base64
from unittest.mock import patch

from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _MAX_TASK_IMAGES, _build_child_goal_message, _normalize_task_images
from tools.delegate_tool_child_run import _ChildRun

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


class _FakeChild:
    provider = "openrouter"
    model = "some/vision-model"


def _png(tmp_path, name="shot.png"):
    p = tmp_path / name
    p.write_bytes(_PNG)
    return str(p)


def test_normalize_task_images_shapes_and_limit():
    assert _normalize_task_images({"goal": "g"}, 0) == (None, None)
    assert _normalize_task_images({"images": []}, 0) == (None, None)
    assert _normalize_task_images({"images": "/tmp/a.png"}, 0) == (["/tmp/a.png"], None)
    assert _normalize_task_images({"images": [" /tmp/a.png ", "https://x.test/b.jpg"]}, 0) == (
        ["/tmp/a.png", "https://x.test/b.jpg"], None,
    )
    for bad in ({"path": "x"}, ["/tmp/a.png", 42], ["  "]):
        cleaned, err = _normalize_task_images({"images": bad}, 2)
        assert cleaned is None and "Task 2" in err
    cleaned, err = _normalize_task_images({"images": [f"/tmp/{i}.png" for i in range(_MAX_TASK_IMAGES + 1)]}, 1)
    assert cleaned is None and str(_MAX_TASK_IMAGES) in err


def test_native_mode_builds_image_parts_for_every_source_kind(tmp_path):
    path = _png(tmp_path)
    url = "https://example.test/mock.png"
    data_url = "data:image/png;base64," + base64.b64encode(_PNG).decode()
    with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        msg = _build_child_goal_message("Inspect the mock", [path, url, data_url], _FakeChild())
    assert msg[0]["type"] == "text" and "Inspect the mock" in msg[0]["text"]
    images = [p["image_url"]["url"] for p in msg if p.get("type") == "image_url"]
    assert len(images) == 3 and url in images
    assert images.count(data_url) == 2  # local file embedded as a data URL + the inline data URL passed verbatim
    assert data_url not in msg[0]["text"]  # inline base64 never leaks into the text part
    # every source unreadable → plain goal, no empty multimodal envelope
    with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        assert _build_child_goal_message("Goal text", [str(tmp_path / "nope.png")], _FakeChild()) == "Goal text"


def test_text_mode_hints_and_failure_degrade(tmp_path):
    path = _png(tmp_path)
    url = "https://example.test/a.png"
    with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
        msg = _build_child_goal_message("Goal", [path, url, str(tmp_path / "gone.png")], _FakeChild())
    assert isinstance(msg, str)
    assert f"[Image attached at: {path}]" in msg and f"[Image attached: {url}]" in msg and "vision_analyze" in msg
    assert "gone.png" not in msg
    with patch("agent.image_routing.decide_image_input_mode", side_effect=RuntimeError("boom")):
        assert _build_child_goal_message("Plain goal", ["/tmp/x.png"], _FakeChild()) == "Plain goal"


def test_child_run_sends_multimodal_goal_turn(tmp_path):
    """The attached images reach ``run_conversation`` as the FIRST user turn's content list."""
    seen = {}

    class _Child(_FakeChild):
        _delegate_images = [_png(tmp_path)]

        def run_conversation(self, user_message, **kw):
            seen["user_message"] = user_message
            return {"final_response": "ok", "completed": True}

    class _Parent:
        _current_task_id = None

    run = _ChildRun(_Child(), _Parent(), 0, "Describe it", None, None)
    with patch("tools.delegate_tool_child_run._create_isolated_worktree", return_value=None), \
            patch("agent.image_routing.decide_image_input_mode", return_value="native"):
        run.seed_workspace()
        result, failure, _ = run.await_child()
    assert failure is None and result["final_response"] == "ok"
    content = seen["user_message"]
    assert isinstance(content, list) and content[0]["type"] == "text"
    assert [p["type"] for p in content].count("image_url") == 1
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_images_advertised_per_task_only():
    item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
    assert item["properties"]["images"]["type"] == "array" and "images" not in item["required"]
    assert "images" not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Local carry-over from the ours-only v0.23.4 lineage (Roomote #1796, b272d133d6).
#
# Deleted as true duplicates — covering upstream ids in this same file:
#   test_absent_is_none / test_empty_list_is_none / test_single_image_passes_through /
#   test_whitespace_only_rejected / test_empty_string_rejected /
#   test_scalar_string_rejected / test_non_string_entry_rejected
#     → test_normalize_task_images_shapes_and_limit (same assertions, consolidated);
#       the scalar-string case is upstream-native now (a bare string is wrapped into a
#       one-entry list, tools/delegate_tool_child_run.py::_normalize_task_images).
#   test_url_images_are_native_content_parts / test_local_paths_have_hints_for_text_mode /
#   test_mixed_urls_and_paths_are_attached / test_vision_pipe_error_keeps_goal_and_hints
#     → test_native_mode_builds_image_parts_for_every_source_kind +
#       test_text_mode_hints_and_failure_degrade. Note: merged forwarding wraps the hint
#       loop in the same try as mode routing, so a routing-pipe error degrades to the
#       text-only goal with NO hint lines — upstream's exception-path assertion is
#       authoritative; the local "pipe error keeps hints" expectation is obsolete.
# Retained below (ours-only, no upstream coverage): the eight-images boundary guard, the
# schema-surface check, and the whole child-payload spawn block (upstream v0.23.4 covers
# normalize/advertise only — the _delegate_images child wiring is asserted nowhere else).
# ---------------------------------------------------------------------------

import inspect
import json
from unittest.mock import MagicMock

from tools.delegate_tool import _normalize_task_images, delegate_task


def _parent():
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def _child():
    child = MagicMock()
    child.provider = "openrouter"
    child.model = "some/vision-model"
    child.requested_provider = "openrouter"
    child._subagent_id = None
    child._delegate_images = None
    child._session_db = None
    child.session_id = "fake-child-session"
    # No auto-attr context: a MagicMock truthy context would ride the first turn inside a
    # CONTEXT brief block (child_task_message embeds any truthy context), breaking the
    # bare-goal assertions in TestSpawnPayload.
    child._delegate_context = None
    child.run_conversation.return_value = {
        "final_response": "ok", "completed": True, "api_calls": 1,
    }
    return child


def _spawn(tasks=None, *, goal=None, images=None):
    # Patch targets on the merged routes architecture: child construction stays on
    # tools.delegate_tool._build_child_preserving_parent_tools; worktree isolation is
    # consumed in tools.delegate_tool_child_run (was a delegate_tool attribute pre-merge).
    children = [_child() for _ in (tasks or [None])]
    with (
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=children) as build,
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])),
        patch("tools.delegate_tool_child_run._get_worktree_isolation", return_value=False),
        patch("agent.image_routing.decide_image_input_mode", return_value="native"),
    ):
        response = json.loads(delegate_task(
            tasks=tasks, goal=goal, images=images, parent_agent=_parent(),
        ))
    return response, children, build


class TestNormalizeTaskImagesBoundary:
    def test_eight_images_allowed(self):  # ours-only boundary guard (limit is >8)
        cleaned, err = _normalize_task_images({"images": ["x.png"] * 8}, 0)
        assert cleaned == ["x.png"] * 8 and err is None


class TestToolSchemaSurfaceLocal:
    def test_per_task_images_schema(self):  # ours-only; maxItems deliberately absent
        # (tools/delegate_tool.py: the runtime limit is enforced with a clear
        # spawn-time error, not advertised as maxItems).
        item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert item["properties"]["images"]["type"] == "array"
        assert item["properties"]["images"]["items"]["type"] == "string"
        assert "images" not in item["required"]


class TestSpawnPayload:
    def test_top_level_images_reach_single_child_first_turn(self):
        url = "https://example.test/first.png"
        response, children, build = _spawn(goal="Inspect the first screenshot", images=[url])
        assert "error" not in response
        build.assert_called_once()
        assert children[0]._delegate_images == [url]
        message = children[0].run_conversation.call_args.kwargs["user_message"]
        assert any(part.get("image_url", {}).get("url") == url for part in message)
        assert children[0].run_conversation.call_count == 1

        # The live run_agent intercept also forwards the legacy top-level shape.
        from run_agent import AIAgent
        assert 'images=function_args.get("images")' in inspect.getsource(AIAgent._dispatch_delegate_task)

    def test_batch_images_are_isolated_on_each_first_turn(self):
        urls = ["https://example.test/first.png", "https://example.test/second.png"]
        response, children, build = _spawn(tasks=[
            {"goal": "Inspect the first screenshot", "images": [urls[0]]},
            {"goal": "Inspect the second screenshot", "images": [urls[1]]},
        ])
        assert "error" not in response
        assert build.call_count == 2
        for i, child in enumerate(children):
            assert child._delegate_images == [urls[i]]
            message = child.run_conversation.call_args.kwargs["user_message"]
            assert [part["image_url"]["url"] for part in message[1:]] == [urls[i]]
            assert child.run_conversation.call_count == 1

    def test_nine_images_rejected_before_child_spawn(self):
        with patch("tools.delegate_tool._build_child_preserving_parent_tools") as build:
            response = json.loads(delegate_task(
                tasks=[{"goal": "Inspect the screenshot", "images": ["x.png"] * 9}],
                parent_agent=_parent(),
            ))
        assert "Trim to the images the child actually needs to see." in response["error"]
        build.assert_not_called()

    def test_absent_images_leave_first_turn_unchanged(self):
        goal = "Review the text-only second task"
        # _delegate_images is read inside the child-run thread (delegate_tool_child_run.py:1000),
        # so the message-builder seam must be patched for the whole run, not just the spawn.
        with patch("tools.delegate_tool_child_run._build_child_goal_message") as make_message:
            response, children, build = _spawn(goal=goal)
            assert "error" not in response
            build.assert_called_once()
            assert children[0].run_conversation.call_args.kwargs["user_message"] == goal
            assert children[0].run_conversation.call_count == 1
            make_message.assert_not_called()
