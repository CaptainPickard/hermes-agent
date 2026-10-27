"""Per-task delegate_task image forwarding on the deployed v2 lineage."""

import inspect
import json
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _MAX_TASK_IMAGES,
    _build_child_goal_message,
    _normalize_task_images,
    delegate_task,
)


class _FakeChild:
    provider = "openrouter"
    model = "some/vision-model"
    requested_provider = "openrouter"


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
    child.run_conversation.return_value = {
        "final_response": "ok", "completed": True, "api_calls": 1,
    }
    return child


def _spawn(tasks=None, *, goal=None, images=None):
    children = [_child() for _ in (tasks or [None])]
    with (
        patch("tools.delegate_tool._build_child_preserving_parent_tools", side_effect=children) as build,
        patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])),
        patch("tools.delegate_tool._get_worktree_isolation", return_value=False),
        patch("agent.image_routing.decide_image_input_mode", return_value="native"),
    ):
        response = json.loads(delegate_task(
            tasks=tasks, goal=goal, images=images, parent_agent=_parent(),
        ))
    return response, children, build


class TestNormalizeTaskImages:
    def test_absent_is_none(self):  # 1
        assert _normalize_task_images({"goal": "g"}, 0) == (None, None)

    def test_empty_list_is_none(self):  # 2
        assert _normalize_task_images({"images": []}, 0) == (None, None)

    def test_single_image_passes_through(self):  # 3
        assert _normalize_task_images({"images": [" foo.png "]}, 0) == (["foo.png"], None)

    def test_eight_images_allowed(self):  # 4
        assert _normalize_task_images({"images": ["a.png"] * 8}, 0) == (["a.png"] * 8, None)

    def test_nine_images_rejected_with_actionable_limit(self):  # 5
        cleaned, err = _normalize_task_images({"images": ["a.png"] * 9}, 1)
        assert cleaned is None
        assert "per-task limit is 8" in err
        assert "Trim to the images the child actually needs to see." in err

    def test_whitespace_only_rejected(self):  # 6
        cleaned, err = _normalize_task_images({"images": ["  "]}, 0)
        assert cleaned is None and "non-empty strings" in err

    def test_empty_string_rejected(self):  # 7
        cleaned, err = _normalize_task_images({"images": [""]}, 0)
        assert cleaned is None and "non-empty strings" in err

    def test_scalar_string_rejected(self):  # 8
        cleaned, err = _normalize_task_images({"images": "foo.png"}, 0)
        assert cleaned is None and "must be an array" in err

    def test_non_string_entry_rejected(self):  # 9
        cleaned, err = _normalize_task_images({"images": [42]}, 0)
        assert cleaned is None and "non-empty strings" in err


class TestBuildChildGoalMessage:
    def test_url_images_are_native_content_parts(self):  # 10
        urls = ["https://example.test/a.png", "http://example.test/b.jpg"]
        with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
            message = _build_child_goal_message("Inspect", urls, _FakeChild())
        assert message[0]["type"] == "text"
        assert "Inspect" in message[0]["text"]
        assert [part["image_url"]["url"] for part in message[1:]] == urls

    def test_local_paths_have_hints_for_text_mode(self, tmp_path):  # 11
        image = tmp_path / "shot.png"
        image.write_bytes(b"image bytes")
        with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
            message = _build_child_goal_message("Inspect", [str(image)], _FakeChild())
        assert isinstance(message, str)
        assert f"[Image attached at: {image}]" in message
        assert "vision_analyze" in message

    def test_mixed_urls_and_paths_are_attached(self, tmp_path):  # 12
        image = tmp_path / "shot.png"
        image.write_bytes(bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360606060000000050001a14a8ef90000000049454e44ae426082"
        ))
        url = "https://example.test/diagram.png"
        with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
            message = _build_child_goal_message("Compare", [str(image), url], _FakeChild())
        assert f"[Image attached at: {image}]" in message[0]["text"]
        assert f"[Image attached: {url}]" in message[0]["text"]
        assert any(part["image_url"]["url"].startswith("data:image/") for part in message[1:])
        assert any(part["image_url"]["url"] == url for part in message[1:])

    def test_vision_pipe_error_keeps_goal_and_hints(self):  # 13
        path = "/tmp/delegate-image.png"
        with patch("agent.image_routing.decide_image_input_mode", side_effect=RuntimeError("boom")):
            message = _build_child_goal_message("Inspect", [path], _FakeChild())
        assert isinstance(message, str)
        assert "Inspect" in message
        assert f"[Image attached at: {path}]" in message
        assert "vision_analyze" in message


class TestToolSchemaSurface:
    def test_per_task_images_schema(self):  # 14
        item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert item["properties"]["images"]["type"] == "array"
        assert item["properties"]["images"]["items"]["type"] == "string"
        assert item["properties"]["images"]["maxItems"] == _MAX_TASK_IMAGES
        assert "images" not in item["required"]


class TestSpawnPayload:
    def test_top_level_images_reach_single_child_first_turn(self):  # 15
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

    def test_batch_images_are_isolated_on_each_first_turn(self):  # 16
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

    def test_nine_images_rejected_before_child_spawn(self):  # 17
        with patch("tools.delegate_tool._build_child_preserving_parent_tools") as build:
            response = json.loads(delegate_task(
                tasks=[{"goal": "Inspect the screenshot", "images": ["x.png"] * 9}],
                parent_agent=_parent(),
            ))
        assert "Trim to the images the child actually needs to see." in response["error"]
        build.assert_not_called()

    def test_absent_images_leave_first_turn_unchanged(self):  # 18
        goal = "Review the text-only second task"
        with patch("tools.delegate_tool._build_child_goal_message") as make_message:
            response, children, build = _spawn(goal=goal)
        assert "error" not in response
        build.assert_called_once()
        assert children[0].run_conversation.call_args.kwargs["user_message"] == goal
        assert children[0].run_conversation.call_count == 1
        make_message.assert_not_called()
