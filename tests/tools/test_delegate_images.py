"""Per-task delegate_task image forwarding, adapted from upstream f3f5c4f7c7."""

import base64
import json
from types import SimpleNamespace
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


class TestNormalizeTaskImages:
    def test_absent_is_none(self):
        cleaned, err = _normalize_task_images({"goal": "g"}, 0)
        assert cleaned is None and err is None

    def test_list_passes_through_stripped(self):
        cleaned, err = _normalize_task_images(
            {"images": [" /tmp/a.png ", "https://x.test/b.jpg"]}, 0
        )
        assert err is None
        assert cleaned == ["/tmp/a.png", "https://x.test/b.jpg"]

    def test_bare_string_wrapped(self):
        cleaned, err = _normalize_task_images({"images": "/tmp/a.png"}, 0)
        assert err is None and cleaned == ["/tmp/a.png"]

    def test_non_list_rejected(self):
        cleaned, err = _normalize_task_images({"images": {"path": "x"}}, 2)
        assert cleaned is None and "Task 2" in err

    def test_non_string_entry_rejected(self):
        cleaned, err = _normalize_task_images({"images": ["/tmp/a.png", 42]}, 0)
        assert cleaned is None and err

    def test_empty_string_entry_rejected(self):
        cleaned, err = _normalize_task_images({"images": ["  "]}, 0)
        assert cleaned is None and err

    def test_over_limit_rejected(self):
        many = [f"/tmp/img{i}.png" for i in range(_MAX_TASK_IMAGES + 1)]
        cleaned, err = _normalize_task_images({"images": many}, 1)
        assert cleaned is None and str(_MAX_TASK_IMAGES) in err

    def test_empty_list_normalizes_to_none(self):
        cleaned, err = _normalize_task_images({"images": []}, 0)
        assert cleaned is None and err is None


def _make_png(tmp_path, name="shot.png"):
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBg"
        "AAAABQABh6FO1AAAAABJRU5ErkJggg=="
    )
    path = tmp_path / name
    path.write_bytes(png)
    return str(path)


class TestBuildChildGoalMessage:
    def test_native_mode_builds_content_parts(self, tmp_path):
        path = _make_png(tmp_path)
        with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
            msg = _build_child_goal_message("Inspect the mock", [path], _FakeChild())
        assert isinstance(msg, list)
        assert msg[0]["type"] == "text"
        assert "Inspect the mock" in msg[0]["text"]
        img = next(part for part in msg if part["type"] == "image_url")
        assert img["image_url"]["url"].startswith("data:image/")

    def test_native_mode_passes_urls_verbatim(self):
        url = "https://example.test/mock.png"
        with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
            msg = _build_child_goal_message("Look", [url], _FakeChild())
        img = next(part for part in msg if part["type"] == "image_url")
        assert img["image_url"]["url"] == url

    def test_native_mode_all_unreadable_falls_back_to_text(self, tmp_path):
        with patch("agent.image_routing.decide_image_input_mode", return_value="native"):
            msg = _build_child_goal_message("Goal text", [str(tmp_path / "nope.png")], _FakeChild())
        assert msg == "Goal text"

    def test_text_mode_appends_hints(self, tmp_path):
        path = _make_png(tmp_path)
        url = "https://example.test/a.png"
        with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
            msg = _build_child_goal_message("Goal", [path, url], _FakeChild())
        assert isinstance(msg, str)
        assert f"[Image attached at: {path}]" in msg
        assert f"[Image attached: {url}]" in msg
        assert "vision_analyze" in msg

    def test_text_mode_missing_path_dropped(self, tmp_path):
        with patch("agent.image_routing.decide_image_input_mode", return_value="text"):
            msg = _build_child_goal_message("Goal", [str(tmp_path / "gone.png")], _FakeChild())
        assert msg == "Goal"

    def test_any_exception_degrades_to_plain_goal(self):
        with patch("agent.image_routing.decide_image_input_mode", side_effect=RuntimeError("boom")):
            msg = _build_child_goal_message("Plain goal", ["/tmp/x.png"], _FakeChild())
        assert msg == "Plain goal"


class TestToolSchemaSurface:
    def test_images_on_task_items(self):
        item = DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        assert item["properties"]["images"]["type"] == "array"
        assert item["properties"]["images"]["items"]["type"] == "string"
        assert item["properties"]["images"]["maxItems"] == _MAX_TASK_IMAGES
        assert "images" not in item["required"]

    def test_images_not_a_top_level_schema_param(self):
        assert "images" not in DELEGATE_TASK_SCHEMA["parameters"]["properties"]


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


class TestSpawnPayload:
    def test_images_attached_per_task_and_absent_on_other_child(self):
        children = [SimpleNamespace(), SimpleNamespace()]
        result_entry = lambda idx: {
            "task_index": idx, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 0.0, "_child_role": None,
        }
        with (
            patch("tools.delegate_tool._build_child_agent", side_effect=children) as build,
            patch("tools.delegate_tool._run_single_child", side_effect=lambda i, *a, **kw: result_entry(i)),
            patch("tools.delegation_live_log.create_live_transcripts", return_value=(None, [], [])),
        ):
            response = json.loads(delegate_task(
                tasks=[
                    {"goal": "Inspect the first screenshot", "images": [" /tmp/mock.png "]},
                    {"goal": "Review the text-only second task"},
                ],
                parent_agent=_parent(),
            ))
        assert "error" not in response
        assert build.call_count == 2
        assert children[0]._delegate_images == ["/tmp/mock.png"]
        assert not hasattr(children[1], "_delegate_images")

    def test_invalid_image_batch_rejects_before_spawn(self):
        with patch("tools.delegate_tool._build_child_agent") as build:
            response = json.loads(delegate_task(
                tasks=[
                    {"goal": "Inspect the first screenshot"},
                    {"goal": "Review the second screenshot", "images": ["x"] * (_MAX_TASK_IMAGES + 1)},
                ],
                parent_agent=_parent(),
            ))
        assert "error" in response
        assert str(_MAX_TASK_IMAGES) in response["error"]
        build.assert_not_called()
