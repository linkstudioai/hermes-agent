"""
Tests for the HTTP-mode Slack interaction path.

When Aileron's slackrouter forwards a Slack interaction (button click,
modal submit, …) to a Hermes-fork instance, the body arrives as
``application/x-www-form-urlencoded`` with a single ``payload`` field.
``SlackAdapter._handle_http_webhook`` must parse that, dispatch into Bolt,
and translate Bolt's response back to aiohttp.

These tests require the real slack-bolt + aiohttp packages (run with
``uv run --extra slack pytest tests/gateway/test_slack_http_interactions.py``).
If slack-bolt is not installed, the module is skipped — the HTTP
interaction path can't be exercised against a MagicMock Bolt app.
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlencode

import pytest

# Ensure the repo root is importable
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# Skip the whole module if slack-bolt isn't actually installed.
slack_bolt = pytest.importorskip("slack_bolt")
pytest.importorskip("slack_bolt.async_app")
pytest.importorskip("slack_bolt.request.async_request")
pytest.importorskip("aiohttp")

from slack_bolt.async_app import AsyncApp  # noqa: E402

import gateway.platforms.slack as _slack_mod  # noqa: E402

_slack_mod.SLACK_AVAILABLE = True

from gateway.platforms.slack import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEBHOOK_SECRET = "test-webhook-secret"


def _sign(body: bytes) -> str:
    import hashlib
    import hmac

    return "sha256=" + hmac.new(
        WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()


class _FakeRequest:
    """Minimal aiohttp-like request object that ``_handle_http_webhook`` accepts."""

    def __init__(self, body: bytes, headers: dict):
        self._body = body
        self.headers = headers

    async def read(self) -> bytes:
        return self._body


def _build_adapter(monkeypatch) -> SlackAdapter:
    """Construct a SlackAdapter with a real AsyncApp and a fake bot token."""
    monkeypatch.setenv("HERMES_WEBHOOK_SECRET", WEBHOOK_SECRET)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

    cfg = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(cfg)
    # Real Bolt app — required so async_dispatch actually invokes handlers.
    # Match production HTTP-mode config: Slack's own signature middleware
    # is disabled because slackrouter validates X-Webhook-Signature for us.
    # A static ``authorize`` callback short-circuits the default behaviour
    # (which would call Slack's auth.test endpoint with our fake token).
    from slack_bolt.authorization import AuthorizeResult

    async def _authorize(client, *args, **kwargs):
        return AuthorizeResult(
            enterprise_id=None,
            team_id="T1",
            bot_token="xoxb-test-token",
            bot_id="B_BOT",
            bot_user_id="U_BOT",
        )

    adapter._app = AsyncApp(
        signing_secret="",
        request_verification_enabled=False,
        authorize=_authorize,
    )
    adapter._bot_user_id = "U_BOT"
    return adapter


def _interaction_body(action_id: str = "hermes_approve_once") -> bytes:
    """Build a Slack-shaped interaction body (form-encoded)."""
    payload = {
        "type": "block_actions",
        "api_app_id": "A12345",
        "team": {"id": "T1", "domain": "test"},
        "user": {"id": "U_USER", "name": "norbert", "team_id": "T1"},
        "channel": {"id": "C1"},
        "message": {"ts": "1234.5678", "blocks": []},
        "actions": [
            {
                "action_id": action_id,
                "value": "agent:main:slack:group:C1:1111",
                "type": "button",
            }
        ],
    }
    form = urlencode({"payload": json.dumps(payload)})
    return form.encode("utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHTTPInteractionDispatch:
    @pytest.mark.asyncio
    async def test_form_encoded_interaction_invokes_registered_handler(
        self, monkeypatch
    ):
        adapter = _build_adapter(monkeypatch)

        called = {}

        async def fake_handler(ack, body, action):
            called["action_id"] = action.get("action_id")
            called["session_key"] = action.get("value")
            called["channel_id"] = body.get("channel", {}).get("id")
            await ack()

        adapter._app.action("hermes_approve_once")(fake_handler)

        body = _interaction_body("hermes_approve_once")
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Webhook-Signature": _sign(body),
            },
        )

        resp = await adapter._handle_http_webhook(req)

        assert resp.status == 200
        assert called == {
            "action_id": "hermes_approve_once",
            "session_key": "agent:main:slack:group:C1:1111",
            "channel_id": "C1",
        }

    @pytest.mark.asyncio
    async def test_form_body_missing_payload_returns_400(self, monkeypatch):
        adapter = _build_adapter(monkeypatch)
        body = b"not_payload=oops"
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Webhook-Signature": _sign(body),
            },
        )

        resp = await adapter._handle_http_webhook(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_form_body_with_garbage_payload_returns_400(self, monkeypatch):
        adapter = _build_adapter(monkeypatch)
        body = urlencode({"payload": "{not-json"}).encode("utf-8")
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Webhook-Signature": _sign(body),
            },
        )

        resp = await adapter._handle_http_webhook(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_unhandled_action_id_does_not_500(self, monkeypatch):
        """Bolt should ack with 404 (its default) for unregistered action_ids,
        not crash. We accept any non-5xx so the test isn't tied to Bolt's
        exact "unhandled" status code."""
        adapter = _build_adapter(monkeypatch)

        body = _interaction_body("nonexistent_action")
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Webhook-Signature": _sign(body),
            },
        )

        resp = await adapter._handle_http_webhook(req)
        assert resp.status < 500

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected_before_dispatch(self, monkeypatch):
        adapter = _build_adapter(monkeypatch)
        # Patch dispatch to fail loudly if reached.
        adapter._app.async_dispatch = AsyncMock(
            side_effect=AssertionError("dispatch should not run for bad sig")
        )

        body = _interaction_body()
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Webhook-Signature": "sha256=deadbeef",
            },
        )

        resp = await adapter._handle_http_webhook(req)
        assert resp.status == 401
        adapter._app.async_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_event_path_still_works(self, monkeypatch):
        """Existing JSON event flow must continue to dispatch into the
        message pipeline (regression guard for the new content-type branch)."""
        adapter = _build_adapter(monkeypatch)
        adapter._handle_slack_message = AsyncMock()

        payload = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "channel": "C1",
                "user": "U_USER",
                "text": "hello",
                "ts": "1.0",
            },
        }
        body = json.dumps(payload).encode("utf-8")

        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": _sign(body),
            },
        )

        resp = await adapter._handle_http_webhook(req)
        assert resp.status == 200

        # Allow the background task to run.
        import asyncio

        for _ in range(5):
            await asyncio.sleep(0)

        adapter._handle_slack_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_url_verification_challenge_still_works(self, monkeypatch):
        adapter = _build_adapter(monkeypatch)
        body = json.dumps(
            {"type": "url_verification", "challenge": "xyz"}
        ).encode("utf-8")
        req = _FakeRequest(
            body,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": _sign(body),
            },
        )
        resp = await adapter._handle_http_webhook(req)
        assert resp.status == 200
        assert b"xyz" in resp.body
